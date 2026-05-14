import os
import json
from datetime import datetime
from collections import deque
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io
import zipfile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import sys
import pandas as pd
import math
import time
import shutil

def sanitize_for_json(obj):
    """
    Recursively sanitize objects to ensure they are strictly JSON serializable.
    Handles NaNs, Infinities, and converts unknown objects to strings.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return obj
    elif obj is None or isinstance(obj, (int, str, bool)):
        return obj
    else:
        # Fallback for complex objects that bypass standard typing mappings
        return str(obj)

# Ensure the parent directory is in the path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import run_dataforge_pipeline

app = FastAPI(title="DataForge API", description="API Gateway for DataForge pipeline")

@app.on_event("startup")
async def cleanup_old_runs():
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return
    now = time.time()
    cutoff = now - (7 * 24 * 3600)  # 7 days
    for run_folder in os.listdir(runs_dir):
        folder_path = os.path.join(runs_dir, run_folder)
        if os.path.isdir(folder_path):
            try:
                mtime = os.path.getmtime(folder_path)
                if mtime < cutoff:
                    shutil.rmtree(folder_path)
                    print(f"Auto-cleanup: Removed old run folder {run_folder}")
            except Exception as e:
                print(f"Auto-cleanup failed for {run_folder}: {e}")

# Mount the runs directory to serve static files (like plots)
if not os.path.exists("runs"):
    os.makedirs("runs", exist_ok=True)
app.mount('/runs', StaticFiles(directory='runs'), name='runs')

# Enable CORS for the Streamlit UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Simple health check endpoint to verify server status."""
    return {"status": "online", "version": "1.0.0"}

class PipelineRequest(BaseModel):
    file_path: str
    target_column: Optional[str] = None
    drop_columns: Optional[List[str]] = None

class InspectRequest(BaseModel):
    file_path: str

@app.post("/inspect")
async def inspect_file(req: InspectRequest):
    """
    Read the headers of the specified file and return them as a list.
    """
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {req.file_path}")
        
    try:
        # Use pandas to read just the headers (nrows=0)
        df = pd.read_csv(req.file_path, nrows=0)
        return {"columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to inspect file: {str(e)}")

@app.post("/start-pipeline")
async def start_pipeline(req: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Start the pipeline in the background and return the run_id immediately.
    """
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {req.file_path}")
        
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    # Let the background task execute the agents
    background_tasks.add_task(
        run_dataforge_pipeline,
        dataset_path=req.file_path,
        target_col=req.target_column,
        drop_columns=req.drop_columns,
        run_id=run_id
    )
    
    return {"run_id": run_id, "status": "started", "message": "Pipeline initiated in the background."}

@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """
    Read the summary.json for the specified run_id and return it to track agent completion.
    """
    summary_path = os.path.join("runs", run_id, "summary.json")
    if not os.path.exists(summary_path):
        # The run just started and hasn't finished, or it doesn't exist
        # Check if the run directory itself was created
        if os.path.exists(os.path.join("runs", run_id)):
            return {"run_id": run_id, "agent_status": {}, "status": "running"}
        raise HTTPException(status_code=404, detail="Run ID not found")
        
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            # Sanitize to ensure robust serialization and no NaN/complex object crashes
            summary = sanitize_for_json(summary)
            # Inject a 'status' field indicating we have the final summary
            summary["status"] = "completed"
            return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read summary: {str(e)}")

@app.get("/logs/{run_id}")
async def get_logs(run_id: str):
    """
    Stream the last 30 lines of pipeline.log for the specified run_id.
    """
    log_path = os.path.join("runs", run_id, "pipeline.log")
    if not os.path.exists(log_path):
        return {"run_id": run_id, "logs": [], "message": "Log file not yet initialized."}
        
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            # Read all lines and take the last 30
            # Note: collection.deque is faster for large files, but reading here directly 
            # is acceptable since the pipeline logs aren't massive.
            last_30_lines = deque(f, maxlen=30)
            return {"run_id": run_id, "logs": list(last_30_lines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")

@app.get("/plots/{run_id}")
async def get_plots(run_id: str):
    plots_dir = os.path.join("runs", run_id, "plots")
    if not os.path.exists(plots_dir):
        return {"plots": []}
    # Only return image files (png, jpg, jpeg, svg)
    files = [f for f in os.listdir(plots_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg"))]
    files.sort()
    return {"plots": files}

@app.get('/explanations/{run_id}')
async def get_explanations(run_id: str):
    path = os.path.join('runs', run_id, 'explanations.json')
    if not os.path.exists(path):
        return {'explanations': {}}
    with open(path, 'r', encoding='utf-8') as fh:
        return {'explanations': json.load(fh)}

@app.get('/guidance/{run_id}')
async def get_guidance(run_id: str):
    path = os.path.join('runs', run_id, 'guidance.json')
    if not os.path.exists(path):
        return {'guidance': []}
    with open(path, 'r', encoding='utf-8') as fh:
        return {'guidance': json.load(fh)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    dest_path = os.path.join("uploads", file.filename)
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"file_path": dest_path}

@app.get("/runs")
async def list_runs():
    runs_root = "runs"
    if not os.path.exists(runs_root):
        return {"runs": []}
    runs = []
    for name in sorted(os.listdir(runs_root), reverse=True):
        run_dir = os.path.join(runs_root, name)
        if os.path.isdir(run_dir):
            meta = {"run_id": name}
            summary_path = os.path.join(run_dir, "summary.json")
            if os.path.exists(summary_path):
                try:
                    import json
                    with open(summary_path) as fh:
                        s = json.load(fh)
                        meta["summary"] = s
                except Exception:
                    meta["summary"] = None
            runs.append(meta)
    return {"runs": runs}

MAX_ZIP_BYTES = 200 * 1024 * 1024  # 200 MB

@app.get("/download/{run_id}")
async def download_run_zip(run_id: str):
    run_dir = os.path.join("runs", run_id)
    if not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail="run_id not found")
    # compute total size
    total = 0
    for root, _, files in os.walk(run_dir):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
            if total > MAX_ZIP_BYTES:
                raise HTTPException(status_code=413, detail="Run artifacts too large to download (over 200MB)")
    # create in-memory zip
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(run_dir):
            for f in files:
                file_path = os.path.join(root, f)
                arcname = os.path.relpath(file_path, run_dir)
                zf.write(file_path, arcname)
    mem_zip.seek(0)
    filename = f"dataforge_run_{run_id}.zip"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return StreamingResponse(mem_zip, media_type="application/zip", headers=headers)
