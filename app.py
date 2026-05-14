# app.py - DataForge Brutalist Master Dashboard
import os, time, requests, json
import streamlit as st
import pandas as pd
import io

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="DataForge — Brutalist", layout="wide")

# Brutalist CSS
st.markdown(
    """
    <style>
    html, body, .stApp { background: #0A0A0A; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; }
    .block { border: none; padding: 12px; margin-bottom: 12px; background: #1C1C1E; border-radius: 8px; transition: all 0.3s ease-in-out; }
    .block:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    .timeline-agent { padding: 4px 0px; margin-bottom: 2px; display: flex; flex-direction: column; transition: all 0.3s ease-in-out; }
    .agent-name { font-weight:500; font-family: 'JetBrains Mono', monospace; font-size: 0.95em; }
    .agent-past { color: #8E8E93; }
    .agent-active { color: #FFFFFF; font-weight: 700; }
    .dot-active { color: #007AFF; animation: pulse-opacity 1.5s infinite ease-in-out; display: inline-block; font-size: 1.1em; }
    .agent-future { color: #555555; }
    .agent-desc { font-size: 0.75em; color: rgba(255, 255, 255, 0.6); margin-top: 2px; margin-left: 24px; word-break: break-all; font-family: 'JetBrains Mono', monospace; }
    @keyframes pulse-opacity { 0% { opacity: 0.3; } 50% { opacity: 1; } 100% { opacity: 0.3; } }
    .terminal { background: #1C1C1E; color:#D1D1D6; padding:16px; font-family: 'Courier New', monospace; border:none; border-radius: 12px; height:220px; overflow:auto; font-size: 0.8em; }
    .terminal p { margin: 0; line-height: 1.3; }
    div.stButton > button { background: #1C1C1E; color: #FFFFFF; border: none !important; box-shadow: none !important; transition: all 0.3s ease-in-out; border-radius: 8px; font-family: 'JetBrains Mono', monospace; }
    div.stButton > button:hover { background: #3A3A3C; color: #FFFFFF; transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True
)

def api_post(path, payload=None, files=None):
    if files:
        return requests.post(API_BASE + path, files=files, timeout=30)
    return requests.post(API_BASE + path, json=payload, timeout=30)

def api_get(path):
    return requests.get(API_BASE + path, timeout=30)

left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("DATAFORGE — Forge Input")
    tab = st.tabs(["CSV", "URL", "PARQUET"])
    file_path = None
    
    with tab[0]:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                up = api_post("/upload", files={"file": (uploaded.name, uploaded.getvalue())})
                if up.status_code == 200:
                    file_path = up.json().get("file_path")
            except Exception:
                local_path = os.path.join("uploads", uploaded.name)
                os.makedirs("uploads", exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                file_path = local_path

    with tab[1]:
        url = st.text_input("Dataset URL (http/https)", "")
        if st.button("Use URL"):
            file_path = url

    with tab[2]:
        uploaded_pq = st.file_uploader("Upload Parquet", type=["parquet"])
        if uploaded_pq is not None:
            try:
                up = api_post("/upload", files={"file": (uploaded_pq.name, uploaded_pq.getvalue())})
                if up.status_code == 200:
                    file_path = up.json().get("file_path")
            except Exception:
                local_path = os.path.join("uploads", uploaded_pq.name)
                os.makedirs("uploads", exist_ok=True)
                with open(local_path, "wb") as f:
                    f.write(uploaded_pq.getbuffer())
                file_path = local_path

    if file_path:
        st.write("Using:", file_path)
        try:
            cols = api_post("/inspect", {"file_path": file_path}).json().get("columns", [])
            target = st.selectbox("Select target column", ["--select--"] + cols)
        except Exception as exc:
            st.error("Inspect failed (ensure API is running at :8000)")
            st.write(exc)
            target = None
    else:
        st.info("Choose a dataset first.")
        target = None

    if st.button("COMMENCE PIPELINE") and file_path and target and target != "--select--":
        start_resp = api_post("/start-pipeline", {"file_path": file_path, "target_column": target, "drop_columns": []})
        if start_resp.status_code == 200:
            run_id = start_resp.json().get("run_id")
            st.session_state.run_id = run_id
            st.session_state['last_dataset'] = file_path
            st.session_state['last_target'] = target
            st.success(f"Started run: {run_id}")
        else:
            st.error("Failed to start pipeline: " + start_resp.text)

    st.markdown("### Past Runs")
    try:
        req = api_get("/runs")
        runs_resp = req.json() if req.status_code == 200 else {}
        runs = runs_resp.get("runs", []) if isinstance(runs_resp, dict) else []
        for r in runs[:5]:
            rid = r.get("run_id")
            c = st.columns([2, 1])
            c[0].write(f"`{rid}`")
            if c[1].button("Load", key=f"load_{rid}"):
                st.session_state.run_id = rid
                st.rerun()
    except Exception:
        pass

@st.cache_data
def load_csv_data(path):
    return pd.read_csv(path)

# Right Column - Real-time fragment (Timeline & Logs ONLY)
@st.fragment(run_every="2s")
def render_monoliths_and_logs(run_id):
    if not run_id:
        st.info("No run selected. Commence pipeline or load a past run.")
        return False
        
    try:
        status_resp = api_get(f"/status/{run_id}").json()
        agent_status = status_resp.get("agent_status", {})
        overall_status = status_resp.get("status", "running")
        warnings = status_resp.get("warnings", [])
    except Exception:
        agent_status = {}
        overall_status = "fetching..."
        warnings = []

    try:
        logs_resp = api_get(f"/logs/{run_id}").json()
        log_lines = logs_resp.get("logs", [])
        last_log = log_lines[-1].strip() if log_lines else "Initializing..."
        # Escape HTML chars to avoid rendering issues in the div
        safe_logs = [line.replace("<", "&lt;").replace(">", "&gt;") for line in log_lines]
        log_text = "<br>".join(safe_logs).replace("\\n", "")
    except Exception:
        log_text = "Awaiting log stream..."
        last_log = "..."

    all_agents = [
        "Data Ingestion", "Data Quality Audit", "Data Cleaning", 
        "EDA", "Feature Engineering", "Modeling", 
        "SHAP Interpretability", "Statistical Testing", "Recommendations"
    ]
    
    st.markdown(f"**Status:** `{overall_status.upper()}` | **Run ID:** `{run_id}`")
    
    if warnings:
        st.markdown("### ⚠️ AI Warnings & Insights")
        for w in warnings:
            st.warning(w)
            
    # AI Explanation Panel
    if run_id:
        try:
            exp_resp = requests.get(f'{API_BASE}/explanations/{run_id}', timeout=10)
            if exp_resp.status_code == 200:
                explanations = exp_resp.json().get('explanations', {})
                if explanations:
                    st.markdown('### 🧠 AI Explanation Engine')
                    agent_labels = {
                        'cleaning': '🧹 Data Cleaning',
                        'eda': '🔍 Exploratory Analysis',
                        'features': '⚙️ Feature Engineering',
                        'modeling': '🤖 Modeling Decision',
                        'warnings': '🚨 Critical Issues'
                    }
                    for key, label in agent_labels.items():
                        if key in explanations:
                            with st.expander(label, expanded=(key == 'modeling')):
                                st.markdown(explanations[key])
        except Exception as e:
            st.error('Could not load explanations: ' + str(e))
            
    # Mentor Mode Panel
    if run_id:
        try:
            mentor_resp = requests.get(f'{API_BASE}/guidance/{run_id}', timeout=10)
            if mentor_resp.status_code == 200:
                guidance = mentor_resp.json().get('guidance', [])
                if guidance:
                    st.markdown('### 🧭 Mentor Mode — What Should I Do Next?')
                    type_colors = {
                        'danger': st.error,
                        'warning': st.warning,
                        'insight': st.info,
                        'next_step': st.success,
                        'learning': st.info
                    }
                    for item in guidance:
                        display_fn = type_colors.get(item.get('type', 'insight'), st.info)
                        with st.expander(item.get('title', 'Guidance'), expanded=(item.get('type') == 'danger')):
                            st.markdown(item.get('message', ''))
                            st.markdown(f"**👉 Action:** {item.get('action', '')}")
        except Exception as e:
            st.error('Could not load mentor guidance: ' + str(e))

    st.markdown('---')
    st.markdown('### 🔁 Ready to Experiment?')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('🔄 Rerun with Same Settings'):
            # trigger new pipeline run with same dataset and target
            if st.session_state.get('last_dataset') and st.session_state.get('last_target'):
                resp = requests.post(f'{API_BASE}/start-pipeline', json={
                    'file_path': st.session_state['last_dataset'],
                    'target_column': st.session_state['last_target'],
                    'drop_columns': []
                })
                if resp.status_code == 200:
                    st.session_state['run_id'] = resp.json().get('run_id')
                    st.rerun()
    with col2:
        if st.button('🆕 Try a New Dataset'):
            st.session_state.clear()
            st.rerun()

    
    for name in all_agents:
        st_status = agent_status.get(name)
        if st_status == "running":
            cls = "agent-active"
            icon = '<span class="dot-active">●</span>'
            desc = f'<div class="agent-desc">{last_log.replace("<", "&lt;").replace(">", "&gt;")}</div>'
        elif st_status == "done":
            cls = "agent-past"
            icon = '<span style="color:#8E8E93;">●</span>'
            desc = ""
        else:
            cls = "agent-future"
            icon = "○"
            desc = ""
            
        html = f'<div class="timeline-agent {cls}"><div class="agent-name">{icon}&nbsp;&nbsp;{name}</div>{desc}</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("### Pipeline Stream")
    st.markdown(f'<div class="terminal">{log_text}</div>', unsafe_allow_html=True)

    # Automatically flag completion to the outer script via session state
    is_done = overall_status in ("completed", "failed")
    if is_done and not st.session_state.get(f"done_{run_id}"):
        st.session_state[f"done_{run_id}"] = True
        st.rerun()

    return is_done

def render_static_theater(run_id):
    st.info("No cleaned CSV available for editing. To enable interactive editing, pipeline must save cleaned dataset to runs/{run_id}/dataset_clean.csv")
    plots = api_get(f"/plots/{run_id}").json().get("plots", [])
    if plots:
        pcols = st.columns(3)
        for idx, p in enumerate(plots):
            url = f"{API_BASE}/runs/{run_id}/plots/{p}"
            with pcols[idx % 3]:
                st.image(url, use_container_width=True, caption=p)
    else:
        st.info("Waiting for plots...")

def render_interactive_theater(run_id, csv_path):
    df = load_csv_data(csv_path)
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.markdown("**Theater Controls**")
        x_col = st.selectbox("X Axis", df.columns)
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns)
        chart_type = st.selectbox("Chart Type", ["scatter", "line", "bar"])
        
        show_reg = False
        if chart_type == "scatter":
            show_reg = st.checkbox("Show regression line")
            
        if st.button("Render Chart", use_container_width=True):
            st.session_state[f"chart_x_{run_id}"] = x_col
            st.session_state[f"chart_y_{run_id}"] = y_col
            st.session_state[f"chart_type_{run_id}"] = chart_type
            st.session_state[f"chart_reg_{run_id}"] = show_reg
                
    with c2:
        cx = st.session_state.get(f"chart_x_{run_id}")
        cy = st.session_state.get(f"chart_y_{run_id}")
        ctype = st.session_state.get(f"chart_type_{run_id}")
        creg = st.session_state.get(f"chart_reg_{run_id}")
        
        if cx and cy and ctype:
            try:
                import plotly.express as px
            except ImportError:
                st.error("Plotly not installed. Run `pip install plotly`")
                return

            try:
                if ctype == "scatter":
                    # statsmodels required for trendline="ols"
                    fig = px.scatter(df, x=cx, y=cy, trendline="ols" if creg else None)
                elif ctype == "line":
                    fig = px.line(df, x=cx, y=cy)
                else:
                    fig = px.bar(df, x=cx, y=cy)
                    
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#0A0A0A", plot_bgcolor="#0A0A0A",
                    font=dict(family="monospace", color="#FFFFFF", size=12)
                )
                fig.update_traces(marker=dict(color="#FF4B2B"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown(f"**Current Render:** `{ctype.upper()}` mapping `{cx}` ➔ `{cy}`")
                
                # Exporters
                e1, e2 = st.columns(2)
                with e1:
                    csv_slice = df[[cx, cy]].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Data Slice (CSV)",
                        data=csv_slice,
                        file_name=f"slice_{cx}_vs_{cy}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with e2:
                    try:
                        png_bytes = fig.to_image(format="png", width=1280, height=720)
                        st.download_button(
                            label="Download Chart (PNG)",
                            data=png_bytes,
                            file_name=f"scatter_{cx}_vs_{cy}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.warning("PNG export requires 'kaleido'. Right-click chart to save manually.")
                        
            except Exception as e:
                st.error(f"Plot generation failed: {e}. Check if you selected valid data types or if 'statsmodels' is installed for regression.")
        else:
            st.info("Configure chart parameters on the left and click 'Render Chart'.")

with right_col:
    st.header("The Monoliths — Agents")
    run_id = st.session_state.get("run_id")
    
    # 1. Background Poller
    render_monoliths_and_logs(run_id)
    
    # 2. Archival & Interactive Theater (Only visible if run_id exists and run is completed)
    if run_id and st.session_state.get(f"done_{run_id}"):
        st.markdown("---")
        st.markdown("### Run Operations")
        
        # Download API Hook
        dl_res = api_get(f"/download/{run_id}")
        if dl_res.status_code == 200:
            st.download_button(
                label="⇓ DOWNLOAD RUN ARCHIVE (ZIP)", 
                data=dl_res.content, 
                file_name=f"dataforge_run_{run_id}.zip", 
                mime="application/zip",
                use_container_width=True
            )
        else:
            st.warning("Run archive ZIP not available from API.")
            
        # Toggle Theater
        theater_key = f"viz_active_{run_id}"
        btn_label = "Hide Interactive Theater" if st.session_state.get(theater_key) else "Show Interactive Visualizations"
        if st.button(btn_label, use_container_width=True):
            st.session_state[theater_key] = not st.session_state.get(theater_key, False)
            st.rerun()
            
        if st.session_state.get(theater_key):
            st.markdown("### Interactive In‑Viz Theater")
            try:
                duration = api_get(f"/status/{run_id}").json().get("duration_seconds")
                if duration:
                    st.caption(f"**Thinking Time:** {duration} seconds")
            except Exception:
                pass
            csv_path = os.path.join("runs", run_id, "dataset_clean.csv")
            if os.path.exists(csv_path):
                render_interactive_theater(run_id, csv_path)
            else:
                render_static_theater(run_id)
