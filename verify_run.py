from main import run_dataforge_pipeline
import os, json, glob

# 1. Run a test pipeline
print("🚀 Starting Test Run...")
ctx = run_dataforge_pipeline('uploads/sample_for_tests.csv', 'Monetary_Impact', drop_columns=[])

# 2. Verify Folder Structure
run_dir = ctx.run_dir
print(f"\n📂 Run Directory Created: {run_dir}")

checks = {
    "Summary JSON": os.path.exists(os.path.join(run_dir, 'summary.json')),
    "Metadata JSON": os.path.exists(os.path.join(run_dir, 'metadata.json')),
    "Pipeline Log": os.path.exists(os.path.join(run_dir, 'pipeline.log')),
    "Plots Folder": os.path.exists(os.path.join(run_dir, 'plots'))
}

for name, exists in checks.items():
    status = "✅" if exists else "❌"
    print(f"{status} {name}")

# 3. Check for actual plots
plots = os.listdir(os.path.join(run_dir, 'plots'))
print(f"📊 Plots Generated: {len(plots)} files found.")

# 4. Final Success Check
if all(checks.values()):
    print("\n✨ MULTI-RUN STORAGE IS ACTIVE AND VERIFIED!")
else:
    print("\n⚠️ Some files are missing. Check the logs.")