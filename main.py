"""DataForge - Personal Data Science Automation Tool"""
import sys

# Virtual environment check (must be first)
if sys.prefix == sys.base_prefix:
    print("⚠️  WARNING: Activate your virtual environment first.")
    print("Run: venv\\Scripts\\activate")
    sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pipeline_context import PipelineContext
from agents import run_data_ingestion

# ASCII Banner
BANNER = r"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   ██████╗  █████╗ ████████╗ █████╗ ███████╗ ██████╗ ██████╗ ███████╗  ║
║   ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝  ║
║   ██║  ██║███████║   ██║   ███████║█████╗  ██║   ██║██████╔╝█████╗    ║
║   ██║  ██║██╔══██║   ██║   ██╔══██║██╔══╝  ██║   ██║██╔══██╗██╔══╝    ║
║   ██████╔╝██║  ██║   ██║   ██║  ██║██║     ╚██████╔╝██║  ██║███████╗  ║
║   ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝  ║
║                                                           ║
║          Personal Data Science Automation Tool            ║
║                        Phase 2                            ║
╚═══════════════════════════════════════════════════════════╝
"""


def display_results(ctx: PipelineContext) -> None:
    """Display the results of data ingestion."""
    print("\n" + "=" * 60)
    print("📊 DATA INGESTION RESULTS")
    print("=" * 60)
    
    if ctx.agent_status.get("Data Ingestion") == "done":
        print(f"\n✅ Status: SUCCESS")
        print(f"📁 Dataset Name: {ctx.dataset_name}")
        print(f"📐 Shape: {ctx.raw_df.shape[0]} rows x {ctx.raw_df.shape[1]} columns")
        print(f"\n📋 Columns: {list(ctx.raw_df.columns)}")
        print(f"\n🔍 First 3 rows:")
        print("-" * 60)
        print(ctx.raw_df.head(3).to_string(index=False))
        print("-" * 60)
        print(f"\n🤖 Agent Status: {ctx.agent_status}")
    else:
        print(f"\n❌ Status: FAILED")
        print(f"🤖 Agent Status: {ctx.agent_status}")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                print(f"   - {error}")


def main():
    """Main entry point - Agent 1 Demo."""
    print(BANNER)
    print("🚀 Phase 2 - Agent 1: Data Ingestion Demo")
    print("=" * 60)
    print("\nSupported formats: CSV, Excel (.xlsx, .xls), Parquet (.parquet, .pq), URL")
    print("Sample data available: uploads/sample_data.csv")
    print("-" * 60)
    
    # Prompt for dataset path
    dataset_path = input("\n📂 Enter dataset path (CSV/Excel/Parquet/URL): ").strip()
    
    if not dataset_path:
        print("❌ No path provided. Exiting.")
        return
    
    # Create pipeline context
    ctx = PipelineContext(dataset_path=dataset_path)
    
    # Run data ingestion agent
    print("\n🔄 Running Data Ingestion Agent...")
    ctx = run_data_ingestion(ctx)
    
    # Display results
    display_results(ctx)
    
    print("\n" + "=" * 60)
    print("Demo complete. Full pipeline coming in later phases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
