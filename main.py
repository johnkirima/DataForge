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
from agents import run_data_ingestion, run_data_quality_audit

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


def display_ingestion_results(ctx: PipelineContext) -> None:
    """Display the results of data ingestion."""
    print("\n" + "=" * 60)
    print("📊 AGENT 1: DATA INGESTION RESULTS")
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
    else:
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                print(f"   - {error}")


def display_quality_audit_results(ctx: PipelineContext) -> None:
    """Display the results of data quality audit."""
    print("\n" + "=" * 60)
    print("🔍 AGENT 2: DATA QUALITY AUDIT RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("Data Quality Audit")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (Agent 1 failed or no data)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        audit = ctx.eda_summary.get("quality_audit", {})
        
        # Missing values
        missing = audit.get("missing_values", {})
        print(f"\n📉 Missing Values:")
        print(f"   - Columns with missing: {missing.get('columns_with_missing', 0)}")
        print(f"   - Total missing cells: {missing.get('total_missing_cells', 0)}")
        if missing.get('details'):
            print(f"   - Details: {dict(list(missing['details'].items())[:5])}")
        
        # Duplicates
        dups = audit.get("duplicates", {})
        print(f"\n📋 Duplicates:")
        print(f"   - Duplicate rows: {dups.get('count', 0)} ({dups.get('percentage', 0)}%)")
        
        # Data types
        types = audit.get("data_types", {}).get("summary", {})
        print(f"\n📊 Data Types:")
        print(f"   - Numeric columns: {types.get('numeric_count', 0)}")
        print(f"   - Categorical columns: {types.get('categorical_count', 0)}")
        
        # Outliers
        outliers = audit.get("outliers", {})
        print(f"\n📈 Outliers (IQR method):")
        print(f"   - Columns with outliers: {outliers.get('columns_with_outliers', 0)}")
        if outliers.get('details'):
            outlier_cols = list(outliers['details'].keys())[:5]
            print(f"   - Affected columns: {outlier_cols}")
        
        # Inconsistencies
        inconsist = audit.get("inconsistencies", {})
        print(f"\n⚠️  Inconsistencies:")
        print(f"   - Columns with issues: {inconsist.get('columns_with_issues', 0)}")
        if inconsist.get('details'):
            for col, info in list(inconsist['details'].items())[:3]:
                print(f"   - {col}: {info.get('issue', 'unknown')}")
        
        # LLM Narrative
        narrative = ctx.llm_narratives.get("quality_audit", "")
        print(f"\n🤖 AI Quality Analysis:")
        print("-" * 60)
        if narrative:
            # Show first 500 chars or full if shorter
            display_text = narrative[:500] + "..." if len(narrative) > 500 else narrative
            print(display_text)
        else:
            print("No narrative generated.")
        print("-" * 60)


def display_agent_status(ctx: PipelineContext) -> None:
    """Display overall agent status."""
    print("\n" + "=" * 60)
    print("📊 AGENT STATUS SUMMARY")
    print("=" * 60)
    for agent, status in ctx.agent_status.items():
        icon = "✅" if status == "done" else "❌" if status == "failed" else "⏭️" if status == "skipped" else "🔄"
        print(f"   {icon} {agent}: {status}")


def main():
    """Main entry point - Agent 1 + Agent 2 Demo."""
    print(BANNER)
    print("🚀 Phase 2 - Data Ingestion + Quality Audit Demo")
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
    
    # === Agent 1: Data Ingestion ===
    print("\n🔄 Running Agent 1: Data Ingestion...")
    ctx = run_data_ingestion(ctx)
    display_ingestion_results(ctx)
    
    # === Agent 2: Data Quality Audit ===
    if ctx.agent_status.get("Data Ingestion") == "done":
        print("\n🔄 Running Agent 2: Data Quality Audit...")
        ctx = run_data_quality_audit(ctx)
        display_quality_audit_results(ctx)
    else:
        print("\n⏭️  Skipping Agent 2: Data Ingestion failed")
        ctx.mark_agent("Data Quality Audit", "skipped")
    
    # Display overall status
    display_agent_status(ctx)
    
    print("\n" + "=" * 60)
    print("Demo complete. Full pipeline coming in later phases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
