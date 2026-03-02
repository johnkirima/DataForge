"""DataForge - Personal Data Science Automation Tool"""
import sys
import os

# Virtual environment check (must be first)
if sys.prefix == sys.base_prefix:
    print("⚠️  WARNING: Activate your virtual environment first.")
    print("Run: venv\\Scripts\\activate")
    sys.exit(1)

# Check for placeholder API keys BEFORE loading .env
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8') as f:
        env_content = f.read()
        if 'your_anthropic_key_here' in env_content or 'your_openai_key_here' in env_content or 'your_deepseek_key_here' in env_content:
            print("⚠️ ERROR: Placeholder API keys detected in .env file.")
            print("Please replace placeholder values with your actual API keys.")
            print("Edit .env and add your real keys before running DataForge.")
            sys.exit(1)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from pipeline_context import PipelineContext
from agents import run_data_ingestion, run_data_quality_audit, run_data_cleaning, run_eda, run_feature_engineering, run_modeling

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


def display_cleaning_results(ctx: PipelineContext) -> None:
    """Display the results of data cleaning."""
    print("\n" + "=" * 60)
    print("🧹 AGENT 3: DATA CLEANING RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("Data Cleaning")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (Previous agents failed or no data)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "cleaning" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        report = ctx.cleaning_report
        
        # Shape comparison
        orig = report.get('original_shape', {})
        clean = report.get('cleaned_shape', {})
        print(f"\n📐 Shape Comparison:")
        print(f"   - Original: {orig.get('rows', 0)} rows x {orig.get('columns', 0)} columns")
        print(f"   - Cleaned:  {clean.get('rows', 0)} rows x {clean.get('columns', 0)} columns")
        
        # Missing values imputation
        missing = report.get('missing_values', {})
        print(f"\n📉 Missing Values Imputed:")
        print(f"   - Total values imputed: {missing.get('total_values_imputed', 0)}")
        print(f"   - Columns affected: {len(missing.get('columns_imputed', []))}")
        if missing.get('columns_imputed'):
            cols_display = missing['columns_imputed'][:5]
            print(f"   - Columns: {cols_display}")
        
        # Duplicates removed
        dups = report.get('duplicates', {})
        print(f"\n📋 Duplicates Removed:")
        print(f"   - Rows removed: {dups.get('duplicates_removed', 0)}")
        
        # Outliers capped
        outliers = report.get('outliers', {})
        print(f"\n📈 Outliers Capped (IQR method):")
        print(f"   - Total values capped: {outliers.get('total_values_capped', 0)}")
        print(f"   - Columns affected: {len(outliers.get('columns_capped', []))}")
        if outliers.get('columns_capped'):
            cols_display = outliers['columns_capped'][:5]
            print(f"   - Columns: {cols_display}")
        
        # Type conversions
        types = report.get('type_conversions', {})
        print(f"\n🔄 Data Type Conversions:")
        print(f"   - Columns converted: {len(types.get('columns_converted', []))}")
        if types.get('columns_converted'):
            print(f"   - Columns: {types['columns_converted']}")
        
        # Inconsistencies fixed
        inconsist = report.get('inconsistencies', {})
        print(f"\n⚠️  Inconsistencies Fixed:")
        print(f"   - Total values fixed: {inconsist.get('total_values_fixed', 0)}")
        print(f"   - Columns affected: {len(inconsist.get('columns_fixed', []))}")
        if inconsist.get('details'):
            for col, info in list(inconsist['details'].items())[:3]:
                fixes = info.get('fixes_applied', [])
                print(f"   - {col}: {', '.join(fixes)}")
        
        # LLM Cleaning Narrative
        narrative = ctx.llm_narratives.get("cleaning", "")
        print(f"\n🤖 AI Cleaning Summary:")
        print("-" * 60)
        if narrative:
            display_text = narrative[:600] + "..." if len(narrative) > 600 else narrative
            print(display_text)
        else:
            print("No narrative generated.")
        print("-" * 60)
        
        # Show cleaned data preview
        if ctx.clean_df is not None:
            print(f"\n🔍 Cleaned Data Preview (first 3 rows):")
            print("-" * 60)
            print(ctx.clean_df.head(3).to_string(index=False))
            print("-" * 60)


def display_eda_results(ctx: PipelineContext) -> None:
    """Display the results of exploratory data analysis."""
    print("\n" + "=" * 60)
    print("📈 AGENT 4: EXPLORATORY DATA ANALYSIS RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("EDA")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (Previous agents failed or no data)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "eda" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        # Descriptive Statistics Summary
        desc_stats = ctx.eda_summary.get('descriptive_stats', {})
        numeric_stats = desc_stats.get('numeric', {})
        categorical_stats = desc_stats.get('categorical', {})
        
        print(f"\n📊 Descriptive Statistics:")
        print(f"   - Numeric columns analyzed: {len(numeric_stats)}")
        print(f"   - Categorical columns analyzed: {len(categorical_stats)}")
        
        # Show top 3 numeric column stats
        if numeric_stats:
            print(f"\n   Top numeric column stats:")
            for i, (col, stats) in enumerate(list(numeric_stats.items())[:3]):
                skew = stats.get('skewness', 0)
                skew_label = "normal" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
                print(f"   - {col}: mean={stats['mean']:.2f}, median={stats['median']:.2f}, std={stats['std']:.2f}, skew={skew_label}")
        
        # Correlation Analysis
        correlations = ctx.eda_summary.get('correlations', {})
        print(f"\n🔗 Correlation Analysis:")
        
        top_positive = correlations.get('top_positive', [])
        top_negative = correlations.get('top_negative', [])
        
        if top_positive:
            print(f"   Top Positive Correlations:")
            for col1, col2, corr in top_positive[:3]:
                print(f"   - {col1} ↔ {col2}: {corr:.3f}")
        else:
            print(f"   - No strong positive correlations found")
        
        if top_negative:
            print(f"   Top Negative Correlations:")
            for col1, col2, corr in top_negative[:3]:
                print(f"   - {col1} ↔ {col2}: {corr:.3f}")
        else:
            print(f"   - No strong negative correlations found")
        
        # Target Analysis (if available)
        target_analysis = ctx.eda_summary.get('target_analysis')
        if target_analysis and 'error' not in target_analysis:
            print(f"\n🎯 Target Variable Analysis:")
            print(f"   - Column: {target_analysis.get('column', 'N/A')}")
            print(f"   - Task Type: {target_analysis.get('task_type', 'N/A')}")
            
            if target_analysis.get('task_type') == 'classification':
                print(f"   - Classes: {target_analysis.get('num_classes', 'N/A')}")
                print(f"   - Balance: {target_analysis.get('balance_status', 'N/A')}")
                class_dist = target_analysis.get('class_distribution', {})
                print(f"   - Distribution: {class_dist}")
            else:
                print(f"   - Mean: {target_analysis.get('mean', 0):.2f}")
                print(f"   - Std: {target_analysis.get('std', 0):.2f}")
        
        # Plots Generated
        plots = ctx.plots
        print(f"\n📊 Plots Generated: {len(plots)}")
        if plots:
            print(f"   Files (first 5):")
            import os
            for plot_path in plots[:5]:
                filename = os.path.basename(plot_path)
                print(f"   - {filename}")
            if len(plots) > 5:
                print(f"   ... and {len(plots) - 5} more")
        
        # LLM EDA Narrative
        narrative = ctx.llm_narratives.get("eda", "")
        print(f"\n🤖 AI EDA Insights:")
        print("-" * 60)
        if narrative:
            display_text = narrative[:700] + "..." if len(narrative) > 700 else narrative
            print(display_text)
        else:
            print("No narrative generated.")
        print("-" * 60)


def display_feature_engineering_results(ctx: PipelineContext) -> None:
    """Display the results of feature engineering."""
    print("\n" + "=" * 60)
    print("🔧 AGENT 5: FEATURE ENGINEERING RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("Feature Engineering")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (Previous agents failed or no data)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "feature" in error.lower() or "engineering" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        fe_report = ctx.eda_summary.get("feature_engineering", {})
        
        # Feature counts
        print(f"\n📊 Feature Summary:")
        print(f"   - Original columns: {fe_report.get('original_columns', 0)}")
        print(f"   - New columns: {fe_report.get('new_columns', 0)}")
        print(f"   - Features created: {fe_report.get('features_created', 0)}")
        
        # Transformations applied
        transformations = fe_report.get('transformations_applied', {})
        print(f"\n🔄 Transformations Applied:")
        
        log_transforms = transformations.get('log_transforms', [])
        if log_transforms:
            print(f"   - Log transforms: {log_transforms[:5]}{'...' if len(log_transforms) > 5 else ''}")
        
        polynomial = transformations.get('polynomial', [])
        if polynomial:
            print(f"   - Polynomial features: {polynomial[:5]}{'...' if len(polynomial) > 5 else ''}")
        
        interactions = transformations.get('interactions', [])
        if interactions:
            print(f"   - Interaction features: {interactions[:3]}{'...' if len(interactions) > 3 else ''}")
        
        binning = transformations.get('binning', [])
        if binning:
            print(f"   - Binned features: {binning[:3]}{'...' if len(binning) > 3 else ''}")
        
        # New feature names
        new_features = fe_report.get('new_feature_names', [])
        print(f"\n📋 New Feature Names (first 10):")
        for i, feat in enumerate(new_features[:10]):
            print(f"   {i+1}. {feat}")
        if len(new_features) > 10:
            print(f"   ... and {len(new_features) - 10} more")
        
        # LLM Feature Engineering Narrative
        narrative = ctx.llm_narratives.get("feature_engineering", "")
        print(f"\n🤖 AI Feature Engineering Summary:")
        print("-" * 60)
        if narrative:
            display_text = narrative[:600] + "..." if len(narrative) > 600 else narrative
            print(display_text)
        else:
            print("No narrative generated.")
        print("-" * 60)
        
        # Updated dataframe preview
        if ctx.clean_df is not None:
            print(f"\n📐 Updated DataFrame Shape: {ctx.clean_df.shape[0]} rows x {ctx.clean_df.shape[1]} columns")


def display_modeling_results(ctx: PipelineContext) -> None:
    """Display the results of modeling."""
    print("\n" + "=" * 60)
    print("🤖 AGENT 6: MODELING RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("Modeling")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (No target variable or previous agents failed)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "modeling" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        results = ctx.model_results
        
        # Task Type and Data Split
        print(f"\n📊 Model Summary:")
        print(f"   - Task Type: {results.get('task_type', 'N/A')}")
        print(f"   - Features after encoding: {results.get('feature_count', 0)}")
        print(f"   - Train set size: {results.get('train_size', 0)}")
        print(f"   - Test set size: {results.get('test_size', 0)}")
        
        # Hyperparameters
        hyperparams = results.get('hyperparameters', {})
        print(f"\n⚙️  Random Forest Hyperparameters:")
        print(f"   - n_estimators: {hyperparams.get('n_estimators', 'N/A')}")
        print(f"   - max_depth: {hyperparams.get('max_depth', 'N/A')}")
        print(f"   - min_samples_leaf: {hyperparams.get('min_samples_leaf', 'N/A')}")
        print(f"   - random_state: {hyperparams.get('random_state', 'N/A')}")
        
        # Metrics
        metrics = results.get('metrics', {})
        print(f"\n📈 Evaluation Metrics:")
        
        if results.get('task_type') == 'classification':
            print(f"   - Train Accuracy: {metrics.get('train_accuracy', 'N/A')}")
            print(f"   - Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
            print(f"   - Precision: {metrics.get('precision', 'N/A')}")
            print(f"   - Recall: {metrics.get('recall', 'N/A')}")
            print(f"   - F1 Score: {metrics.get('f1_score', 'N/A')}")
            
            # Confusion Matrix
            conf_matrix = results.get('confusion_matrix')
            if conf_matrix:
                print(f"\n   Confusion Matrix:")
                for row in conf_matrix:
                    print(f"   {row}")
        else:
            print(f"   - Train RMSE: {metrics.get('train_rmse', 'N/A')}")
            print(f"   - Test RMSE: {metrics.get('test_rmse', 'N/A')}")
            print(f"   - Test MAE: {metrics.get('test_mae', 'N/A')}")
            print(f"   - Test R² Score: {metrics.get('test_r2', 'N/A')}")
            if 'test_mape' in metrics:
                print(f"   - Test MAPE: {metrics.get('test_mape')}%")
        
        # Feature Importance
        importance = results.get('feature_importance', [])
        print(f"\n🔝 Top 5 Feature Importances:")
        for i, feat in enumerate(importance[:5]):
            print(f"   {i+1}. {feat['feature']}: {feat['importance']:.4f}")


def display_agent_status(ctx: PipelineContext) -> None:
    """Display overall agent status."""
    print("\n" + "=" * 60)
    print("📊 AGENT STATUS SUMMARY")
    print("=" * 60)
    for agent, status in ctx.agent_status.items():
        icon = "✅" if status == "done" else "❌" if status == "failed" else "⏭️" if status == "skipped" else "🔄"
        print(f"   {icon} {agent}: {status}")


def main():
    """Main entry point - Agent 1 through Agent 6 Demo."""
    print(BANNER)
    print("🚀 Phase 2 - Data Ingestion + Quality Audit + Data Cleaning + EDA + Feature Engineering + Modeling Demo")
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
    
    # === Agent 3: Data Cleaning ===
    if ctx.agent_status.get("Data Ingestion") == "done":
        print("\n🔄 Running Agent 3: Data Cleaning...")
        ctx = run_data_cleaning(ctx)
        display_cleaning_results(ctx)
    else:
        print("\n⏭️  Skipping Agent 3: Previous agents failed")
        ctx.mark_agent("Data Cleaning", "skipped")
    
    # === Agent 4: Exploratory Data Analysis ===
    if ctx.agent_status.get("Data Cleaning") == "done":
        print("\n🔄 Running Agent 4: Exploratory Data Analysis...")
        ctx = run_eda(ctx)
        display_eda_results(ctx)
    else:
        print("\n⏭️  Skipping Agent 4: Previous agents failed")
        ctx.mark_agent("EDA", "skipped")
    
    # === Agent 5: Feature Engineering ===
    if ctx.agent_status.get("EDA") == "done":
        print("\n🔄 Running Agent 5: Feature Engineering...")
        ctx = run_feature_engineering(ctx)
        display_feature_engineering_results(ctx)
    else:
        print("\n⏭️  Skipping Agent 5: Previous agents failed")
        ctx.mark_agent("Feature Engineering", "skipped")
    
    # === Agent 6: Modeling ===
    if ctx.agent_status.get("Feature Engineering") == "done" and ctx.has_target:
        print("\n🔄 Running Agent 6: Modeling...")
        ctx = run_modeling(ctx)
        display_modeling_results(ctx)
    else:
        if not ctx.has_target:
            print("\n⏭️  Skipping Agent 6: No target variable defined")
        else:
            print("\n⏭️  Skipping Agent 6: Previous agents failed")
        ctx.mark_agent("Modeling", "skipped")
    
    # Display overall status
    display_agent_status(ctx)
    
    print("\n" + "=" * 60)
    print("Demo complete. Full pipeline coming in later phases.")
    print("=" * 60)


if __name__ == "__main__":
    main()
