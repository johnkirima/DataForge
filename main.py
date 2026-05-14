"""DataForge - Personal Data Science Automation Tool"""
import sys
import os
import json

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
from agents import (
    run_data_ingestion, run_data_quality_audit, run_data_cleaning, 
    run_eda, run_feature_engineering, run_modeling, run_shap_interpretability,
    run_statistical_testing, run_recommendations
)
from logger import get_logger

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
    """Display the results of modeling with all metrics rounded to 4 decimal places."""
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
        if hyperparams.get('class_weight'):
            print(f"   - class_weight: {hyperparams.get('class_weight')}")
        
        # Metrics
        metrics = results.get('metrics', {})
        print(f"\n📈 Evaluation Metrics:")
        
        if results.get('task_type') == 'classification':
            print(f"   - Train Accuracy: {metrics.get('train_accuracy', 'N/A'):.4f}" if isinstance(metrics.get('train_accuracy'), (int, float)) else f"   - Train Accuracy: {metrics.get('train_accuracy', 'N/A')}")
            print(f"   - Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}" if isinstance(metrics.get('test_accuracy'), (int, float)) else f"   - Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
            print(f"   - Precision: {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), (int, float)) else f"   - Precision: {metrics.get('precision', 'N/A')}")
            print(f"   - Recall: {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), (int, float)) else f"   - Recall: {metrics.get('recall', 'N/A')}")
            print(f"   - F1 Score: {metrics.get('f1_score', 'N/A'):.4f}" if isinstance(metrics.get('f1_score'), (int, float)) else f"   - F1 Score: {metrics.get('f1_score', 'N/A')}")
            
            # Cross-Validation
            if 'cv_mean' in metrics and metrics['cv_mean'] is not None:
                print(f"\n📊 Cross-Validation (5-fold):")
                print(f"   - CV Accuracy: {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}")
            
            # Confusion Matrix
            conf_matrix = results.get('confusion_matrix')
            if conf_matrix:
                print(f"\n   Confusion Matrix:")
                for row in conf_matrix:
                    print(f"   {row}")
        else:
            print(f"   - Train RMSE: {metrics.get('train_rmse', 'N/A'):.4f}" if isinstance(metrics.get('train_rmse'), (int, float)) else f"   - Train RMSE: {metrics.get('train_rmse', 'N/A')}")
            print(f"   - Test RMSE: {metrics.get('test_rmse', 'N/A'):.4f}" if isinstance(metrics.get('test_rmse'), (int, float)) else f"   - Test RMSE: {metrics.get('test_rmse', 'N/A')}")
            print(f"   - Test MAE: {metrics.get('test_mae', 'N/A'):.4f}" if isinstance(metrics.get('test_mae'), (int, float)) else f"   - Test MAE: {metrics.get('test_mae', 'N/A')}")
            print(f"   - Test R² Score: {metrics.get('test_r2', 'N/A'):.4f}" if isinstance(metrics.get('test_r2'), (int, float)) else f"   - Test R² Score: {metrics.get('test_r2', 'N/A')}")
            if 'test_mape' in metrics:
                print(f"   - Test MAPE: {metrics.get('test_mape'):.4f}%" if isinstance(metrics.get('test_mape'), (int, float)) else f"   - Test MAPE: {metrics.get('test_mape')}%")
            
            # Cross-Validation
            if 'cv_mean' in metrics and metrics['cv_mean'] is not None:
                print(f"\n📊 Cross-Validation (5-fold):")
                print(f"   - CV R² Score: {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}")
        
        # Feature Importance
        importance = results.get('feature_importance', [])
        print(f"\n🔝 Top 5 Feature Importances:")
        for i, feat in enumerate(importance[:5]):
            print(f"   {i+1}. {feat['feature']}: {feat['importance']:.4f}")


def display_shap_results(ctx: PipelineContext) -> None:
    """Display the results of SHAP interpretability analysis."""
    print("\n" + "=" * 60)
    print("🔬 AGENT 7: SHAP INTERPRETABILITY RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("SHAP Interpretability")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (No model available or previous agents failed)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "shap" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        shap_results = ctx.shap_results
        
        # Method used
        method = shap_results.get('method', 'N/A')
        print(f"\n📊 Analysis Method:")
        if method == "shap":
            print(f"   - SHAP TreeExplainer (full SHAP values computed)")
        else:
            print(f"   - Fallback: model.feature_importances_ (SHAP unavailable)")
        
        # Top features
        top_features = shap_results.get('top_features', [])
        print(f"\n🔝 Top 10 Feature Importances ({method.upper()}):")
        for i, feat in enumerate(top_features[:10]):
            print(f"   {i+1}. {feat['feature']}: {feat['importance']:.4f}")
        
        # Plots generated
        plots = shap_results.get('plots', [])
        print(f"\n📊 SHAP Plots Generated: {len(plots)}")
        if plots:
            for plot_path in plots:
                filename = os.path.basename(plot_path)
                print(f"   - {filename}")


def display_statistical_testing_results(ctx: PipelineContext) -> None:
    """Display the results of statistical testing."""
    print("\n" + "=" * 60)
    print("📊 AGENT 8: STATISTICAL TESTING RESULTS")
    print("=" * 60)
    
    status = ctx.agent_status.get("Statistical Testing")
    
    if status == "skipped":
        reason = ctx.statistical_results.get("skipped_reason", "Previous agents failed or regression task")
        print(f"\n⏭️  Status: SKIPPED ({reason})")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "statistical" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        stat_results = ctx.statistical_results
        
        # Tests performed
        tests_performed = stat_results.get("tests_performed", [])
        print(f"\n🔬 Tests Performed:")
        for test in tests_performed:
            print(f"   - {test['type']}: {test['count']} tests")
        
        print(f"\n📈 Summary:")
        print(f"   - Total tests: {stat_results.get('total_tests', 0)}")
        print(f"   - Significant features (p<0.05): {stat_results.get('significant_count', 0)}")
        
        # Significant features
        significant = stat_results.get("significant_features", [])
        if significant:
            print(f"\n✨ Significant Features:")
            for i, feat in enumerate(significant[:10]):
                print(f"   {i+1}. {feat}")
            if len(significant) > 10:
                print(f"   ... and {len(significant) - 10} more")
        
        # Test details (show top 5 from each test type)
        test_details = stat_results.get("test_details", {})
        
        ttest = test_details.get("ttest", [])
        if ttest:
            print(f"\n📋 T-Test Results (top 5):")
            for r in sorted(ttest, key=lambda x: x.get('p_value', 1))[:5]:
                sig = "✓" if r.get('significant') else ""
                print(f"   - {r['feature']}: t={r['statistic']}, p={r['p_value']:.6f} {sig}")
        
        anova = test_details.get("anova", [])
        if anova:
            print(f"\n📋 ANOVA Results (top 5):")
            for r in sorted(anova, key=lambda x: x.get('p_value', 1))[:5]:
                sig = "✓" if r.get('significant') else ""
                print(f"   - {r['feature']}: F={r['statistic']}, p={r['p_value']:.6f} {sig}")
        
        chi_square = test_details.get("chi_square", [])
        if chi_square:
            print(f"\n📋 Chi-Square Results (top 5):")
            for r in sorted(chi_square, key=lambda x: x.get('p_value', 1))[:5]:
                sig = "✓" if r.get('significant') else ""
                print(f"   - {r['feature']}: χ²={r['statistic']}, p={r['p_value']:.6f} {sig}")
        
        # DeepSeek Narrative
        narrative = ctx.llm_narratives.get("statistical_testing", "")
        print(f"\n🤖 AI Statistical Analysis (DeepSeek):")
        print("-" * 60)
        if narrative:
            display_text = narrative[:700] + "..." if len(narrative) > 700 else narrative
            print(display_text)
        else:
            print("No narrative generated.")
        print("-" * 60)


def display_recommendations_results(ctx: PipelineContext) -> None:
    """Display the final recommendations."""
    print("\n" + "=" * 60)
    print("🎯 AGENT 9: RECOMMENDATIONS")
    print("=" * 60)
    
    status = ctx.agent_status.get("Recommendations")
    
    if status == "skipped":
        print(f"\n⏭️  Status: SKIPPED (No data available)")
        return
    
    if status == "failed":
        print(f"\n❌ Status: FAILED")
        if ctx.errors:
            print(f"\n⚠️  Errors:")
            for error in ctx.errors:
                if "recommendations" in error.lower():
                    print(f"   - {error}")
        return
    
    if status == "done":
        print(f"\n✅ Status: SUCCESS")
        
        recs = ctx.recommendations
        
        # Overall Assessment
        assessment = recs.get("overall_assessment", "")
        print(f"\n📋 Overall Assessment:")
        print("-" * 60)
        print(assessment if assessment else "Not available")
        print("-" * 60)
        
        # Strengths
        strengths = recs.get("strengths", [])
        if strengths:
            print(f"\n💪 Key Strengths:")
            for i, s in enumerate(strengths):
                print(f"   {i+1}. {s}")
        
        # Areas for Improvement
        improvements = recs.get("improvements", [])
        if improvements:
            print(f"\n🔧 Areas for Improvement:")
            for i, imp in enumerate(improvements):
                print(f"   {i+1}. {imp}")
        
        # Next Steps
        next_steps = recs.get("next_steps", [])
        if next_steps:
            print(f"\n📌 Actionable Next Steps:")
            for i, step in enumerate(next_steps):
                print(f"   {i+1}. {step}")
        
        # Deployment Readiness
        deployment_ready = recs.get("deployment_ready", False)
        deployment_icon = "✅" if deployment_ready else "⚠️"
        deployment_text = "YES" if deployment_ready else "NO"
        print(f"\n🚀 Deployment Readiness: {deployment_icon} {deployment_text}")
        reasoning = recs.get("deployment_reasoning", "")
        if reasoning:
            print(f"   {reasoning[:300]}")
        
        # Full Claude Narrative
        narrative = ctx.llm_narratives.get("recommendations", "")
        print(f"\n🤖 Full AI Recommendations (Claude Sonnet 4.6):")
        print("-" * 60)
        if narrative:
            display_text = narrative[:1000] + "..." if len(narrative) > 1000 else narrative
            print(display_text)
        else:
            print("No narrative generated.")
        print("-" * 60)


def display_agent_status(ctx: PipelineContext) -> None:
    """Display overall agent status."""
    print("\n" + "=" * 60)
    print("📊 AGENT STATUS SUMMARY (All 9 Agents)")
    print("=" * 60)
    
    # Define agent order for display
    agent_order = [
        "Data Ingestion",
        "Data Quality Audit", 
        "Data Cleaning",
        "EDA",
        "Feature Engineering",
        "Modeling",
        "SHAP Interpretability",
        "Statistical Testing",
        "Recommendations"
    ]
    
    for agent in agent_order:
        status = ctx.agent_status.get(agent, "not run")
        icon = "✅" if status == "done" else "❌" if status == "failed" else "⏭️" if status == "skipped" else "🔄"
        print(f"   {icon} {agent}: {status}")


def safe_run(agent_fn, ctx: PipelineContext, display_name: str) -> PipelineContext:
    """Safely run an agent, buffering START/DONE/ERROR into ctx.agent_logs."""
    ctx.append_log(f"START: {display_name}")
    try:
        result = agent_fn(ctx)
        ctx.append_log(f"DONE: {display_name}")
        return result
    except Exception as e:
        short_msg = str(e)[:200]
        get_logger().exception(f"ERROR: {display_name}: {short_msg}")
        ctx.append_log(f"ERROR: {display_name}: {short_msg}")
        ctx.errors.append(f"{display_name} Error: {str(e)}")
        ctx.mark_agent(display_name, "failed")
        return ctx


def run_dataforge_pipeline(dataset_path: str, target_col: str = None, drop_columns: list = None, run_id: str = None) -> PipelineContext:
    """Programmatic entry point for DataForge pipeline."""
    from datetime import datetime, timezone
    start_time_dt = datetime.now(timezone.utc)
    start_time = start_time_dt.isoformat()
    
    ctx = PipelineContext(dataset_path=dataset_path, run_id=run_id if run_id else "")
    
    # === Agent 1: Data Ingestion ===
    ctx = safe_run(run_data_ingestion, ctx, "Data Ingestion")
    if ctx.agent_status.get("Data Ingestion") != "done" or ctx.raw_df is None:
        pass # allow writing summary even if failed
    else:
        # === Handle Columns to Drop ===
        if drop_columns:
            cols_to_drop = [c for c in drop_columns if c in ctx.raw_df.columns]
            if cols_to_drop:
                ctx.raw_df.drop(columns=cols_to_drop, inplace=True)

        # === Target Variable Configuration ===
        if target_col and target_col in ctx.raw_df.columns:
            ctx.target_column = target_col
            ctx.has_target = True
        else:
            ctx.has_target = False

        # === Execute Remaining Pipeline ===
        ctx = safe_run(run_data_quality_audit, ctx, "Data Quality Audit")
        ctx = safe_run(run_data_cleaning, ctx, "Data Cleaning")
        ctx = safe_run(run_eda, ctx, "EDA")
        ctx = safe_run(run_feature_engineering, ctx, "Feature Engineering")
        
        if ctx.has_target:
            ctx = safe_run(run_modeling, ctx, "Modeling")
            ctx = safe_run(run_shap_interpretability, ctx, "SHAP Interpretability")
            ctx = safe_run(run_statistical_testing, ctx, "Statistical Testing")
        else:
            ctx.mark_agent("Modeling", "skipped")
            ctx.mark_agent("SHAP Interpretability", "skipped")
            ctx.mark_agent("Statistical Testing", "skipped")
            
        ctx = safe_run(run_recommendations, ctx, "Recommendations")

    # ── Write per-run summary.json ─────────────────────────────────────────
    _NON_SERIAL = ("model", "X_test", "X_train", "y_test", "y_train")
    try:
        _safe_model = {
            k: v for k, v in ctx.model_results.items()
            if k not in _NON_SERIAL
        } if ctx.model_results else {}
        
        end_time_dt = datetime.now(timezone.utc)
        end_time = end_time_dt.isoformat()
        duration_seconds = round((end_time_dt - start_time_dt).total_seconds(), 2)
        
        summary_data = {
            "run_id": ctx.run_id,
            "agent_status": ctx.agent_status,
            "model_results": _safe_model,
            "errors": ctx.errors,
            "warnings": ctx.warnings,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds
        }
        
        from agents.explainer import ExplanationEngine
        engine = ExplanationEngine(ctx)
        explanations = engine.generate_all()
        summary_data['explanations'] = explanations
        
        explanations_path = os.path.join(ctx.run_dir, 'explanations.json')
        with open(explanations_path, 'w', encoding='utf-8') as fh:
            json.dump(explanations, fh, indent=2)

        from agents.mentor import MentorEngine
        mentor = MentorEngine(ctx)
        guidance = mentor.generate_all()
        summary_data['guidance'] = guidance
        
        guidance_path = os.path.join(ctx.run_dir, 'guidance.json')
        with open(guidance_path, 'w', encoding='utf-8') as fh:
            json.dump(guidance, fh, indent=2)

        with open(os.path.join(ctx.run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=4, default=str)
    except Exception as exc:
        ctx.errors.append(f"summary.json write failed: {exc}")

    # ── Write per-run metadata.json ────────────────────────────────────────
    try:
        metadata = {
            "run_id": ctx.run_id,
            "dataset_path": ctx.dataset_path,
            "target_column": ctx.target_column,
        }
        with open(os.path.join(ctx.run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    except Exception as exc:
        ctx.errors.append(f"metadata.json write failed: {exc}")

    # ── Write Cleaned Dataset ────────────────────────────────────────────────
    run_id = ctx.run_id
    run_dir = ctx.run_dir
    if getattr(ctx, "clean_df", None) is not None and not ctx.clean_df.empty:
        from tempfile import NamedTemporaryFile
        tmp_path = None
        try:
            with NamedTemporaryFile(mode="w", delete=False, dir=run_dir, suffix=".csv") as tf:
                tmp_path = tf.name
                ctx.clean_df.to_csv(tmp_path, index=False, encoding="utf-8")
            final_path = os.path.join(run_dir, "dataset_clean.csv")
            os.replace(tmp_path, final_path)
            # update summary.json
            summary_path = os.path.join(run_dir, "summary.json")
            try:
                summary = {}
                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as fh:
                        summary = json.load(fh)
                summary.setdefault("artifacts", []).append("dataset_clean.csv")
                with open(summary_path, "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=4)
            except Exception:
                pass
            ctx.append_log(f"Artifact Saved: dataset_clean.csv")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    else:
        ctx.append_log(f"WARNING: No cleaned dataframe available to save for run_id={run_id}")

    ctx.close()  # flush and release the per-run pipeline.log FileHandler
    return ctx



def main():
    """Main entry point - Complete 9-Agent Pipeline."""
    print(BANNER)
    print("🚀 Phase 2 - Full Pipeline: Ingestion → Quality → Cleaning → EDA → Features → Model → SHAP → Stats → Recommendations")
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
    
    # === Prompt for Target Column (after successful ingestion) ===
    if ctx.agent_status.get("Data Ingestion") == "done" and ctx.raw_df is not None:
        print("\n" + "-" * 60)
        print("🎯 TARGET VARIABLE CONFIGURATION")
        print("-" * 60)
        print(f"Available columns: {list(ctx.raw_df.columns)}")
        
        while True:
            target_col = input("\nEnter target column name (or press Enter to skip modeling): ").strip()
            
            if target_col == "":
                # User wants to skip modeling
                print("ℹ️  Skipping modeling (no target variable specified).")
                ctx.has_target = False
                break
            elif target_col in ctx.raw_df.columns:
                # Valid column
                ctx.target_column = target_col
                ctx.has_target = True
                print(f"✅ Target column set to: '{target_col}'")
                break
            else:
                # Invalid column
                print(f"❌ Error: Column '{target_col}' not found in dataset.")
                print(f"Available columns: {list(ctx.raw_df.columns)}")
                # Loop continues, ask again
    
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
    
    # === Agent 7: SHAP Interpretability ===
    if ctx.agent_status.get("Modeling") == "done":
        print("\n🔄 Running Agent 7: SHAP Interpretability...")
        ctx = run_shap_interpretability(ctx)
        display_shap_results(ctx)
    else:
        print("\n⏭️  Skipping Agent 7: Modeling not completed")
        ctx.mark_agent("SHAP Interpretability", "skipped")
    
    # === Agent 8: Statistical Testing ===
    if ctx.agent_status.get("Modeling") == "done" and ctx.has_target:
        print("\n🔄 Running Agent 8: Statistical Testing...")
        ctx = run_statistical_testing(ctx)
        display_statistical_testing_results(ctx)
    else:
        if not ctx.has_target:
            print("\n⏭️  Skipping Agent 8: No target variable defined")
        else:
            print("\n⏭️  Skipping Agent 8: Modeling not completed")
        ctx.mark_agent("Statistical Testing", "skipped")
    
    # === Agent 9: Recommendations ===
    # Agent 9 can run with partial results, but needs some data
    if ctx.clean_df is not None or ctx.raw_df is not None:
        print("\n🔄 Running Agent 9: Recommendations...")
        ctx = run_recommendations(ctx)
        display_recommendations_results(ctx)
    else:
        print("\n⏭️  Skipping Agent 9: No data available")
        ctx.mark_agent("Recommendations", "skipped")
    
    # Display overall status
    display_agent_status(ctx)
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline complete. All 9 agents executed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
