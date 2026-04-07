"""DataForge Agent 8: Statistical Testing - Hypothesis tests with DeepSeek narrative."""
import numpy as np
import pandas as pd
from scipy import stats
from openai import OpenAI

import config
from pipeline_context import PipelineContext
from logger import get_logger

logger = get_logger("StatisticalTesting")


def _perform_ttest(df: pd.DataFrame, target_column: str, numeric_cols: list) -> list:
    """
    Perform t-tests for binary classification targets.
    Compare means of numeric features between two classes.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        numeric_cols: List of numeric column names
        
    Returns:
        List of test results
    """
    results = []
    classes = df[target_column].unique()
    
    if len(classes) != 2:
        return results
    
    class_0 = df[df[target_column] == classes[0]]
    class_1 = df[df[target_column] == classes[1]]
    
    for col in numeric_cols:
        if col == target_column:
            continue
        try:
            # Get values for each class, dropping NaN
            vals_0 = class_0[col].dropna()
            vals_1 = class_1[col].dropna()
            
            if len(vals_0) < 2 or len(vals_1) < 2:
                continue
            
            t_stat, p_value = stats.ttest_ind(vals_0, vals_1, equal_var=False)
            
            results.append({
                "feature": col,
                "test_type": "t-test",
                "statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05
            })
        except Exception as e:
            logger.warning(f"T-test failed for {col}: {e}")
    
    return results


def _perform_anova(df: pd.DataFrame, target_column: str, numeric_cols: list) -> list:
    """
    Perform ANOVA for multi-class classification targets.
    Compare means of numeric features across multiple classes.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        numeric_cols: List of numeric column names
        
    Returns:
        List of test results
    """
    results = []
    classes = df[target_column].unique()
    
    if len(classes) <= 2:
        return results
    
    for col in numeric_cols:
        if col == target_column:
            continue
        try:
            # Get values for each class
            groups = [df[df[target_column] == cls][col].dropna() for cls in classes]
            
            # Filter out empty groups
            groups = [g for g in groups if len(g) >= 2]
            
            if len(groups) < 2:
                continue
            
            f_stat, p_value = stats.f_oneway(*groups)
            
            results.append({
                "feature": col,
                "test_type": "anova",
                "statistic": round(float(f_stat), 4),
                "p_value": round(float(p_value), 6),
                "significant": p_value < 0.05
            })
        except Exception as e:
            logger.warning(f"ANOVA failed for {col}: {e}")
    
    return results


def _perform_chi_square(df: pd.DataFrame, target_column: str, categorical_cols: list) -> list:
    """
    Perform chi-square tests for categorical features vs categorical target.
    Test independence between categorical variables.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        categorical_cols: List of categorical column names
        
    Returns:
        List of test results
    """
    results = []
    
    for col in categorical_cols:
        if col == target_column:
            continue
        try:
            # Create contingency table
            contingency = pd.crosstab(df[col], df[target_column])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            results.append({
                "feature": col,
                "test_type": "chi-square",
                "statistic": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "degrees_of_freedom": int(dof),
                "significant": p_value < 0.05
            })
        except Exception as e:
            logger.warning(f"Chi-square test failed for {col}: {e}")
    
    return results


def _build_statistical_prompt(ctx: PipelineContext, ttest_results: list, 
                               anova_results: list, chi_square_results: list) -> str:
    """Build prompt for DeepSeek narrative generation."""
    
    # Format test results
    ttest_summary = ""
    if ttest_results:
        ttest_summary = "T-Test Results (comparing means between classes):\n"
        for r in ttest_results[:10]:  # Limit to 10
            ttest_summary += f"- {r['feature']}: t={r['statistic']}, p={r['p_value']}, significant={r['significant']}\n"
    
    anova_summary = ""
    if anova_results:
        anova_summary = "ANOVA Results (comparing means across classes):\n"
        for r in anova_results[:10]:
            anova_summary += f"- {r['feature']}: F={r['statistic']}, p={r['p_value']}, significant={r['significant']}\n"
    
    chi_square_summary = ""
    if chi_square_results:
        chi_square_summary = "Chi-Square Results (testing independence):\n"
        for r in chi_square_results[:10]:
            chi_square_summary += f"- {r['feature']}: chi2={r['statistic']}, p={r['p_value']}, significant={r['significant']}\n"
    
    # Count significant features
    all_results = ttest_results + anova_results + chi_square_results
    significant_count = sum(1 for r in all_results if r.get('significant', False))
    
    prompt = f"""You are a statistician analyzing hypothesis test results. Summarize the following statistical tests concisely.

Dataset: {ctx.dataset_name}
Target: {ctx.target_column}
Task: {ctx.task_type}
Total Tests Performed: {len(all_results)}
Significant Features (p < 0.05): {significant_count}

{ttest_summary}
{anova_summary}
{chi_square_summary}

Provide:
1. Summary of significant findings (2-3 sentences)
2. Which features show strongest relationships with target (list top 3-5)
3. Brief statistical interpretation and implications for modeling

Keep response under 300 words."""

    return prompt


def _call_deepseek_api(prompt: str) -> str:
    """Call DeepSeek API for narrative generation."""
    if not config.DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not configured")
        return "Statistical narrative not available (API key not configured)."
    
    try:
        client = OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a data science expert providing statistical analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"DeepSeek API call failed: {e}")
        return f"Statistical narrative generation failed: {str(e)}"


def run_statistical_testing(ctx: PipelineContext) -> PipelineContext:
    """
    Perform hypothesis tests (t-test, chi-square, ANOVA) on features vs target.
    Generates narrative using DeepSeek V3.2.
    Only runs for classification tasks.
    
    Args:
        ctx: PipelineContext with clean_df and target_column
        
    Returns:
        PipelineContext with statistical_results populated
    """
    logger.info("=" * 50)
    logger.info("Starting Agent 8: Statistical Testing")
    logger.info("=" * 50)
    
    ctx.mark_agent("Statistical Testing", "running")
    
    # === Check Prerequisites ===
    if ctx.clean_df is None:
        logger.warning("No clean_df available. Skipping statistical testing.")
        ctx.mark_agent("Statistical Testing", "skipped")
        ctx.errors.append("Statistical Testing: No clean_df available")
        return ctx
    
    if not ctx.has_target:
        logger.warning("No target variable defined. Skipping statistical testing.")
        ctx.mark_agent("Statistical Testing", "skipped")
        ctx.errors.append("Statistical Testing: No target variable defined")
        return ctx
    
    if ctx.target_column not in ctx.clean_df.columns:
        logger.warning(f"Target column '{ctx.target_column}' not found. Skipping statistical testing.")
        ctx.mark_agent("Statistical Testing", "skipped")
        ctx.errors.append(f"Statistical Testing: Target column '{ctx.target_column}' not found")
        return ctx
    
    if ctx.task_type == "regression":
        logger.info("Task type is regression. Statistical tests designed for classification. Skipping.")
        ctx.mark_agent("Statistical Testing", "skipped")
        ctx.statistical_results = {
            "skipped_reason": "Regression task - statistical tests only applicable for classification",
            "tests_performed": [],
            "significant_features": [],
            "test_details": {},
            "narrative": "Statistical hypothesis tests (t-test, ANOVA, chi-square) are designed for classification tasks and were skipped for this regression analysis."
        }
        ctx.llm_narratives["statistical_testing"] = ctx.statistical_results["narrative"]
        return ctx
    
    try:
        df = ctx.clean_df.copy()
        logger.info(f"Working with DataFrame shape: {df.shape}")
        
        # === Identify Feature Types ===
        numeric_cols = df.select_dtypes(include=['int64', 'int32', 'float64', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists
        if ctx.target_column in numeric_cols:
            numeric_cols.remove(ctx.target_column)
        if ctx.target_column in categorical_cols:
            categorical_cols.remove(ctx.target_column)
        
        logger.info(f"Numeric features: {len(numeric_cols)}")
        logger.info(f"Categorical features: {len(categorical_cols)}")
        
        # === Determine Number of Classes ===
        n_classes = df[ctx.target_column].nunique()
        logger.info(f"Target classes: {n_classes}")
        
        # === Perform Tests ===
        ttest_results = []
        anova_results = []
        chi_square_results = []
        
        # T-tests for binary classification
        if n_classes == 2 and numeric_cols:
            logger.info("Performing t-tests for binary classification...")
            ttest_results = _perform_ttest(df, ctx.target_column, numeric_cols)
            logger.info(f"T-tests completed: {len(ttest_results)} tests")
        
        # ANOVA for multi-class classification
        if n_classes > 2 and numeric_cols:
            logger.info("Performing ANOVA for multi-class classification...")
            anova_results = _perform_anova(df, ctx.target_column, numeric_cols)
            logger.info(f"ANOVA tests completed: {len(anova_results)} tests")
        
        # Chi-square for categorical features
        if categorical_cols:
            logger.info("Performing chi-square tests for categorical features...")
            chi_square_results = _perform_chi_square(df, ctx.target_column, categorical_cols)
            logger.info(f"Chi-square tests completed: {len(chi_square_results)} tests")
        
        # === Collect Results ===
        all_results = ttest_results + anova_results + chi_square_results
        significant_features = [r['feature'] for r in all_results if r.get('significant', False)]
        
        logger.info(f"Total tests performed: {len(all_results)}")
        logger.info(f"Significant features found: {len(significant_features)}")
        
        # === Generate DeepSeek Narrative ===
        logger.info("Generating statistical narrative with DeepSeek...")
        prompt = _build_statistical_prompt(ctx, ttest_results, anova_results, chi_square_results)
        narrative = _call_deepseek_api(prompt)
        logger.info("Narrative generated successfully")
        
        # === Store Results ===
        ctx.statistical_results = {
            "tests_performed": [
                {"type": "t-test", "count": len(ttest_results)},
                {"type": "anova", "count": len(anova_results)},
                {"type": "chi-square", "count": len(chi_square_results)}
            ],
            "significant_features": significant_features,
            "test_details": {
                "ttest": ttest_results,
                "anova": anova_results,
                "chi_square": chi_square_results
            },
            "total_tests": len(all_results),
            "significant_count": len(significant_features),
            "narrative": narrative
        }
        
        ctx.llm_narratives["statistical_testing"] = narrative
        
        ctx.mark_agent("Statistical Testing", "done")
        logger.info("Agent 8: Statistical Testing completed successfully")
        
    except Exception as e:
        logger.error(f"Statistical Testing failed with error: {str(e)}", exc_info=True)
        ctx.errors.append(f"Statistical Testing: {str(e)}")
        ctx.mark_agent("Statistical Testing", "failed")
    
    return ctx
