"""
Agent 2: Data Quality Audit
===========================
Analyze data quality, flag issues, and generate narrative summary using DeepSeek V3.2.
Checks: missing values, duplicates, data types, outliers, inconsistencies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from openai import OpenAI

from pipeline_context import PipelineContext
from config import DEEPSEEK_API_KEY
from logger import get_logger

logger = get_logger("DataQualityAudit")

# Columns that should contain positive values
POSITIVE_COLUMNS = {'age', 'income', 'salary', 'price', 'quantity', 'count', 'amount', 
                    'weight', 'height', 'duration', 'years', 'months', 'days'}


def _check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Check missing values per column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_data = {}
    for col in df.columns:
        if missing[col] > 0:
            missing_data[col] = {
                'count': int(missing[col]),
                'percentage': float(missing_pct[col])
            }
    
    return {
        'columns_with_missing': len(missing_data),
        'total_missing_cells': int(missing.sum()),
        'details': missing_data
    }


def _check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate rows."""
    duplicate_count = df.duplicated().sum()
    duplicate_pct = round(duplicate_count / len(df) * 100, 2)
    
    return {
        'count': int(duplicate_count),
        'percentage': float(duplicate_pct)
    }


def _identify_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify numeric vs categorical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols,
        'boolean': bool_cols,
        'summary': {
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols),
            'datetime_count': len(datetime_cols),
            'boolean_count': len(bool_cols)
        }
    }


def _detect_outliers_iqr(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect outliers using IQR method for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        # Skip columns with <=10 unique values (likely ID/binary/discrete)
        if df[col].nunique() <= 10:
            continue
        
        # Skip columns with all NaN
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round(outlier_count / len(col_data) * 100, 2),
                'lower_bound': round(float(lower_bound), 2),
                'upper_bound': round(float(upper_bound), 2)
            }
    
    return {
        'columns_with_outliers': len(outliers),
        'details': outliers
    }


def _check_inconsistencies(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for inconsistencies like negative values in positive columns."""
    issues = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_lower = col.lower()
        # Check if column should be positive
        should_be_positive = any(pos_term in col_lower for pos_term in POSITIVE_COLUMNS)
        
        if should_be_positive:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues[col] = {
                    'issue': 'negative_values_found',
                    'count': int(neg_count),
                    'expected': 'positive values only'
                }
        
        # Check for unrealistic age values
        if 'age' in col_lower:
            unrealistic = ((df[col] < 0) | (df[col] > 150)).sum()
            if unrealistic > 0:
                issues[col] = {
                    'issue': 'unrealistic_age_values',
                    'count': int(unrealistic),
                    'expected': '0-150 range'
                }
    
    return {
        'columns_with_issues': len(issues),
        'details': issues
    }


def _compute_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic statistics for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        stats[col] = {
            'mean': round(float(col_data.mean()), 2),
            'std': round(float(col_data.std()), 2),
            'min': round(float(col_data.min()), 2),
            'max': round(float(col_data.max()), 2),
            'q25': round(float(col_data.quantile(0.25)), 2),
            'q50': round(float(col_data.quantile(0.50)), 2),
            'q75': round(float(col_data.quantile(0.75)), 2)
        }
    
    return stats


def _compute_cardinality(df: pd.DataFrame) -> Dict[str, Any]:
    """Count unique values per column."""
    cardinality = {}
    
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality[col] = {
            'unique_values': int(unique_count),
            'cardinality_ratio': round(unique_count / len(df) * 100, 2)
        }
    
    # Identify high-cardinality categoricals (>50% unique for object cols)
    high_cardinality = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if cardinality[col]['cardinality_ratio'] > 50:
            high_cardinality.append(col)
    
    return {
        'details': cardinality,
        'high_cardinality_categoricals': high_cardinality
    }


def _build_llm_prompt(quality_report: Dict[str, Any], total_rows: int, total_cols: int) -> str:
    """Build a concise prompt for DeepSeek V3.2."""
    # Summarize missing values
    missing_info = quality_report.get('missing_values', {})
    missing_summary = f"{missing_info.get('columns_with_missing', 0)} columns with missing values"
    if missing_info.get('details'):
        top_missing = list(missing_info['details'].items())[:3]
        missing_details = ", ".join([f"{col}: {info['count']}" for col, info in top_missing])
        missing_summary += f" (top: {missing_details})"
    
    # Summarize duplicates
    dup_info = quality_report.get('duplicates', {})
    dup_summary = f"{dup_info.get('count', 0)} duplicate rows ({dup_info.get('percentage', 0)}%)"
    
    # Summarize outliers
    outlier_info = quality_report.get('outliers', {})
    outlier_cols = list(outlier_info.get('details', {}).keys())[:5]
    outlier_summary = f"{outlier_info.get('columns_with_outliers', 0)} columns with outliers"
    if outlier_cols:
        outlier_summary += f" ({', '.join(outlier_cols)})"
    
    # Summarize inconsistencies
    inconsist_info = quality_report.get('inconsistencies', {})
    inconsist_summary = f"{inconsist_info.get('columns_with_issues', 0)} columns with data issues"
    
    # Data types summary
    types_info = quality_report.get('data_types', {}).get('summary', {})
    types_summary = f"{types_info.get('numeric_count', 0)} numeric, {types_info.get('categorical_count', 0)} categorical"
    
    prompt = f"""You are a data quality analyst. Analyze the following data quality report and provide:
1. A brief executive summary (2-3 sentences)
2. Key issues found (bullet points)
3. Severity assessment (Low/Medium/High)
4. Recommendations for data cleaning

Data Quality Report:
- Total rows: {total_rows}
- Total columns: {total_cols}
- Data types: {types_summary}
- Missing values: {missing_summary}
- Duplicates: {dup_summary}
- Outliers detected: {outlier_summary}
- Data inconsistencies: {inconsist_summary}

Provide your analysis in a clear, professional tone. Keep it concise."""

    return prompt


def _call_deepseek_api(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call DeepSeek V3.2 API for narrative generation."""
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set, skipping LLM narrative generation")
        return None
    
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Calling DeepSeek API (attempt {attempt + 1}/{max_retries + 1})")
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful data quality analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                narrative = response.choices[0].message.content
                logger.info("DeepSeek API call successful")
                return narrative
                
            except Exception as api_error:
                logger.warning(f"DeepSeek API attempt {attempt + 1} failed: {str(api_error)}")
                if attempt == max_retries:
                    raise
        
    except Exception as e:
        error_msg = f"DeepSeek API error: {str(e)}"
        logger.error(error_msg)
        return None
    
    return None


def run_data_quality_audit(ctx: PipelineContext) -> PipelineContext:
    """
    Analyze data quality of ctx.raw_df and generate narrative using DeepSeek V3.2.
    Checks: missing values, duplicates, types, outliers, inconsistencies.
    
    Args:
        ctx: PipelineContext with raw_df populated by Agent 1
        
    Returns:
        Updated PipelineContext with quality audit results
    """
    logger.info("=" * 50)
    logger.info("Starting Data Quality Audit Agent")
    logger.info("=" * 50)
    
    # Check if Agent 1 succeeded
    if ctx.raw_df is None:
        logger.warning("raw_df is None - Agent 1 likely failed. Skipping quality audit.")
        ctx.mark_agent("Data Quality Audit", "skipped")
        ctx.errors.append("Data Quality Audit skipped: No data available from Agent 1")
        return ctx
    
    # Check for empty DataFrame
    if ctx.raw_df.empty:
        logger.warning("raw_df is empty. Skipping quality audit.")
        ctx.mark_agent("Data Quality Audit", "skipped")
        ctx.errors.append("Data Quality Audit skipped: Dataset is empty")
        return ctx
    
    ctx.mark_agent("Data Quality Audit", "running")
    
    df = ctx.raw_df
    total_rows = len(df)
    total_cols = len(df.columns)
    
    logger.info(f"Analyzing dataset: {total_rows} rows x {total_cols} columns")
    
    try:
        # Perform quality checks
        logger.info("Checking missing values...")
        missing_values = _check_missing_values(df)
        
        logger.info("Checking duplicates...")
        duplicates = _check_duplicates(df)
        
        logger.info("Identifying data types...")
        data_types = _identify_data_types(df)
        
        logger.info("Detecting outliers (IQR method)...")
        outliers = _detect_outliers_iqr(df)
        
        logger.info("Checking data inconsistencies...")
        inconsistencies = _check_inconsistencies(df)
        
        logger.info("Computing summary statistics...")
        summary_stats = _compute_summary_stats(df)
        
        logger.info("Computing cardinality...")
        cardinality = _compute_cardinality(df)
        
        # Compile quality report
        quality_report = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'data_types': data_types,
            'outliers': outliers,
            'inconsistencies': inconsistencies,
            'summary_stats': summary_stats,
            'cardinality': cardinality
        }
        
        # Store in context
        ctx.eda_summary["quality_audit"] = quality_report
        
        logger.info(f"Quality checks completed:")
        logger.info(f"  - Missing values in {missing_values['columns_with_missing']} columns")
        logger.info(f"  - {duplicates['count']} duplicate rows ({duplicates['percentage']}%)")
        logger.info(f"  - {outliers['columns_with_outliers']} columns with outliers")
        logger.info(f"  - {inconsistencies['columns_with_issues']} columns with inconsistencies")
        
        # Generate LLM narrative
        logger.info("Generating LLM narrative using DeepSeek V3.2...")
        prompt = _build_llm_prompt(quality_report, total_rows, total_cols)
        narrative = _call_deepseek_api(prompt)
        
        if narrative:
            ctx.llm_narratives["quality_audit"] = narrative
            logger.info("LLM narrative generated successfully")
        else:
            ctx.llm_narratives["quality_audit"] = "LLM narrative generation failed or API key not configured."
            logger.warning("LLM narrative not generated")
        
        ctx.mark_agent("Data Quality Audit", "done")
        logger.info("Data Quality Audit Agent completed successfully")
        
    except Exception as e:
        error_msg = f"Error during data quality audit: {str(e)}"
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Data Quality Audit", "failed")
    
    return ctx
