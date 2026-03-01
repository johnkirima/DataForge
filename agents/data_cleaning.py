"""
Agent 3: Data Cleaning
======================
Clean data based on quality audit results and generate narrative using DeepSeek V3.2.
Operations: missing value imputation, duplicate removal, outlier capping, type conversions, inconsistency fixes.
"""

import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI

from pipeline_context import PipelineContext
from config import DEEPSEEK_API_KEY, RANDOM_SEED
from logger import get_logger

logger = get_logger("DataCleaning")

# Columns that should contain positive values
POSITIVE_COLUMNS = {'age', 'income', 'salary', 'price', 'quantity', 'count', 'amount', 
                    'weight', 'height', 'duration', 'years', 'months', 'days'}


def _cap_outliers_iqr(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, int]:
    """
    Cap outliers using IQR method. Skip if <=10 unique values.
    
    Args:
        df: DataFrame to modify
        column: Column name to cap outliers in
        
    Returns:
        Tuple of (modified DataFrame, number of outliers capped)
    """
    if df[column].nunique() <= 10:
        return df, 0  # Skip ID/binary/discrete columns
    
    col_data = df[column].dropna()
    if len(col_data) == 0:
        return df, 0
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers before capping
    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    
    # Cap values
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    
    return df, int(outliers)


def _impute_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Impute missing values: median for numeric, mode for categorical.
    
    Returns:
        Tuple of (modified DataFrame, imputation report)
    """
    report = {
        'columns_imputed': [],
        'total_values_imputed': 0,
        'details': {}
    }
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric: use median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            report['details'][col] = {
                'method': 'median',
                'fill_value': float(median_val) if pd.notna(median_val) else 0,
                'count': int(missing_count)
            }
        else:
            # Categorical: use mode or "Unknown"
            mode_result = df[col].mode()
            if len(mode_result) > 0:
                fill_value = mode_result.iloc[0]
                method = 'mode'
            else:
                fill_value = "Unknown"
                method = 'unknown_fill'
            df[col] = df[col].fillna(fill_value)
            report['details'][col] = {
                'method': method,
                'fill_value': str(fill_value),
                'count': int(missing_count)
            }
        
        report['columns_imputed'].append(col)
        report['total_values_imputed'] += missing_count
    
    return df, report


def _remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove exact duplicate rows, keeping first occurrence.
    
    Returns:
        Tuple of (modified DataFrame, duplicate removal report)
    """
    original_rows = len(df)
    df = df.drop_duplicates(keep='first')
    duplicates_removed = original_rows - len(df)
    
    report = {
        'duplicates_removed': int(duplicates_removed),
        'original_rows': original_rows,
        'final_rows': len(df)
    }
    
    return df, report


def _cap_all_outliers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cap outliers using IQR method for all numeric columns.
    
    Returns:
        Tuple of (modified DataFrame, outlier capping report)
    """
    report = {
        'columns_capped': [],
        'total_values_capped': 0,
        'details': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df, capped_count = _cap_outliers_iqr(df, col)
        if capped_count > 0:
            report['columns_capped'].append(col)
            report['total_values_capped'] += capped_count
            report['details'][col] = {'values_capped': capped_count}
    
    return df, report


def _convert_data_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convert string numbers to numeric and detect date strings.
    
    Returns:
        Tuple of (modified DataFrame, conversion report)
    """
    report = {
        'columns_converted': [],
        'details': {}
    }
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                # If >80% values can be converted, use it
                non_null_original = df[col].notna().sum()
                non_null_converted = numeric_converted.notna().sum()
                
                if non_null_original > 0 and (non_null_converted / non_null_original) > 0.8:
                    df[col] = numeric_converted
                    report['columns_converted'].append(col)
                    report['details'][col] = {
                        'from': 'object',
                        'to': 'numeric',
                        'successful_conversions': int(non_null_converted)
                    }
                    continue
            except Exception:
                pass
            
            # Try to convert to datetime (check for common date patterns)
            try:
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    # Check if values look like dates
                    date_keywords = ['date', 'time', 'day', 'month', 'year']
                    col_lower = col.lower()
                    looks_like_date = any(kw in col_lower for kw in date_keywords)
                    
                    if looks_like_date:
                        datetime_converted = pd.to_datetime(df[col], errors='coerce')
                        non_null_converted = datetime_converted.notna().sum()
                        non_null_original = df[col].notna().sum()
                        
                        if non_null_original > 0 and (non_null_converted / non_null_original) > 0.8:
                            df[col] = datetime_converted
                            report['columns_converted'].append(col)
                            report['details'][col] = {
                                'from': 'object',
                                'to': 'datetime',
                                'successful_conversions': int(non_null_converted)
                            }
            except Exception:
                pass
    
    return df, report


def _fix_inconsistencies(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fix negative values in positive columns and cap unrealistic values.
    
    Returns:
        Tuple of (modified DataFrame, inconsistency fix report)
    """
    report = {
        'columns_fixed': [],
        'total_values_fixed': 0,
        'details': {}
    }
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        col_lower = col.lower()
        values_fixed = 0
        fixes_applied = []
        
        # Check if column should contain positive values
        should_be_positive = any(pos_term in col_lower for pos_term in POSITIVE_COLUMNS)
        
        if should_be_positive:
            # Fix negative values by taking absolute value
            neg_mask = df[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                df.loc[neg_mask, col] = df.loc[neg_mask, col].abs()
                values_fixed += neg_count
                fixes_applied.append(f"converted {neg_count} negative values to absolute")
        
        # Cap unrealistic age values
        if 'age' in col_lower:
            # Cap ages above 120 to 120
            high_age_mask = df[col] > 120
            high_age_count = high_age_mask.sum()
            if high_age_count > 0:
                df.loc[high_age_mask, col] = 120
                values_fixed += high_age_count
                fixes_applied.append(f"capped {high_age_count} ages > 120 to 120")
            
            # Set ages below 0 to 0 (in case absolute didn't apply)
            low_age_mask = df[col] < 0
            low_age_count = low_age_mask.sum()
            if low_age_count > 0:
                df.loc[low_age_mask, col] = 0
                values_fixed += low_age_count
                fixes_applied.append(f"set {low_age_count} negative ages to 0")
        
        if values_fixed > 0:
            report['columns_fixed'].append(col)
            report['total_values_fixed'] += values_fixed
            report['details'][col] = {
                'values_fixed': int(values_fixed),
                'fixes_applied': fixes_applied
            }
    
    return df, report


def _build_cleaning_llm_prompt(cleaning_report: Dict[str, Any], 
                                original_shape: Tuple[int, int],
                                cleaned_shape: Tuple[int, int]) -> str:
    """Build a concise prompt for DeepSeek V3.2 to summarize cleaning actions."""
    
    missing = cleaning_report.get('missing_values', {})
    dups = cleaning_report.get('duplicates', {})
    outliers = cleaning_report.get('outliers', {})
    types = cleaning_report.get('type_conversions', {})
    inconsist = cleaning_report.get('inconsistencies', {})
    
    prompt = f"""You are a data cleaning specialist. Summarize the following data cleaning operations in a clear, professional narrative:

Cleaning Report:
- Missing values imputed: {missing.get('total_values_imputed', 0)} values across {len(missing.get('columns_imputed', []))} columns
- Duplicates removed: {dups.get('duplicates_removed', 0)} rows
- Outliers capped: {outliers.get('total_values_capped', 0)} values across {len(outliers.get('columns_capped', []))} columns (IQR method)
- Data types converted: {len(types.get('columns_converted', []))} columns ({', '.join(types.get('columns_converted', [])[:5]) or 'none'})
- Inconsistencies fixed: {inconsist.get('total_values_fixed', 0)} values across {len(inconsist.get('columns_fixed', []))} columns

Original shape: {original_shape[0]} rows x {original_shape[1]} columns
Cleaned shape: {cleaned_shape[0]} rows x {cleaned_shape[1]} columns

Provide a concise summary (3-4 sentences) of what was done and why it improves data quality."""

    return prompt


def _call_deepseek_api(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call DeepSeek V3.2 API for cleaning narrative generation."""
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
                        {"role": "system", "content": "You are a helpful data cleaning specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
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


def run_data_cleaning(ctx: PipelineContext) -> PipelineContext:
    """
    Clean ctx.raw_df based on quality audit results.
    Creates ctx.clean_df and generates narrative using DeepSeek V3.2.
    
    Cleaning operations:
    1. Missing value imputation (median for numeric, mode for categorical)
    2. Duplicate removal (keep first)
    3. Outlier capping using IQR method
    4. Data type conversions
    5. Inconsistency fixes (negative values, unrealistic ranges)
    
    Args:
        ctx: PipelineContext with raw_df populated by Agent 1
        
    Returns:
        Updated PipelineContext with clean_df and cleaning report
    """
    logger.info("=" * 50)
    logger.info("Starting Data Cleaning Agent")
    logger.info("=" * 50)
    
    # Check if raw_df is available
    if ctx.raw_df is None:
        logger.warning("raw_df is None - Agent 1 likely failed. Skipping data cleaning.")
        ctx.mark_agent("Data Cleaning", "skipped")
        ctx.errors.append("Data Cleaning skipped: No data available from Agent 1")
        return ctx
    
    # Check for empty DataFrame
    if ctx.raw_df.empty:
        logger.warning("raw_df is empty. Skipping data cleaning.")
        ctx.mark_agent("Data Cleaning", "skipped")
        ctx.errors.append("Data Cleaning skipped: Dataset is empty")
        return ctx
    
    ctx.mark_agent("Data Cleaning", "running")
    
    # Store original shape
    original_shape = ctx.raw_df.shape
    logger.info(f"Original dataset shape: {original_shape[0]} rows x {original_shape[1]} columns")
    
    try:
        # Create a copy for cleaning
        df = ctx.raw_df.copy()
        
        # === 1. Impute Missing Values ===
        logger.info("Step 1: Imputing missing values...")
        df, missing_report = _impute_missing_values(df)
        logger.info(f"  - Imputed {missing_report['total_values_imputed']} values in {len(missing_report['columns_imputed'])} columns")
        
        # === 2. Remove Duplicates ===
        logger.info("Step 2: Removing duplicate rows...")
        df, dup_report = _remove_duplicates(df)
        logger.info(f"  - Removed {dup_report['duplicates_removed']} duplicate rows")
        
        # === 3. Cap Outliers (IQR method) ===
        logger.info("Step 3: Capping outliers using IQR method...")
        df, outlier_report = _cap_all_outliers(df)
        logger.info(f"  - Capped {outlier_report['total_values_capped']} values in {len(outlier_report['columns_capped'])} columns")
        
        # === 4. Convert Data Types ===
        logger.info("Step 4: Converting data types...")
        df, type_report = _convert_data_types(df)
        logger.info(f"  - Converted {len(type_report['columns_converted'])} columns")
        
        # === 5. Fix Inconsistencies ===
        logger.info("Step 5: Fixing data inconsistencies...")
        df, inconsist_report = _fix_inconsistencies(df)
        logger.info(f"  - Fixed {inconsist_report['total_values_fixed']} values in {len(inconsist_report['columns_fixed'])} columns")
        
        # Store cleaned DataFrame
        ctx.clean_df = df
        cleaned_shape = ctx.clean_df.shape
        logger.info(f"Cleaned dataset shape: {cleaned_shape[0]} rows x {cleaned_shape[1]} columns")
        
        # Compile cleaning report
        cleaning_report = {
            'original_shape': {'rows': original_shape[0], 'columns': original_shape[1]},
            'cleaned_shape': {'rows': cleaned_shape[0], 'columns': cleaned_shape[1]},
            'missing_values': missing_report,
            'duplicates': dup_report,
            'outliers': outlier_report,
            'type_conversions': type_report,
            'inconsistencies': inconsist_report
        }
        ctx.cleaning_report = cleaning_report
        
        # === Memory Management ===
        logger.info("Performing memory cleanup (deleting raw_df)...")
        ctx.raw_df = None
        gc.collect()
        logger.info("Memory cleanup completed")
        
        # === Generate LLM Narrative ===
        logger.info("Generating LLM narrative using DeepSeek V3.2...")
        prompt = _build_cleaning_llm_prompt(cleaning_report, original_shape, cleaned_shape)
        narrative = _call_deepseek_api(prompt)
        
        if narrative:
            ctx.llm_narratives["cleaning"] = narrative
            logger.info("LLM cleaning narrative generated successfully")
        else:
            ctx.llm_narratives["cleaning"] = "LLM narrative generation failed or API key not configured."
            logger.warning("LLM cleaning narrative not generated")
        
        ctx.mark_agent("Data Cleaning", "done")
        logger.info("Data Cleaning Agent completed successfully")
        
    except Exception as e:
        error_msg = f"Error during data cleaning: {str(e)}"
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Data Cleaning", "failed")
    
    return ctx
