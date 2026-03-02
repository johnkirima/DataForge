"""
Agent 5: Feature Engineering
=============================
Suggest and apply feature transformations using DeepSeek V3.2.
Operations: log transforms, polynomial features, interaction features, binning.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from pipeline_context import PipelineContext
import config
from config import DEEPSEEK_API_KEY, RANDOM_SEED
from logger import get_logger

logger = get_logger("FeatureEngineering")

# Constants
MAX_POLYNOMIAL_DEGREE = 2
MAX_INTERACTIONS = 5
MAX_LOG_TRANSFORMS = 10
MAX_BINNING_COLS = 5
SKEWNESS_THRESHOLD = 1.0


def _get_column_skewness(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate skewness for all numeric columns."""
    skewness_dict = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 2:
            skewness_dict[col] = float(col_data.skew())
    
    return skewness_dict


def _get_correlations_summary(ctx: PipelineContext) -> List[Tuple[str, str, float]]:
    """Extract top correlations from EDA summary."""
    correlations = ctx.eda_summary.get('correlations', {})
    top_positive = correlations.get('top_positive', [])
    top_negative = correlations.get('top_negative', [])
    
    all_corrs = []
    for col1, col2, corr in top_positive:
        all_corrs.append((col1, col2, corr))
    for col1, col2, corr in top_negative:
        all_corrs.append((col1, col2, corr))
    
    return all_corrs


def _build_feature_engineering_prompt(ctx: PipelineContext, skewness: Dict[str, float]) -> str:
    """Build a prompt for DeepSeek to suggest feature engineering transformations."""
    df = ctx.clean_df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Get correlations
    correlations = _get_correlations_summary(ctx)
    
    # Get target correlations if available
    target_corrs = ""
    target_analysis = ctx.eda_summary.get('target_analysis', {})
    if target_analysis and 'feature_correlations' in target_analysis:
        feature_corrs = target_analysis['feature_correlations']
        target_corrs = "\n".join([f"- {col}: {corr:.2f}" for col, corr in list(feature_corrs.items())[:5]])
    
    prompt = f"""You are a feature engineering expert. Based on the following EDA results, suggest feature engineering transformations:

Dataset: {ctx.dataset_name}
Shape: {df.shape[0]} rows x {df.shape[1]} columns

Numeric columns ({len(numeric_cols)}): {numeric_cols[:15]}
Categorical columns ({len(categorical_cols)}): {categorical_cols[:10]}

Key Statistics (skewness for numeric columns):
"""
    
    # Add skewness info
    for col, skew in list(skewness.items())[:10]:
        skew_label = "highly right-skewed" if skew > 1 else ("right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "normal"))
        prompt += f"- {col}: skewness={skew:.2f} ({skew_label})\n"
    
    # Add top correlations
    prompt += "\nTop Feature Correlations:\n"
    for col1, col2, corr in correlations[:5]:
        prompt += f"- {col1} ↔ {col2}: {corr:.2f}\n"
    
    if target_corrs:
        prompt += f"\nTarget Variable Correlations:\n{target_corrs}\n"
    
    prompt += """
Based on this analysis, suggest:
1. Which numeric features should be log-transformed (if skewness > 1.0)?
2. Which feature interactions (product of two features) would be meaningful based on correlations?
3. Which features should be polynomial (squared) - focus on features with moderate target correlation?
4. Which numeric features should be binned into categories?

IMPORTANT: 
- Only suggest columns that actually exist in the dataset
- Keep suggestions practical (max 5-10 features per category)
- For binning, provide sensible bin edges based on typical data ranges

Respond ONLY with valid JSON in this exact format (no additional text):
{
    "log_transforms": ["column1", "column2"],
    "interactions": [["col1", "col2"], ["col3", "col4"]],
    "polynomial": ["column1", "column2"],
    "binning": [{"column": "age", "bins": [0, 25, 50, 75, 100], "labels": ["young", "adult", "middle", "senior"]}]
}
"""
    
    return prompt


def _call_deepseek_api(prompt: str, max_retries: int = 2) -> Optional[Dict]:
    """Call DeepSeek V3.2 API for feature engineering suggestions."""
    if not DEEPSEEK_API_KEY:
        logger.warning("DEEPSEEK_API_KEY not set, using default feature suggestions")
        return None
    
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Calling DeepSeek API for feature suggestions (attempt {attempt + 1}/{max_retries + 1})")
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a feature engineering expert. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.2
                )
                
                content = response.choices[0].message.content
                logger.info("DeepSeek API call successful")
                
                # Parse JSON from response
                suggestions = _parse_json_response(content)
                if suggestions:
                    return suggestions
                else:
                    logger.warning(f"Failed to parse JSON from response on attempt {attempt + 1}")
                    if attempt == max_retries:
                        return None
                    
            except Exception as api_error:
                logger.warning(f"DeepSeek API attempt {attempt + 1} failed: {str(api_error)}")
                if attempt == max_retries:
                    raise
        
    except Exception as e:
        error_msg = f"DeepSeek API error: {str(e)}"
        logger.error(error_msg)
        return None
    
    return None


def _parse_json_response(content: str) -> Optional[Dict]:
    """Parse JSON from LLM response, handling common issues."""
    if not content:
        return None
    
    # Try direct JSON parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            try:
                # Clean up common issues
                cleaned = match.strip()
                if not cleaned.startswith('{'):
                    continue
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    
    logger.warning("Could not parse JSON from response")
    return None


def _validate_suggestions(suggestions: Dict, df: pd.DataFrame) -> Dict:
    """Validate and filter suggestions to only include existing columns."""
    validated = {
        'log_transforms': [],
        'interactions': [],
        'polynomial': [],
        'binning': []
    }
    
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns.tolist())
    
    # Validate log transforms
    for col in suggestions.get('log_transforms', [])[:MAX_LOG_TRANSFORMS]:
        if col in numeric_cols:
            validated['log_transforms'].append(col)
    
    # Validate interactions
    for pair in suggestions.get('interactions', [])[:MAX_INTERACTIONS]:
        if isinstance(pair, list) and len(pair) == 2:
            if pair[0] in numeric_cols and pair[1] in numeric_cols:
                validated['interactions'].append(pair)
    
    # Validate polynomial
    for col in suggestions.get('polynomial', [])[:MAX_POLYNOMIAL_DEGREE * 5]:
        if col in numeric_cols:
            validated['polynomial'].append(col)
    
    # Validate binning
    for bin_spec in suggestions.get('binning', [])[:MAX_BINNING_COLS]:
        if isinstance(bin_spec, dict) and 'column' in bin_spec:
            col = bin_spec['column']
            if col in numeric_cols and 'bins' in bin_spec and 'labels' in bin_spec:
                validated['binning'].append(bin_spec)
    
    return validated


def _get_default_suggestions(df: pd.DataFrame, skewness: Dict[str, float]) -> Dict:
    """Generate default feature engineering suggestions when API fails."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Default log transforms for highly skewed columns
    log_transforms = [col for col, skew in skewness.items() if skew > SKEWNESS_THRESHOLD][:5]
    
    # Default polynomial for first 2-3 numeric columns with high variance
    polynomial = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    
    # No default interactions or binning (require domain knowledge)
    return {
        'log_transforms': log_transforms,
        'interactions': [],
        'polynomial': polynomial[:2],
        'binning': []
    }


def _apply_log_transforms(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Apply log transformation to specified columns."""
    new_features = []
    
    for col in columns:
        try:
            # Only apply if all values are positive
            col_data = df[col].dropna()
            if (col_data > 0).all():
                new_col_name = f"log_{col}"
                df[new_col_name] = np.log1p(df[col])  # log(1+x) to handle zeros
                new_features.append(new_col_name)
                logger.info(f"Created log transform: {new_col_name}")
            else:
                # For columns with negative/zero values, shift then log
                min_val = df[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    new_col_name = f"log_shifted_{col}"
                    df[new_col_name] = np.log(df[col] + shift)
                    new_features.append(new_col_name)
                    logger.info(f"Created shifted log transform: {new_col_name}")
        except Exception as e:
            logger.warning(f"Failed to apply log transform to {col}: {e}")
    
    return df, new_features


def _apply_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> Tuple[pd.DataFrame, List[str]]:
    """Create polynomial features (squared terms)."""
    new_features = []
    
    for col in columns:
        try:
            new_col_name = f"{col}_squared"
            df[new_col_name] = df[col] ** 2
            new_features.append(new_col_name)
            logger.info(f"Created polynomial feature: {new_col_name}")
        except Exception as e:
            logger.warning(f"Failed to create polynomial for {col}: {e}")
    
    return df, new_features


def _apply_interaction_features(df: pd.DataFrame, interactions: List[List[str]]) -> Tuple[pd.DataFrame, List[str]]:
    """Create interaction features (product of two columns)."""
    new_features = []
    
    for pair in interactions:
        try:
            col1, col2 = pair
            new_col_name = f"{col1}_x_{col2}"
            df[new_col_name] = df[col1] * df[col2]
            new_features.append(new_col_name)
            logger.info(f"Created interaction feature: {new_col_name}")
        except Exception as e:
            logger.warning(f"Failed to create interaction for {pair}: {e}")
    
    return df, new_features


def _apply_binning(df: pd.DataFrame, binning_specs: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
    """Apply binning/discretization to specified columns."""
    new_features = []
    
    for spec in binning_specs:
        try:
            col = spec['column']
            bins = spec['bins']
            labels = spec['labels']
            
            # Ensure bins and labels are compatible
            if len(labels) != len(bins) - 1:
                logger.warning(f"Bins and labels mismatch for {col}, skipping")
                continue
            
            new_col_name = f"{col}_binned"
            df[new_col_name] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            new_features.append(new_col_name)
            logger.info(f"Created binned feature: {new_col_name}")
        except Exception as e:
            logger.warning(f"Failed to apply binning to {spec.get('column', 'unknown')}: {e}")
    
    return df, new_features


def _build_narrative_prompt(ctx: PipelineContext, applied_features: Dict, new_feature_names: List[str]) -> str:
    """Build prompt for DeepSeek to generate feature engineering narrative."""
    prompt = f"""You performed feature engineering on the {ctx.dataset_name} dataset.

Features Created:
- Log transforms: {applied_features.get('log_transforms', [])}
- Polynomial features: {applied_features.get('polynomial', [])}
- Interaction features: {applied_features.get('interactions', [])}
- Binned features: {[s.get('column', '') for s in applied_features.get('binning', [])]}

Total new features created: {len(new_feature_names)}
New feature names: {new_feature_names[:10]}{'...' if len(new_feature_names) > 10 else ''}

Provide a brief summary (100-150 words) explaining:
1. What transformations were applied and why
2. How these new features might help with modeling
3. Any recommendations for feature selection
"""
    return prompt


def run_feature_engineering(ctx: PipelineContext) -> PipelineContext:
    """
    Suggest and apply feature engineering transformations using DeepSeek V3.2.
    Creates new features: polynomial, interactions, binning, log transforms.
    
    Args:
        ctx: PipelineContext with clean_df populated by Agent 3
        
    Returns:
        Updated PipelineContext with new features added to clean_df
    """
    logger.info("=" * 50)
    logger.info("Starting Feature Engineering Agent")
    logger.info("=" * 50)
    
    # Check if clean_df is available
    if ctx.clean_df is None:
        logger.warning("clean_df is None - Previous agents failed. Skipping Feature Engineering.")
        ctx.mark_agent("Feature Engineering", "skipped")
        ctx.errors.append("Feature Engineering skipped: No cleaned data available")
        return ctx
    
    # Check for empty DataFrame
    if ctx.clean_df.empty:
        logger.warning("clean_df is empty. Skipping Feature Engineering.")
        ctx.mark_agent("Feature Engineering", "skipped")
        ctx.errors.append("Feature Engineering skipped: Dataset is empty")
        return ctx
    
    ctx.mark_agent("Feature Engineering", "running")
    
    df = ctx.clean_df.copy()
    original_columns = df.columns.tolist()
    all_new_features = []
    
    logger.info(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    try:
        # === Step 1: Calculate skewness for all numeric columns ===
        logger.info("Step 1: Calculating column skewness...")
        skewness = _get_column_skewness(df)
        logger.info(f"  - Calculated skewness for {len(skewness)} numeric columns")
        
        # === Step 2: Get feature suggestions from DeepSeek ===
        logger.info("Step 2: Getting feature suggestions from DeepSeek V3.2...")
        prompt = _build_feature_engineering_prompt(ctx, skewness)
        suggestions = _call_deepseek_api(prompt)
        
        if suggestions is None:
            logger.info("  - API failed or unavailable, using default suggestions")
            suggestions = _get_default_suggestions(df, skewness)
        
        # === Step 3: Validate suggestions ===
        logger.info("Step 3: Validating feature suggestions...")
        validated = _validate_suggestions(suggestions, df)
        logger.info(f"  - Log transforms: {len(validated['log_transforms'])} columns")
        logger.info(f"  - Polynomial: {len(validated['polynomial'])} columns")
        logger.info(f"  - Interactions: {len(validated['interactions'])} pairs")
        logger.info(f"  - Binning: {len(validated['binning'])} columns")
        
        # === Step 4: Apply transformations ===
        logger.info("Step 4: Applying feature transformations...")
        
        # 4.1 Log transforms
        if validated['log_transforms']:
            df, log_features = _apply_log_transforms(df, validated['log_transforms'])
            all_new_features.extend(log_features)
        
        # 4.2 Polynomial features
        if validated['polynomial']:
            df, poly_features = _apply_polynomial_features(df, validated['polynomial'])
            all_new_features.extend(poly_features)
        
        # 4.3 Interaction features
        if validated['interactions']:
            df, interaction_features = _apply_interaction_features(df, validated['interactions'])
            all_new_features.extend(interaction_features)
        
        # 4.4 Binning
        if validated['binning']:
            df, binned_features = _apply_binning(df, validated['binning'])
            all_new_features.extend(binned_features)
        
        logger.info(f"  - Total new features created: {len(all_new_features)}")
        
        # === Step 5: Update context ===
        logger.info("Step 5: Updating pipeline context...")
        ctx.clean_df = df
        
        # Store feature engineering report
        feature_report = {
            'original_columns': len(original_columns),
            'new_columns': len(df.columns),
            'features_created': len(all_new_features),
            'new_feature_names': all_new_features,
            'transformations_applied': {
                'log_transforms': validated['log_transforms'],
                'polynomial': validated['polynomial'],
                'interactions': validated['interactions'],
                'binning': [s.get('column', '') for s in validated['binning']]
            },
            'skewness_before': skewness
        }
        ctx.eda_summary["feature_engineering"] = feature_report
        
        # === Step 6: Generate narrative ===
        logger.info("Step 6: Generating feature engineering narrative...")
        narrative_prompt = _build_narrative_prompt(ctx, validated, all_new_features)
        
        if DEEPSEEK_API_KEY:
            try:
                client = OpenAI(
                    api_key=DEEPSEEK_API_KEY,
                    base_url="https://api.deepseek.com"
                )
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a data scientist explaining feature engineering results concisely."},
                        {"role": "user", "content": narrative_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                
                narrative = response.choices[0].message.content
                ctx.llm_narratives["feature_engineering"] = narrative
                logger.info("Feature engineering narrative generated successfully")
                
            except Exception as e:
                logger.warning(f"Failed to generate narrative: {e}")
                ctx.llm_narratives["feature_engineering"] = f"Feature engineering completed. Created {len(all_new_features)} new features."
        else:
            ctx.llm_narratives["feature_engineering"] = f"Feature engineering completed. Created {len(all_new_features)} new features including log transforms, polynomial terms, and interaction features."
        
        ctx.mark_agent("Feature Engineering", "done")
        logger.info("Feature Engineering Agent completed successfully")
        
    except Exception as e:
        error_msg = f"Error during Feature Engineering: {str(e)}"
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("Feature Engineering", "failed")
    
    return ctx
