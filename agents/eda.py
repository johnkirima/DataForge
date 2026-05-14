"""
Agent 4: Exploratory Data Analysis (EDA)
=========================================
Perform comprehensive EDA with statistics, plots, and narrative using DeepSeek V3.2.
Operations: descriptive stats, correlation analysis, visualizations, target analysis.
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from openai import OpenAI

from pipeline_context import PipelineContext
import config
from config import DEEPSEEK_API_KEY, RANDOM_SEED
from logger import get_logger

logger = get_logger("EDA")

# Set seaborn style globally
sns.set_style("whitegrid")

# Constants for plot limits
MAX_TOTAL_PLOTS = 12
MAX_NUMERIC_DIST_PLOTS = 5
MAX_CATEGORICAL_COUNT_PLOTS = 3
MAX_BOXPLOTS = 3
MAX_CATEGORIES_FOR_PLOT = 50


def _get_timestamp_prefix() -> str:
    """Generate timestamp prefix for plot filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _compute_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute descriptive statistics for all columns.
    
    Returns:
        Dictionary with numeric and categorical statistics.
    """
    stats_result = {
        'numeric': {},
        'categorical': {}
    }
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        q1 = col_data.quantile(0.25)
        q2 = col_data.quantile(0.50)
        q3 = col_data.quantile(0.75)
        
        stats_result['numeric'][col] = {
            'mean': float(col_data.mean()),
            'median': float(col_data.median()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'Q1': float(q1),
            'Q2': float(q2),
            'Q3': float(q3),
            'skewness': float(col_data.skew()) if len(col_data) > 2 else 0.0,
            'kurtosis': float(col_data.kurtosis()) if len(col_data) > 3 else 0.0
        }
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        value_counts = col_data.value_counts()
        top_10 = value_counts.head(10).to_dict()
        
        stats_result['categorical'][col] = {
            'unique_count': int(col_data.nunique()),
            'top_10_values': {str(k): int(v) for k, v in top_10.items()},
            'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None
        }
    
    return stats_result


def _compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute correlation matrix and identify top correlations.
    
    Returns:
        Dictionary with correlation matrix info and top correlations.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {
            'matrix': None,
            'top_positive': [],
            'top_negative': [],
            'error': 'Insufficient numeric columns for correlation analysis'
        }
    
    corr_matrix = df[numeric_cols].corr()
    
    # Extract upper triangle (exclude diagonal)
    correlations = []
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if i < j:  # Upper triangle only
                corr_val = corr_matrix.loc[col1, col2]
                if pd.notna(corr_val):
                    correlations.append((col1, col2, float(corr_val)))
    
    # Sort by absolute correlation value
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Top 5 positive and negative correlations
    positive_corrs = [(c[0], c[1], c[2]) for c in correlations if c[2] > 0][:5]
    negative_corrs = [(c[0], c[1], c[2]) for c in correlations if c[2] < 0][:5]
    
    return {
        'matrix': corr_matrix.to_dict(),
        'top_positive': positive_corrs,
        'top_negative': negative_corrs
    }


def _compute_target_analysis(df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
    """
    Analyze target variable distribution and relationships.
    
    Args:
        df: DataFrame with clean data
        target_column: Name of target column
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with target analysis results.
    """
    if target_column not in df.columns:
        return {'error': f'Target column "{target_column}" not found'}
    
    target_data = df[target_column].dropna()
    
    result = {
        'column': target_column,
        'task_type': task_type,
        'total_count': len(target_data),
        'missing_count': df[target_column].isnull().sum()
    }
    
    if task_type == 'classification':
        value_counts = target_data.value_counts()
        value_percentages = (value_counts / len(target_data) * 100).round(2)
        
        result['class_distribution'] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
        result['class_percentages'] = {str(k): float(v) for k, v in value_percentages.to_dict().items()}
        result['num_classes'] = len(value_counts)
        
        # Check class balance (imbalanced if any class < 20% or > 80%)
        min_pct = value_percentages.min()
        max_pct = value_percentages.max()
        result['is_balanced'] = min_pct >= 20.0 and max_pct <= 80.0
        result['balance_status'] = 'Balanced' if result['is_balanced'] else 'Imbalanced'
        
    elif task_type == 'regression':
        result['mean'] = float(target_data.mean())
        result['median'] = float(target_data.median())
        result['std'] = float(target_data.std())
        result['min'] = float(target_data.min())
        result['max'] = float(target_data.max())
        result['skewness'] = float(target_data.skew()) if len(target_data) > 2 else 0.0
    
    # Feature-target correlations (for numeric features)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols and len(numeric_cols) > 1:
        feature_correlations = {}
        for col in numeric_cols:
            if col != target_column:
                corr = df[col].corr(df[target_column])
                if pd.notna(corr):
                    feature_correlations[col] = float(corr)
        
        # Sort by absolute correlation
        sorted_corrs = sorted(feature_correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        result['feature_correlations'] = dict(sorted_corrs[:10])
    
    return result


def _save_plot(fig: plt.Figure, filename: str, reports_dir: str) -> str:
    """Save plot and close figure. Returns full path."""
    plot_path = os.path.join(reports_dir, filename)
    fig.savefig(plot_path, dpi=config.PLOT_DPI, bbox_inches='tight')
    plt.close('all')  # CRITICAL: prevent memory leak
    return plot_path


def _generate_correlation_heatmap(df: pd.DataFrame, reports_dir: str, timestamp: str) -> Optional[str]:
    """Generate and save correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        logger.warning("Insufficient numeric columns for correlation heatmap")
        return None
    
    # Limit columns for readability
    cols_to_plot = numeric_cols[:15]  # Max 15 columns for heatmap
    
    try:
        fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
        corr_matrix = df[cols_to_plot].corr()
        
        # Adjust font size based on number of columns
        annot_size = 8 if len(cols_to_plot) <= 8 else 6
        
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt='.2f',
            annot_kws={'size': annot_size},
            ax=ax
        )
        ax.set_title('Correlation Heatmap', fontsize=10)
        plt.tight_layout()
        
        filename = f"{timestamp}_correlation_heatmap.png"
        plot_path = _save_plot(fig, filename, reports_dir)
        logger.info(f"Saved correlation heatmap: {filename}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")
        plt.close('all')
        return None


def _generate_distribution_plots(df: pd.DataFrame, reports_dir: str, timestamp: str, max_plots: int = 5) -> List[str]:
    """Generate histogram distribution plots for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_paths = []
    
    # Prioritize columns with more variation (skip low-cardinality)
    cols_with_variation = [col for col in numeric_cols if df[col].nunique() > 10]
    cols_to_plot = cols_with_variation[:max_plots]
    
    for col in cols_to_plot:
        try:
            fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
            
            data = df[col].dropna()
            sns.histplot(data=data, kde=True, ax=ax, color='steelblue')
            ax.set_title(f'Distribution of {col}', fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            
            # Sanitize column name for filename
            safe_col_name = col.replace(' ', '_').replace('/', '_')[:30]
            filename = f"{timestamp}_{safe_col_name}_distribution.png"
            plot_path = _save_plot(fig, filename, reports_dir)
            plot_paths.append(plot_path)
            logger.info(f"Saved distribution plot: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating distribution plot for {col}: {e}")
            plt.close('all')
    
    return plot_paths


def _generate_count_plots(df: pd.DataFrame, reports_dir: str, timestamp: str, max_plots: int = 3) -> List[str]:
    """Generate count plots for categorical columns (top 10 categories)."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    plot_paths = []
    
    # Filter columns with reasonable number of unique values
    cols_to_plot = [col for col in categorical_cols if df[col].nunique() <= MAX_CATEGORIES_FOR_PLOT][:max_plots]
    
    for col in cols_to_plot:
        try:
            fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
            
            # Get top 10 categories
            value_counts = df[col].value_counts().head(10)
            
            sns.barplot(x=value_counts.values, y=value_counts.index, hue=value_counts.index, ax=ax, palette='viridis', legend=False)
            ax.set_title(f'Top Categories: {col}', fontsize=10)
            ax.set_xlabel('Count')
            ax.set_ylabel(col)
            plt.tight_layout()
            
            safe_col_name = col.replace(' ', '_').replace('/', '_')[:30]
            filename = f"{timestamp}_{safe_col_name}_counts.png"
            plot_path = _save_plot(fig, filename, reports_dir)
            plot_paths.append(plot_path)
            logger.info(f"Saved count plot: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating count plot for {col}: {e}")
            plt.close('all')
    
    return plot_paths


def _generate_boxplots(df: pd.DataFrame, reports_dir: str, timestamp: str, max_plots: int = 3) -> List[str]:
    """Generate box plots for numeric columns to show outlier ranges."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_paths = []
    
    # Prioritize columns with variation
    cols_with_variation = [col for col in numeric_cols if df[col].nunique() > 10]
    cols_to_plot = cols_with_variation[:max_plots]
    
    for col in cols_to_plot:
        try:
            fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
            
            data = df[col].dropna()
            sns.boxplot(x=data, ax=ax, color='lightcoral')
            ax.set_title(f'Box Plot: {col}', fontsize=10)
            ax.set_xlabel(col)
            plt.tight_layout()
            
            safe_col_name = col.replace(' ', '_').replace('/', '_')[:30]
            filename = f"{timestamp}_{safe_col_name}_boxplot.png"
            plot_path = _save_plot(fig, filename, reports_dir)
            plot_paths.append(plot_path)
            logger.info(f"Saved box plot: {filename}")
            
        except Exception as e:
            logger.error(f"Error generating box plot for {col}: {e}")
            plt.close('all')
    
    return plot_paths


def _generate_target_distribution_plot(df: pd.DataFrame, target_column: str, task_type: str, 
                                        reports_dir: str, timestamp: str) -> Optional[str]:
    """Generate target variable distribution plot."""
    if target_column not in df.columns:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=config.PLOT_FIGSIZE)
        target_data = df[target_column].dropna()
        
        if task_type == 'classification':
            value_counts = target_data.value_counts()
            sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, hue=value_counts.index.astype(str), ax=ax, palette='Set2', legend=False)
            ax.set_title(f'Target Distribution: {target_column}', fontsize=10)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
        else:  # regression
            sns.histplot(data=target_data, kde=True, ax=ax, color='forestgreen')
            ax.set_title(f'Target Distribution: {target_column}', fontsize=10)
            ax.set_xlabel(target_column)
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        
        filename = f"{timestamp}_target_distribution.png"
        plot_path = _save_plot(fig, filename, reports_dir)
        logger.info(f"Saved target distribution plot: {filename}")
        return plot_path
        
    except Exception as e:
        logger.error(f"Error generating target distribution plot: {e}")
        plt.close('all')
        return None


def _build_eda_llm_prompt(ctx: PipelineContext, desc_stats: Dict, correlations: Dict, 
                          target_analysis: Optional[Dict]) -> str:
    """Build a concise prompt for DeepSeek V3.2 to summarize EDA findings."""
    
    df = ctx.clean_df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    prompt = f"""You are a data scientist performing Exploratory Data Analysis. Analyze the following summary and provide insights:

Dataset: {ctx.dataset_name}
Shape: {df.shape[0]} rows, {df.shape[1]} columns
Numeric columns: {len(numeric_cols)}
Categorical columns: {len(categorical_cols)}

Key Statistics (top 5 numeric columns):
"""
    
    # Add key stats for top numeric columns
    numeric_stats = desc_stats.get('numeric', {})
    for i, (col, stats_dict) in enumerate(list(numeric_stats.items())[:5]):
        skew = stats_dict.get('skewness', 0)
        skew_label = "normal" if abs(skew) < 0.5 else ("right-skewed" if skew > 0 else "left-skewed")
        prompt += f"- {col}: mean={stats_dict['mean']:.2f}, median={stats_dict['median']:.2f}, std={stats_dict['std']:.2f}, skewness={skew:.2f} ({skew_label})\n"
    
    # Add correlation info
    prompt += "\nTop 5 Positive Correlations:\n"
    for col1, col2, corr in correlations.get('top_positive', []):
        prompt += f"- {col1} ↔ {col2}: {corr:.2f}\n"
    
    if not correlations.get('top_positive'):
        prompt += "- No strong positive correlations found\n"
    
    prompt += "\nTop 5 Negative Correlations:\n"
    for col1, col2, corr in correlations.get('top_negative', []):
        prompt += f"- {col1} ↔ {col2}: {corr:.2f}\n"
    
    if not correlations.get('top_negative'):
        prompt += "- No strong negative correlations found\n"
    
    # Add target analysis if available
    if target_analysis and 'error' not in target_analysis:
        prompt += f"\nTarget Variable: {target_analysis.get('column', 'N/A')}\n"
        if target_analysis.get('task_type') == 'classification':
            prompt += f"- Classes: {target_analysis.get('num_classes', 'N/A')}\n"
            prompt += f"- Distribution: {target_analysis.get('class_distribution', {})}\n"
            prompt += f"- Balance: {target_analysis.get('balance_status', 'N/A')}\n"
        else:
            prompt += f"- Mean: {target_analysis.get('mean', 0):.2f}, Std: {target_analysis.get('std', 0):.2f}\n"
    
    prompt += """
Provide:
1. Key insights (3-4 bullet points)
2. Interesting patterns or relationships
3. Recommendations for modeling

Keep response concise (150-200 words)."""
    
    return prompt


def _call_deepseek_api(prompt: str, max_retries: int = 2) -> Optional[str]:
    """Call DeepSeek V3.2 API for EDA narrative generation."""
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
                logger.info(f"Calling DeepSeek API for EDA narrative (attempt {attempt + 1}/{max_retries + 1})")
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful data scientist providing concise EDA insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
                
                narrative = response.choices[0].message.content
                logger.info("DeepSeek API call successful for EDA narrative")
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


def run_eda(ctx: PipelineContext) -> PipelineContext:
    """
    Perform comprehensive EDA on ctx.clean_df.
    Generates stats, plots, and narrative using DeepSeek V3.2.
    
    Args:
        ctx: PipelineContext with clean_df populated by Agent 3
        
    Returns:
        Updated PipelineContext with eda_summary, plots, and llm_narratives["eda"]
    """
    logger.info("=" * 50)
    logger.info("Starting EDA Agent")
    logger.info("=" * 50)
    
    # Check if clean_df is available
    if ctx.clean_df is None:
        logger.warning("clean_df is None - Agent 3 likely failed. Skipping EDA.")
        ctx.mark_agent("EDA", "skipped")
        ctx.errors.append("EDA skipped: No cleaned data available from Agent 3")
        return ctx
    
    # Check for empty DataFrame
    if ctx.clean_df.empty:
        logger.warning("clean_df is empty. Skipping EDA.")
        ctx.mark_agent("EDA", "skipped")
        ctx.errors.append("EDA skipped: Dataset is empty")
        return ctx
    
    ctx.mark_agent("EDA", "running")
    
    # Setup
    df = ctx.clean_df
    reports_dir = os.path.join(ctx.run_dir, "plots")
    # Directory already created by PipelineContext.__post_init__
    timestamp = _get_timestamp_prefix()
    plot_paths = []
    
    logger.info(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Missing Value Check
    missing_ratio = df.isnull().mean().mean()
    if missing_ratio > 0.3:
        ctx.warnings.append(
            "⚠️ High Missing Values: More than 30% of data is missing."
        )
    
    try:
        # === 1. Descriptive Statistics ===
        logger.info("Step 1: Computing descriptive statistics...")
        desc_stats = _compute_descriptive_stats(df)
        ctx.eda_summary["descriptive_stats"] = desc_stats
        logger.info(f"  - Computed stats for {len(desc_stats['numeric'])} numeric and {len(desc_stats['categorical'])} categorical columns")
        
        # === 2. Correlation Analysis ===
        logger.info("Step 2: Computing correlation analysis...")
        correlations = _compute_correlations(df)
        ctx.eda_summary["correlations"] = correlations
        logger.info(f"  - Found {len(correlations.get('top_positive', []))} positive and {len(correlations.get('top_negative', []))} negative correlations")
        
        # === 3. Target Analysis (if applicable) ===
        target_analysis = None
        if ctx.has_target and ctx.target_column:
            logger.info(f"Step 3: Analyzing target variable '{ctx.target_column}'...")
            target_analysis = _compute_target_analysis(df, ctx.target_column, ctx.task_type or 'classification')
            ctx.eda_summary["target_analysis"] = target_analysis
            logger.info(f"  - Target analysis completed for '{ctx.target_column}'")
        else:
            logger.info("Step 3: No target variable specified - skipping target analysis")
        
        # === 4. Generate Visualizations ===
        logger.info("Step 4: Generating visualizations...")
        plots_generated = 0
        
        # 4.1 Correlation Heatmap
        if plots_generated < MAX_TOTAL_PLOTS:
            heatmap_path = _generate_correlation_heatmap(df, reports_dir, timestamp)
            if heatmap_path:
                plot_paths.append(heatmap_path)
                plots_generated += 1
        
        # 4.2 Distribution Plots (top 5 numeric columns)
        remaining_slots = MAX_TOTAL_PLOTS - plots_generated
        num_dist_plots = min(MAX_NUMERIC_DIST_PLOTS, remaining_slots)
        if num_dist_plots > 0:
            dist_paths = _generate_distribution_plots(df, reports_dir, timestamp, max_plots=num_dist_plots)
            plot_paths.extend(dist_paths)
            plots_generated += len(dist_paths)
        
        # 4.3 Count Plots (top 3 categorical columns)
        remaining_slots = MAX_TOTAL_PLOTS - plots_generated
        num_count_plots = min(MAX_CATEGORICAL_COUNT_PLOTS, remaining_slots)
        if num_count_plots > 0:
            count_paths = _generate_count_plots(df, reports_dir, timestamp, max_plots=num_count_plots)
            plot_paths.extend(count_paths)
            plots_generated += len(count_paths)
        
        # 4.4 Box Plots (top 3 numeric columns)
        remaining_slots = MAX_TOTAL_PLOTS - plots_generated
        num_boxplots = min(MAX_BOXPLOTS, remaining_slots)
        if num_boxplots > 0:
            box_paths = _generate_boxplots(df, reports_dir, timestamp, max_plots=num_boxplots)
            plot_paths.extend(box_paths)
            plots_generated += len(box_paths)
        
        # 4.5 Target Distribution Plot (if applicable)
        if ctx.has_target and ctx.target_column and plots_generated < MAX_TOTAL_PLOTS:
            target_plot_path = _generate_target_distribution_plot(
                df, ctx.target_column, ctx.task_type or 'classification', reports_dir, timestamp
            )
            if target_plot_path:
                plot_paths.append(target_plot_path)
                plots_generated += 1
        
        ctx.plots = plot_paths
        logger.info(f"  - Generated {len(plot_paths)} plots")
        
        # === 5. Generate LLM Narrative ===
        logger.info("Step 5: Generating LLM narrative using DeepSeek V3.2...")
        prompt = _build_eda_llm_prompt(ctx, desc_stats, correlations, target_analysis)
        narrative = _call_deepseek_api(prompt)
        
        if narrative:
            ctx.llm_narratives["eda"] = narrative
            logger.info("LLM EDA narrative generated successfully")
        else:
            ctx.llm_narratives["eda"] = "LLM narrative generation failed or API key not configured."
            logger.warning("LLM EDA narrative not generated")
        
        ctx.mark_agent("EDA", "done")
        logger.info("EDA Agent completed successfully")
        
    except Exception as e:
        error_msg = f"Error during EDA: {str(e)}"
        logger.error(error_msg)
        ctx.errors.append(error_msg)
        ctx.mark_agent("EDA", "failed")
        plt.close('all')  # Ensure all plots are closed on error
    
    return ctx
