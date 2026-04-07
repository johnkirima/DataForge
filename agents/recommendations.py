"""DataForge Agent 9: Recommendations - Final recommendations using Claude Sonnet 4.6."""
import time
from anthropic import Anthropic

import config
from pipeline_context import PipelineContext
from logger import get_logger

logger = get_logger("Recommendations")


def _build_comprehensive_prompt(ctx: PipelineContext) -> str:
    """Build comprehensive prompt summarizing entire pipeline execution."""
    
    # === Dataset Info ===
    dataset_info = f"""
Dataset: {ctx.dataset_name}
Shape: {ctx.clean_df.shape[0] if ctx.clean_df is not None else 'N/A'} rows x {ctx.clean_df.shape[1] if ctx.clean_df is not None else 'N/A'} columns
Target: {ctx.target_column if ctx.target_column else 'Not specified'}
Task: {ctx.task_type if ctx.task_type else 'Not determined'}
"""

    # === Data Quality Summary ===
    quality_audit = ctx.eda_summary.get("quality_audit", {})
    missing = quality_audit.get("missing_values", {})
    duplicates = quality_audit.get("duplicates", {})
    outliers = quality_audit.get("outliers", {})
    
    quality_summary = f"""
DATA QUALITY:
- Missing values: {missing.get('total_missing_cells', 0)} cells across {missing.get('columns_with_missing', 0)} columns
- Duplicates: {duplicates.get('count', 0)} rows ({duplicates.get('percentage', 0)}%)
- Columns with outliers: {outliers.get('columns_with_outliers', 0)}
"""

    # === Cleaning Summary ===
    cleaning = ctx.cleaning_report
    cleaning_summary = f"""
DATA CLEANING:
- Values imputed: {cleaning.get('missing_values', {}).get('total_values_imputed', 0)}
- Duplicates removed: {cleaning.get('duplicates', {}).get('duplicates_removed', 0)}
- Outliers capped: {cleaning.get('outliers', {}).get('total_values_capped', 0)}
"""

    # === EDA Summary ===
    correlations = ctx.eda_summary.get('correlations', {})
    top_positive = correlations.get('top_positive', [])
    top_negative = correlations.get('top_negative', [])
    
    corr_text = ""
    if top_positive:
        corr_text += f"Top positive correlations: {[(c[0], c[1], round(c[2], 3)) for c in top_positive[:3]]}\n"
    if top_negative:
        corr_text += f"Top negative correlations: {[(c[0], c[1], round(c[2], 3)) for c in top_negative[:3]]}"
    
    target_analysis = ctx.eda_summary.get('target_analysis', {})
    target_text = ""
    if target_analysis and 'error' not in target_analysis:
        if target_analysis.get('task_type') == 'classification':
            target_text = f"Classes: {target_analysis.get('num_classes', 'N/A')}, Balance: {target_analysis.get('balance_status', 'N/A')}"
        else:
            target_text = f"Target mean: {target_analysis.get('mean', 0):.2f}, std: {target_analysis.get('std', 0):.2f}"
    
    eda_summary = f"""
EXPLORATORY ANALYSIS:
{corr_text}
Target analysis: {target_text}
Plots generated: {len(ctx.plots)}
"""

    # === Feature Engineering Summary ===
    fe_report = ctx.eda_summary.get("feature_engineering", {})
    fe_summary = f"""
FEATURE ENGINEERING:
- Original columns: {fe_report.get('original_columns', 'N/A')}
- New features created: {fe_report.get('features_created', 0)}
- Final columns: {fe_report.get('new_columns', 'N/A')}
"""

    # === Model Performance ===
    model_results = ctx.model_results
    metrics = model_results.get('metrics', {})
    
    if ctx.task_type == 'classification':
        model_summary = f"""
MODEL PERFORMANCE:
- Algorithm: Random Forest Classifier
- Test Accuracy: {metrics.get('test_accuracy', 'N/A')}
- Precision: {metrics.get('precision', 'N/A')}
- Recall: {metrics.get('recall', 'N/A')}
- F1-score: {metrics.get('f1_score', 'N/A')}
- Cross-validation: {metrics.get('cv_mean', 'N/A')} ± {metrics.get('cv_std', 'N/A')}
"""
    else:
        model_summary = f"""
MODEL PERFORMANCE:
- Algorithm: Random Forest Regressor
- Test RMSE: {metrics.get('test_rmse', 'N/A')}
- Test MAE: {metrics.get('test_mae', 'N/A')}
- Test R²: {metrics.get('test_r2', 'N/A')}
- Cross-validation R²: {metrics.get('cv_mean', 'N/A')} ± {metrics.get('cv_std', 'N/A')}
"""

    # Top features
    top_features = model_results.get('feature_importance', [])[:5]
    if top_features:
        model_summary += f"Top features: {[f['feature'] for f in top_features]}\n"

    # === SHAP Interpretability ===
    shap_results = ctx.shap_results
    shap_method = shap_results.get('method', 'N/A')
    shap_top = shap_results.get('top_features', [])[:5]
    
    shap_summary = f"""
INTERPRETABILITY:
- Method: {shap_method}
- SHAP top features: {[f['feature'] for f in shap_top] if shap_top else 'N/A'}
"""

    # === Statistical Testing ===
    stat_results = ctx.statistical_results
    stat_summary = f"""
STATISTICAL TESTS:
- Tests performed: {stat_results.get('total_tests', 0)}
- Significant features (p<0.05): {len(stat_results.get('significant_features', []))}
- Significant features: {stat_results.get('significant_features', [])[:10]}
"""

    # === Previous Narratives ===
    narratives = ctx.llm_narratives
    narratives_summary = ""
    if narratives.get('quality_audit'):
        narratives_summary += f"Quality Audit Insight: {narratives['quality_audit'][:200]}...\n"
    if narratives.get('eda'):
        narratives_summary += f"EDA Insight: {narratives['eda'][:200]}...\n"
    if narratives.get('statistical_testing'):
        narratives_summary += f"Statistical Insight: {narratives['statistical_testing'][:200]}...\n"

    # === Build Full Prompt ===
    prompt = f"""You are a senior data scientist reviewing a complete ML pipeline execution. Provide actionable recommendations and next steps.

PIPELINE SUMMARY:
{dataset_info}
{quality_summary}
{cleaning_summary}
{eda_summary}
{fe_summary}
{model_summary}
{shap_summary}
{stat_summary}

PREVIOUS AI INSIGHTS:
{narratives_summary}

Based on this comprehensive analysis, provide:

1. **Overall Assessment** (2-3 sentences): How well did the pipeline perform? Is the model reliable?

2. **Key Strengths**: List 3-4 strengths of the current analysis

3. **Areas for Improvement**: List 3-4 areas that need improvement

4. **Actionable Next Steps**: Provide 5-7 specific, prioritized recommendations for improving the model or analysis

5. **Deployment Readiness**: Is this model ready for production deployment? Answer with Yes/No and provide brief reasoning (2-3 sentences)

Be specific and actionable. Focus on practical improvements the user can implement."""

    return prompt


def _call_claude_api(prompt: str, max_retries: int = 2) -> dict:
    """
    Call Claude Sonnet 4.6 API for recommendations.
    
    Args:
        prompt: The comprehensive prompt
        max_retries: Maximum number of retries
        
    Returns:
        dict with parsed recommendations
    """
    if not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not configured")
        return {
            "overall_assessment": "Recommendations not available (API key not configured).",
            "strengths": [],
            "improvements": [],
            "next_steps": [],
            "deployment_ready": False,
            "deployment_reasoning": "Unable to assess without API access.",
            "full_narrative": "Claude Sonnet 4.6 recommendations not available - API key not configured."
        }
    
    for attempt in range(max_retries + 1):
        try:
            client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            full_narrative = response.content[0].text.strip()
            
            # Parse the response into structured format
            parsed = _parse_recommendations(full_narrative)
            parsed["full_narrative"] = full_narrative
            
            return parsed
        
        except Exception as e:
            logger.error(f"Claude API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return {
                    "overall_assessment": f"Recommendations generation failed: {str(e)}",
                    "strengths": [],
                    "improvements": [],
                    "next_steps": [],
                    "deployment_ready": False,
                    "deployment_reasoning": "Unable to assess due to API error.",
                    "full_narrative": f"Claude Sonnet 4.6 API error: {str(e)}"
                }


def _parse_recommendations(narrative: str) -> dict:
    """
    Parse Claude's narrative response into structured format.
    
    Args:
        narrative: Raw text response from Claude
        
    Returns:
        dict with structured recommendations
    """
    result = {
        "overall_assessment": "",
        "strengths": [],
        "improvements": [],
        "next_steps": [],
        "deployment_ready": False,
        "deployment_reasoning": ""
    }
    
    try:
        lines = narrative.split('\n')
        current_section = None
        
        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Detect sections
            if 'overall assessment' in line_lower:
                current_section = 'assessment'
                continue
            elif 'key strengths' in line_lower or 'strengths' in line_lower and '**' in line:
                current_section = 'strengths'
                continue
            elif 'areas for improvement' in line_lower or 'improvement' in line_lower and '**' in line:
                current_section = 'improvements'
                continue
            elif 'actionable next steps' in line_lower or 'next steps' in line_lower and '**' in line:
                current_section = 'next_steps'
                continue
            elif 'deployment readiness' in line_lower or 'deployment' in line_lower and '**' in line:
                current_section = 'deployment'
                continue
            
            # Parse content based on current section
            if not line_stripped:
                continue
            
            if current_section == 'assessment':
                if result["overall_assessment"]:
                    result["overall_assessment"] += " " + line_stripped
                else:
                    result["overall_assessment"] = line_stripped
            
            elif current_section == 'strengths':
                if line_stripped.startswith(('-', '•', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    # Clean up the line
                    item = line_stripped.lstrip('-•*0123456789. ').strip()
                    if item:
                        result["strengths"].append(item)
            
            elif current_section == 'improvements':
                if line_stripped.startswith(('-', '•', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    item = line_stripped.lstrip('-•*0123456789. ').strip()
                    if item:
                        result["improvements"].append(item)
            
            elif current_section == 'next_steps':
                if line_stripped.startswith(('-', '•', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                    item = line_stripped.lstrip('-•*0123456789. ').strip()
                    if item:
                        result["next_steps"].append(item)
            
            elif current_section == 'deployment':
                if 'yes' in line_lower:
                    result["deployment_ready"] = True
                elif 'no' in line_lower:
                    result["deployment_ready"] = False
                
                # Capture reasoning
                if result["deployment_reasoning"]:
                    result["deployment_reasoning"] += " " + line_stripped
                else:
                    result["deployment_reasoning"] = line_stripped
        
        # Clean up deployment reasoning
        result["deployment_reasoning"] = result["deployment_reasoning"][:500]  # Limit length
        
    except Exception as e:
        logger.warning(f"Failed to parse recommendations: {e}")
        # Return raw narrative as assessment if parsing fails
        result["overall_assessment"] = narrative[:500]
    
    return result


def run_recommendations(ctx: PipelineContext) -> PipelineContext:
    """
    Generate final recommendations and next steps using Claude Sonnet 4.6.
    Analyzes entire pipeline execution and provides actionable insights.
    
    Args:
        ctx: PipelineContext with all previous agent results
        
    Returns:
        PipelineContext with recommendations populated
    """
    logger.info("=" * 50)
    logger.info("Starting Agent 9: Recommendations")
    logger.info("=" * 50)
    
    ctx.mark_agent("Recommendations", "running")
    
    # === Check Prerequisites ===
    # Agent 9 can run even with partial results, but needs some data
    if ctx.clean_df is None and ctx.raw_df is None:
        logger.warning("No data available. Skipping recommendations.")
        ctx.mark_agent("Recommendations", "skipped")
        ctx.errors.append("Recommendations: No data available")
        return ctx
    
    try:
        # === Build Comprehensive Prompt ===
        logger.info("Building comprehensive pipeline summary...")
        prompt = _build_comprehensive_prompt(ctx)
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # === Call Claude Sonnet 4.6 ===
        logger.info("Generating recommendations with Claude Sonnet 4.6...")
        recommendations = _call_claude_api(prompt)
        logger.info("Recommendations generated successfully")
        
        # === Store Results ===
        ctx.recommendations = recommendations
        ctx.llm_narratives["recommendations"] = recommendations.get("full_narrative", "")
        
        # Log summary
        logger.info(f"Strengths identified: {len(recommendations.get('strengths', []))}")
        logger.info(f"Improvements identified: {len(recommendations.get('improvements', []))}")
        logger.info(f"Next steps provided: {len(recommendations.get('next_steps', []))}")
        logger.info(f"Deployment ready: {recommendations.get('deployment_ready', False)}")
        
        ctx.mark_agent("Recommendations", "done")
        logger.info("Agent 9: Recommendations completed successfully")
        
    except Exception as e:
        logger.error(f"Recommendations failed with error: {str(e)}", exc_info=True)
        ctx.errors.append(f"Recommendations: {str(e)}")
        ctx.mark_agent("Recommendations", "failed")
    
    return ctx
