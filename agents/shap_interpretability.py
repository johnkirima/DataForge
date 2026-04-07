"""DataForge Agent 7: SHAP Interpretability - Feature importance using SHAP values."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import config
from pipeline_context import PipelineContext
from logger import get_logger

logger = get_logger("SHAPInterpretability")


def run_shap_interpretability(ctx: PipelineContext) -> PipelineContext:
    """
    Compute SHAP values for feature importance and interpretability.
    Uses TreeExplainer for Random Forest, falls back to feature_importances_ if needed.
    
    Args:
        ctx: PipelineContext with model_results from Agent 6
        
    Returns:
        PipelineContext with shap_results populated
    """
    logger.info("=" * 50)
    logger.info("Starting Agent 7: SHAP Interpretability")
    logger.info("=" * 50)
    
    ctx.mark_agent("SHAP Interpretability", "running")
    
    # === Check Prerequisites ===
    if ctx.model_results is None:
        logger.warning("No model_results available. Skipping SHAP analysis.")
        ctx.mark_agent("SHAP Interpretability", "skipped")
        ctx.errors.append("SHAP Interpretability: No model_results available")
        return ctx
    
    model = ctx.model_results.get('model')
    X_test = ctx.model_results.get('X_test')
    
    if model is None:
        logger.warning("No trained model available. Skipping SHAP analysis.")
        ctx.mark_agent("SHAP Interpretability", "skipped")
        ctx.errors.append("SHAP Interpretability: No trained model available")
        return ctx
    
    if X_test is None:
        logger.warning("No X_test data available. Skipping SHAP analysis.")
        ctx.mark_agent("SHAP Interpretability", "skipped")
        ctx.errors.append("SHAP Interpretability: No X_test data available")
        return ctx
    
    # Initialize shap_results
    ctx.shap_results = {
        "shap_values": None,
        "feature_importance": {},
        "top_features": [],
        "plots": [],
        "method": None
    }
    
    try:
        # === Sample Data if Too Large ===
        if len(X_test) > config.SHAP_SAMPLE_SIZE:
            logger.info(f"Sampling {config.SHAP_SAMPLE_SIZE} rows from {len(X_test)} for SHAP analysis")
            X_sample = X_test.sample(n=config.SHAP_SAMPLE_SIZE, random_state=config.RANDOM_SEED)
        else:
            X_sample = X_test
        
        feature_names = ctx.model_results.get('feature_names', X_sample.columns.tolist())
        task_type = ctx.model_results.get('task_type', ctx.task_type)
        
        # === Try SHAP Analysis ===
        try:
            import shap
            logger.info("Computing SHAP values using TreeExplainer...")
            
            # Use TreeExplainer for Random Forest
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            logger.info("SHAP values computed successfully")
            
            # Handle classification (list of arrays) vs regression (single array)
            if isinstance(shap_values, list):
                # For classification: use absolute mean across all classes
                shap_importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
            else:
                # For regression: use absolute mean
                shap_importance = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance dict
            feature_importance = {}
            for i, name in enumerate(feature_names):
                feature_importance[name] = round(float(shap_importance[i]), 4)
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": k, "importance": v} for k, v in sorted_features[:10]]
            
            ctx.shap_results["shap_values"] = shap_values
            ctx.shap_results["feature_importance"] = feature_importance
            ctx.shap_results["top_features"] = top_features
            ctx.shap_results["method"] = "shap"
            
            logger.info(f"Top 5 SHAP features: {[f['feature'] for f in top_features[:5]]}")
            
            # === Generate SHAP Plots ===
            plots_generated = []
            reports_dir = os.path.join(ctx.run_dir, "plots")
            # Directory already created by PipelineContext.__post_init__
            
            # Summary Plot (Beeswarm)
            try:
                logger.info("Generating SHAP summary plot...")
                plt.figure(figsize=config.PLOT_FIGSIZE)
                
                if isinstance(shap_values, list):
                    # For classification, use class 1 or the positive class
                    shap.summary_plot(
                        shap_values[1] if len(shap_values) > 1 else shap_values[0],
                        X_sample,
                        feature_names=feature_names,
                        show=False,
                        plot_size=config.PLOT_FIGSIZE
                    )
                else:
                    shap.summary_plot(
                        shap_values,
                        X_sample,
                        feature_names=feature_names,
                        show=False,
                        plot_size=config.PLOT_FIGSIZE
                    )
                
                summary_path = os.path.join(reports_dir, "shap_summary_plot.png")
                plt.tight_layout()
                plt.savefig(summary_path, dpi=config.PLOT_DPI, bbox_inches='tight')
                plt.close('all')
                plots_generated.append(summary_path)
                logger.info(f"SHAP summary plot saved: {summary_path}")
            except Exception as e:
                logger.warning(f"Failed to generate SHAP summary plot: {e}")
                plt.close('all')
            
            # Bar Plot (Feature Importance)
            try:
                logger.info("Generating SHAP bar plot...")
                plt.figure(figsize=config.PLOT_FIGSIZE)
                
                if isinstance(shap_values, list):
                    shap.summary_plot(
                        shap_values[1] if len(shap_values) > 1 else shap_values[0],
                        X_sample,
                        feature_names=feature_names,
                        plot_type="bar",
                        show=False,
                        plot_size=config.PLOT_FIGSIZE
                    )
                else:
                    shap.summary_plot(
                        shap_values,
                        X_sample,
                        feature_names=feature_names,
                        plot_type="bar",
                        show=False,
                        plot_size=config.PLOT_FIGSIZE
                    )
                
                bar_path = os.path.join(reports_dir, "shap_feature_importance.png")
                plt.tight_layout()
                plt.savefig(bar_path, dpi=config.PLOT_DPI, bbox_inches='tight')
                plt.close('all')
                plots_generated.append(bar_path)
                logger.info(f"SHAP bar plot saved: {bar_path}")
            except Exception as e:
                logger.warning(f"Failed to generate SHAP bar plot: {e}")
                plt.close('all')
            
            ctx.shap_results["plots"] = plots_generated
            ctx.plots.extend(plots_generated)
            
        except ImportError:
            logger.warning("SHAP library not installed. Falling back to feature_importances_.")
            raise Exception("SHAP not available")
        except Exception as shap_error:
            logger.warning(f"SHAP analysis failed: {shap_error}. Falling back to feature_importances_.")
            
            # === Fallback to feature_importances_ ===
            logger.info("Using model.feature_importances_ as fallback...")
            
            importance = model.feature_importances_
            feature_importance = {}
            for i, name in enumerate(feature_names):
                feature_importance[name] = round(float(importance[i]), 4)
            
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = [{"feature": k, "importance": v} for k, v in sorted_features[:10]]
            
            ctx.shap_results["shap_values"] = None
            ctx.shap_results["feature_importance"] = feature_importance
            ctx.shap_results["top_features"] = top_features
            ctx.shap_results["method"] = "feature_importances"
            
            logger.info(f"Top 5 features (fallback): {[f['feature'] for f in top_features[:5]]}")
            
            # Generate fallback bar plot
            try:
                reports_dir = os.path.join(ctx.run_dir, "plots")
                # Directory already created by PipelineContext.__post_init__
                
                plt.figure(figsize=config.PLOT_FIGSIZE)
                top_10 = sorted_features[:10]
                features = [f[0] for f in top_10]
                importances = [f[1] for f in top_10]
                
                plt.barh(range(len(features)), importances, align='center')
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title('Top 10 Feature Importances (Random Forest)')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                fallback_path = os.path.join(reports_dir, "feature_importance_fallback.png")
                plt.savefig(fallback_path, dpi=config.PLOT_DPI, bbox_inches='tight')
                plt.close('all')
                
                ctx.shap_results["plots"] = [fallback_path]
                ctx.plots.append(fallback_path)
                logger.info(f"Fallback feature importance plot saved: {fallback_path}")
            except Exception as plot_error:
                logger.warning(f"Failed to generate fallback plot: {plot_error}")
                plt.close('all')
        
        ctx.mark_agent("SHAP Interpretability", "done")
        logger.info("Agent 7: SHAP Interpretability completed successfully")
        
    except Exception as e:
        logger.error(f"SHAP Interpretability failed with error: {str(e)}", exc_info=True)
        ctx.errors.append(f"SHAP Interpretability: {str(e)}")
        ctx.mark_agent("SHAP Interpretability", "failed")
        plt.close('all')  # Clean up any open plots
    
    return ctx
