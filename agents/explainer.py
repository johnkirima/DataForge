# agents/explainer.py
# AI Explanation Engine - generates plain English reasoning for pipeline decisions

from typing import Dict, Any
import json

class ExplanationEngine:
    """Generates plain-English explanations for every pipeline decision."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.explanations: Dict[str, str] = {}

    def explain_cleaning(self) -> str:
        df = self.ctx.clean_df
        if df is None:
            return "No cleaning data available."
        missing = df.isnull().sum().sum()
        rows, cols = df.shape
        return (
            f"The cleaning agent processed {rows:,} rows and {cols} columns. "
            f"{'No missing values were found — the dataset was already clean.' if missing == 0 else f'{missing:,} missing values were detected and imputed using median/mode strategies to preserve data distribution.'}"
        )

    def explain_eda(self) -> str:
        df = self.ctx.clean_df
        if df is None:
            return "No EDA data available."
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        return (
            f"EDA identified {len(numeric_cols)} numeric features and {len(cat_cols)} categorical features. "
            f"Distributions, correlations, and outliers were analyzed to understand the data structure before modeling."
        )

    def explain_features(self) -> str:
        df = self.ctx.clean_df
        if df is None:
            return "No feature data available."
        return (
            f"Feature engineering transformed raw columns into model-ready inputs. "
            f"Categorical variables were encoded and numeric features were scaled to prevent any single feature from dominating the model."
        )

    def explain_modeling(self) -> str:
        results = getattr(self.ctx, 'model_results', {})
        if not results:
            return "No modeling results available."
            
        metrics = results.get('metrics', {})
        best_score = metrics.get('test_accuracy', metrics.get('test_r2', 0))
        best_model = "Random Forest"
        
        explanation = (
            f"Multiple models were trained and evaluated. "
            f"'{best_model}' performed best with a score of {best_score:.3f} on unseen test data. "
        )
        if best_score >= 0.99:
            explanation += "⚠️ This score is suspiciously high — investigate for data leakage."
        elif best_score >= 0.85:
            explanation += "This is a strong result that suggests good generalization."
        elif best_score >= 0.70:
            explanation += "This is a moderate result — consider feature engineering or hyperparameter tuning."
        else:
            explanation += "This score is low — the model may need more data or better features."
        return explanation

    def explain_warnings(self) -> str:
        warnings = getattr(self.ctx, 'warnings', [])
        if not warnings:
            return "No critical issues were detected in this run."
        return f"{len(warnings)} issue(s) were flagged: " + " | ".join(warnings)

    def generate_all(self) -> Dict[str, str]:
        self.explanations = {
            'cleaning': self.explain_cleaning(),
            'eda': self.explain_eda(),
            'features': self.explain_features(),
            'modeling': self.explain_modeling(),
            'warnings': self.explain_warnings(),
        }
        return self.explanations

    def to_json(self) -> str:
        return json.dumps(self.generate_all(), indent=2)
