# agents/mentor.py
# Guidance & Mentor Mode - suggests next steps and guides user thinking

from typing import Dict, List, Any
import json

class MentorEngine:
    """Guides the user with actionable next steps based on pipeline results."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.guidance: List[Dict[str, str]] = []

    def _check_leakage(self):
        results = getattr(self.ctx, 'model_results', {})
        if not results:
            return
        metrics = results.get('metrics', {})
        score = metrics.get('test_accuracy', metrics.get('test_r2', 0))
        if score >= 0.99:
            self.guidance.append({
                'type': 'danger',
                'title': '🔴 Investigate Data Leakage',
                'message': (
                    f"Your model scored {score:.3f} — this is almost certainly data leakage. "
                    "A real-world model rarely achieves this. "
                    "Check if any feature directly encodes or derives from your target column."
                ),
                'action': "Remove features one by one and rerun. Start with the highest-correlated feature."
            })

    def _check_overfitting(self):
        results = getattr(self.ctx, 'model_results', {})
        if not results:
            return
        metrics = results.get('metrics', {})
        train = metrics.get('train_accuracy', metrics.get('train_r2', None))
        test = metrics.get('test_accuracy', metrics.get('test_r2', None))
        if train and test and (train - test) > 0.15:
            self.guidance.append({
                'type': 'warning',
                'title': '🟡 Reduce Overfitting',
                'message': (
                    f"'Random Forest' scores {train:.3f} on training but only {test:.3f} on test data. "
                    "This gap means the model memorized training data instead of learning patterns."
                ),
                'action': "Try: (1) Reduce model complexity, (2) Add more training data, (3) Apply regularization, (4) Remove noisy features."
            })

    def _check_small_dataset(self):
        df = getattr(self.ctx, 'clean_df', None)
        if df is not None and len(df) < 500:
            self.guidance.append({
                'type': 'warning',
                'title': '🟡 Small Dataset Risk',
                'message': (
                    f"Your dataset has only {len(df):,} rows. "
                    "Small datasets produce unreliable models that may not generalize."
                ),
                'action': "Try: (1) Collect more data, (2) Use cross-validation instead of a single train/test split, (3) Use simpler models."
            })

    def _check_feature_importance(self):
        shap = getattr(self.ctx, 'shap_results', {})
        if shap:
            top_features = shap.get('top_features', [])
            if top_features:
                top_feature = top_features[0].get('feature')
                if top_feature:
                    self.guidance.append({
                        'type': 'insight',
                        'title': '💡 Feature Insight',
                        'message': (
                            f"'{top_feature}' is your most influential feature. "
                            "This means the model relies heavily on it for predictions."
                        ),
                        'action': f"Ask yourself: Does '{top_feature}' make business sense as a predictor? If not, investigate why the model depends on it."
                    })

    def _suggest_next_experiment(self):
        results = getattr(self.ctx, 'model_results', {})
        if results:
            metrics = results.get('metrics', {})
            score = metrics.get('test_accuracy', metrics.get('test_r2', 0))
            self.guidance.append({
                'type': 'next_step',
                'title': '🚀 Suggested Next Experiment',
                'message': (
                    f"Your model scored {score:.3f}. "
                    "Understanding model performance helps you learn what patterns exist in your data."
                ),
                'action': (
                    f"Next experiment: Try hyperparameter tuning on your model "
                    "or try a different target column to explore a new business question."
                )
            })

    def _add_learning_prompt(self):
        self.guidance.append({
            'type': 'learning',
            'title': '📚 Think Like a Data Scientist',
            'message': (
                "Before trusting any model result, always ask: "
                "(1) Does this make business sense? "
                "(2) Would a domain expert agree? "
                "(3) What would happen if I removed the top feature?"
            ),
            'action': "Discuss your results with a stakeholder before making decisions. Data Science is a conversation, not a calculation."
        })

    def generate_all(self) -> List[Dict[str, str]]:
        self._check_leakage()
        self._check_overfitting()
        self._check_small_dataset()
        self._check_feature_importance()
        self._suggest_next_experiment()
        self._add_learning_prompt()
        return self.guidance

    def to_json(self) -> str:
        return json.dumps(self.generate_all(), indent=2)
