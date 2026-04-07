"""DataForge Agents Package - Pipeline Agents for Data Science Automation"""

from agents.data_ingestion import run_data_ingestion
from agents.data_quality_audit import run_data_quality_audit
from agents.data_cleaning import run_data_cleaning
from agents.eda import run_eda
from agents.feature_engineering import run_feature_engineering
from agents.modeling import run_modeling
from agents.shap_interpretability import run_shap_interpretability
from agents.statistical_testing import run_statistical_testing
from agents.recommendations import run_recommendations

__all__ = [
    'run_data_ingestion',
    'run_data_quality_audit',
    'run_data_cleaning',
    'run_eda',
    'run_feature_engineering',
    'run_modeling',
    'run_shap_interpretability',
    'run_statistical_testing',
    'run_recommendations'
]
