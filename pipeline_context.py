"""DataForge Pipeline Context - Shared state across agents"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class PipelineContext:
    """Central state object passed through the agent pipeline."""
    
    # Dataset info
    dataset_path: str = ""
    dataset_name: str = ""
    
    # DataFrames
    raw_df: Optional[pd.DataFrame] = None
    clean_df: Optional[pd.DataFrame] = None
    
    # Target configuration
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # "classification" or "regression"
    has_target: bool = False
    
    # Future-proofing
    scaling_applied: bool = False  # for v2
    
    # Results storage
    eda_summary: Dict = field(default_factory=dict)
    cleaning_report: Dict = field(default_factory=dict)
    model_results: Dict = field(default_factory=dict)
    shap_results: Dict = field(default_factory=dict)
    plots: List[str] = field(default_factory=list)
    llm_narratives: Dict = field(default_factory=dict)
    recommendations: Dict = field(default_factory=dict)
    
    # Tracking
    errors: List[str] = field(default_factory=list)
    agent_status: Dict[str, str] = field(default_factory=dict)
    
    def mark_agent(self, name: str, status: str) -> None:
        """Update agent status.
        
        Args:
            name: Agent name
            status: One of 'pending', 'running', 'done', 'skipped', 'failed'
        """
        self.agent_status[name] = status
