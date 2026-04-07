"""DataForge Pipeline Context - Shared state across agents"""
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from logger import get_logger

_PIPE_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


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
    statistical_results: Dict = field(default_factory=dict)  # Agent 8: Statistical testing results
    plots: List[str] = field(default_factory=list)
    llm_narratives: Dict = field(default_factory=dict)
    recommendations: Dict = field(default_factory=dict)  # Agent 9: Final recommendations

    # Advisory warnings (leakage flags, schema issues, etc.) — never blocks pipeline
    warnings: Dict = field(default_factory=dict)

    # Tracking
    errors: List[str] = field(default_factory=list)
    agent_status: Dict[str, str] = field(default_factory=dict)

    # Streaming log buffer for UI consumption (capped at 500 lines)
    agent_logs: deque = field(default_factory=lambda: deque(maxlen=500))

    # Run identity (auto-generated in __post_init__ if empty)
    run_id: str = field(default="")
    run_dir: str = field(default="")

    def __post_init__(self) -> None:
        """Auto-generate run identity and wire a per-run FileHandler."""
        if not self.run_id:
            self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        if not self.run_dir:
            self.run_dir = f"runs/{self.run_id}"
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "plots"), exist_ok=True)
        # Per-run FileHandler — non-field attr, not serialised by dataclass
        log_path = os.path.join(self.run_dir, "pipeline.log")
        self._log_handler: logging.FileHandler = logging.FileHandler(
            log_path, encoding="utf-8"
        )
        self._log_handler.setLevel(logging.INFO)
        self._log_handler.setFormatter(logging.Formatter(_PIPE_FMT))
        logging.getLogger().addHandler(self._log_handler)
        self._handler_closed = False

    def close(self) -> None:
        """Release the per-run FileHandler immediately. Safe to call multiple times."""
        if getattr(self, "_handler_closed", True):
            return
        try:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler.close()
        except Exception:
            pass
        self._handler_closed = True

    def __del__(self) -> None:
        """Fallback cleanup in case close() was never called explicitly."""
        self.close()

    def mark_agent(self, name: str, status: str) -> None:
        """Update agent status.

        Args:
            name: Agent name
            status: One of 'pending', 'running', 'done', 'skipped', 'failed'
        """
        self.agent_status[name] = status

    def append_log(self, msg: str) -> None:
        """Append a log message to the UI buffer and emit to the file logger."""
        self.agent_logs.append(msg)
        get_logger().info(msg)

    def get_agent_logs(self, last_n: int = 100) -> List[str]:
        """Return the last N log lines as a list."""
        logs = list(self.agent_logs)
        return logs[-last_n:]
