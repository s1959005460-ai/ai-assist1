"""
monitoring.py

Wrapper utilities to log metrics to Weights & Biases (wandb) if available,
otherwise fall back to logger.secure_log. Keeps the rest of code agnostic.
"""

from typing import Dict, Any, Optional
from . import logger
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

_wandb_run = None

def init_wandb(project: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
    global _wandb_run
    if not _WANDB_AVAILABLE:
        logger.secure_log("info", "wandb not available; skipping init", project=project, run_name=name)
        return
    _wandb_run = wandb.init(project=project, name=name, config=config)
    logger.secure_log("info", "wandb initialized", project=project, run_name=name)

def log_metrics(step: int, metrics: Dict[str, Any]):
    if _WANDB_AVAILABLE and _wandb_run is not None:
        wandb.log({"step": step, **metrics})
    else:
        # fallback: structured logger
        logger.secure_log("info", "metrics", step=step, metrics=metrics)

def finish():
    if _WANDB_AVAILABLE and _wandb_run is not None:
        wandb.finish()
        logger.secure_log("info", "wandb finished")
