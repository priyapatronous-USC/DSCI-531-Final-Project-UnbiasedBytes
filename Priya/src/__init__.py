"""
DSCI 531 Bias Evaluation Package

This package contains modules for evaluating bias in Large Language Models
using established benchmarks (StereoSet, CrowS-Pairs, BBQ).
"""

# Core data loading function (most commonly used)
from .data_preprocessing import load_all_preprocessed

# Optional imports for advanced usage
try:
    from .config import *
    from .model_scoring import BiasEvaluatorModel, DecodeConfig
    from .evaluation_pipeline import ensure_dir, summarize_all, run_full_pipeline
except ImportError:
    # Allow basic data loading even if other modules have import issues
    pass

__version__ = "1.0.0"
__author__ = "DSCI 531 Student"