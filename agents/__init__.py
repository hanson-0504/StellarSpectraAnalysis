"""
Agent package for automating the spectral analysis pipeline.

Public API:
- preprocess_spectra(), train_models(), predict(), evaluate_results()
- modules: tools, state, rule_agent, llm_agent
"""

from . import tools, state, rule_agent, llm_agent

# Re-export the main tool functions at the package root
from .tools import (
    preprocess_spectra,
    train_models,
    predict_new,
    evaluate_results,
)

__all__ = [
    # modules
    "tools",
    "state",
    "rule_agent",
    "llm_agent",
    # functions
    "preprocess_spectra",
    "train_models",
    "predict_new",
    "evaluate_results",
]