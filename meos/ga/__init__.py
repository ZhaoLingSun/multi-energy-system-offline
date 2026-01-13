"""
MEOS GA Module - 遗传算法优化模块
"""

from .codec import (
    Codec,
    CodecConfig,
    Individual,
    PLAN18_BOUNDS,
    PRICE12_BOUNDS,
    PLAN18_BASE_CAPACITY,
)

from .ga_core import (
    GeneticAlgorithm,
    GAConfig,
    Population,
)

from .evaluator import (
    Candidate,
    EvalResult,
    EvaluatorConfig,
    CandidateEvaluator,
    EvalCache,
    compute_cache_key,
    evaluate_candidate,
    evaluate_candidates,
    PLAN18_DEVICE_NAMES,
)

__all__ = [
    # Codec
    "Codec",
    "CodecConfig",
    "Individual",
    "PLAN18_BOUNDS",
    "PRICE12_BOUNDS",
    "PLAN18_BASE_CAPACITY",
    # GA Core
    "GeneticAlgorithm",
    "GAConfig",
    "Population",
    # Evaluator
    "Candidate",
    "EvalResult",
    "EvaluatorConfig",
    "CandidateEvaluator",
    "EvalCache",
    "compute_cache_key",
    "evaluate_candidate",
    "evaluate_candidates",
    "PLAN18_DEVICE_NAMES",
]
