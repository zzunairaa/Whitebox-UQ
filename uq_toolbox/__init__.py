__version__ = "0.1.0"

from .core.pipeline import (
    compute_dataset_uq_scores,
    compute_batch_uqlm_scores,
)
from .core.uq_engine import evaluate_uncertainty
from .core.response_evaluator import (
    BaseResponseEvaluator,
    SubstringMatchEvaluator,
)

from .registry import UQ_REGISTRY

from .managers.llama_cpp_manager import LlamaCppManager
from .managers.model_manager import (
    UQModelManager,
    initialize_uq_models,
)

from .learned_uq import (
    SupervisedUQManager,
    evaluate_supervised_uncertainty,
    evaluate_supervised_batch,
)

__all__ = [
    "compute_dataset_uq_scores",
    "compute_batch_uqlm_scores",
    "evaluate_uncertainty",
    "UQ_REGISTRY",
    "BaseResponseEvaluator",
    "SubstringMatchEvaluator",
    "LlamaCppManager",
    "UQModelManager",
    "initialize_uq_models",
    "SupervisedUQManager",
    "evaluate_supervised_uncertainty",
    "evaluate_supervised_batch",
    "__version__",
]
