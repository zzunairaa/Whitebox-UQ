from typing import Any, Dict, List, Optional

from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.estimators import RDESeq
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.factory_stat_calculator import (
    StatCalculatorContainer,
)
from lm_polygraph.utils.manager import UEManager
from omegaconf import OmegaConf


def _build_rde_training_config(
    training_size: int,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    """Build the reference-dataset configuration required by RDE."""

    return {
        "dataset": ["qiaojin/PubMedQA", "pqa_labeled"],
        "train_dataset": ["qiaojin/PubMedQA", "pqa_labeled"],
        "text_column": "question",
        "label_column": "final_decision",
        "train_split": "train",
        "few_shot_split": "train",
        "prompt": "{question}",

        # Lightweight reference fitting for the tutorial
        "size": training_size,
        "subsample_train_dataset": training_size,
        "batch_size": batch_size,
        "seed": seed,

        # Required by the current LM-Polygraph configuration
        "description": "",
        "load_from_disk": False,
        "train_test_split": False,
        "n_shot": 0,
        "bg_size": 1,

        "background_train_dataset": "allenai/c4",
        "background_train_dataset_text_column": "text",
        "background_train_dataset_label_column": "url",
        "background_train_dataset_data_files": (
            "en/c4-train.00000-of-01024.json.gz"
        ),
        "background_load_from_disk": False,
        "subsample_background_train_dataset": 1,
    }


def _extract_rde_scores(
    manager: UEManager,
) -> List[float]:
    """Extract sequence-level RDE scores from a completed manager run."""

    for key, scores in manager.estimations.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue

        level, estimator_name = key

        if level == "sequence" and "RDESeq" in str(estimator_name):
            return [float(score) for score in scores]

    available = [
        str(key)
        for key in manager.estimations.keys()
    ]

    raise RuntimeError(
        "RDESeq sequence-level scores were not found in "
        f"manager.estimations. Available keys: {available}"
    )


def run_rde_sequence(
    uq_context: Any,
    prompts: List[str],
    prompt_labels: Optional[List[str]] = None,
    model_alias: str = "qwen",
    training_size: int = 20,
    batch_size: int = 1,
    max_new_tokens: int = 8,
    seed: int = 42,
    layer: str = "decoder",
    training_config_overrides: Optional[
        Dict[str, Any]
    ] = None,
    return_manager: bool = False,
) -> Dict[str, Any]:
    """
    Run LM-Polygraph Robust Density Estimation at sequence level.

    Args:
        uq_context:
            Model manager containing registered LM-Polygraph models.
        prompts:
            Prompts to evaluate.
        prompt_labels:
            Optional readable labels for the prompts.
        model_alias:
            Alias used in uq_context.polygraph_models.
        training_size:
            Number of PubMedQA examples used to fit the reference
            distribution.
        batch_size:
            Batch size for reference fitting and evaluation.
        max_new_tokens:
            Maximum number of generated tokens.
        seed:
            Dataset-sampling seed.
        layer:
            Representation layer used by RDESeq.
        training_config_overrides:
            Optional values overriding the default training configuration.
        return_manager:
            Include the completed UEManager object in the result.

    Returns:
        Dictionary containing:
            - scores: sequence-level RDE uncertainty scores
            - rows: result rows for display and comparison
            - settings: configuration used for the experiment
            - estimations: raw LM-Polygraph estimations
            - manager: included only when return_manager=True
    """

    if not prompts:
        raise ValueError(
            "prompts must contain at least one prompt."
        )

    if not all(
        isinstance(prompt, str) and prompt.strip()
        for prompt in prompts
    ):
        raise ValueError(
            "Every prompt must be a non-empty string."
        )

    if training_size <= 0:
        raise ValueError(
            "training_size must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than zero."
        )

    if prompt_labels is None:
        prompt_labels = prompts

    if len(prompt_labels) != len(prompts):
        raise ValueError(
            "prompt_labels and prompts must contain "
            "the same number of items."
        )

    if model_alias not in uq_context.polygraph_models:
        available = list(
            uq_context.polygraph_models.keys()
        )

        raise KeyError(
            f"No Polygraph model registered as '{model_alias}'. "
            f"Available aliases: {available}"
        )

    polygraph_model = (
        uq_context.polygraph_models[model_alias]
    )

    evaluation_dataset = Dataset(
        prompts,
        [""] * len(prompts),
        batch_size=batch_size,
    )

    training_config = _build_rde_training_config(
        training_size=training_size,
        batch_size=batch_size,
        seed=seed,
    )

    if training_config_overrides:
        training_config.update(
            training_config_overrides
        )

    calculators = {
        calculator.name: calculator
        for calculator in register_default_stat_calculators(
            "Whitebox"
        )
    }

    calculators[
        "TrainingStatisticExtractionCalculator"
    ] = StatCalculatorContainer(
        name="TrainingStatisticExtractionCalculator",
        cfg=OmegaConf.create(training_config),
        stats=["train_embeddings"],
        dependencies=[],
        builder=(
            "lm_polygraph.defaults.stat_calculator_builders."
            "default_TrainingStatisticExtractionCalculator"
        ),
    )

    manager = UEManager(
        data=evaluation_dataset,
        model=polygraph_model,
        estimators=[RDESeq(layer)],
        builder_env_stat_calc=(
            BuilderEnvironmentStatCalculator(
                model=polygraph_model,
            )
        ),
        available_stat_calculators=list(
            calculators.values()
        ),
        generation_metrics=[],
        ue_metrics=[],
        processors=[],
        ignore_exceptions=False,
        max_new_tokens=max_new_tokens,
    )

    print(
        f"Running RDE with {training_size} reference "
        f"examples for {len(prompts)} prompt(s)..."
    )

    manager()

    rde_scores = _extract_rde_scores(manager)

    if len(rde_scores) != len(prompts):
        raise RuntimeError(
            "The number of RDE scores does not match "
            "the number of prompts: "
            f"{len(rde_scores)} scores for "
            f"{len(prompts)} prompts."
        )

    rows = [
        {
            "Technique": "Robust Density Estimation",
            "Category": "Density-based",
            "Granularity": "sequence",
            "Prompt": label,
            "Uncertainty": score,
        }
        for label, score in zip(
            prompt_labels,
            rde_scores,
        )
    ]

    settings = {
        "estimator": "RDESeq",
        "category": "Density-based",
        "granularity": "sequence",
        "model_alias": model_alias,
        "model_name": getattr(
            polygraph_model,
            "model_path",
            type(polygraph_model).__name__,
        ),
        "reference_dataset": "qiaojin/PubMedQA",
        "reference_subset": "pqa_labeled",
        "reference_text_column": "question",
        "training_size": training_size,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "representation_layer": layer,
        "number_of_prompts": len(prompts),
    }

    result: Dict[str, Any] = {
        "scores": rde_scores,
        "rows": rows,
        "settings": settings,
        "estimations": manager.estimations,
    }

    if return_manager:
        result["manager"] = manager

    print("RDE calculation complete.")

    return result
