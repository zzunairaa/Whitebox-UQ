# --- Standard Library ---
import gc
import inspect
from typing import Optional, Any, Dict, List

# --- Third-Party Libraries ---
import pandas as pd
import torch

# --- Local Imports ---
from .response_evaluator import BaseResponseEvaluator
from .uq_engine import evaluate_uncertainty


async def _run_evaluator(
    evaluator: BaseResponseEvaluator,
    *,
    prompt: str,
    generated_text: str,
    ground_truth: str,
) -> float:
    """
    Run synchronous or asynchronous evaluators while preserving
    compatibility with older evaluators that do not accept `prompt`.
    """
    try:
        result = evaluator(
            prompt=prompt,
            generated_text=generated_text,
            ground_truth=ground_truth,
        )
    except TypeError:
        # Backward compatibility with evaluators using:
        # __call__(generated_text, ground_truth)
        result = evaluator(
            generated_text=generated_text,
            ground_truth=ground_truth,
        )

    if inspect.isawaitable(result):
        result = await result

    return float(result)


def _archive_metrics(
    registry: Dict[str, Dict[str, List[Any]]],
    tech: str,
    score: Any,
    quality: float,
    prompt: str,
    response: str,
    question: str,
    ground_truth: str,
) -> None:
    """Archive extracted metrics into the provided registry."""
    registry[tech]["scores"].append(score)
    registry[tech]["qualities"].append(quality)
    registry[tech]["prompts"].append(prompt)
    registry[tech]["responses"].append(response)
    registry[tech]["questions"].append(question)
    registry[tech]["ground_truths"].append(ground_truth)


async def compute_dataset_uq_scores(
    dataset: pd.DataFrame,
    uq_techniques: List[str],
    uq_context: Any,
    evaluator: BaseResponseEvaluator,
    granularity: str = "sequence",
    tech_kwargs_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    **global_uq_kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute sequential LLM inference and aggregate raw uncertainty metrics.

    Supports:
    - pandas DataFrames;
    - iterable dictionaries;
    - raw strings;
    - synchronous evaluators;
    - asynchronous evaluators;
    - older evaluators that do not accept a prompt argument.
    """
    tech_kwargs_registry = tech_kwargs_registry or {}

    if hasattr(dataset, "iterrows"):
        samples = [
            row.to_dict()
            for _, row in dataset.iterrows()
        ]
    else:
        samples = list(dataset)

    is_multimodal = (
        hasattr(dataset, "columns")
        and (
            "image" in dataset.columns
            or "image_base64" in dataset.columns
        )
    )

    raw_registry = {
        tech: {
            "scores": [],
            "qualities": [],
            "prompts": [],
            "responses": [],
            "questions": [],
            "ground_truths": [],
        }
        for tech in uq_techniques
    }

    print(
        f"⏳ Extracting raw scores from {len(samples)} samples "
        f"using {evaluator.__class__.__name__}..."
    )

    for sample in samples:
        if isinstance(sample, dict):
            base_prompt = (
                sample.get("question")
                or sample.get("prompt")
                or sample.get("text")
                or ""
            )

            true_answer = str(
                sample.get("answer", "N/A")
            ).strip()

            image_payload = sample.get("image_base64")
        else:
            base_prompt = str(sample)
            true_answer = "N/A"
            image_payload = None

        prompt_input = (
            f"<image>\n{base_prompt}"
            if is_multimodal and image_payload
            else base_prompt
        )

        for tech in uq_techniques:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            specific_tech_kwargs = tech_kwargs_registry.get(
                tech,
                {},
            )

            merged_uq_kwargs = {
                **global_uq_kwargs,
                **specific_tech_kwargs,
            }

            if image_payload:
                merged_uq_kwargs["image_base64"] = image_payload

            result = await evaluate_uncertainty(
                prompt=prompt_input,
                technique_name=tech,
                granularity=granularity,
                uq_context=uq_context,
                **merged_uq_kwargs,
            )

            raw_score = result.get(
                "uncertainty_score",
                0.0,
            )

            generated_text = str(
                result.get(
                    "generated_text",
                    "No response",
                )
            ).strip()

            is_correct = await _run_evaluator(
                evaluator,
                prompt=base_prompt,
                generated_text=generated_text,
                ground_truth=true_answer,
            )

            _archive_metrics(
                registry=raw_registry,
                tech=tech,
                score=raw_score,
                quality=is_correct,
                prompt=prompt_input,
                response=generated_text,
                question=base_prompt,
                ground_truth=true_answer,
            )

    print("✅ Raw scores extraction completed successfully.")
    return raw_registry


async def compute_batch_uqlm_scores(
    prompts: List[str],
    reference_answers: List[str],
    uq_techniques: List[str],
    granularity: str,
    uq_context: Any,
    evaluator: BaseResponseEvaluator,
    model_alias: str = "default",
    batch_size: int = 16,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute UQLM uncertainty scores in batches.

    Supports both synchronous and asynchronous evaluators.
    """
    if len(prompts) != len(reference_answers):
        raise ValueError(
            "prompts and reference_answers must have equal lengths."
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    raw_registry = {
        tech: {
            "scores": [],
            "qualities": [],
            "prompts": [],
            "responses": [],
            "questions": [],
            "ground_truths": [],
        }
        for tech in uq_techniques
    }

    total_prompts = len(prompts)

    print(
        f"🚀 Starting extraction. Total: {total_prompts} samples "
        f"| Batch: {batch_size} "
        f"| Evaluator: {evaluator.__class__.__name__}"
    )

    for start_index in range(
        0,
        total_prompts,
        batch_size,
    ):
        end_index = min(
            start_index + batch_size,
            total_prompts,
        )

        chunk_prompts = prompts[start_index:end_index]
        chunk_references = reference_answers[
            start_index:end_index
        ]

        for tech in uq_techniques:
            batch_outputs = await evaluate_uncertainty(
                prompt=chunk_prompts,
                technique_name=tech,
                granularity=granularity,
                uq_context=uq_context,
                model_alias=model_alias,
                **kwargs,
            )

            if isinstance(batch_outputs, dict):
                batch_outputs = [batch_outputs]

            for output, prompt, reference in zip(
                batch_outputs,
                chunk_prompts,
                chunk_references,
            ):
                raw_score = output.get(
                    "uncertainty_score",
                    0.0,
                )

                generated_text = str(
                    output.get(
                        "generated_text",
                        "No response",
                    )
                ).strip()

                is_correct = await _run_evaluator(
                    evaluator,
                    prompt=prompt,
                    generated_text=generated_text,
                    ground_truth=reference,
                )

                _archive_metrics(
                    registry=raw_registry,
                    tech=tech,
                    score=raw_score,
                    quality=is_correct,
                    prompt=prompt,
                    response=generated_text,
                    question=prompt,
                    ground_truth=reference,
                )

    print("✅ Raw scores extraction completed successfully.")
    return raw_registry