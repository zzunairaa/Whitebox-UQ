import os

import nltk
import pandas as pd
from openai import OpenAI

from lm_polygraph.stat_calculators import (
    GreedyProbsCalculator,
    ClaimsExtractor,
    GreedyAlternativesNLICalculator,
    EntropyCalculator,
    GreedyLMProbsCalculator,
    ClaimPromptCalculator,
)
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.estimators import (
    MaximumClaimProbability,
    MaxTokenEntropyClaim,
    PerplexityClaim,
    PointwiseMutualInformationClaim,
    PTrueClaim,
    ClaimConditionedProbabilityClaim,
)


class OpenAIChat:
    """Adapter that allows LM-Polygraph to use OpenAI for claim extraction."""

    def __init__(
        self,
        model_name="gpt-4.1-mini",
        max_tokens=250,
        api_key=None,
        base_url=None,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY or pass "
                "openai_api_key to run_claim_level_uq()."
            )

        client_options = {
            "api_key": resolved_api_key,
        }

        resolved_base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL")
        )

        if resolved_base_url:
            client_options["base_url"] = resolved_base_url

        self.client = OpenAI(**client_options)

    def ask(self, message):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or ""


def run_claim_level_uq(
    model,
    tokenizer,
    base_model,
    prompts,
    prompt_labels,
    max_new_tokens=50,
    nli_device="cuda:0",
    claim_extractor_model="gpt-4.1-mini",
    claim_extractor_max_tokens=250,
    openai_api_key=None,
    openai_base_url=None,
):
    """Run six LM-Polygraph claim-level uncertainty estimators."""

    if len(prompts) != len(prompt_labels):
        raise ValueError(
            "prompts and prompt_labels must have the same length."
        )

    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)

    claim_extractor = OpenAIChat(
        model_name=claim_extractor_model,
        max_tokens=claim_extractor_max_tokens,
        api_key=openai_api_key,
        base_url=openai_base_url,
    )

    calculators = {
        "greedy": GreedyProbsCalculator(),
        "entropy": EntropyCalculator(),
        "lm_probs": GreedyLMProbsCalculator(),
        "claims": ClaimsExtractor(claim_extractor),
        "nli": GreedyAlternativesNLICalculator(
            Deberta(device=nli_device)
        ),
        "ptrue": ClaimPromptCalculator(),
    }

    estimators = {
        "Maximum Claim Probability":
            MaximumClaimProbability(),
        "Max Token Entropy":
            MaxTokenEntropyClaim(),
        "Perplexity":
            PerplexityClaim(),
        "PMI":
            PointwiseMutualInformationClaim(),
        "p(True)":
            PTrueClaim(),
        "Claim-Conditioned Probability":
            ClaimConditionedProbabilityClaim(),
    }

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(base_model.device)

    stats = {
        "model_inputs": encoded,
    }

    # Generate local responses and collect token probabilities.
    stats.update(
        calculators["greedy"](
            stats,
            texts=prompts,
            model=model,
            max_new_tokens=max_new_tokens,
        )
    )

    stats["greedy_texts"] = tokenizer.batch_decode(
        stats["greedy_tokens"],
        skip_special_tokens=True,
    )

    # Prepare the statistics required by claim-level estimators.
    stats.update(
        calculators["entropy"](
            stats,
            texts=prompts,
            model=model,
        )
    )

    stats.update(
        calculators["lm_probs"](
            stats,
            texts=prompts,
            model=model,
        )
    )

    stats.update(
        calculators["claims"](
            stats,
            texts=prompts,
            model=model,
        )
    )

    stats.update(
        calculators["nli"](
            stats,
            texts=None,
            model=model,
        )
    )

    stats.update(
        calculators["ptrue"](
            stats,
            texts=prompts,
            model=model,
        )
    )

    rows = []

    for technique, estimator in estimators.items():
        scores = estimator(stats)

        for label, claims, claim_scores in zip(
            prompt_labels,
            stats["claims"],
            scores,
        ):
            for claim, score in zip(
                claims,
                claim_scores,
            ):
                rows.append(
                    {
                        "technique": technique,
                        "prompt": label,
                        "claim": claim.claim_text,
                        "uncertainty": float(score),
                    }
                )

    dataframe = pd.DataFrame(rows)

    return {
        "dataframe": dataframe,
        "claims": stats["claims"],
        "responses": stats["greedy_texts"],
        "settings": {
            "generation_model": getattr(
                model,
                "model_path",
                type(model).__name__,
            ),
            "claim_extractor": claim_extractor_model,
            "claim_extractor_provider": "OpenAI",
            "nli_model": "DeBERTa",
            "nli_device": nli_device,
            "granularity": "claim",
            "estimators": list(estimators),
            "max_new_tokens": max_new_tokens,
        },
    }
