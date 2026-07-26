# 🤖 White-Box Uncertainty Quantification for Large Language Models

A hands-on tutorial notebook covering practical **white-box uncertainty quantification (UQ)** for LLMs, using three open-source libraries — **UQLM**, **LM-Polygraph**, and **LLM Uncertainty Head** — unified through a lightweight internal utility called `uq_toolbox`.

---

## 📋 Overview

Large language models can produce fluent, confident-sounding answers even when those answers are wrong. Uncertainty quantification addresses this by estimating how reliable a model's response is — producing a confidence or uncertainty score alongside the generated text.

This tutorial focuses exclusively on **white-box methods**, which require access to the model's internal signals: token probabilities, attention weights, hidden representations, and sampled generations.

> **Score direction awareness:**
> - UQLM reports **confidence** — higher = more reliable
> - LM-Polygraph reports **uncertainty** — higher = less reliable
>
> Always check the direction before interpreting or comparing values across libraries.

---

## 📁 Repository Structure

```
Whitebox-UQ/
│
├── UQWhitebox.ipynb              # Main tutorial notebook
│
└── uq_toolbox/                   # Internal infrastructure utility
    ├── __init__.py
    ├── registry.py               # UQ technique registry (all supported methods)
    ├── core/
    │   ├── pipeline.py           # Dataset-level and batch scoring pipelines
    │   ├── uq_engine.py          # Central routing engine (UQLM ↔ LM-Polygraph)
    │   ├── density_uq.py         # RDE reference-fitting workflow
    │   ├── claim_uq.py           # Claim-level pipeline (extract → NLI → score)
    │   └── response_evaluator.py # Correctness evaluators (substring match etc.)
    ├── managers/
    │   ├── model_manager.py      # Model loading, aliasing, white/black-box modes
    │   └── llama_cpp_manager.py  # llama.cpp backend support
    └── learned_uq/
        ├── engine.py             # Token/sequence-level supervised UQ inference
        └── manager.py            # Mistral + uncertainty head loader
```

---

## 🗂️ Tutorial Structure

| Section | Content |
|---|---|
| ⚙️ Global Setup | Dependencies, credentials, environment flags, model initialisation |
| 1️⃣ UQLM White-Box Scorers | Single-generation, self-reflection, and multi-generation scorers |
| 2️⃣ UQLM Ensemble | Off-the-shelf ensemble combining multiple confidence signals |
| 3️⃣ LM-Polygraph | Information-based, attention-based, semantic, reflexive, and density-based estimators |
| 🧩 Claim-Level Uncertainty | Per-claim uncertainty via a 5-stage pipeline (generate → extract → NLI → score) |
| 4️⃣ Supervised Uncertainty Head | Learned token-level uncertainty using a pretrained head on Mistral-7B |

> Benchmarking and score normalisation are covered in a separate notebook by another team member.

---

## 🎨 UQ Methods Covered

| Category | Library | Methods |
|---|---|---|
| 🔵 Information-based | LM-Polygraph | MaximumSequenceProbability, Perplexity, MonteCarloSequenceEntropy, BoostedProbSequence, CocoaMSP, CocoaMTE, CocoaPPL |
| 🟢 Semantic / Meaning-diversity | LM-Polygraph | SemanticEntropy, SemanticDensity, CocoaMSP, CocoaPPL, EigenScore |
| 🟣 Reflexive | LM-Polygraph / UQLM | P(True), SelfCheckGPT |
| 🟤 Attention-based | LM-Polygraph | RAUQ, AttentionScore |
| 🟠 Density-based | LM-Polygraph | RobustDensityEstimation (requires reference fitting) |
| 🧩 Claim-level | LM-Polygraph + UQLM | ClaimConditionedProbability, PTrueClaim, claim-level pipeline |
| ⚫ Ensemble | UQLM | UQEnsemble (BSDetector-style, off-the-shelf) |
| 🔴 Learned | LLM Uncertainty Head | Supervised UHead on Mistral-7B-Instruct-v0.2 (token + sequence level) |

---

## 🧰 uq_toolbox

`uq_toolbox` is **not** a UQ library and introduces no new uncertainty methods. It is a small internal utility built for this tutorial to remove infrastructure boilerplate. It handles:

- loading and registering UQLM and LM-Polygraph models under simple aliases
- routing uncertainty calls to the correct backend via a central `evaluate_uncertainty()` function
- the RDE reference-fitting workflow (`density_uq.py`)
- the claim-level 5-stage pipeline (`claim_uq.py`)
- the supervised uncertainty head loading and inference (`learned_uq/`)
- batch and dataset-level scoring pipelines (`pipeline.py`)

All actual uncertainty estimates come from **UQLM**, **LM-Polygraph**, and **LLM Uncertainty Head**.

---

## ⚙️ Requirements

### API Keys
- **OpenAI API key** — required for UQLM white-box scorers and for GPT-4o-based claim extraction in the claim-level section
- **Hugging Face token** — required to download Qwen, Mistral, DeBERTa, and the pretrained uncertainty head

### Hardware
- A **CUDA-enabled GPU** is strongly recommended for all sections
- Most sections run on an **NVIDIA T4** or equivalent (Colab free tier)
- The supervised uncertainty head section loads **Mistral-7B-Instruct-v0.2** together with its uncertainty head and is significantly more memory-intensive — an **A100** is recommended

### Software
- Python 3.9+
- Google Colab (recommended) or a local Jupyter environment with GPU access

---

## 🚀 Getting Started

### Option 1 — Google Colab (recommended)

1. Upload `uq_toolbox.zip` to `/content/` in your Colab session
2. Open `UQWhitebox.ipynb` in Colab
3. Select **Runtime → Change runtime type → T4 GPU** (A100 for the supervised section)
4. Run the install cell, then **restart the session** when prompted
5. Run all cells from the top — do not skip the setup cells
6. Enter your OpenAI API key and Hugging Face token when prompted

### Option 2 — Local environment

```bash
git clone <repo-url>
cd Whitebox-UQ

pip install "pip<24.1"
pip install uqlm langchain-openai pandas plotly matplotlib \
  transformers accelerate sentence-transformers datasets \
  scikit-learn nltk ipywidgets
pip install "git+https://github.com/IINemo/lm-polygraph.git@dev"
pip install "git+https://github.com/IINemo/llm-uncertainty-head.git"
pip install --force-reinstall "protobuf==5.28.3"
pip install langchain-anthropic langchain-google-genai langchain-ollama
```

Set environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export HF_TOKEN="your-huggingface-token"
```

Then launch:

```bash
jupyter notebook UQWhitebox.ipynb
```

---

## 📦 Dependencies

| Library | Version / Source | Purpose |
|---|---|---|
| [`uqlm`](https://github.com/cvs-health/uqlm) | PyPI | White-box, black-box, ensemble, LLM-judge scorers |
| [`lm-polygraph`](https://github.com/IINemo/lm-polygraph) | `@dev` branch | Broad UQ estimator suite |
| [`llm-uncertainty-head`](https://github.com/IINemo/llm-uncertainty-head) | GitHub | Supervised uncertainty head |
| `transformers` + `accelerate` | PyPI | Local model loading (Qwen, Mistral) |
| `langchain-openai` | PyPI | GPT-4o integration for UQLM and claim extraction |
| `sentence-transformers` | PyPI | Semantic similarity for meaning-diversity methods |
| `protobuf==5.28.3` | PyPI (pinned) | Required for compatibility with lm-polygraph |
| `plotly` / `matplotlib` | PyPI | Visualisation |
| `datasets` | PyPI | PubMedQA reference data for RDE section |

---

## 📖 References

- Fadeeva et al. (2023). *LM-Polygraph: Uncertainty estimation for language models.* EMNLP. https://aclanthology.org/2023.emnlp-demo.41/
- Kuhn et al. (2023). *Semantic uncertainty: Linguistic invariances for uncertainty estimation in NLG.* ICLR. https://openreview.net/forum?id=VD-AYtP0dve
- Farquhar et al. (2024). *Detecting hallucinations using semantic consistency.* https://arxiv.org/abs/2406.15927
- Kadavath et al. (2022). *Language models (mostly) know what they know.* https://arxiv.org/abs/2207.05221
- Manakul et al. (2023). *SelfCheckGPT: Zero-resource black-box hallucination detection.* https://arxiv.org/abs/2303.08896
- Shelmanov et al. (2025). *A head to predict and a head to question.* EMNLP. https://aclanthology.org/2025.emnlp-main.1809/
- Vashurin et al. (2025). *Benchmarking UQ methods with LM-Polygraph.* TACL. https://aclanthology.org/2025.tacl-1.11/
- Vashurin et al. (2025). *CoCoA: Consistency and confidence aggregation.* https://arxiv.org/abs/2502.04964
- Vazhentsev et al. (2025). *RAUQ.* https://arxiv.org/abs/2502.04964
- Sriramanan et al. (2024). *Attention-based uncertainty.* https://arxiv.org/abs/2406.10209
- Chen & Mueller (2023). *BSDetector.* https://arxiv.org/abs/2308.16175
- Fomicheva et al. (2020). *Unsupervised quality estimation for NMT.* TACL. https://aclanthology.org/2020.tacl-1.35/
- ACL 2025 Tutorial on UQ for NLP. https://aclanthology.org/2025.acl-tutorials.3/

Full references with links are listed at the end of `UQWhitebox.ipynb`.

