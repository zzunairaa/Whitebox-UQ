# 🤖 Uncertainty Quantification for Large Language Models: White-Box Methods

A hands-on tutorial notebook covering **white-box uncertainty quantification (UQ)** for LLMs, using three open-source libraries — **UQLM**, **LM-Polygraph**, and **LLM Uncertainty Head** — unified through a lightweight internal utility called `uq_toolbox`.

---

## 📋 Overview

Large language models can produce fluent, confident-sounding answers even when those answers are wrong. Uncertainty quantification addresses this by estimating how reliable a model's response is — producing a confidence or uncertainty score alongside the generated text.

This notebook focuses exclusively on **white-box methods**, which require access to the model's internal signals: token probabilities, attention weights, hidden representations, and sampled generations.

> **Score direction:**
> - UQLM reports **confidence** — higher = more reliable, range `[0, 1]`
> - LM-Polygraph reports **uncertainty** — higher = less reliable, estimator-specific scale
>
> Always check the direction before interpreting or comparing values across libraries.

---

## 📁 Repository Structure

```
Whitebox-UQ/
│
├── Whitebox_UQ.ipynb             # Main tutorial notebook
│
└── uq_toolbox/                   # Internal infrastructure utility
    ├── __init__.py
    ├── registry.py               # UQ technique registry
    ├── core/
    │   ├── pipeline.py           # Dataset-level and batch scoring pipelines
    │   ├── uq_engine.py          # Central routing engine (UQLM ↔ LM-Polygraph)
    │   ├── density_uq.py         # RDE reference-fitting workflow
    │   ├── claim_uq.py           # Claim-level pipeline (generate → extract → NLI → score)
    │   └── response_evaluator.py # Correctness evaluators
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
| 2️⃣ UQLM Ensemble Scoring | Off-the-shelf ensemble combining multiple confidence signals |
| 3️⃣ LM-Polygraph White-Box UQ | Information-based, attention-based, meaning-diversity, reflexive, and density-based estimators |
| 🧩 Claim-Level Uncertainty | Per-claim uncertainty via a 5-stage pipeline (generate → extract → NLI → score) |
| 4️⃣ Supervised Uncertainty Head | Learned token-level uncertainty using a pretrained head on Mistral-7B |

> Benchmarking , black box, multimodal and score normalisation will be covered by other team members.

---

## 🎨 UQ Methods Covered

| Category | Library | Methods |
|---|---|---|
| 🔵 Information-based | UQLM + LM-Polygraph | Sequence Probability, Min Token Probability, Mean/Min Token Negentropy, Probability Margin, Perplexity, Monte Carlo Sequence Entropy, Max Sequence Probability |
| 🟢 Meaning-diversity | UQLM + LM-Polygraph | Monte Carlo Sequence Probability, CoCoA (MSP/PPL), Semantic Negentropy, Semantic Density, Semantic Entropy, Claim-Conditioned Probability |
| 🟣 Reflexive | UQLM + LM-Polygraph | P(True), P(True) Claim |
| 🟤 Attention-based | LM-Polygraph | RAUQ, Attention Score |
| 🟠 Density-based | LM-Polygraph | Robust Density Estimation (requires reference fitting on PubMedQA) |
| 🧩 Claim-level | LM-Polygraph | MaximumClaimProbability, MaxTokenEntropyClaim, PerplexityClaim, PointwiseMutualInformationClaim, PTrueClaim, ClaimConditionedProbabilityClaim |
| ⚫ Ensemble | UQLM | UQEnsemble (off-the-shelf, BSDetector-style) |
| 🔴 Learned | LLM Uncertainty Head | Supervised UHead on Mistral-7B-Instruct-v0.2 (token + sequence level) |

---

## 🧰 uq_toolbox

`uq_toolbox` is **not** a UQ library and introduces no new uncertainty methods. It is a small internal utility built for this tutorial to remove infrastructure boilerplate. It handles:

- loading and registering UQLM and LM-Polygraph models under simple aliases
- routing uncertainty calls to the correct backend
- the RDE two-stage reference-fitting workflow (`density_uq.py`)
- the claim-level 5-stage pipeline (`claim_uq.py`)
- the supervised uncertainty head loading and inference (`learned_uq/`)

All actual uncertainty estimates come from **UQLM**, **LM-Polygraph**, and **LLM Uncertainty Head**. The notebook shows the native library API for every method before introducing the wrapper.

---

## ⚙️ Requirements

### API Keys
- **OpenAI API key** — required for UQLM white-box scorers and GPT-4.1-mini claim extraction
- **Hugging Face token** — required to download Qwen, Mistral, DeBERTa, and the pretrained uncertainty head

### Hardware
- A **CUDA-enabled GPU** is strongly recommended for all sections
- Most sections run on an **NVIDIA T4** (Colab free tier)
- The supervised uncertainty head section loads **Mistral-7B-Instruct-v0.2** with its uncertainty head — a **high-memory GPU** is recommended for that section

### Software
- Python 3.9+
- Google Colab (recommended) or a local Jupyter environment with GPU

---

## 🚀 Getting Started

### Option 1 — Google Colab (recommended)

1. Open `Whitebox_UQ.ipynb` in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run the install cell, then **restart the session** when prompted
4. Run all cells from the top — do not skip the setup cells
5. Enter your OpenAI API key and Hugging Face token when prompted

### Option 2 — Local environment

```bash
git clone https://github.com/zzunairaa/Whitebox-UQ.git
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
jupyter notebook Whitebox_UQ.ipynb
```

> ⚠️ **After running the install cell, restart the runtime before continuing.** In Colab: Runtime → Restart session. Then run all cells from the top.

---

## 📦 Dependencies

| Library | Source | Purpose |
|---|---|---|
| [`uqlm`](https://github.com/cvs-health/uqlm) | PyPI | White-box, black-box, ensemble, LLM-judge scorers |
| [`lm-polygraph`](https://github.com/IINemo/lm-polygraph) | `@dev` branch | Broad UQ estimator suite |
| [`llm-uncertainty-head`](https://github.com/IINemo/llm-uncertainty-head) | GitHub | Supervised uncertainty head |
| `transformers` + `accelerate` | PyPI | Local model loading (Qwen, Mistral) |
| `langchain-openai` | PyPI | GPT-4o for UQLM; GPT-4.1-mini for claim extraction |
| `sentence-transformers` | PyPI | Semantic similarity for meaning-diversity methods |
| `protobuf==5.28.3` | PyPI (pinned) | Required for lm-polygraph compatibility |
| `plotly` / `matplotlib` | PyPI | Visualisation |
| `datasets` | PyPI | PubMedQA reference data for RDE section |

---

## 📖 References

1. Shelmanov et al. (2025). *Uncertainty Quantification for Large Language Models.* ACL Tutorial. https://aclanthology.org/2025.acl-tutorials.3/
2. Bouchard & Chauhan (2025). *UQ for Language Models: Black-Box, White-Box, LLM Judge, and Ensemble.* TMLR. https://openreview.net/forum?id=WOFspd4lq5
3. Fadeeva et al. (2023). *LM-Polygraph: Uncertainty Estimation for Language Models.* EMNLP. https://aclanthology.org/2023.emnlp-demo.41/
4. Vashurin et al. (2025). *Benchmarking UQ Methods with LM-Polygraph.* TACL. https://aclanthology.org/2025.tacl-1.11/
5. Kuhn et al. (2023). *Semantic Uncertainty.* ICLR. https://openreview.net/forum?id=VD-AYtP0dve
6. Farquhar et al. (2024). *Detecting Hallucinations Using Semantic Entropy.* Nature. https://www.nature.com/articles/s41586-024-07421-0
7. Kadavath et al. (2022). *Language Models (Mostly) Know What They Know.* https://arxiv.org/abs/2207.05221
8. Manakul et al. (2023). *SelfCheckGPT.* EMNLP. https://arxiv.org/abs/2303.08896
9. Shelmanov et al. (2025). *A Head to Predict and a Head to Question.* EMNLP. https://aclanthology.org/2025.emnlp-main.1809/
10. Vashurin et al. (2025). *CoCoA.* https://arxiv.org/abs/2502.04964
11. Vazhentsev et al. (2025). *RAUQ.* https://arxiv.org/abs/2505.20045
12. Sriramanan et al. (2024). *Attention-Based Uncertainty.* https://arxiv.org/abs/2406.10209
13. Qiu & Miikkulainen (2024). *Semantic Density.* https://arxiv.org/abs/2405.13845
14. Scalena et al. (2025). *EAGer: Entropy-Aware Generation.* https://arxiv.org/abs/2510.11170
15. Fomicheva et al. (2020). *Unsupervised Quality Estimation for NMT.* TACL. https://aclanthology.org/2020.tacl-1.35/
16. Yoo et al. (2022). *Robust Density Estimation.* Findings of ACL. https://aclanthology.org/2022.findings-acl.289/
17. Chen & Mueller (2023). *BSDetector.* https://arxiv.org/abs/2308.16175

Full references with links are listed at the end of `Whitebox_UQ.ipynb`.


