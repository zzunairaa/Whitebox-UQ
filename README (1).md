# Uncertainty Quantification for Large Language Models

A hands-on tutorial notebook covering practical white-box uncertainty quantification (UQ) for LLMs using three open-source libraries: **UQLM**, **LM-Polygraph**, and **LLM Uncertainty Head**.

---

## 📋 Overview

Large language models can produce fluent, confident-sounding answers even when those answers are wrong. Uncertainty quantification addresses this by estimating how reliable a model's response is — returning a confidence or uncertainty score alongside the generated text.

This tutorial focuses exclusively on **white-box methods**, which require access to the model's internal information such as token probabilities, attention weights, and hidden representations.

---

## 📓 Notebooks

| Notebook | Description |
|---|---|
| `UQWhitebox.ipynb` | White-box UQ methods: UQLM scorers, UQLM ensemble, LM-Polygraph estimators, claim-level uncertainty, and supervised uncertainty head |

> The benchmarking and score normalisation parts are covered in a separate notebook by another team member.

---

## 🗂️ Tutorial Structure

| Section | Content |
|---|---|
| ⚙️ Global Setup | Dependencies, credentials, model initialisation |
| 1️⃣ UQLM White-Box Scorers | Single-generation, self-reflection, and multi-generation scorers |
| 2️⃣ UQLM Ensemble | Off-the-shelf ensemble combining multiple confidence signals |
| 3️⃣ LM-Polygraph | Information-based, attention-based, semantic, reflexive, and density-based estimators |
| 🧩 Claim-Level Uncertainty | Per-claim uncertainty scoring using LM-Polygraph |
| 4️⃣ Supervised Uncertainty Head | Learned uncertainty via a pretrained head attached to Mistral-7B |

---

## 🎨 UQ Methods Covered

| Category | Methods |
|---|---|
| 🔵 Information-based | Sequence Probability, Min Token Probability, Token Negentropy, Probability Margin, Perplexity, Monte Carlo Entropy |
| 🟢 Semantic | Semantic Entropy, Semantic Density, CoCoA (MSP/PPL), Semantic Negentropy |
| 🟣 Reflexive | P(True) |
| 🟤 Attention-based | RAUQ, Attention Score |
| 🟠 Density-based | Robust Density Estimation (RDE) |
| ⚫ Ensemble | UQLM off-the-shelf ensemble (BSDetector) |
| 🔴 Learned | Supervised Uncertainty Head (Mistral-7B + uhead6) |

> **Score direction:** UQLM reports **confidence** (↑ = more reliable). LM-Polygraph reports **uncertainty** (↑ = less reliable).

---

## ⚙️ Requirements

### API Keys
- **OpenAI API key** — required for UQLM white-box scorers and claim extraction
- **Hugging Face token** — required to download local models and the pretrained uncertainty head

### Hardware
- A **CUDA-enabled GPU** is strongly recommended
- Most sections run on an **NVIDIA T4** or equivalent
- The supervised uncertainty head section loads **Mistral-7B** and requires more GPU memory — an A100 is recommended for that section

### Environment
- Python 3.9+
- Google Colab (recommended) or a local Jupyter environment with GPU access

---

## 🚀 Getting Started

### Option 1 — Google Colab (recommended)

1. Open the notebook in Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run the setup cell and follow the restart prompt
4. Enter your API keys when prompted
5. Run all cells from top to bottom

### Option 2 — Local environment

```bash
git clone <repo-url>
cd <repo-folder>
pip install -q "pip<24.1"
pip install -q uqlm langchain-openai pandas plotly matplotlib \
  transformers accelerate sentence-transformers datasets \
  scikit-learn nltk ipywidgets
pip install -q "git+https://github.com/IINemo/lm-polygraph.git@dev"
pip install -q "git+https://github.com/IINemo/llm-uncertainty-head.git"
pip install -q --force-reinstall "protobuf==5.28.3"
```

Then set your environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export HF_TOKEN="your-token-here"
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| [`uqlm`](https://github.com/cvs-health/uqlm) | White-box, black-box, ensemble, and LLM-judge scorers |
| [`lm-polygraph`](https://github.com/IINemo/lm-polygraph) | Broad UQ estimator suite |
| [`llm-uncertainty-head`](https://github.com/IINemo/llm-uncertainty-head) | Supervised uncertainty head |
| `transformers` | Local model loading (Qwen, Mistral) |
| `langchain-openai` | GPT-4o integration for UQLM |
| `sentence-transformers` | Semantic similarity for meaning-diversity methods |
| `plotly` / `matplotlib` | Visualisation |

---

## 🧰 uq_toolbox

This repo includes a small internal utility called `uq_toolbox` (distributed as `uq_toolbox.zip`). It is **not** a UQ library — it handles shared infrastructure such as model loading, aliasing, and configuration so the notebook can focus on the methods being taught. All actual uncertainty estimates come from UQLM, LM-Polygraph, and LLM Uncertainty Head.

---

## 📖 Key References

- Fadeeva et al. (2023). *LM-Polygraph: Uncertainty estimation for language models.* EMNLP. https://aclanthology.org/2023.emnlp-demo.41/
- Kuhn et al. (2023). *Semantic uncertainty.* ICLR. https://openreview.net/forum?id=VD-AYtP0dve
- Farquhar et al. (2024). *Detecting hallucinations using semantic consistency.* https://arxiv.org/abs/2406.15927
- Kadavath et al. (2022). *Language models (mostly) know what they know.* https://arxiv.org/abs/2207.05221
- Shelmanov et al. (2025). *A head to predict and a head to question.* EMNLP. https://aclanthology.org/2025.emnlp-main.1809/
- Vashurin et al. (2025). *Benchmarking UQ methods with LM-Polygraph.* TACL. https://aclanthology.org/2025.tacl-1.11/
- ACL 2025 Tutorial on UQ for NLP. https://aclanthology.org/2025.acl-tutorials.3/

Full references are listed at the end of the notebook.

---

## 👥 Authors

> Add your names / student IDs here.

---

## 📄 License

> Add your license here if applicable.
