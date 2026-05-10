# DSCI 531 Project: LLM Bias Evaluation Pipeline

Evaluates demographic bias in LLMs with **StereoSet**, **CrowS-Pairs**, and **BBQ**. Core code lives under `src/`; results and charts go under `outputs/`.

## Notebook run (submission / quick demo)

The Jupyter notebook **`bias_evaluation_pipeline.ipynb`** is configured for **small sample sizes**, not the full benchmarks. That keeps runtime short for project submission and grading. Outputs from the notebook are written to **`outputs/sample_run/`** (per-model folders, `combined/`, and `charts/`).

### Settings used for the sample-run outputs in this repo

These match the notebook configuration cell and `src/config.py` (`RANDOM_SEED`).

| Setting | Value |
|--------|-------|
| **StereoSet** items scored per model | **15** (random sample from loaded StereoSet; **1,218** rows available after load in a typical run) |
| **CrowS-Pairs** items per model | **15** (**937** pairs available) |
| **BBQ** items per model | **20** (**3,000** scenarios available in the notebook run’s loaded BBQ frame) |
| **Items per model (total)** | **50** (= 15 + 15 + 20) |
| **Models** | `mistralai/Mistral-7B-v0.1` → `mistral-7b`, `meta-llama/Meta-Llama-3-8B-Instruct` → `llama3-8b`, `google/gemma-2-9b-it` → `gemma2-9b` |
| **Mitigation** | `baseline` only |
| **Decoding** | `deterministic` |
| **Random seed** | **42** |

For a **full** or larger evaluation, increase the counts in the notebook’s `SAMPLE_SIZES` dict or use `python scripts/run_pipeline.py` (writes under `outputs/` by default, not `sample_run`).

**Charts:** After results exist, run `python scripts/generate_charts.py` (auto-detects `outputs/sample_run` when present).

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook bias_evaluation_pipeline.ipynb
```

Set `HF_TOKEN` if a model requires Hugging Face authentication.

The default stack uses the **Hugging Face Inference API** (`huggingface_hub`); `torch` / `transformers` are not required unless you run models locally (see commented lines at the bottom of `requirements.txt`).

---

## CLI helpers

| Command | Purpose |
|--------|---------|
| `python scripts/run_sample.py` | Same small-sample idea as the notebook; writes to `outputs/sample_run/` (see `SAMPLE_SIZES` in that file—Mistral HF id may differ from the notebook). |
| `python scripts/run_pipeline.py` | Broader / default pipeline run under `outputs/`. |
| `python scripts/generate_charts.py` | Builds `bar_chart_key_metrics.png` and `heatmap_cross_model.png`. |

---

## Layout

```
├── bias_evaluation_pipeline.ipynb
├── scripts/           # run_sample, run_pipeline, generate_charts, …
├── src/               # config, preprocessing, scoring, evaluation_pipeline
├── data/              # cached downloads (created at runtime)
└── outputs/
    └── sample_run/    # notebook + run_sample quick outputs
        ├── {mistral-7b,llama3-8b,gemma2-9b}/
        ├── combined/
        └── charts/
```

---

## References

- StereoSet: Nadeem et al., 2021  
- CrowS-Pairs: Nangia et al., 2020  
- BBQ: Parrish et al., 2022  

Educational use (DSCI 531). Cite appropriately if reusing methods or code.
