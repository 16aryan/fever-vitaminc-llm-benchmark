# FEVER & VitaminC Fact-Verification Benchmark

**Author / Lead Contributor**: Aryan Goel  
**Course / Project**: ANLP Group Assignment  

A reproducible evaluation framework benchmarking **Qwen 2.5 (7B)** and **Llama 3.1 (8B)** on factual verification tasks using local [Ollama](https://ollama.ai) inference on macOS (Apple Silicon).

Models classify natural language claims into three standard categories:
- **`SUPPORTS`**: The claim is proven factually true by evidence.
- **`REFUTES`**: The claim is contradicted or disproven by evidence.
- **`NOT ENOUGH INFO`**: Available facts are insufficient to verify or refute the claim.

---

## 1. Project Directory Structure

```text
fever-benchmark/
├── .venv/                         # Python virtual environment (Python 3.13)
├── config.json                    # Central configuration for models, datasets & paths
├── requirements.txt               # Locked project dependencies
├── README.md                      # Complete project documentation
├── fever_benchmark.py             # Standalone pilot script (FEVER 30-claim experiment)
│
├── data/
│   ├── fever/
│   │   └── shared_task_dev.jsonl  # Official FEVER dev set (19,998 samples)
│   ├── vitaminc/
│   │   └── dev.jsonl              # Official VitaminC dev set (63,054 samples)
│   ├── sample_manifest.csv        # Frozen 60-claim balanced manifest (seed 42)
│   └── wiki_cache.json            # Persistent Wikipedia article text cache
│
├── src/
│   ├── inspect_data.py            # Verifies dataset schemas, column mappings & class counts
│   ├── prepare_samples.py         # Generates stratified manifests across datasets
│   ├── retrieve_evidence.py       # Rate-limited MediaWiki search & TF-IDF sentence ranker
│   ├── run_benchmark.py           # Unified runner (FEVER + VitaminC across 3 conditions)
│   ├── evaluate_results.py        # Computes metrics, Macro-F1 & generates confusion matrices
│   └── analyse_errors.py          # Disagreement matrix, failure taxonomy & case studies
│
├── results/
│   ├── fever_predictions.csv      # Pilot predictions, reasoning, latency & retrieved context
│   ├── fever_metrics.csv          # Pilot evaluation metrics
│   ├── predictions.csv            # Unified benchmark output file
│   ├── metrics.csv                # Unified aggregate metrics
│   ├── error_analysis.csv         # Categorized error analysis
│   └── figures/                   # Confusion matrix heatmaps & F1 bar plots
│
└── report/
    └── benchmark_report.md        # Formatted project report & scorecard template
```

---

## 2. Environment & Prerequisites

### 2.1 Virtual Environment
The virtual environment is pre-configured in `.venv/`. To activate:
```bash
cd fever-benchmark
source .venv/bin/activate
```

Dependencies include `requests`, `pandas`, `scikit-learn`, `tqdm`, `datasets`, `matplotlib`, and `seaborn`. If reinstalling:
```bash
pip install -r requirements.txt
```

### 2.2 Ollama Local LLMs
Ensure Ollama is running and both models are installed:
```bash
ollama list
```
Required models:
- **`llama3.1:8b`** (4.9 GB, Q4_K_M)
- **`qwen2.5:7b`** (4.7 GB, Q4_K_M)

If missing:
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

---

## 3. Dataset Status

Both datasets have been downloaded, verified, and mapped:

| Dataset | File | Total Rows | Classes | Evidence Source |
| :--- | :--- | :---: | :---: | :--- |
| **FEVER** | `data/fever/shared_task_dev.jsonl` | 19,998 | 6,666 / class | Real-time Wikipedia (Retrieved) |
| **VitaminC** | `data/vitaminc/dev.jsonl` | 63,054 | 31.4k / 22.5k / 9.0k | Wikipedia revisions (Provided Gold) |

### Frozen Evaluation Manifest
The frozen evaluation sample in `data/sample_manifest.csv` contains 60 claims (10 per class for each dataset, seed `42`):
- FEVER: 10 `SUPPORTS`, 10 `REFUTES`, 10 `NOT ENOUGH INFO`
- VitaminC: 10 `SUPPORTS`, 10 `REFUTES`, 10 `NOT ENOUGH INFO`

To re-sample or adjust sample size per class:
```bash
python src/prepare_samples.py --n-per-class 10 --seed 42
```

---

## 4. How to Run the Experiments

You have two execution workflows depending on your objective:

### Option A: FEVER Pilot (Quick Validation)
Runs the 30-claim pilot script across 4 conditions (120 model evaluations):
```bash
python fever_benchmark.py
```
- Output predictions: `results/fever_predictions.csv`
- Output metrics: `results/fever_metrics.csv`

---

### Option B: Unified Multi-Dataset Benchmark (FEVER + VitaminC)
Runs all models and experimental conditions defined in `config.json`:

#### Step 1: Inspect Dataset Schemas
```bash
python src/inspect_data.py --dataset all
```

#### Step 2: Run Benchmark Inference
```bash
# Run all datasets and conditions:
python src/run_benchmark.py

# Or filter by dataset:
python src/run_benchmark.py --dataset fever
python src/run_benchmark.py --dataset vitaminc

# Or filter by condition:
python src/run_benchmark.py --mode claim_only
python src/run_benchmark.py --mode gold_evidence
```
*Note: Predictions are saved incrementally to `results/predictions.csv`. If interrupted, rerun the command to resume.*

#### Step 3: Compute Evaluation Metrics & Generate Figures
```bash
python src/evaluate_results.py
```
Generates:
- `results/metrics.csv`
- `results/figures/macro_f1_comparison.png`
- `results/figures/cm_*.png` (confusion matrix heatmaps)

#### Step 4: Run Disagreement & Error Analysis
```bash
python src/analyse_errors.py
```
Identifies:
- Concrete disagreements between Llama and Qwen
- Error taxonomy: `OVER_CAUTIOUS_NEI`, `HALLUCINATED_VERDICT`, `FALSE_REFUTATION`, `FALSE_SUPPORT`
- Evidence sensitivity: instances where evidence flipped a model's prediction

---

## 5. Experimental Conditions Explained

| Condition | Input Given to Model | Purpose |
| :--- | :--- | :--- |
| **`claim_only`** | Claim text only | Measures parametric knowledge without external text. |
| **`retrieved_evidence`** | Claim + Top-5 Wikipedia sentences (TF-IDF) | Evaluates end-to-end information retrieval and reasoning. |
| **`gold_evidence`** | Claim + Authoritative revision evidence (VitaminC) | Isolates reasoning ability from retrieval errors. |

---

## 6. Pilot Results Summary (N=30)

Baseline metrics obtained from the initial FEVER run:

| Experiment | Condition | Model | Valid Coverage | Accuracy | Macro-F1 | Mean Latency |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **A** | `claim_only` | **Qwen 2.5 (7B)** | **100%** | **33.3%** | **0.251** | 2.54 s |
| **B** | `claim_only` | **Llama 3.1 (8B)** | **100%** | **36.7%** | **0.232** | **1.83 s** |
| **C** | `retrieved_evidence` | **Qwen 2.5 (7B)** | **100%** | **33.3%** | **0.167** | 2.37 s |
| **D** | `retrieved_evidence` | **Llama 3.1 (8B)** | **100%** | **26.7%** | **0.140** | **1.84 s** |

### Key Analytical Takeaways
1. **Perfect JSON Adherence**: Both models achieved 100% coverage, producing well-formed JSON objects with zero formatting errors.
2. **Inference Speed**: Llama 3.1 (8B) was ~28% faster than Qwen 2.5 (7B) on Apple Silicon (1.83s vs. 2.54s per claim).
3. **Retrieval Quality Impact**: Un-throttled Wikipedia queries during initial runs triggered `429 Too Many Requests` on un-cached items, prompting conservative `NOT ENOUGH INFO` fallbacks. [src/retrieve_evidence.py](file:///Users/aryan/ANLP_Group_AT2/fever-benchmark/src/retrieve_evidence.py) has been upgraded with request pacing (`0.3s`) and backoff retries to guarantee clean evidence retrieval.

---

## 7. Report Handover

A complete Markdown report template is available at [report/benchmark_report.md](file:///Users/aryan/ANLP_Group_AT2/fever-benchmark/report/benchmark_report.md). When experiments finish, update the report tables and copy confusion matrix figures into your presentation deck.
