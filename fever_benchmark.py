"""FEVER Benchmark Runner.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import json
import re
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = (
    Path("data/fever/shared_task_dev.jsonl")
    if Path("data/fever/shared_task_dev.jsonl").exists()
    else Path("data/shared_task_dev.jsonl")
)
CACHE_PATH = Path("data/wiki_cache.json")
OUTPUT_PATH = Path("results/fever_predictions.csv")
METRICS_PATH = Path("results/fever_metrics.csv")

MODELS = [
    "qwen2.5:7b",
    "llama3.1:8b",
]

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
SAMPLE_PER_CLASS = 10
RANDOM_SEED = 42
TOP_K_SENTENCES = 5

OLLAMA_URL = "http://localhost:11434/api/chat"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# =========================================================
# DATA AND CACHE
# =========================================================

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8"
    )


# =========================================================
# WIKIPEDIA RETRIEVAL
# =========================================================

session = requests.Session()
session.headers.update({
    "User-Agent": "FEVERBenchmarkStudentProject/1.0"
})


def wikipedia_search(claim, limit=3):
    """Find a few current Wikipedia articles related to the claim."""
    response = session.get(
        WIKI_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": claim,
            "srlimit": limit,
            "format": "json",
        },
        timeout=30,
    )
    response.raise_for_status()

    return [
        item["title"]
        for item in response.json()["query"]["search"]
    ]


def get_wikipedia_text(title, cache):
    """Fetch and cache plain article text."""
    if title in cache:
        return cache[title]

    response = session.get(
        WIKI_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": title,
            "format": "json",
            "redirects": 1,
        },
        timeout=30,
    )
    response.raise_for_status()

    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    text = page.get("extract", "")

    cache[title] = text
    return text


def split_sentences(text):
    """Simple sentence splitter suitable for this pilot."""
    text = re.sub(r"\s+", " ", text)
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) > 30
    ]


def retrieve_evidence(claim, cache, top_k=TOP_K_SENTENCES):
    """
    Retrieve current Wikipedia pages, then rank their sentences
    by TF-IDF cosine similarity to the claim.

    This is an approximate retriever, not the original FEVER
    2017 Wikipedia snapshot or the official FEVER retrieval system.
    """
    titles = wikipedia_search(claim, limit=3)
    candidates = []

    for title in titles:
        try:
            article = get_wikipedia_text(title, cache)
            sentences = split_sentences(article)

            for sentence in sentences:
                candidates.append({
                    "title": title,
                    "sentence": sentence,
                })
        except requests.RequestException as exc:
            print(f"Wikipedia retrieval warning ({title}): {exc}")

    if not candidates:
        return []

    texts = [claim] + [x["sentence"] for x in candidates]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    try:
        matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(
            matrix[0:1], matrix[1:]
        ).flatten()
    except ValueError:
        return []

    ranked = similarities.argsort()[::-1]
    selected = []

    for idx in ranked:
        if similarities[idx] <= 0:
            continue

        candidate = candidates[idx]
        selected.append({
            **candidate,
            "similarity": round(float(similarities[idx]), 4),
        })

        if len(selected) >= top_k:
            break

    return selected


# =========================================================
# MODEL PREDICTION
# =========================================================

def call_ollama(model, claim, evidence=None):
    if evidence is None:
        evidence_text = (
            "No external evidence was provided. "
            "Use your existing knowledge, but choose NOT ENOUGH INFO "
            "if you cannot confidently determine the verdict."
        )
    elif evidence:
        evidence_text = "\n".join(
            f"[{i}] Wikipedia: {item['title']}\n"
            f"{item['sentence']}"
            for i, item in enumerate(evidence, start=1)
        )
    else:
        evidence_text = (
            "No relevant evidence was retrieved. "
            "Do not assume the claim is true or false."
        )

    prompt = f"""
You are a careful fact-verification system.

Choose exactly one label:
SUPPORTS
REFUTES
NOT ENOUGH INFO

Definitions:
- SUPPORTS: the supplied evidence establishes the claim.
- REFUTES: the supplied evidence contradicts the claim.
- NOT ENOUGH INFO: the evidence does not establish either verdict.

Treat the evidence as the source of truth. Do not use outside
knowledge to override the evidence.

Claim:
{claim}

Evidence:
{evidence_text}

Return a JSON object with exactly these keys:
"label": one of the three allowed labels
"reason": a brief explanation based only on the evidence
"""

    response = session.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Follow the requested JSON format. "
                        "Never invent evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    response.raise_for_status()

    raw = response.json()["message"]["content"]

    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().upper()
        reason = str(parsed.get("reason", ""))
    except (json.JSONDecodeError, AttributeError):
        label = ""
        reason = raw

    if label not in LABELS:
        match = re.search(
            r"\b(NOT ENOUGH INFO|SUPPORTS|REFUTES)\b",
            raw.upper(),
        )
        label = match.group(1) if match else "INVALID"

    return label, reason, raw


# =========================================================
# SAMPLE SELECTION
# =========================================================

def build_sample():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download the official labelled "
            "FEVER development JSONL first."
        )

    df = pd.DataFrame(load_jsonl(DATA_PATH))
    df = df[df["label"].isin(LABELS)].copy()

    counts = df["label"].value_counts()
    print("Available labels:\n", counts)

    if any(counts.get(label, 0) < SAMPLE_PER_CLASS for label in LABELS):
        raise ValueError("Not enough examples in one or more classes.")

    sample = (
        df.groupby("label", group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    return sample


# =========================================================
# BENCHMARK
# =========================================================

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Verify the local Ollama API is reachable.
    health = session.get("http://localhost:11434/api/tags", timeout=10)
    health.raise_for_status()
    installed_models = {
        item["name"]
        for item in health.json().get("models", [])
    }
    print("Ollama models:", sorted(installed_models))

    for model in MODELS:
        if not any(
            name == model or name.startswith(model + "-")
            for name in installed_models
        ):
            raise RuntimeError(
                f"{model} is not installed. Run ollama pull {model}, "
                "or update MODELS to match `ollama list`."
            )

    sample = build_sample()
    cache = load_cache()
    evidence_by_id = {}

    # Retrieve evidence once per claim so models get identical context.
    print("\nRetrieving evidence from Wikipedia...")
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        try:
            evidence = retrieve_evidence(row["claim"], cache)
            retrieval_error = ""
        except requests.RequestException as exc:
            evidence = []
            retrieval_error = str(exc)

        evidence_by_id[str(row["id"])] = {
            "evidence": evidence,
            "retrieval_error": retrieval_error,
        }
        save_cache(cache)

    results = []

    for mode in ["claim_only", "retrieved_evidence"]:
        print(f"\n===== Experiment: {mode} =====")

        for model in MODELS:
            print(f"\nModel: {model}")

            for _, row in tqdm(sample.iterrows(), total=len(sample)):
                start = time.perf_counter()
                claim_id = str(row["id"])
                retrieval = evidence_by_id[claim_id]

                try:
                    if mode == "claim_only":
                        evidence = None
                    else:
                        evidence = retrieval["evidence"]

                    pred, reason, raw = call_ollama(
                        model=model,
                        claim=row["claim"],
                        evidence=evidence,
                    )
                    error = ""
                except Exception as exc:
                    pred, reason, raw = "ERROR", "", ""
                    error = str(exc)

                results.append({
                    "id": row["id"],
                    "claim": row["claim"],
                    "gold_label": row["label"],
                    "model": model,
                    "mode": mode,
                    "prediction": pred,
                    "correct": pred == row["label"],
                    "latency_seconds": round(
                        time.perf_counter() - start, 3
                    ),
                    "retrieved_evidence": json.dumps(
                        retrieval["evidence"],
                        ensure_ascii=False,
                    ),
                    "retrieval_error": retrieval["retrieval_error"],
                    "reason": reason,
                    "raw_response": raw,
                    "error": error,
                })

                # Save incrementally so a long run is recoverable.
                pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    # Invalid outputs and API errors count against coverage,
    # but are excluded from the conditional classification metrics.
    valid = results_df[results_df["prediction"].isin(LABELS)].copy()

    metrics = []
    for (mode, model), group in results_df.groupby(["mode", "model"]):
        valid_group = group[group["prediction"].isin(LABELS)]

        accuracy = (
            accuracy_score(
                valid_group["gold_label"],
                valid_group["prediction"],
            )
            if len(valid_group) else float("nan")
        )

        macro_f1 = (
            f1_score(
                valid_group["gold_label"],
                valid_group["prediction"],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
            if len(valid_group) else float("nan")
        )

        metrics.append({
            "mode": mode,
            "model": model,
            "total_examples": len(group),
            "valid_predictions": len(valid_group),
            "coverage": len(valid_group) / len(group),
            "accuracy_valid_only": accuracy,
            "macro_f1_valid_only": macro_f1,
            "mean_latency_seconds": group["latency_seconds"].mean(),
        })

        print(f"\n{mode} | {model}")
        print(f"Valid predictions: {len(valid_group)}/{len(group)}")

        if len(valid_group):
            print(
                classification_report(
                    valid_group["gold_label"],
                    valid_group["prediction"],
                    labels=LABELS,
                    zero_division=0,
                )
            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\nSaved predictions:", OUTPUT_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("\nSummary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
