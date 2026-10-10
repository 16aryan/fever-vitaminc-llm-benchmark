#!/usr/bin/env python3
"""Unified Benchmark Runner for FEVER and VitaminC using Ollama.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from retrieve_evidence import load_cache, retrieve_evidence, save_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_ollama(url: str, required_models: List[str]) -> None:
    tags_url = url.replace("/api/chat", "/api/tags").replace("/api/generate", "/api/tags")
    try:
        res = requests.get(tags_url, timeout=10)
        res.raise_for_status()
        installed = {m["name"] for m in res.json().get("models", [])}
    except Exception as exc:
        raise RuntimeError(f"Ollama is unreachable at {tags_url}. Ensure ollama is running: {exc}")

    missing = []
    for model in required_models:
        if not any(name == model or name.startswith(model + "-") or name.startswith(model + ":") for name in installed):
            missing.append(model)

    if missing:
        raise RuntimeError(
            f"Missing required models in Ollama: {missing}. Run: "
            + " && ".join(f"ollama pull {m}" for m in missing)
        )


def build_evidence_block(mode: str, retrieved: List[Dict[str, Any]], gold_text: str) -> str:
    if mode == "claim_only":
        return (
            "No external evidence was provided. "
            "Use your existing knowledge, but choose NOT ENOUGH INFO "
            "if you cannot confidently determine the verdict."
        )
    elif mode == "gold_evidence":
        if gold_text:
            return f"Official Ground-Truth Evidence:\n{gold_text}"
        return "No ground truth evidence available for this sample."
    elif mode == "retrieved_evidence":
        if retrieved:
            return "\n".join(
                f"[{i}] Wikipedia ({item['title']}): {item['sentence']}"
                for i, item in enumerate(retrieved, start=1)
            )
        return (
            "No relevant external evidence was retrieved. "
            "Do not assume the claim is true or false."
        )
    return ""


def call_model(
    session: requests.Session,
    url: str,
    model: str,
    claim: str,
    evidence_text: str,
    temperature: float = 0.0,
    timeout: int = 300,
) -> Tuple[str, str, str, str]:
    prompt = f"""You are a strict, objective fact-verification system.

Choose exactly one verdict label:
SUPPORTS
REFUTES
NOT ENOUGH INFO

Definitions:
- SUPPORTS: the supplied evidence proves the claim to be factually true.
- REFUTES: the supplied evidence contradicts or disproves the claim.
- NOT ENOUGH INFO: the evidence does not contain sufficient facts to confirm or deny the claim.

Treat the provided evidence as the sole ground truth. Do not invent facts.

Claim:
{claim}

Evidence:
{evidence_text}

Respond ONLY with a JSON object adhering to this schema:
{{
  "label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
  "reason": "concise explanation grounded exclusively in the evidence"
}}"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a factual verification assistant. Always output valid JSON and never invent facts.",
            },
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        response = session.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        raw = data.get("message", {}).get("content", "").strip()
        error = ""
    except Exception as exc:
        return "ERROR", "", "", str(exc)

    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().upper()
        reason = str(parsed.get("reason", "")).strip()
    except Exception:
        label = ""
        reason = raw

    if label not in VALID_LABELS:
        match = re.search(r"\b(NOT ENOUGH INFO|SUPPORTS|REFUTES)\b", raw.upper())
        label = match.group(1) if match else "INVALID"

    return label, reason, raw, error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full fact-verification benchmark.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--dataset", type=str, default=None, help="Filter to specific dataset")
    parser.add_argument("--model", type=str, default=None, help="Filter to specific model")
    parser.add_argument("--mode", type=str, default=None, help="Filter to specific experiment mode")
    args = parser.parse_args()

    config = load_config(args.config)
    ollama_url = config.get("ollama_url", "http://localhost:11434/api/chat")
    models = [args.model] if args.model else config.get("models", ["qwen2.5:7b", "llama3.1:8b"])
    temp = config.get("temperature", 0.0)
    top_k = config.get("top_k_sentences", 5)

    manifest_path = Path(config["paths"]["sample_manifest"])
    cache_path = Path(config["paths"]["wiki_cache"])
    output_path = Path(config["paths"]["predictions"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        logger.info("Manifest not found at %s. Generating from data files...", manifest_path)
        from prepare_samples import main as prep_main
        prep_main()

    manifest_df = pd.read_csv(manifest_path)
    if args.dataset:
        manifest_df = manifest_df[manifest_df["dataset"] == args.dataset].copy()

    check_ollama(ollama_url, models)
    cache = load_cache(cache_path)
    session = requests.Session()

    # Pre-retrieve evidence for all claims in manifest to guarantee identical context across models
    logger.info("Verifying Wikipedia evidence cache for %d claims...", len(manifest_df))
    evidence_cache = {}
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Retrieving Wikipedia"):
        cid = str(row["sample_id"])
        claim = str(row["claim"])
        ev = retrieve_evidence(claim, cache, top_k=top_k)
        evidence_cache[cid] = ev
    save_cache(cache, cache_path)

    # Check for existing results to allow graceful recovery
    results = []
    if output_path.exists():
        try:
            results = pd.read_csv(output_path).to_dict("records")
            logger.info("Found %d existing records in %s", len(results), output_path)
        except Exception:
            results = []

    existing_keys = {
        (str(r.get("sample_id")), str(r.get("dataset")), str(r.get("model")), str(r.get("mode")))
        for r in results
    }

    datasets_cfg = config.get("datasets", {})

    for ds_name, ds_meta in datasets_cfg.items():
        if args.dataset and ds_name != args.dataset:
            continue

        ds_rows = manifest_df[manifest_df["dataset"] == ds_name]
        if ds_rows.empty:
            continue

        available_modes = ds_meta.get("modes", ["claim_only", "retrieved_evidence"])
        if args.mode:
            available_modes = [m for m in available_modes if m == args.mode]

        for mode in available_modes:
            for model in models:
                logger.info("=== Running Dataset: %s | Mode: %s | Model: %s ===", ds_name, mode, model)

                for _, row in tqdm(ds_rows.iterrows(), total=len(ds_rows), desc=f"{model} [{mode}]"):
                    sid = str(row["sample_id"])
                    claim = str(row["claim"])
                    gold_label = str(row["gold_label"])
                    gold_ev = str(row.get("gold_evidence", ""))

                    key = (sid, ds_name, model, mode)
                    if key in existing_keys:
                        continue

                    retrieved = evidence_cache.get(sid, [])
                    evidence_text = build_evidence_block(mode, retrieved, gold_ev)

                    start_time = time.perf_counter()
                    pred, reason, raw, error = call_model(
                        session=session,
                        url=ollama_url,
                        model=model,
                        claim=claim,
                        evidence_text=evidence_text,
                        temperature=temp,
                    )
                    latency = round(time.perf_counter() - start_time, 3)

                    record = {
                        "dataset": ds_name,
                        "sample_id": sid,
                        "claim": claim,
                        "gold_label": gold_label,
                        "model": model,
                        "mode": mode,
                        "prediction": pred,
                        "correct": (pred == gold_label),
                        "latency_seconds": latency,
                        "evidence_text": evidence_text,
                        "reason": reason,
                        "raw_response": raw,
                        "error": error,
                    }
                    results.append(record)
                    existing_keys.add(key)

                    # Incremental save
                    pd.DataFrame(results).to_csv(output_path, index=False)

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    logger.info("Finished benchmark. Saved %d total predictions to: %s", len(final_df), output_path)


if __name__ == "__main__":
    main()
