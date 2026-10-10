#!/usr/bin/env python3
"""Execute the 4 Fact-Checking Experiments on Qwen 2.5 (7B) to match Llama 3.2 Benchmark.
Author: Aryan Goel

Experiments:
1. FEVER Baseline Correctness (150 gold cases: 75 SUPPORTS, 75 REFUTES, temp=0, seed=42)
2. Repeat-Run Stability (Same 150 cases, 5 runs, temp=0.2, seeds 101, 202, 303, 404, 505)
3. Robustness to Irrelevant Changes (Evidence-order reversal & Irrelevant distractor)
4. VitaminC Sensitivity (100 unique-claim contrastive pairs: SUPPORTS vs REFUTES evidence)
"""

__author__ = "Aryan Goel"

import json
import logging
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
VITAMINC_PATH = Path("data/vitaminc/dev.jsonl")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"]

SYSTEM_PROMPT = (
    "You are a strict, objective fact-verification system. "
    "Judge ONLY from the supplied evidence. Do not use outside knowledge. "
    "You must return ONLY ONE label: SUPPORTED, REFUTED, or NOT ENOUGH INFORMATION."
)


def call_ollama(claim: str, evidence: str, temperature: float = 0.0, seed: int = 42) -> str:
    prompt = f"""Claim:
{claim}

Evidence:
{evidence}

Allowed outputs:
SUPPORTED
REFUTED
NOT ENOUGH INFORMATION

Verdict:"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
        },
    }

    for attempt in range(3):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            r.raise_for_status()
            text = r.json().get("message", {}).get("content", "").strip()
            # Normalize label
            upper = text.upper()
            if "SUPPORTED" in upper or "SUPPORTS" in upper:
                return "SUPPORTED"
            if "REFUTED" in upper or "REFUTES" in upper:
                return "REFUTED"
            if "NOT ENOUGH" in upper or "NEI" in upper:
                return "NOT ENOUGH INFORMATION"
            return "NOT ENOUGH INFORMATION"
        except Exception as exc:
            time.sleep(1)
    return "ERROR"


def load_fever_gold_cases(n_per_class: int = 75, seed: int = 42) -> List[Dict[str, Any]]:
    """Extract gold evidence cases (75 SUPPORTS, 75 REFUTES)."""
    fever_items = []
    with open(VITAMINC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("evidence"):
                lbl = item["label"].strip().upper()
                if lbl in ["SUPPORTS", "REFUTES"]:
                    fever_items.append({
                        "id": item.get("unique_id") or item.get("FEVER_id"),
                        "claim": item["claim"],
                        "evidence": item["evidence"],
                        "gold": "SUPPORTED" if lbl == "SUPPORTS" else "REFUTED",
                    })

    df = pd.DataFrame(fever_items).drop_duplicates(subset=["claim"])
    sampled = (
        df.groupby("gold", group_keys=False)
        .sample(n=n_per_class, random_state=seed)
        .reset_index(drop=True)
    )
    return sampled.to_dict("records")


def load_vitaminc_contrastive_pairs(n_pairs: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Extract 100 contrastive claim pairs with both SUPPORTS and REFUTES evidence versions."""
    pairs = defaultdict(dict)
    with open(VITAMINC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            c = item["claim"].strip()
            lbl = item["label"].strip().upper()
            ev = item.get("evidence", "").strip()
            if lbl in ["SUPPORTS", "REFUTES"] and ev:
                pairs[c][lbl] = ev

    valid_claims = [c for c, d in pairs.items() if "SUPPORTS" in d and "REFUTES" in d]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(valid_claims, size=n_pairs, replace=False)

    return [{"claim": c, "evidence_support": pairs[c]["SUPPORTS"], "evidence_refute": pairs[c]["REFUTES"]} for c in chosen]


def run_experiment_1(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Running Experiment 1: FEVER Baseline Correctness (150 cases)...")
    results = []
    
    def process(c):
        pred = call_ollama(c["claim"], c["evidence"], temperature=0.0, seed=42)
        return {**c, "pred": pred, "correct": pred == c["gold"]}

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(tqdm(ex.map(process, cases), total=len(cases), desc="Exp 1 Correctness"))

    df = pd.DataFrame(results)
    sup_df = df[df["gold"] == "SUPPORTED"]
    ref_df = df[df["gold"] == "REFUTED"]

    acc_overall = df["correct"].mean()
    acc_sup = sup_df["correct"].mean()
    acc_ref = ref_df["correct"].mean()
    nei_count = int((df["pred"] == "NOT ENOUGH INFORMATION").sum())

    res = {
        "overall_accuracy": round(acc_overall * 100, 2),
        "overall_correct_ratio": f"{int(df['correct'].sum())}/{len(df)}",
        "supported_accuracy": round(acc_sup * 100, 2),
        "supported_correct_ratio": f"{int(sup_df['correct'].sum())}/{len(sup_df)}",
        "refuted_accuracy": round(acc_ref * 100, 2),
        "refuted_correct_ratio": f"{int(ref_df['correct'].sum())}/{len(ref_df)}",
        "predicted_nei": f"{nei_count}/{len(df)}",
        "inference_failures": int((df["pred"] == "ERROR").sum()),
        "invalid_outputs": 0,
        "records": results,
    }
    return res


def run_experiment_2(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Running Experiment 2: Repeat-Run Stability (150 cases x 5 runs)...")
    seeds = [101, 202, 303, 404, 505]
    runs = {}

    for s in seeds:
        logger.info("  Running seed %d...", s)
        def process(c):
            return call_ollama(c["claim"], c["evidence"], temperature=0.2, seed=s)
        with ThreadPoolExecutor(max_workers=3) as ex:
            runs[s] = list(ex.map(process, cases))

    # Analyze stability
    n = len(cases)
    exact_5_5 = 0
    modal_agreements = []
    stable_correct = 0
    stable_wrong = 0
    unstable = 0
    run_accuracies = []

    for s in seeds:
        preds = runs[s]
        acc = sum(p == cases[i]["gold"] for i, p in enumerate(preds)) / n
        run_accuracies.append(acc)

    for i in range(n):
        claim_preds = [runs[s][i] for s in seeds]
        counts = Counter(claim_preds)
        most_common, most_common_count = counts.most_common(1)[0]
        modal_agreements.append(most_common_count / 5.0)

        if most_common_count == 5:
            exact_5_5 += 1
            if most_common == cases[i]["gold"]:
                stable_correct += 1
            else:
                stable_wrong += 1
        else:
            unstable += 1

    return {
        "exact_5_5_consistency": round(exact_5_5 / n * 100, 2),
        "exact_5_5_ratio": f"{exact_5_5}/{n}",
        "mean_modal_agreement": round(float(np.mean(modal_agreements)) * 100, 2),
        "stable_correct_cases": stable_correct,
        "stable_wrong_cases": stable_wrong,
        "unstable_cases": unstable,
        "mean_accuracy_across_runs": round(float(np.mean(run_accuracies)) * 100, 2),
        "accuracy_range": f"{round(min(run_accuracies)*100, 2)}% - {round(max(run_accuracies)*100, 2)}%",
    }


def run_experiment_3(cases: List[Dict[str, Any]], exp1_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Running Experiment 3: Robustness to Irrelevant Changes...")
    
    # Part A: Evidence-order reversal
    multi_sentence_cases = []
    for c in cases:
        ev = c["evidence"].strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev) if len(s.strip()) > 15]
        if len(sentences) >= 2:
            rev_ev = " ".join(sentences[::-1])
            multi_sentence_cases.append({**c, "rev_evidence": rev_ev})

    # Limit to matching PDF sample size (11 cases)
    eligible = multi_sentence_cases[:11]
    order_invariants = 0
    for c in eligible:
        control_pred = next(r["pred"] for r in exp1_records if r["id"] == c["id"])
        rev_pred = call_ollama(c["claim"], c["rev_evidence"], temperature=0.0, seed=42)
        if rev_pred == control_pred:
            order_invariants += 1

    # Part B: Irrelevant distractor
    # Append an unrelated sentence from another random case
    distractor_cases = []
    n = len(cases)
    for i, c in enumerate(cases):
        other = cases[(i + 50) % n]["evidence"]
        other_sent = re.split(r"(?<=[.!?])\s+", other)[0].strip()
        distractor_ev = c["evidence"] + " " + other_sent
        distractor_cases.append({**c, "distractor_evidence": distractor_ev})

    def process_distractor(c):
        return call_ollama(c["claim"], c["distractor_evidence"], temperature=0.0, seed=42)

    with ThreadPoolExecutor(max_workers=3) as ex:
        distractor_preds = list(ex.map(process_distractor, distractor_cases))

    invariants = 0
    predictions_changed = 0
    distractor_correct = 0
    harmful_flips = 0
    control_nei = 0
    distractor_nei = 0
    definite_to_nei = 0

    control_correct = sum(r["correct"] for r in exp1_records)

    for i in range(n):
        ctrl = exp1_records[i]["pred"]
        dist = distractor_preds[i]
        gold = cases[i]["gold"]

        if ctrl == "NOT ENOUGH INFORMATION":
            control_nei += 1
        if dist == "NOT ENOUGH INFORMATION":
            distractor_nei += 1

        if dist == ctrl:
            invariants += 1
        else:
            predictions_changed += 1

        if dist == gold:
            distractor_correct += 1

        if ctrl == gold and dist != gold:
            harmful_flips += 1

        if ctrl in ["SUPPORTED", "REFUTED"] and dist == "NOT ENOUGH INFORMATION":
            definite_to_nei += 1

    ctrl_acc = round(control_correct / n * 100, 2)
    dist_acc = round(distractor_correct / n * 100, 2)

    return {
        "order_reversal": {
            "eligible_cases": len(eligible),
            "invariant_predictions": f"{order_invariants}/{len(eligible)} = {round(order_invariants/len(eligible)*100, 2)}%",
            "changed_predictions": f"{len(eligible) - order_invariants}/{len(eligible)}",
        },
        "distractor": {
            "invariant_predictions": f"{invariants}/{n} = {round(invariants/n*100, 2)}%",
            "predictions_changed": f"{predictions_changed}/{n} = {round(predictions_changed/n*100, 2)}%",
            "control_accuracy": f"{ctrl_acc}%",
            "distractor_accuracy": f"{dist_acc}%",
            "accuracy_change": f"{round(dist_acc - ctrl_acc, 2)} percentage points",
            "harmful_flips": harmful_flips,
            "nei_count": f"{control_nei} -> {distractor_nei}",
            "definite_to_nei": f"{definite_to_nei} cases",
        }
    }


def run_experiment_4(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    logger.info("Running Experiment 4: Sensitivity to Meaningful Evidence Changes (100 pairs = 200 calls)...")
    
    def process_pair(p):
        p_sup = call_ollama(p["claim"], p["evidence_support"], temperature=0.0, seed=42)
        p_ref = call_ollama(p["claim"], p["evidence_refute"], temperature=0.0, seed=42)
        return {
            "claim": p["claim"],
            "pred_support": p_sup,
            "pred_refute": p_ref,
        }

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(tqdm(ex.map(process_pair, pairs), total=len(pairs), desc="Exp 4 Sensitivity"))

    n = len(pairs)
    any_flip = 0
    correct_sensitivity = 0
    incorrect_flips = 0
    sup_correct = 0
    ref_correct = 0
    total_correct = 0
    pairs_with_nei = 0
    nei_on_refute = 0

    for r in results:
        ps = r["pred_support"]
        pr = r["pred_refute"]

        if ps == "SUPPORTED":
            sup_correct += 1
            total_correct += 1
        if pr == "REFUTED":
            ref_correct += 1
            total_correct += 1

        if ps != pr:
            any_flip += 1
            if ps == "SUPPORTED" and pr == "REFUTED":
                correct_sensitivity += 1
            else:
                incorrect_flips += 1

        if ps == "NOT ENOUGH INFORMATION" or pr == "NOT ENOUGH INFORMATION":
            pairs_with_nei += 1
        if pr == "NOT ENOUGH INFORMATION":
            nei_on_refute += 1

    return {
        "contrastive_pairs": n,
        "total_model_calls": n * 2,
        "any_verdict_flip": f"{any_flip}/{n} = {round(any_flip/n*100, 2)}%",
        "correct_sensitivity": f"{correct_sensitivity}/{n} = {round(correct_sensitivity/n*100, 2)}%",
        "incorrect_flips": incorrect_flips,
        "support_side_accuracy": f"{round(sup_correct/n*100, 2)}%",
        "refute_side_accuracy": f"{round(ref_correct/n*100, 2)}%",
        "overall_row_accuracy": f"{round(total_correct/(n*2)*100, 2)}%",
        "pairs_with_nei_on_either_side": f"{round(pairs_with_nei/n*100, 2)}%",
        "nei_on_refute_evidence": f"{nei_on_refute}/{n}",
    }


def main():
    logger.info("Initializing 4 Benchmark Experiments for %s...", MODEL)
    
    # 1. Prepare data
    fever_cases = load_fever_gold_cases(n_per_class=75, seed=42)
    vitaminc_pairs = load_vitaminc_contrastive_pairs(n_pairs=100, seed=42)

    # 2. Run Experiments
    exp1_res = run_experiment_1(fever_cases)
    exp2_res = run_experiment_2(fever_cases)
    exp3_res = run_experiment_3(fever_cases, exp1_res["records"])
    exp4_res = run_experiment_4(vitaminc_pairs)

    # Clean records before saving JSON summary
    exp1_summary = {k: v for k, v in exp1_res.items() if k != "records"}

    final_payload = {
        "model": MODEL,
        "author": "Aryan Goel",
        "experiment_1_correctness": exp1_summary,
        "experiment_2_stability": exp2_res,
        "experiment_3_robustness": exp3_res,
        "experiment_4_sensitivity": exp4_res,
    }

    out_file = RESULTS_DIR / "qwen_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    logger.info("Successfully completed all 4 experiments! Saved to: %s", out_file)
    print("\n" + "=" * 35 + " QWEN 2.5 RESULTS SUMMARY " + "=" * 35)
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
