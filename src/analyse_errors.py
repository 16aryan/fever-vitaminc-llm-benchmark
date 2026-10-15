#!/usr/bin/env python3
"""Error and Disagreement Analysis for fact-verification benchmarks.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model errors and disagreements.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    preds_path = Path(args.predictions or config["paths"]["predictions"])
    if not preds_path.exists():
        fallback = Path("results/fever_predictions.csv")
        if fallback.exists():
            preds_path = fallback

    error_out_path = Path(config["paths"]["error_analysis"])
    error_out_path.parent.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        print(f"Predictions file not found: {preds_path}")
        return

    df = pd.read_csv(preds_path)

    # Categorize error types
    def classify_error(row):
        gold = str(row["gold_label"])
        pred = str(row["prediction"])
        if pred == gold:
            return "CORRECT"
        if pred in ["INVALID", "ERROR"]:
            return f"SYSTEM_{pred}"
        if gold in ["SUPPORTS", "REFUTES"] and pred == "NOT ENOUGH INFO":
            return "OVER_CAUTIOUS_NEI"
        if gold == "NOT ENOUGH INFO" and pred in ["SUPPORTS", "REFUTES"]:
            return "HALLUCINATED_VERDICT"
        if gold == "SUPPORTS" and pred == "REFUTES":
            return "FALSE_REFUTATION"
        if gold == "REFUTES" and pred == "SUPPORTS":
            return "FALSE_SUPPORT"
        return "OTHER_MISMATCH"

    if "dataset" not in df.columns:
        df["dataset"] = "FEVER"
    if "sample_id" not in df.columns:
        df["sample_id"] = df["id"] if "id" in df.columns else df.index

    df["error_category"] = df.apply(classify_error, axis=1)

    print("\n" + "=" * 25 + " ERROR CATEGORY BREAKDOWN " + "=" * 25)
    breakdown = df.groupby(["dataset", "model", "mode", "error_category"]).size().unstack(fill_value=0)
    print(breakdown.to_string())

    # Model Disagreements
    print("\n" + "=" * 25 + " MODEL DISAGREEMENT SUMMARY " + "=" * 25)
    pivot_models = df.pivot(
        index=["dataset", "sample_id", "claim", "gold_label", "mode"],
        columns="model",
        values="prediction",
    ).reset_index()

    model_cols = [c for c in pivot_models.columns if c not in ["dataset", "sample_id", "claim", "gold_label", "mode"]]
    if len(model_cols) >= 2:
        m1, m2 = model_cols[0], model_cols[1]
        disagreements = pivot_models[pivot_models[m1] != pivot_models[m2]].copy()
        print(f"Total Disagreements between {m1} and {m2}: {len(disagreements)} / {len(pivot_models)} samples")

        # Save all errors and disagreements
        df.to_csv(error_out_path, index=False)
        print(f"\nSaved detailed analysis records to: {error_out_path}")

        if not disagreements.empty:
            print(f"\nTop 3 concrete disagreements:")
            for idx, r in disagreements.head(3).iterrows():
                print(f" - Claim: \"{r['claim']}\"")
                print(f"   Gold: {r['gold_label']} | {m1}: {r[m1]} | {m2}: {r[m2]} (Mode: {r['mode']})")
    else:
        df.to_csv(error_out_path, index=False)
        print(f"Saved analysis records to: {error_out_path}")


if __name__ == "__main__":
    main()
