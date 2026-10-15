#!/usr/bin/env python3
"""Dataset Inspection Script for FEVER and VitaminC.
Author: Aryan Goel

Checks row counts, unique IDs, available columns, label distributions,
and displays sample entries to verify dataset schemas.
"""

__author__ = "Aryan Goel"

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def inspect_dataset(name: str, meta: Dict[str, Any]) -> None:
    data_path = Path(meta["path"])
    if not data_path.exists() and "fallback_path" in meta:
        fallback = Path(meta["fallback_path"])
        if fallback.exists():
            data_path = fallback

    print(f"\n{'='*20} Inspecting: {name.upper()} {'='*20}")
    print(f"Path: {data_path.resolve()}")

    if not data_path.exists():
        print(f"❌ File NOT found: {data_path}")
        print("Please check dataset file path.")
        return

    records = inspect_jsonl(data_path)
    df = pd.DataFrame(records)

    print(f"✅ Total rows: {len(df):,}")
    print(f"Available columns: {list(df.columns)}")

    id_col = meta.get("id_field", "id")
    if id_col in df.columns:
        print(f"Unique IDs: {df[id_col].nunique():,}")

    claim_col = meta.get("claim_field", "claim")
    if claim_col in df.columns:
        empty_claims = df[claim_col].isna().sum() + (df[claim_col].str.strip() == "").sum()
        print(f"Claim field: '{claim_col}' (Missing/empty: {empty_claims})")
    else:
        print(f"⚠️ Warning: Configured claim field '{claim_col}' not found!")

    label_col = meta.get("label_field", "label")
    if label_col not in df.columns and "gold_label" in df.columns:
        label_col = "gold_label"
    elif label_col not in df.columns and "label" in df.columns:
        label_col = "label"

    if label_col in df.columns:
        print(f"\nLabel distribution (field: '{label_col}'):")
        print(df[label_col].value_counts(dropna=False).to_string())
    else:
        print(f"⚠️ Warning: Configured label field '{label_col}' not found!")

    evidence_col = meta.get("evidence_field")
    if evidence_col and evidence_col in df.columns:
        print(f"Evidence field: '{evidence_col}' present.")

    print("\nSample entry (first row):")
    sample_row = records[0]
    print(json.dumps(sample_row, indent=2, ensure_ascii=False)[:600] + ("..." if len(str(sample_row)) > 600 else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect fact-verification datasets.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "fever", "vitaminc"],
        help="Which dataset to inspect",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = config.get("datasets", {})

    if args.dataset == "all":
        for name, meta in datasets.items():
            inspect_dataset(name, meta)
    else:
        if args.dataset in datasets:
            inspect_dataset(args.dataset, datasets[args.dataset])
        else:
            print(f"Unknown dataset '{args.dataset}'. Choose from: {list(datasets.keys())}")


if __name__ == "__main__":
    main()
