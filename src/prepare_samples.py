#!/usr/bin/env python3
"""Prepare and freeze balanced evaluation sample manifest across datasets.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

LABEL_NORMALIZATION = {
    "supports": "SUPPORTS",
    "support": "SUPPORTS",
    "supported": "SUPPORTS",
    "true": "SUPPORTS",
    "refutes": "REFUTES",
    "refute": "REFUTES",
    "refuted": "REFUTES",
    "false": "REFUTES",
    "not enough info": "NOT ENOUGH INFO",
    "not enough information": "NOT ENOUGH INFO",
    "nei": "NOT ENOUGH INFO",
}


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def sample_dataset(
    dataset_name: str,
    meta: Dict[str, Any],
    n_per_class: int,
    seed: int,
    allowed_labels: List[str],
) -> pd.DataFrame:
    path = Path(meta["path"])
    if not path.exists():
        print(f"Skipping {dataset_name}: {path} not found.")
        return pd.DataFrame()

    records = load_jsonl(path)
    df = pd.DataFrame(records)

    claim_col = meta.get("claim_field", "claim")
    label_col = meta.get("label_field", "label")
    id_col = meta.get("id_field", "id")
    evidence_col = meta.get("evidence_field")

    # Standardize label
    df["gold_label"] = (
        df[label_col].astype(str).str.strip().str.lower().map(lambda x: LABEL_NORMALIZATION.get(x, x.upper()))
    )
    df = df[df["gold_label"].isin(allowed_labels)].copy()

    # Stratified sampling
    sampled = (
        df.groupby("gold_label", group_keys=False)
        .sample(n=min(len(df), n_per_class), random_state=seed)
        .reset_index(drop=True)
    )

    result_rows = []
    for _, row in sampled.iterrows():
        gold_ev = ""
        if evidence_col and evidence_col in row and pd.notna(row[evidence_col]):
            gold_ev = str(row[evidence_col]).strip()

        result_rows.append({
            "dataset": dataset_name,
            "sample_id": str(row[id_col]) if id_col in row else str(_),
            "claim": str(row[claim_col]).strip(),
            "gold_label": row["gold_label"],
            "gold_evidence": gold_ev,
        })

    out_df = pd.DataFrame(result_rows)
    print(f"Prepared {len(out_df)} samples for '{dataset_name}' (n={n_per_class} per class).")
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed sample manifest.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--n-per-class", type=int, default=None, help="Override sample per class")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    n_per_class = args.n_per_class or config.get("sample_per_class", 10)
    seed = args.seed or config.get("random_seed", 42)
    allowed_labels = config.get("labels", ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    manifest_path = Path(config["paths"]["sample_manifest"])

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for name, meta in config.get("datasets", {}).items():
        sub_df = sample_dataset(name, meta, n_per_class, seed, allowed_labels)
        if not sub_df.empty:
            all_dfs.append(sub_df)

    if not all_dfs:
        print("No datasets were sampled. Check dataset files.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(manifest_path, index=False)
    print(f"\nManifest successfully saved to: {manifest_path}")
    print(f"Total frozen claims across all datasets: {len(final_df)}")
    print(final_df.groupby(["dataset", "gold_label"]).size().to_string())


if __name__ == "__main__":
    main()
