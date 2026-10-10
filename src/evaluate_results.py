#!/usr/bin/env python3
"""Evaluate predictions: Accuracy, Macro-F1, Coverage, Confusion Matrices, and Publication Figures.
Author: Aryan Goel
"""

__author__ = "Aryan Goel"

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title(title, fontsize=13, weight="bold", pad=12)
    plt.ylabel("Gold Label (True)", fontsize=11, weight="bold")
    plt.xlabel("Predicted Label", fontsize=11, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metric_bar(df: pd.DataFrame, metric_col: str, title: str, ylabel: str, save_path: Path, ylim=(0, 1.05)) -> None:
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="mode",
        y=metric_col,
        hue="model",
        palette=["#2b5c8f", "#d95f02"],
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title(title, fontsize=14, weight="bold", pad=14)
    plt.ylabel(ylabel, fontsize=11, weight="bold")
    plt.xlabel("Experimental Condition", fontsize=11, weight="bold")
    plt.ylim(ylim)
    plt.legend(title="Model", frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(
                f"{val:.3f}" if ylim[1] <= 1.05 else f"{val:.2f}s",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_f1(per_class_data: list, save_path: Path) -> None:
    if not per_class_data:
        return
    df_cls = pd.DataFrame(per_class_data)
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_cls,
        x="class",
        y="f1",
        hue="experiment",
        palette="tab10",
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title("Per-Class F1 Score Comparison", fontsize=14, weight="bold", pad=14)
    plt.ylabel("F1 Score", fontsize=11, weight="bold")
    plt.xlabel("Label Class", fontsize=11, weight="bold")
    plt.ylim(0, 1.05)
    plt.legend(title="Experiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    preds_path = Path(args.predictions) if args.predictions else Path(config["paths"]["predictions"])
    
    # Fallback to fever_predictions.csv if predictions.csv is missing
    if not preds_path.exists():
        fallback_path = Path("results/fever_predictions.csv")
        if fallback_path.exists():
            preds_path = fallback_path
        else:
            print(f"❌ Predictions file not found at {preds_path} or {fallback_path}.")
            return

    metrics_path = Path(config["paths"]["metrics"])
    figures_dir = Path(config["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(preds_path)
    print(f"\nLoaded {len(df)} predictions from {preds_path}")

    if "dataset" not in df.columns:
        df["dataset"] = "FEVER"

    metrics_list = []
    per_class_records = []
    group_cols = ["dataset", "model", "mode"]

    for (dataset, model, mode), group in df.groupby(group_cols):
        total = len(group)
        valid_df = group[group["prediction"].isin(LABELS)]
        valid_count = len(valid_df)
        coverage = round(valid_count / max(total, 1), 4)

        if valid_count > 0:
            acc = round(accuracy_score(valid_df["gold_label"], valid_df["prediction"]), 4)
            macro_f1 = round(
                f1_score(
                    valid_df["gold_label"],
                    valid_df["prediction"],
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                4,
            )
            # Detailed per-class report
            rep = classification_report(
                valid_df["gold_label"],
                valid_df["prediction"],
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            )
            for lbl in LABELS:
                per_class_records.append({
                    "experiment": f"{model} ({mode})",
                    "class": lbl,
                    "f1": round(rep[lbl]["f1-score"], 3),
                    "precision": round(rep[lbl]["precision"], 3),
                    "recall": round(rep[lbl]["recall"], 3),
                })

            # Plot confusion matrix
            clean_model = model.replace(":", "_").replace("/", "_")
            cm_filename = f"cm_{dataset.lower()}_{clean_model}_{mode}.png"
            plot_confusion_matrix(
                valid_df["gold_label"],
                valid_df["prediction"],
                title=f"Confusion Matrix: {model} [{mode}]",
                save_path=figures_dir / cm_filename,
            )
        else:
            acc = 0.0
            macro_f1 = 0.0

        mean_lat = round(group["latency_seconds"].mean(), 3)

        metrics_list.append({
            "dataset": dataset,
            "model": model,
            "mode": mode,
            "total_samples": total,
            "valid_predictions": valid_count,
            "coverage": coverage,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "mean_latency_s": mean_lat,
        })

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(metrics_path, index=False)

    # Generate Figures
    print("\nGenerating figures in:", figures_dir)
    plot_metric_bar(
        metrics_df,
        metric_col="macro_f1",
        title="Macro-F1 Performance by Model & Condition",
        ylabel="Macro-F1 Score",
        save_path=figures_dir / "macro_f1_comparison.png",
        ylim=(0, 0.45),
    )
    plot_metric_bar(
        metrics_df,
        metric_col="accuracy",
        title="Accuracy Performance by Model & Condition",
        ylabel="Accuracy",
        save_path=figures_dir / "accuracy_comparison.png",
        ylim=(0, 0.50),
    )
    plot_metric_bar(
        metrics_df,
        metric_col="mean_latency_s",
        title="Inference Latency Comparison (Apple Silicon)",
        ylabel="Seconds per Claim",
        save_path=figures_dir / "latency_comparison.png",
        ylim=(0, max(metrics_df["mean_latency_s"]) * 1.25),
    )
    plot_per_class_f1(
        per_class_records,
        save_path=figures_dir / "per_class_f1_breakdown.png",
    )

    print("\n" + "=" * 32 + " EVALUATION SUMMARY " + "=" * 32)
    print(metrics_df.to_string(index=False))
    print(f"\n✅ All figures successfully generated in: {figures_dir}")
    for fig in figures_dir.glob("*.png"):
        print(f"  - {fig.name}")


if __name__ == "__main__":
    main()
