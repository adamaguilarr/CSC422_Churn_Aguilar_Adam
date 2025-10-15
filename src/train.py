from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .config import Config
from .preprocess import (
    load_raw,
    clean_dataframe,
    split_features_target,
    build_preprocessor,
    train_test_split_strat,
)
from .models import get_models
from .baseline import MajorityClassBaseline


def evaluate_and_save(y_true, y_pred, name: str, out_dir: Path):
    """Compute metrics + save metrics row and confusion matrix image."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    metrics = {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    # Save metrics row-wise in a CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        pd.concat([old, pd.DataFrame([metrics])], ignore_index=True).to_csv(
            metrics_path, index=False
        )
    else:
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"cm_{name}.png", dpi=150)
    plt.close(fig)


def main(data_path: str, out_dir: str):
    cfg = Config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and clean
    df = load_raw(data_path)

    # Telco quirk: TotalCharges has some blanks -> NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    df = clean_dataframe(df, cfg)

    # Quick distribution plot for target
    target_counts = df[cfg.target_col].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5, 3))
    target_counts.plot(kind="bar", ax=ax)
    ax.set_title("Churn distribution")
    ax.set_xlabel("Churn (0=No, 1=Yes)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "churn_distribution.png", dpi=150)
    plt.close(fig)

    # Split
    X, y = split_features_target(df, cfg)
    preprocessor, num_cols, cat_cols = build_preprocessor(X)
    X_train, X_test, y_train, y_test = train_test_split_strat(X, y, cfg)

    # Baseline (majority class)
    baseline = MajorityClassBaseline().fit(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    evaluate_and_save(y_test, y_pred_base, "baseline_majority", out_dir)

    # Shallow models
    models = get_models()

    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

        # Try CV, but don't let it crash the run
        cv_summary = {}
        try:
            cv_scores = cross_val_score(
                pipe, X_train, y_train, cv=cfg.cv_folds, scoring="f1"
            )
            cv_summary = {
                "cv_f1_mean": float(np.mean(cv_scores)),
                "cv_f1_std": float(np.std(cv_scores)),
                "folds": cfg.cv_folds,
            }
        except Exception as e:
            cv_summary = {"error": str(e), "folds": cfg.cv_folds}

        # Fit on train and evaluate on holdout
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            evaluate_and_save(y_test, y_pred, name, out_dir)
        except Exception as e:
            # Save an error note so it's visible in results/
            with open(out_dir / f"error_{name}.txt", "w") as f:
                f.write(str(e))
            print(f"[WARN] {name} failed: {e}")

        # Save CV summary (or error)
        with open(out_dir / f"cv_{name}.json", "w") as f:
            json.dump(cv_summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to Telco-Customer-Churn.csv"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results", help="Where to write metrics and plots"
    )
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
