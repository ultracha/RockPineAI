"""
Rock Pine optimal growing environment recommender.

This script trains two models using the available Rock Pine greenhouse data:
- A regression model to predict `Height_cm` (growth height).
- A classification model to predict `Health_Status` (with special focus on the
  `정상` / healthy class).

After training, the script searches through the observed combinations of input
environment variables to surface the scenarios that simultaneously maximise
expected height and the probability of a healthy outcome.

Usage:
    python rock_pine_recommender.py --data-path /absolute/path/to/dataset.csv

Dependencies:
    pandas
    numpy
    scikit-learn
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, f1_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None

INPUT_FEATURES: List[str] = [
    "Soil_Type",
    "Fertilizer",
    "Plant_Age",
    "Dripper_Count",
    "Water_Daily_cc",
    "Ventilation",
    "Temp_Low",
    "Temp_High",
    "Humidity_Low",
    "Humidity_High",
    "CO2_Low",
    "CO2_High",
]
HEIGHT_TARGET = "Height_cm"
HEALTH_TARGET = "Health_Status"
HEALTHY_LABEL = "0" # "정상"
EXPECTED_HEIGHT_LABEL = "Expected Height (cm)"
HEALTHY_PROBABILITY_LABEL = "Healthy Probability"


@dataclass
class ModelArtifacts:
    height_pipeline: Pipeline
    health_pipeline: Pipeline
    feature_columns: List[str]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Loads the dataset and enforces expected columns."""
    df = pd.read_csv(csv_path)

    required_columns = set(INPUT_FEATURES + [HEIGHT_TARGET, HEALTH_TARGET])
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    return df


def build_preprocessor() -> ColumnTransformer:
    """Creates the shared preprocessor for both pipelines."""
    categorical_features = [
        "Soil_Type",
        "Fertilizer",
        "Plant_Age",
    ]
    numeric_features = [
        "Dripper_Count",
        "Water_Daily_cc",
        "Ventilation",
        "Temp_Low",
        "Temp_High",
        "Humidity_Low",
        "Humidity_High",
        "CO2_Low",
        "CO2_High",
    ]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ]
    )

    return preprocessor


def train_models(df: pd.DataFrame) -> Tuple[ModelArtifacts, dict]:
    """Trains the height and health models and returns evaluation metrics."""
    preprocessor = build_preprocessor()

    features = df[INPUT_FEATURES].copy()
    height_targets = df[HEIGHT_TARGET].astype(float)
    health_targets = df[HEALTH_TARGET].astype(str)

    (
        x_train,
        x_valid,
        height_train,
        height_valid,
        health_train,
        health_valid,
    ) = train_test_split(
        features,
        height_targets,
        health_targets,
        test_size=0.2,
        random_state=42,
        stratify=health_targets,
    )

    height_pipeline = Pipeline(
        memory=None,
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ],
    )

    health_pipeline = Pipeline(
        memory=None,
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ],
    )

    height_pipeline.fit(x_train, height_train)
    health_pipeline.fit(x_train, health_train)

    # Evaluate on validation split
    height_pred = height_pipeline.predict(x_valid)
    height_mae = mean_absolute_error(height_valid, height_pred)
    height_rmse = mean_squared_error(height_valid, height_pred) #, squared=False)

    health_pred = health_pipeline.predict(x_valid)
    health_f1 = f1_score(
        health_valid == HEALTHY_LABEL,
        health_pred == HEALTHY_LABEL,
    )

    metrics = {
        "height_mae": height_mae,
        "height_rmse": height_rmse,
        "health_f1": health_f1,
        "health_classification_report": classification_report(
            health_valid,
            health_pred,
            digits=3,
        ),
    }

    # Refit both models on full data to leverage all observations
    height_pipeline.fit(features, height_targets)
    health_pipeline.fit(features, health_targets)

    return ModelArtifacts(height_pipeline, health_pipeline, INPUT_FEATURES), metrics


def recommend_environments(
    artifacts: ModelArtifacts,
    df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """Runs a grid-search over unique feature combinations observed in the data."""
    feature_space = (
        df[artifacts.feature_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    expected_height = artifacts.height_pipeline.predict(feature_space)
    health_proba_all = artifacts.health_pipeline.predict_proba(feature_space)
    healthy_idx = list(artifacts.health_pipeline.named_steps["model"].classes_).index(HEALTHY_LABEL)
    healthy_probability = health_proba_all[:, healthy_idx]

    recommendations = feature_space.copy()
    recommendations["expected_height_cm"] = expected_height
    recommendations["healthy_probability"] = healthy_probability
    recommendations["score"] = recommendations["expected_height_cm"] * recommendations["healthy_probability"]

    recommendations = recommendations.sort_values(
        by=["healthy_probability", "expected_height_cm", "score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return recommendations.head(top_k)


def _environment_label(row: pd.Series) -> str:
    return (
        f"Soil:{row['Soil_Type']} | Fert:{row['Fertilizer']} | Age:{row['Plant_Age']} | "
        f"Dripper:{row['Dripper_Count']} | Water:{row['Water_Daily_cc']} | Vent:{row['Ventilation']}"
    )


def visualize_results(
    recommendations: pd.DataFrame,
    metrics: dict,
    plot_dir: str,
) -> List[Path]:
    if plt is None:
        raise ImportError(
            "matplotlib is required for visualization. Install it or omit --plot-dir."
        )

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    if not recommendations.empty:
        labels = recommendations.apply(_environment_label, axis=1)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        ax1.bar(
            labels,
            recommendations["expected_height_cm"],
            color="#4B8BBE",
            alpha=0.8,
            label=EXPECTED_HEIGHT_LABEL,
        )
        ax2.plot(
            labels,
            recommendations["healthy_probability"],
            color="#FFB000",
            marker="o",
            label=HEALTHY_PROBABILITY_LABEL,
        )
        ax1.set_ylabel(EXPECTED_HEIGHT_LABEL)
        ax2.set_ylabel(HEALTHY_PROBABILITY_LABEL)
        ax1.set_title("Top Recommended Environments")
        ax1.set_xticklabels(labels.tolist(), rotation=45, ha="right")
        #ax1.tick_params(axis="x", rotation=45, ha="right")
        ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        handles, labels_handles = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels_handles.extend(l)
        ax1.legend(handles, labels_handles, loc="upper right")
        dual_axis_path = plot_path / "top_recommendations.png"
        fig.savefig(dual_axis_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(dual_axis_path)

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            recommendations["expected_height_cm"],
            recommendations["healthy_probability"],
            c=recommendations["score"],
            cmap="viridis",
            s=120,
            edgecolor="k",
        )
        for idx, row in recommendations.iterrows():
            ax.annotate(
                str(idx + 1),
                (row["expected_height_cm"], row["healthy_probability"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
            )
        ax.set_xlabel(EXPECTED_HEIGHT_LABEL)
        ax.set_ylabel(HEALTHY_PROBABILITY_LABEL)
        ax.set_title("Height vs. Healthy Probability")
        fig.colorbar(sc, label="Composite Score")
        scatter_path = plot_path / "height_vs_health_prob.png"
        fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(scatter_path)

    # Metrics summary figure
    fig, ax = plt.subplots(figsize=(6, 4))
    metric_items = [
        ("Height MAE", metrics.get("height_mae")),
        ("Height RMSE", metrics.get("height_rmse")),
        ("Health F1", metrics.get("health_f1")),
    ]
    names = [m[0] for m in metric_items]
    values = [m[1] for m in metric_items]
    bars = ax.bar(names, values, color=["#4B8BBE", "#306998", "#FFB000"])
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    ax.set_title("Model Validation Metrics")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    metrics_path = plot_path / "model_metrics.png"
    fig.savefig(metrics_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(metrics_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train models and recommend optimal Rock Pine environments."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Absolute path to the Rock Pine dataset CSV.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommended environments to display.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump the recommendations as JSON.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory to store visualizations (requires matplotlib).",
    )

    args = parser.parse_args()

    df = load_dataset(args.data_path)
    artifacts, metrics = train_models(df)
    recommendations = recommend_environments(artifacts, df, top_k=args.top_k)

    print("=== Model evaluation metrics (validation split) ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\n=== Top environment recommendations ===")
    print(recommendations.to_string(index=False))

    if args.output_json:
        recommendations.to_json(args.output_json, orient="records", force_ascii=False, indent=2)
        print(f"\nRecommendations saved to: {args.output_json}")

    if args.plot_dir:
        plot_paths = visualize_results(recommendations, metrics, args.plot_dir)
        print("\nVisualization artifacts:")
        for plot in plot_paths:
            print(f"- {plot}")


if __name__ == "__main__":
    main()

