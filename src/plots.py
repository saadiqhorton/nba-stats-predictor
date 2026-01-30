"""Visualization functions for the NBA Stats Predictor."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

from src.constants import FEATURE_IMPORTANCE_THRESHOLD, TOP_N_FEATURES

logger = logging.getLogger(__name__)


def _add_importance_label(
    ax: plt.Axes,
    importance: float,
    bar: plt.Rectangle,
    threshold: float = FEATURE_IMPORTANCE_THRESHOLD,
) -> None:
    """Add a label to a bar in the feature importance chart.

    Places label inside bar (white text) if importance > threshold,
    otherwise places it outside bar (black text).
    """
    y_pos = bar.get_y() + bar.get_height() / 2
    if importance > threshold:
        ax.text(
            importance - 0.01,
            y_pos,
            f"{importance:.4f}",
            va="center",
            ha="right",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
    else:
        ax.text(
            importance + 0.002,
            y_pos,
            f"{importance:.4f}",
            va="center",
            ha="left",
            fontsize=12,
        )


def plot_feature_importance(
    model: xgb.XGBRegressor,
    features: np.ndarray,
    sort: bool = True,
    title: str = "Feature Importance",
    figsize: tuple[int, int] = (10, 12),
) -> None:
    """Plot feature importance from an XGBoost model."""
    importances = model.feature_importances_

    if sort:
        sorted_idx = np.argsort(importances)
        features = np.array(features)[sorted_idx][-TOP_N_FEATURES:]
        importances = importances[sorted_idx][-TOP_N_FEATURES:]

    fig, ax = plt.subplots(figsize=figsize)
    st.subheader("Feature Importance")
    bars = ax.barh(features, importances, color="skyblue")

    for importance, bar in zip(importances, bars, strict=True):
        _add_importance_label(ax, importance, bar)

    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)


def plot_actual_vs_predicted(
    y_test: pd.Series, y_pred: np.ndarray, player_name: str, mae: float
) -> None:
    """Plot actual vs predicted values comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    st.subheader("Actual vs. Predicted Points")
    ax.plot(y_test.values, label="Actual Points", marker="o")
    ax.plot(y_pred, label="Predicted Points", marker="x")
    ax.set_title(f"{player_name.title()}: Actual vs Predicted Points")
    ax.set_xlabel("Test Game Index")
    ax.set_ylabel("Points")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    actual_vs_pred = pd.DataFrame(
        {
            "Actual Points": y_test.reset_index(drop=True).round(2).astype(str),
            "Predicted Points": pd.Series(y_pred).round(2).astype(str),
        }
    )

    actual_vs_pred.index = [f"Game {i + 1}" for i in range(len(actual_vs_pred))]

    st.subheader("Actual vs Predicted Games")
    st.table(actual_vs_pred)

    st.write("Mean Absolute Error:", round(mae, 2))


def plot_residuals(errors: pd.Series) -> None:
    """Plot prediction error residuals."""
    fig, ax = plt.subplots(figsize=(10, 4))
    st.write("---")
    st.subheader("Prediction Errors")
    ax.bar(range(len(errors)), errors)
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_title("Prediction Errors (Residuals)")
    ax.set_xlabel("Test Game Index")
    ax.set_ylabel("Error (Actual - Predicted)")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


def plot_points_vs_teams(
    teams_clean: list[str], avg_stats: list[float], player_name: str, metric_label: str
) -> None:
    """Plot player performance against specific teams."""
    fig, ax = plt.subplots(figsize=(10, 5))
    st.subheader("Performance Against High Impact Teams")
    ax.bar(teams_clean, avg_stats)
    ax.set_title(f"{player_name.title()} Performance vs. Impact Teams - {metric_label}")
    ax.set_xlabel("Teams")
    ax.set_ylabel(f"Average {metric_label}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_shap(model: xgb.XGBRegressor, X: pd.DataFrame) -> None:
    """Plot SHAP feature importance summary."""
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        st.subheader("SHAP Feature Importance Summary")
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, max_display=15, show=False)
        st.pyplot(fig)
    except (ValueError, RuntimeError, TypeError) as e:
        logger.warning("SHAP plot failed: %s", e)
        st.warning(f"SHAP plot failed: {e}")
