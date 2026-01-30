"""Tests for plotting functions."""

import pytest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.plots import (
    _add_importance_label,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_points_vs_teams,
    plot_residuals,
    plot_shap,
)

matplotlib.use("Agg")


@pytest.fixture
def trained_model():
    """Create a small trained XGBoost model for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "FG%": np.random.rand(30),
            "FG3%": np.random.rand(30),
            "FT%": np.random.rand(30),
            "OPP_BOS": np.random.randint(0, 2, 30),
        }
    )
    y = pd.Series(np.random.rand(30) * 30)
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestPlotFeatureImportance:
    """Tests for plot_feature_importance."""

    @pytest.mark.unit
    def test_runs_without_error(self, trained_model):
        """Test that the function completes without raising."""
        model, X, _ = trained_model
        plot_feature_importance(model, X.columns)
        plt.close("all")

    @pytest.mark.unit
    def test_unsorted(self, trained_model):
        """Test plotting without sorting."""
        model, X, _ = trained_model
        plot_feature_importance(model, X.columns, sort=False)
        plt.close("all")


class TestPlotActualVsPredicted:
    """Tests for plot_actual_vs_predicted."""

    @pytest.mark.unit
    def test_runs_without_error(self):
        """Test basic plot creation."""
        y_test = pd.Series([20, 25, 30, 22, 28])
        y_pred = np.array([21, 24, 29, 23, 27])
        plot_actual_vs_predicted(y_test, y_pred, "Test Player", 2.0)
        plt.close("all")


class TestPlotResiduals:
    """Tests for plot_residuals."""

    @pytest.mark.unit
    def test_runs_without_error(self):
        """Test basic residuals plot."""
        errors = pd.Series([1.5, -2.0, 0.5, -1.0, 3.0])
        plot_residuals(errors)
        plt.close("all")


class TestPlotPointsVsTeams:
    """Tests for plot_points_vs_teams."""

    @pytest.mark.unit
    def test_runs_without_error(self):
        """Test team performance bar chart."""
        plot_points_vs_teams(
            ["BOS", "GSW", "MIA"], [25.0, 28.5, 22.0], "Test Player", "Points"
        )
        plt.close("all")


class TestPlotShap:
    """Tests for plot_shap."""

    @pytest.mark.unit
    def test_runs_without_error(self, trained_model):
        """Test SHAP plot generation."""
        model, X, _ = trained_model
        plot_shap(model, X)
        plt.close("all")

    @pytest.mark.unit
    def test_handles_shap_failure(self):
        """Test that SHAP failure is handled gracefully."""
        # Pass invalid model to trigger exception
        plot_shap(None, pd.DataFrame({"a": [1]}))
        plt.close("all")


class TestAddImportanceLabelExtended:
    """Extended tests for _add_importance_label covering both branches."""

    @pytest.fixture
    def ax_and_bar(self):
        """Create axes and bar for testing."""
        fig, ax = plt.subplots()
        bars = ax.barh([0], [0.15])
        return ax, bars[0]

    @pytest.mark.unit
    def test_high_importance_inside_bar(self, ax_and_bar):
        """Test label placement inside bar for high importance."""
        ax, bar = ax_and_bar
        _add_importance_label(ax, 0.15, bar, threshold=0.1)
        assert len(ax.texts) == 1
        assert ax.texts[0].get_color() == "white"
        plt.close("all")

    @pytest.mark.unit
    def test_low_importance_outside_bar(self, ax_and_bar):
        """Test label placement outside bar for low importance."""
        ax, bar = ax_and_bar
        _add_importance_label(ax, 0.05, bar, threshold=0.1)
        assert len(ax.texts) == 1
        plt.close("all")
