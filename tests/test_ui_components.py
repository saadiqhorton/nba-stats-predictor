"""Tests for UI component functions."""

import pytest
import numpy as np
import pandas as pd

from src.data_processing import make_features_and_target, preprocess_game_data
from src.ui_components import (
    display_last_n_games,
    display_player_header,
    run_metric_predictions,
    _display_team_performance,
    _run_single_metric_prediction,
)


@pytest.fixture
def sample_player_info():
    """Sample player info dict."""
    return {
        "id": 2544,
        "full_name": "LeBron James",
        "first_name": "LeBron",
        "last_name": "James",
    }


@pytest.fixture
def sample_games_sorted():
    """Sample sorted game data."""
    data = {
        "GAME_DATE": pd.to_datetime(
            ["2024-01-09", "2024-01-07", "2024-01-05", "2024-01-03", "2024-01-01"]
        ),
        "MATCHUP": [
            "LAL vs. PHX",
            "LAL @ BKN",
            "LAL vs. MIA",
            "LAL @ GSW",
            "LAL vs. BOS",
        ],
        "PTS": [35, 22, 28, 30, 25],
        "REB": [11, 7, 9, 10, 8],
        "AST": [8, 4, 6, 7, 5],
        "MIN": [40, 32, 36, 38, 35],
        "FGM": [14, 8, 11, 12, 10],
        "FGA": [28, 18, 22, 24, 20],
        "FG3M": [4, 1, 2, 3, 2],
        "FG3A": [10, 5, 7, 8, 6],
        "FTM": [3, 5, 4, 3, 3],
        "FTA": [4, 6, 5, 4, 4],
        "PLUS_MINUS": [10, -5, 8, -3, 5],
    }
    return pd.DataFrame(data)


class TestDisplayPlayerHeader:
    """Tests for display_player_header."""

    @pytest.mark.unit
    def test_renders_without_error(self, sample_player_info):
        """Test that player header renders."""
        display_player_header(sample_player_info, 2544)

    @pytest.mark.unit
    def test_escapes_html_characters(self):
        """Test that special characters in names are escaped."""
        info = {"id": 1, "full_name": '<script>alert("xss")</script>'}
        display_player_header(info, 1)


class TestDisplayLastNGames:
    """Tests for display_last_n_games."""

    @pytest.mark.unit
    def test_renders_without_error(self, sample_games_sorted):
        """Test that last N games table renders."""
        display_last_n_games(sample_games_sorted, 3)

    @pytest.mark.unit
    def test_renders_all_games(self, sample_games_sorted):
        """Test with N larger than available games."""
        display_last_n_games(sample_games_sorted, 10)


class TestDisplayTeamPerformance:
    """Tests for _display_team_performance."""

    @pytest.mark.unit
    def test_returns_teams_and_stats(self, sample_games_sorted):
        """Test team performance extraction."""
        recent_games = preprocess_game_data(sample_games_sorted)
        X, y = make_features_and_target(recent_games, target_col="PTS")

        importances = np.random.rand(len(X.columns))
        teams_clean, avg_stats = _display_team_performance(
            recent_games, X.columns, importances, "PTS", "Points"
        )

        assert isinstance(teams_clean, list)
        assert isinstance(avg_stats, list)
        assert len(teams_clean) == len(avg_stats)

    @pytest.mark.unit
    def test_empty_when_no_opp_features(self):
        """Test when no OPP_ features have importance > 0."""
        df = pd.DataFrame({"PTS": [25], "FG%": [0.5]})
        features = pd.Index(["FG%"])
        importances = np.array([0.5])
        teams_clean, avg_stats = _display_team_performance(
            df, features, importances, "PTS", "Points"
        )
        assert teams_clean == []
        assert avg_stats == []


class TestRunSingleMetricPrediction:
    """Tests for _run_single_metric_prediction."""

    @pytest.mark.unit
    def test_runs_points_prediction(self, sample_games_sorted):
        """Test full prediction pipeline for points."""
        recent_games = preprocess_game_data(sample_games_sorted)
        last_10 = sample_games_sorted.head(5).copy()
        _run_single_metric_prediction(
            recent_games, last_10, "Test Player", "PTS", "Points"
        )

    @pytest.mark.unit
    def test_runs_rebounds_prediction(self, sample_games_sorted):
        """Test full prediction pipeline for rebounds."""
        recent_games = preprocess_game_data(sample_games_sorted)
        last_10 = sample_games_sorted.head(5).copy()
        _run_single_metric_prediction(
            recent_games, last_10, "Test Player", "REB", "Rebounds"
        )


class TestRunMetricPredictions:
    """Tests for run_metric_predictions."""

    @pytest.mark.unit
    def test_runs_all_tabs(self, sample_games_sorted):
        """Test that all three metric tabs render."""
        recent_games = preprocess_game_data(sample_games_sorted)
        last_10 = sample_games_sorted.head(5).copy()
        run_metric_predictions(recent_games, last_10, "Test Player")
