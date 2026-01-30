"""Tests for API integration functions (fetch_and_combine_game_logs) and coverage gaps."""

import pytest
import pandas as pd
from unittest.mock import patch

from src.api import fetch_and_combine_game_logs
from src.data_processing import make_features_and_target


class TestFetchAndCombineGameLogs:
    """Tests for fetch_and_combine_game_logs function."""

    @pytest.fixture
    def mock_game_logs(self):
        """Create mock game log data."""
        return pd.DataFrame(
            {
                "GAME_DATE": ["2024-01-01", "2024-01-03"],
                "MATCHUP": ["LAL vs. BOS", "LAL @ GSW"],
                "PTS": [25, 30],
                "MIN": [35, 38],
                "FGM": [10, 12],
                "FGA": [20, 24],
                "FG3M": [2, 3],
                "FG3A": [6, 8],
                "FTM": [3, 3],
                "FTA": [4, 4],
                "AST": [5, 7],
                "REB": [8, 10],
                "PLUS_MINUS": [5, -3],
            }
        )

    @pytest.mark.unit
    @patch("src.api.fetch_player_game_logs")
    @patch("src.api.get_recent_seasons")
    def test_combines_multiple_seasons(self, mock_seasons, mock_fetch, mock_game_logs):
        """Test combining game logs from multiple seasons."""
        mock_seasons.return_value = ["2024-25", "2023-24"]
        mock_fetch.return_value = mock_game_logs

        result = fetch_and_combine_game_logs(player_id=2544, num_seasons=2)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 2 games per season * 2 seasons

    @pytest.mark.unit
    @patch("src.api.fetch_player_game_logs")
    @patch("src.api.get_recent_seasons")
    def test_returns_none_when_no_data(self, mock_seasons, mock_fetch):
        """Test returns None when no game data is available."""
        mock_seasons.return_value = ["2024-25"]
        mock_fetch.return_value = pd.DataFrame()

        result = fetch_and_combine_game_logs(player_id=9999, num_seasons=1)

        assert result is None

    @pytest.mark.unit
    @patch("src.api.fetch_player_game_logs")
    @patch("src.api.get_recent_seasons")
    def test_skips_empty_seasons(self, mock_seasons, mock_fetch, mock_game_logs):
        """Test that empty seasons are skipped but valid ones are kept."""
        mock_seasons.return_value = ["2024-25", "2023-24"]
        mock_fetch.side_effect = [mock_game_logs, pd.DataFrame()]

        result = fetch_and_combine_game_logs(player_id=2544, num_seasons=2)

        assert result is not None
        assert len(result) == 2  # Only first season has data

    @pytest.mark.unit
    @patch("src.api.fetch_player_game_logs")
    @patch("src.api.get_recent_seasons")
    def test_single_season(self, mock_seasons, mock_fetch, mock_game_logs):
        """Test with a single season."""
        mock_seasons.return_value = ["2024-25"]
        mock_fetch.return_value = mock_game_logs

        result = fetch_and_combine_game_logs(player_id=2544, num_seasons=1)

        assert result is not None
        assert len(result) == 2


class TestMakeFeaturesAndTargetEdgeCases:
    """Coverage gap: test KeyError branch in make_features_and_target."""

    @pytest.mark.unit
    def test_invalid_target_column_raises(self):
        """Test that an invalid target column raises KeyError."""
        df = pd.DataFrame(
            {
                "GAME_DATE": ["2024-01-01"],
                "PTS": [25],
                "FG%": [0.5],
            }
        )
        with pytest.raises(KeyError, match="INVALID_COL"):
            make_features_and_target(df, target_col="INVALID_COL")
