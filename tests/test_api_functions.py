"""Tests for API functions."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from src.api import find_player, fetch_player_game_logs


class TestFindPlayer:
    """Tests for find_player function."""

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    def test_find_existing_player(self, mock_find):
        """Test finding an existing player."""
        # Mock the API response
        mock_find.return_value = [
            {
                "id": 2544,
                "full_name": "LeBron James",
                "first_name": "LeBron",
                "last_name": "James",
            }
        ]

        result = find_player("LeBron James")

        assert result is not None
        assert result["id"] == 2544
        assert result["full_name"] == "LeBron James"
        mock_find.assert_called_once_with("LeBron James")

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    def test_find_nonexistent_player(self, mock_find):
        """Test finding a player that doesn't exist."""
        # Mock empty response
        mock_find.return_value = []

        result = find_player("Fake Player")

        assert result is None
        mock_find.assert_called_once_with("Fake Player")

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    def test_returns_first_match(self, mock_find):
        """Test that function returns first match when multiple players found."""
        # Mock multiple players (edge case)
        mock_find.return_value = [
            {"id": 1, "full_name": "Player One"},
            {"id": 2, "full_name": "Player Two"},
        ]

        result = find_player("Player")

        assert result["id"] == 1
        assert result["full_name"] == "Player One"

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    def test_whitespace_handling(self, mock_find):
        """Test handling of whitespace in player names."""
        # Mock the API response
        mock_find.return_value = [
            {
                "id": 2544,
                "full_name": "LeBron James",
                "first_name": "LeBron",
                "last_name": "James",
            }
        ]

        # Test various whitespace scenarios
        test_cases = [
            "LeBron James",  # Normal case
            "LeBron James ",  # Trailing space
            " LeBron James",  # Leading space
            "  LeBron James  ",  # Multiple spaces
            "\tLeBron James\n",  # Whitespace characters
        ]

        for test_name in test_cases:
            result = find_player(test_name)
            assert result is not None
            assert result["id"] == 2544
            assert result["full_name"] == "LeBron James"
            # Should be called with cleaned name (no extra whitespace)
            mock_find.assert_called_with("LeBron James")
            mock_find.reset_mock()

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    @patch("src.api.players.get_players")
    def test_case_insensitive_fallback(self, mock_get_players, mock_find):
        """Test case-insensitive fallback search when exact match fails."""
        # Mock exact search as empty
        mock_find.return_value = []

        # Mock all players for fallback search
        mock_get_players.return_value = [
            {
                "id": 2544,
                "full_name": "LeBron James",
                "first_name": "LeBron",
                "last_name": "James",
            },
            {
                "id": 201939,
                "full_name": "Stephen Curry",
                "first_name": "Stephen",
                "last_name": "Curry",
            },
            {
                "id": 203507,
                "full_name": "Giannis Antetokounmpo",
                "first_name": "Giannis",
                "last_name": "Antetokounmpo",
            },
        ]

        # Test different case variations
        test_cases = [
            "lebron james",  # Lowercase
            "LEBRON JAMES",  # Uppercase
            "LeBron",  # Partial match
            "james",  # Partial match (last name)
            "lEbRoN jAmEs",  # Mixed case
        ]

        for test_name in test_cases:
            result = find_player(test_name)
            assert result is not None
            assert result["id"] == 2544
            assert result["full_name"] == "LeBron James"

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    @patch("src.api.players.get_players")
    def test_partial_match_preference(self, mock_get_players, mock_find):
        """Test that exact name matches are preferred over partial matches."""
        # Mock exact search as empty
        mock_find.return_value = []

        # Mock players with similar names
        mock_get_players.return_value = [
            {"id": 1, "full_name": "LeBron James"},
            {"id": 2, "full_name": "LeBron Williams"},
            {"id": 3, "full_name": "Bron James"},
        ]

        # Search for "LeBron James" - should return exact match
        result = find_player("LeBron James")

        assert result is not None
        assert result["id"] == 1
        assert result["full_name"] == "LeBron James"

    @pytest.mark.unit
    def test_empty_input_handling(self):
        """Test handling of empty or None input."""
        # Test empty string
        result = find_player("")
        assert result is None

        # Test whitespace only
        result = find_player("   ")
        assert result is None

        # Test None input
        result = find_player(None)
        assert result is None

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    @patch("src.api.players.get_players")
    def test_fallback_search_error_handling(self, mock_get_players, mock_find):
        """Test that fallback search gracefully handles errors."""
        # Mock both searches to fail
        mock_find.return_value = []
        mock_get_players.side_effect = Exception("API Error")

        result = find_player("LeBron James")

        # Should return None when both searches fail
        assert result is None

    @pytest.mark.unit
    @patch("src.api.players.find_players_by_full_name")
    @patch("src.api.players.get_players")
    def test_lebron_james_specific_cases(self, mock_get_players, mock_find):
        """Test specific LeBron James scenarios that were failing."""
        # Test the original failing case: trailing space
        mock_find.return_value = []

        # Mock fallback data
        mock_get_players.return_value = [
            {
                "id": 2544,
                "full_name": "LeBron James",
                "first_name": "LeBron",
                "last_name": "James",
            }
        ]

        # These should now work (previously would return None)
        test_cases = [
            "LeBron James ",  # Trailing space
            " LeBron James",  # Leading space
            " LeBron James ",  # Both spaces
        ]

        for test_name in test_cases:
            result = find_player(test_name)
            assert result is not None
            assert result["id"] == 2544
            assert result["full_name"] == "LeBron James"


class TestFetchPlayerGameLogs:
    """Tests for fetch_player_game_logs function."""

    @pytest.fixture
    def mock_game_logs(self):
        """Create mock game log data."""
        data = {
            "GAME_DATE": ["2024-01-01", "2024-01-03", "2024-01-05"],
            "MATCHUP": ["LAL vs. BOS", "LAL @ GSW", "LAL vs. MIA"],
            "PTS": [25, 30, 28],
            "MIN": [35, 38, 36],
            "FGM": [10, 12, 11],
            "FGA": [20, 24, 22],
            "FG3M": [2, 3, 2],
            "FG3A": [6, 8, 7],
            "FTM": [3, 3, 4],
            "FTA": [4, 4, 5],
            "AST": [5, 7, 6],
            "REB": [8, 10, 9],
            "PLUS_MINUS": [5, -3, 8],
        }
        return pd.DataFrame(data)

    @pytest.mark.unit
    @patch("src.api.playergamelog.PlayerGameLog")
    def test_fetch_successful(self, mock_gamelog, mock_game_logs):
        """Test successful game log fetch."""
        # Mock the API response
        mock_instance = Mock()
        mock_instance.get_data_frames.return_value = [mock_game_logs]
        mock_gamelog.return_value = mock_instance

        result = fetch_player_game_logs(player_id=2544, season="2023-24")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "PTS" in result.columns
        assert "MATCHUP" in result.columns

        # Verify API was called with correct parameters
        mock_gamelog.assert_called_once_with(
            player_id=2544, season="2023-24", season_type_all_star="Regular Season"
        )

    @pytest.mark.unit
    @patch("src.api.playergamelog.PlayerGameLog")
    def test_fetch_api_error(self, mock_gamelog):
        """Test handling of API errors."""
        # Mock API error
        mock_gamelog.side_effect = Exception("API Error")

        result = fetch_player_game_logs(player_id=2544, season="2023-24")

        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.unit
    @patch("src.api.playergamelog.PlayerGameLog")
    def test_fetch_empty_response(self, mock_gamelog):
        """Test handling of empty game logs."""
        # Mock empty response
        mock_instance = Mock()
        mock_instance.get_data_frames.return_value = [pd.DataFrame()]
        mock_gamelog.return_value = mock_instance

        result = fetch_player_game_logs(player_id=2544, season="2023-24")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.unit
    @patch("src.api.playergamelog.PlayerGameLog")
    def test_different_seasons(self, mock_gamelog, mock_game_logs):
        """Test fetching different seasons."""
        mock_instance = Mock()
        mock_instance.get_data_frames.return_value = [mock_game_logs]
        mock_gamelog.return_value = mock_instance

        # Test multiple seasons
        seasons = ["2022-23", "2023-24", "2024-25"]

        for season in seasons:
            result = fetch_player_game_logs(player_id=2544, season=season)
            assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    @patch("src.api.playergamelog.PlayerGameLog")
    def test_different_player_ids(self, mock_gamelog, mock_game_logs):
        """Test fetching for different player IDs."""
        mock_instance = Mock()
        mock_instance.get_data_frames.return_value = [mock_game_logs]
        mock_gamelog.return_value = mock_instance

        # Test multiple player IDs
        player_ids = [2544, 201939, 203507]  # LeBron, Curry, Giannis

        for player_id in player_ids:
            result = fetch_player_game_logs(player_id=player_id, season="2023-24")
            assert isinstance(result, pd.DataFrame)
