"""Tests for utility functions."""
import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import is_home_game, get_current_season, get_recent_seasons


class TestIsHomeGame:
    """Tests for is_home_game function."""
    
    @pytest.mark.unit
    def test_home_game_vs(self):
        """Test that 'vs' in matchup returns 1 (home game)."""
        assert is_home_game("LAL vs BOS") == 1
        assert is_home_game("GSW vs BKN") == 1
    
    @pytest.mark.unit
    def test_away_game_at(self):
        """Test that '@' in matchup returns 0 (away game)."""
        assert is_home_game("LAL @ BOS") == 0
        assert is_home_game("GSW @ BKN") == 0
    
    @pytest.mark.unit
    def test_edge_cases(self):
        """Test edge cases."""
        # Multiple 'vs' should still return 1
        assert is_home_game("vs vs vs") == 1
        # No 'vs' should return 0
        assert is_home_game("LAL - BOS") == 0


class TestGetCurrentSeason:
    """Tests for get_current_season function."""
    
    @pytest.mark.unit
    def test_season_during_nba_season(self):
        """Test season calculation during NBA season (Oct-July)."""
        # January 2024 should be 2023-24 season
        test_date = pd.Timestamp(year=2024, month=1, day=15)
        assert get_current_season(test_date) == "2023-24"
        
        # March 2025 should be 2024-25 season
        test_date = pd.Timestamp(year=2025, month=3, day=20)
        assert get_current_season(test_date) == "2024-25"
    
    @pytest.mark.unit
    def test_season_during_offseason(self):
        """Test season calculation during offseason (August-September)."""
        # August 2024 should be 2024-25 season (new season starts)
        test_date = pd.Timestamp(year=2024, month=8, day=1)
        assert get_current_season(test_date) == "2024-25"
        
        # September 2024 should be 2024-25 season
        test_date = pd.Timestamp(year=2024, month=9, day=15)
        assert get_current_season(test_date) == "2024-25"
    
    @pytest.mark.unit
    def test_season_boundary(self):
        """Test season calculation at year boundaries."""
        # July (end of season) should still be previous year's season
        test_date = pd.Timestamp(year=2024, month=7, day=31)
        assert get_current_season(test_date) == "2023-24"
        
        # August (start of new season)
        test_date = pd.Timestamp(year=2024, month=8, day=1)
        assert get_current_season(test_date) == "2024-25"


class TestGetRecentSeasons:
    """Tests for get_recent_seasons function."""
    
    @pytest.mark.unit
    def test_default_two_seasons(self):
        """Test that default returns 2 seasons."""
        seasons = get_recent_seasons(num_seasons=2)
        assert len(seasons) == 2
        assert isinstance(seasons, list)
    
    @pytest.mark.unit
    def test_multiple_seasons(self):
        """Test requesting multiple seasons."""
        seasons = get_recent_seasons(num_seasons=3)
        assert len(seasons) == 3
        
        seasons = get_recent_seasons(num_seasons=5)
        assert len(seasons) == 5
    
    @pytest.mark.unit
    def test_season_format(self):
        """Test that seasons are in correct format."""
        seasons = get_recent_seasons(num_seasons=2)
        
        for season in seasons:
            # Should be in format "YYYY-YY"
            assert isinstance(season, str)
            assert len(season) == 7
            assert season[4] == "-"
            
            # Extract years
            start_year = int(season[:4])
            end_year_suffix = season[5:]
            
            # End year should be start year + 1
            expected_end_suffix = str(start_year + 1)[-2:]
            assert end_year_suffix == expected_end_suffix
    
    @pytest.mark.unit
    def test_seasons_in_descending_order(self):
        """Test that seasons are returned in descending order (most recent first)."""
        seasons = get_recent_seasons(num_seasons=3)
        
        # Extract start years
        years = [int(season[:4]) for season in seasons]
        
        # Should be in descending order
        assert years == sorted(years, reverse=True)
    
    @pytest.mark.unit
    def test_single_season(self):
        """Test requesting single season."""
        seasons = get_recent_seasons(num_seasons=1)
        assert len(seasons) == 1
