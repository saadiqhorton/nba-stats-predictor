"""Shared test configuration and fixtures."""

import sys
import os

import pytest

# Add project root to path (replaces sys.path.insert in every test file)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """Clear all Streamlit caches before each test to prevent interference."""
    from src.api import find_player, fetch_player_game_logs
    from src.data_processing import (
        preprocess_game_data,
        get_current_season,
        get_recent_seasons,
    )

    # Clear cached functions before each test
    find_player.clear()
    fetch_player_game_logs.clear()
    preprocess_game_data.clear()
    get_current_season.clear()
    get_recent_seasons.clear()
    yield
