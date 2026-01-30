"""NBA API interaction functions for the NBA Stats Predictor."""

import logging

import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

from src.data_processing import get_recent_seasons

logger = logging.getLogger(__name__)


@st.cache_data
def find_player(player_name: str) -> dict | None:
    """Find player by name and return player info."""
    if not player_name or not player_name.strip():
        return None

    cleaned_name = player_name.strip()

    player_list = players.find_players_by_full_name(cleaned_name)
    if player_list:
        return player_list[0]

    try:
        all_players = players.get_players()
        matching_players = [
            p
            for p in all_players
            if cleaned_name.lower() in p["full_name"].lower()
            or p["full_name"].lower() in cleaned_name.lower()
        ]
        if matching_players:
            for player in matching_players:
                if player["full_name"].lower() == cleaned_name.lower():
                    return player
            return matching_players[0]
    except Exception as e:
        logger.warning("Player search fallback failed: %s", e)

    return None


@st.cache_data(ttl=3600)
def fetch_player_game_logs(player_id: int, season: str) -> pd.DataFrame:
    """Fetch game logs for a player in a specific season."""
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season"
        ).get_data_frames()[0]
        return logs
    except Exception as e:
        logger.warning("Error fetching data for season %s: %s", season, e)
        st.warning(f"Error fetching data for season {season}: {e}")
        return pd.DataFrame()


def fetch_and_combine_game_logs(
    player_id: int, num_seasons: int
) -> pd.DataFrame | None:
    """Fetch and combine game logs from multiple seasons."""
    seasons = get_recent_seasons(num_seasons=num_seasons)
    dfs = []
    for season in seasons:
        logs = fetch_player_game_logs(player_id, season)
        if logs.empty:
            st.warning(f"No game logs found for season {season}.")
            continue
        dfs.append(logs)

    if not dfs:
        st.error(
            "No game data available for recent seasons. Please try a different player."
        )
        return None

    all_games = pd.concat(dfs, ignore_index=True)
    st.write(
        f"<br>Total games combined from {len(seasons)} seasons: {len(all_games)}",
        unsafe_allow_html=True,
    )
    return all_games
