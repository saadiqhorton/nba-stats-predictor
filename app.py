"""NBA Player Points Predictor - Streamlit Application Entry Point."""

import time

import pandas as pd
import streamlit as st

from src.api import fetch_and_combine_game_logs, find_player
from src.constants import LAST_N_GAMES, NUM_SEASONS, SPINNER_DELAY
from src.data_processing import preprocess_game_data
from src.ui_components import (
    display_last_n_games,
    display_player_header,
    run_metric_predictions,
)

st.markdown(
    "<h1 style='color: white;'>NBA Player Points Predictor</h1>", unsafe_allow_html=True
)


def main() -> None:
    """Main application entry point."""
    if "show_success" not in st.session_state:
        st.session_state.show_success = False

    player_name = st.text_input(
        "Enter NBA Player Name (e.g. LeBron James):", key="player_input"
    )

    if player_name != st.session_state.get("last_input"):
        st.session_state.show_success = False
        st.session_state.last_input = player_name

    if not player_name:
        return

    with st.spinner(f"Running prediction for {player_name.title()}..."):
        time.sleep(SPINNER_DELAY)

    player_info = find_player(player_name)
    if not player_info:
        st.error("Player not found.")
        return

    st.session_state.show_success = True
    if st.session_state.show_success:
        st.markdown(
            "<span style='color: #50fa7b; font-size: 18px;'>Player found successfully</span>",
            unsafe_allow_html=True,
        )

    player_id = player_info["id"]
    display_player_header(player_info, player_id)

    all_games = fetch_and_combine_game_logs(player_id, NUM_SEASONS)
    if all_games is None:
        return

    all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])
    all_games_sorted = all_games.sort_values(
        by="GAME_DATE", ascending=False
    ).reset_index(drop=True)

    display_last_n_games(all_games_sorted, LAST_N_GAMES)

    recent_games = preprocess_game_data(all_games)
    last_10 = all_games_sorted.head(LAST_N_GAMES).copy()

    run_metric_predictions(recent_games, last_10, player_name)


if __name__ == "__main__":
    main()
