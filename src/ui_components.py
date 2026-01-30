"""Streamlit UI rendering components for the NBA Stats Predictor."""

import html

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from src.constants import CROSS_VAL_FOLDS, PLAYER_IMAGE_WIDTH, TARGETS, TEAM_LOGO_WIDTH
from src.config.team_mappings import ABBR_REMAP
from src.data_processing import (
    _format_date_with_suffix,
    _prepare_prediction_input,
    make_features_and_target,
    train_model,
)
from src.plots import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_points_vs_teams,
    plot_residuals,
    plot_shap,
)


def display_player_header(player_info: dict, player_id: int) -> None:
    """Display the player header with image and name."""
    image_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"
    escaped_name = html.escape(player_info["full_name"])

    st.markdown(
        f"""
    <img src="{image_url}" alt="{escaped_name}" style="
            width: {PLAYER_IMAGE_WIDTH}px;
            border-radius: 10px;
            position: absolute;
            right: 0;
            top: 0;
        " />
    <div style="
            display: inline-block;
            background-color: #111111;
            padding: 20px 18px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            color: white;
            font-size: 28px;
            line-height: 1.6;
            margin-right: 190px;
        ">
        <strong>{escaped_name}<span style="margin-left: 16px;">(ID: {player_id})</span></strong>
    </div>
    """,
        unsafe_allow_html=True,
    )


def display_last_n_games(games_sorted: pd.DataFrame, n: int) -> None:
    """Display a table of the last N games."""
    last_n_games_display = (
        games_sorted[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]].head(n).copy()
    )
    last_n_games_display["GAME_DATE"] = last_n_games_display["GAME_DATE"].apply(
        _format_date_with_suffix
    )
    last_n_games_display.index = range(1, len(last_n_games_display) + 1)

    st.subheader(f"Last {n} Games")
    st.dataframe(last_n_games_display)


def _display_team_performance(
    recent_games: pd.DataFrame,
    features: pd.Index,
    importances: np.ndarray,
    target_col: str,
    metric_label: str,
) -> tuple[list[str], list[float]]:
    """Display performance breakdown against high-impact teams."""
    impact_teams = [
        feature
        for i, feature in enumerate(features)
        if "OPP_" in feature and importances[i] > 0
    ]

    avg_stats_list = []
    teams_clean = []

    st.write("#### Performance Against High Impact Teams")
    for team in impact_teams:
        team_abbr = team.replace("OPP_", "")
        slug = ABBR_REMAP.get(team_abbr, team_abbr.lower())
        logo_url = f"https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png"

        team_games = recent_games[recent_games[team] == 1]
        if team_games.empty:
            continue

        avg = team_games[target_col].mean()
        all_values = team_games[target_col].tolist()
        num_games = len(team_games)

        avg_stats_list.append(avg)
        teams_clean.append(team_abbr)

        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo_url, width=TEAM_LOGO_WIDTH)
        with col2:
            st.markdown(
                f'<div style="display:flex; align-items:center; height:85px; font-size:20px;">'
                f"<span>{num_games} games, avg {metric_label.lower()} = {avg:.2f}, {all_values}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    return teams_clean, avg_stats_list


def _run_single_metric_prediction(
    recent_games: pd.DataFrame,
    last_10: pd.DataFrame,
    player_name: str,
    target_col: str,
    metric_label: str,
) -> None:
    """Run prediction and display results for a single metric."""
    # Section 1: Next Game Prediction
    st.subheader("Next Game Prediction (Based on Last 10 Games Avg)")

    X_target, y_target = make_features_and_target(recent_games, target_col=target_col)
    model_target, _, _, _, _ = train_model(X_target, y_target)

    avg_stats_pred = _prepare_prediction_input(last_10, X_target)
    prediction_from_avg = model_target.predict(avg_stats_pred)[0]
    st.markdown(
        f"<h2 style='color: #50fa7b;'>Predicted {metric_label}: {prediction_from_avg:.2f}</h2>",
        unsafe_allow_html=True,
    )

    # Section 2: Model Evaluation
    st.write("---")
    st.subheader("Model Evaluation")

    X, y = make_features_and_target(recent_games, target_col=target_col)
    model, X_train, X_test, y_train, y_test = train_model(X, y)

    scores = cross_val_score(
        model, X, y, cv=CROSS_VAL_FOLDS, scoring="neg_mean_absolute_error"
    )
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"Cross-validated MAE scores: {(-scores).round(2).tolist()}")
    st.write("Average MAE:", round(-scores.mean(), 2))
    st.write(f"Test MAE: {round(mae, 2)}")

    # Section 3: Performance Against High Impact Teams
    importances = model.feature_importances_
    features = X.columns
    teams_clean, avg_stats_list = _display_team_performance(
        recent_games, features, importances, target_col, metric_label
    )

    # Section 4: Visualizations
    errors = y_test - y_pred
    st.write("---")
    st.subheader("Visualizations")
    plot_feature_importance(model, features)
    plot_actual_vs_predicted(y_test, y_pred, player_name, mae)
    plot_residuals(errors)
    plot_points_vs_teams(teams_clean, avg_stats_list, player_name, metric_label)
    plot_shap(model, X)


def run_metric_predictions(
    recent_games: pd.DataFrame, last_10: pd.DataFrame, player_name: str
) -> None:
    """Run predictions for all metrics in tabbed interface."""
    st.write("---")
    tabs = st.tabs([label for _, label in TARGETS])

    for (target_col, metric_label), tab in zip(TARGETS, tabs, strict=True):
        with tab:
            _run_single_metric_prediction(
                recent_games, last_10, player_name, target_col, metric_label
            )
