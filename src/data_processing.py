"""Data processing, feature engineering, and model training for the NBA Stats Predictor."""

import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.constants import RANDOM_STATE, ROLLING_WINDOW_SIZE, TEST_SIZE


def is_home_game(matchup: str) -> int:
    """Return 1 if home game (vs), else 0 for away (@)."""
    return 1 if " vs " in matchup else 0


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Safely divide two series, returning 0 where denominator is 0."""
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    mask = denominator == 0
    result = pd.Series(0.0, index=numerator.index)
    result[~mask] = numerator[~mask] / denominator[~mask]
    return result


@st.cache_data(ttl=86400)
def get_current_season(today=None):
    """Return current NBA season string in format YYYY-YY based on today's date."""
    today = today or pd.Timestamp.today()
    season_start_year = today.year if today.month >= 8 else today.year - 1
    season_end_year = season_start_year + 1
    return f"{season_start_year}-{str(season_end_year)[-2:]}"


@st.cache_data(ttl=86400)
def get_recent_seasons(num_seasons=2):
    """Return a list of season strings for current season plus given number of past seasons."""
    current_start_year = int(get_current_season().split("-")[0])
    seasons = []
    for i in range(num_seasons):
        start_year = current_start_year - i
        end_year = start_year + 1
        seasons.append(f"{start_year}-{str(end_year)[-2:]}")
    return seasons


@st.cache_data
def preprocess_game_data(all_games: pd.DataFrame) -> pd.DataFrame:
    """Preprocess game data and create features for model training."""
    recent_games = all_games.copy()

    recent_games["FG%"] = safe_divide(recent_games["FGM"], recent_games["FGA"])
    recent_games["FG3%"] = safe_divide(recent_games["FG3M"], recent_games["FG3A"])
    recent_games["FT%"] = safe_divide(recent_games["FTM"], recent_games["FTA"])

    recent_games["FGA_5game_avg"] = (
        recent_games["FGA"]
        .shift(1)
        .rolling(window=ROLLING_WINDOW_SIZE, min_periods=1)
        .mean()
    )

    recent_games["OPPONENT"] = recent_games["MATCHUP"].apply(lambda x: x.split()[-1])
    opponent_dummies = pd.get_dummies(recent_games["OPPONENT"], prefix="OPP")

    recent_games = pd.concat([recent_games, opponent_dummies], axis=1)
    recent_games.drop(columns=["OPPONENT"], inplace=True)

    recent_games["HOME/AWAY"] = recent_games["MATCHUP"].apply(is_home_game)

    base_cols = [
        "GAME_DATE",
        "PTS",
        "MIN",
        "FGA",
        "FGA_5game_avg",
        "FGM",
        "FG%",
        "FG3A",
        "FG3M",
        "FG3%",
        "FTA",
        "FTM",
        "FT%",
        "AST",
        "REB",
        "PLUS_MINUS",
        "HOME/AWAY",
    ]

    recent_games = recent_games[base_cols + list(opponent_dummies.columns)]
    recent_games = recent_games.sample(frac=1, random_state=RANDOM_STATE).reset_index(
        drop=True
    )

    return recent_games


def make_features_and_target(
    recent_games: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix and target variable from preprocessed games data."""
    if target_col not in recent_games.columns:
        raise KeyError(f"Target column {target_col!r} not found in recent_games")

    X = recent_games.drop(columns=[target_col, "GAME_DATE"])
    y = recent_games[target_col]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = y.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


@st.cache_resource
def train_model(
    X: pd.DataFrame, y: pd.Series
) -> tuple[xgb.XGBRegressor, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train XGBoost model on preprocessed data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def _format_date_with_suffix(date_obj: pd.Timestamp) -> str:
    """Format a date with ordinal suffix (1st, 2nd, 3rd, etc.)."""
    day = date_obj.day
    if 11 <= day <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return date_obj.strftime(f"%B {day}{suffix}, %Y")


def _prepare_prediction_input(
    last_10: pd.DataFrame, X_target: pd.DataFrame
) -> pd.DataFrame:
    """Prepare the last 10 games data for prediction input."""
    last_10_processed = last_10.copy()
    last_10_processed["FG%"] = safe_divide(
        last_10_processed["FGM"], last_10_processed["FGA"]
    )
    last_10_processed["FG3%"] = safe_divide(
        last_10_processed["FG3M"], last_10_processed["FG3A"]
    )
    last_10_processed["FT%"] = safe_divide(
        last_10_processed["FTM"], last_10_processed["FTA"]
    )
    last_10_processed["FGA_5game_avg"] = (
        last_10_processed["FGA"]
        .rolling(window=ROLLING_WINDOW_SIZE, min_periods=1)
        .mean()
    )
    last_10_processed["HOME/AWAY"] = last_10_processed["MATCHUP"].apply(is_home_game)
    last_10_processed["OPPONENT"] = last_10_processed["MATCHUP"].apply(
        lambda x: x.split()[-1]
    )
    opponent_dummies_10 = pd.get_dummies(last_10_processed["OPPONENT"], prefix="OPP")
    last_10_processed = pd.concat([last_10_processed, opponent_dummies_10], axis=1)

    numeric_cols = last_10_processed.select_dtypes(include=["number"]).columns
    avg_stats_pred = last_10_processed[numeric_cols].mean().to_frame().T

    for col in X_target.columns:
        if col not in avg_stats_pred.columns:
            avg_stats_pred[col] = 0

    return avg_stats_pred[X_target.columns]
