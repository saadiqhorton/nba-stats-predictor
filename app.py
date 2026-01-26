import streamlit as st
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import numpy as np
import time

# Configuration Constants
NUM_SEASONS = 2  # Number of past seasons to fetch for training
ROLLING_WINDOW_SIZE = 5  # Window size for rolling averages
TEST_SIZE = 0.3  # Proportion of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility
CROSS_VAL_FOLDS = 5  # Number of folds for cross-validation
LAST_N_GAMES = 10  # Number of recent games to display and use for prediction
TOP_N_FEATURES = 20  # Number of top features to display in importance plot
FEATURE_IMPORTANCE_THRESHOLD = 0.1  # Threshold for displaying feature importance labels
SPINNER_DELAY = 2  # Seconds to display loading spinner
PLAYER_IMAGE_WIDTH = 280  # Width of player headshot image in pixels
TEAM_LOGO_WIDTH = 80  # Width of team logo in pixels

# Target variables for prediction tabs
TARGETS = [
    ("PTS", "Points"),
    ("REB", "Rebounds"),
    ("AST", "Assists"),
]

st.markdown(
    "<h1 style='color: white;'>NBA Player Points Predictor</h1>", unsafe_allow_html=True
)


TEAM_LOGOS = {
    "ATL": "https://a.espncdn.com/i/teamlogos/nba/500/atl.png",
    "BKN": "https://a.espncdn.com/i/teamlogos/nba/500/bkn.png",
    "BOS": "https://a.espncdn.com/i/teamlogos/nba/500/bos.png",
    "CHA": "https://a.espncdn.com/i/teamlogos/nba/500/cha.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nba/500/chi.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nba/500/cle.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nba/500/dal.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nba/500/den.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nba/500/det.png",
    "GSW": "https://a.espncdn.com/i/teamlogos/nba/500/gs.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nba/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nba/500/ind.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nba/500/lac.png",
    "LAL": "https://a.espncdn.com/i/teamlogos/nba/500/lal.png",
    "MEM": "https://a.espncdn.com/i/teamlogos/nba/500/mem.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nba/500/mia.png",
    "MIL": "https://a.espncdn.com/i/teamlogos/nba/500/mil.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nba/500/min.png",
    "NOP": "https://a.espncdn.com/i/teamlogos/nba/500/no.png",
    "NYK": "https://a.espncdn.com/i/teamlogos/nba/500/ny.png",
    "OKC": "https://a.espncdn.com/i/teamlogos/nba/500/okc.png",
    "ORL": "https://a.espncdn.com/i/teamlogos/nba/500/orl.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nba/500/phi.png",
    "PHX": "https://a.espncdn.com/i/teamlogos/nba/500/phx.png",
    "POR": "https://a.espncdn.com/i/teamlogos/nba/500/por.png",
    "SAC": "https://a.espncdn.com/i/teamlogos/nba/500/sac.png",
    "SAS": "https://a.espncdn.com/i/teamlogos/nba/500/sas.png",
    "TOR": "https://a.espncdn.com/i/teamlogos/nba/500/tor.png",
    "UTA": "https://a.espncdn.com/i/teamlogos/nba/500/utah.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png",
}

ABBR_REMAP = {
    "ATL": "atl",  # Atlanta Hawks
    "BKN": "bkn",  # Brooklyn Nets
    "BOS": "bos",  # Boston Celtics
    "CHA": "cha",  # Charlotte Hornets
    "CHI": "chi",  # Chicago Bulls
    "CLE": "cle",  # Cleveland Cavaliers
    "DAL": "dal",  # Dallas Mavericks
    "DEN": "den",  # Denver Nuggets
    "DET": "det",  # Detroit Pistons
    "GSW": "gs",  # Golden State Warriors
    "HOU": "hou",  # Houston Rockets
    "IND": "ind",  # Indiana Pacers
    "LAC": "lac",  # LA Clippers
    "LAL": "lal",  # Los Angeles Lakers
    "MEM": "mem",  # Memphis Grizzlies
    "MIA": "mia",  # Miami Heat
    "MIL": "mil",  # Milwaukee Bucks
    "MIN": "min",  # Minnesota Timberwolves
    "NOP": "no",  # New Orleans Pelicans
    "NYK": "ny",  # New York Knicks
    "OKC": "okc",  # Oklahoma City Thunder
    "ORL": "orl",  # Orlando Magic
    "PHI": "phi",  # Philadelphia 76ers
    "PHX": "phx",  # Phoenix Suns
    "POR": "por",  # Portland Trail Blazers
    "SAC": "sac",  # Sacramento Kings
    "SAS": "sas",  # San Antonio Spurs
    "TOR": "tor",  # Toronto Raptors
    "UTA": "utah",  # Utah Jazz
    "WAS": "wsh",  # Washington Wizards
}


def plot_feature_importance(
    model, features, sort=True, title="Feature Importance", figsize=(10, 12)
):
    importances = model.feature_importances_

    if sort:
        sorted_idx = np.argsort(importances)
        features = np.array(features)[sorted_idx][-TOP_N_FEATURES:]
        importances = importances[sorted_idx][-TOP_N_FEATURES:]

    fig, ax = plt.subplots(figsize=figsize)
    st.subheader("Feature Importance")
    bars = ax.barh(features, importances, color="skyblue")

    for i, (feature, importance, bar) in enumerate(zip(features, importances, bars)):
        if importance > FEATURE_IMPORTANCE_THRESHOLD:
            ax.text(
                importance - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.4f}",
                va="center",
                ha="right",
                color="white",
                fontsize=12,
                fontweight="bold",
            )
        else:
            # Put label outside bar on the right
            ax.text(
                importance + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{importance:.4f}",
                va="center",
                ha="left",
                fontsize=12,
            )

    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)


def plot_actual_vs_predicted(y_test, y_pred, player_name, mae):
    fig, ax = plt.subplots(figsize=(10, 6))
    st.subheader("Actual vs. Predicted Points")
    ax.plot(y_test.values, label="Actual Points", marker="o")
    ax.plot(y_pred, label="Predicted Points", marker="x")
    ax.set_title(f"{player_name.title()}: Actual vs Predicted Points")
    ax.set_xlabel("Test Game Index")
    ax.set_ylabel("Points")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    actual_vs_pred = pd.DataFrame(
        {
            "Actual Points": y_test.reset_index(drop=True).round(2).astype(str),
            "Predicted Points": pd.Series(y_pred).round(2).astype(str),
        }
    )

    actual_vs_pred.index = [f"Game {i + 1}" for i in range(len(actual_vs_pred))]

    st.subheader("Actual vs Predicted Games")
    st.table(actual_vs_pred)

    st.write("Mean Absolute Error:", round(mae, 2))


def plot_residuals(errors):
    fig, ax = plt.subplots(figsize=(10, 4))
    st.write("---")
    st.subheader("Prediction Errors")
    ax.bar(range(len(errors)), errors)
    ax.axhline(y=0, color="black", linestyle="--")
    ax.set_title("Prediction Errors (Residuals)")
    ax.set_xlabel("Test Game Index")
    ax.set_ylabel("Error (Actual - Predicted)")
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)


def plot_points_vs_teams(teams_clean, avg_stats, player_name, metric_label):
    fig, ax = plt.subplots(figsize=(10, 5))
    st.subheader("Performance Against High Impact Teams")
    ax.bar(teams_clean, avg_stats)
    ax.set_title(f"{player_name.title()} Performance vs. Impact Teams - {metric_label}")
    ax.set_xlabel("Teams")
    ax.set_ylabel(f"Average {metric_label}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_shap(model, X):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        st.subheader("SHAP Feature Importance Summary")
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, max_display=15, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP plot failed: {e}")


# Return 1 if home game (vs), else 0 for away (@).
def is_home_game(matchup):
    return 1 if " vs " in matchup else 0


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    mask = denominator == 0
    result = pd.Series(0.0, index=numerator.index)
    result[~mask] = numerator[~mask] / denominator[~mask]
    return result


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_current_season(today=None):
    """Return current NBA season string in format YYYY-YY based on today's date."""
    today = today or pd.Timestamp.today()
    season_start_year = today.year if today.month >= 8 else today.year - 1
    season_end_year = season_start_year + 1
    return f"{season_start_year}-{str(season_end_year)[-2:]}"


@st.cache_data(ttl=86400)  # Cache for 24 hours
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
def find_player(player_name: str):
    """Find player by name and return player info."""
    if not player_name or not player_name.strip():
        return None

    # Clean the input - strip whitespace and normalize
    cleaned_name = player_name.strip()

    # Try exact match first
    player_list = players.find_players_by_full_name(cleaned_name)
    if player_list:
        return player_list[0]

    # If exact match fails, try case-insensitive search
    try:
        all_players = players.get_players()
        # Case-insensitive search
        matching_players = [
            p
            for p in all_players
            if cleaned_name.lower() in p["full_name"].lower()
            or p["full_name"].lower() in cleaned_name.lower()
        ]
        if matching_players:
            # Return the best match (prefer exact name match first)
            for player in matching_players:
                if player["full_name"].lower() == cleaned_name.lower():
                    return player
            return matching_players[0]  # Return first match as fallback
    except Exception as e:
        # If fallback search fails, return None
        pass

    return None


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_player_game_logs(player_id: int, season: str):
    """Fetch game logs for a player in a specific season."""
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=season, season_type_all_star="Regular Season"
        ).get_data_frames()[0]
        return logs
    except Exception as e:
        st.warning(f"Error fetching data for season {season}: {e}")
        return pd.DataFrame()


@st.cache_data
def preprocess_game_data(all_games: pd.DataFrame):
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


def make_features_and_target(recent_games: pd.DataFrame, target_col: str):
    """Create feature matrix and target variable from preprocessed games data."""
    if target_col not in recent_games.columns:
        raise KeyError(f"Target column {target_col!r} not found in recent_games")

    # Create feature matrix by dropping target and GAME_DATE
    X = recent_games.drop(columns=[target_col, "GAME_DATE"])
    y = recent_games[target_col]

    # Convert to numeric and fill NaN
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = y.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y


@st.cache_resource
def train_model(X: pd.DataFrame, y: pd.Series):
    """Train XGBoost model on preprocessed data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def main():
    if "show_success" not in st.session_state:
        st.session_state.show_success = False

    # Input
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

    # Use cached player search
    player_info = find_player(player_name)

    if not player_info:
        st.error("Player not found.")
        return
    else:
        st.session_state.show_success = True

    if st.session_state.show_success:
        st.markdown(
            "<span style='color: #50fa7b; font-size: 18px;'>Player found successfully</span>",
            unsafe_allow_html=True,
        )

    player_id = player_info["id"]

    image_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

    st.markdown(
        f"""
                
    <img src="{image_url}" alt="{player_info["full_name"]}" style="
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
            margin-right: 190px;  /* Reserve space for image */
        ">
        <strong>{player_info["full_name"]}<span style="margin-left: 16px;">(ID: {player_id})</span></strong>
    </div>
    """,
        unsafe_allow_html=True,
    )
    # Fetch and concatenate player game logs from NBA API for given seasons (CACHED)
    seasons = get_recent_seasons(num_seasons=NUM_SEASONS)
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
        return

    all_games = pd.concat(dfs, ignore_index=True)

    st.write(
        f"<br>Total games combined from {len(seasons)} seasons: {len(all_games)}",
        unsafe_allow_html=True,
    )

    # Convert GAME_DATE to actual datetime format
    all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"])

    # Sort all games in descending order to get most recent games from current season
    all_games_sorted = all_games.sort_values(
        by="GAME_DATE", ascending=False
    ).reset_index(drop=True)

    def format_date_with_suffix(date_obj):
        day = date_obj.day
        if 11 <= day <= 13:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return date_obj.strftime(f"%B {day}{suffix}, %Y")

    # Format GAME_DATE for display
    last_10_debug = (
        all_games_sorted[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]]
        .head(LAST_N_GAMES)
        .copy()
    )
    last_10_debug["GAME_DATE"] = last_10_debug["GAME_DATE"].apply(
        format_date_with_suffix
    )

    last_10_debug.index = range(1, len(last_10_debug) + 1)

    # Show formatted table
    st.subheader("Last 10 Games")
    st.dataframe(last_10_debug)

    # Preprocess game data using cached function
    recent_games = preprocess_game_data(all_games)

    # Define last N games for prediction
    last_10 = all_games_sorted.head(LAST_N_GAMES).copy()

    # Create single tab system for all metrics
    st.write("---")
    tabs = st.tabs([label for _, label in TARGETS])

    for (target_col, metric_label), tab in zip(TARGETS, tabs, strict=True):
        with tab:
            # ===== SECTION 1: NEXT GAME PREDICTION =====
            st.subheader(f"Next Game Prediction (Based on Last 10 Games Avg)")

            # Train model for this specific target
            X_target, y_target = make_features_and_target(
                recent_games, target_col=target_col
            )
            model_target, _, _, _, _ = train_model(X_target, y_target)

            # Preprocess last_10 same as model input
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
            last_10_processed["HOME/AWAY"] = last_10_processed["MATCHUP"].apply(
                is_home_game
            )
            last_10_processed["OPPONENT"] = last_10_processed["MATCHUP"].apply(
                lambda x: x.split()[-1]
            )
            opponent_dummies_10 = pd.get_dummies(
                last_10_processed["OPPONENT"], prefix="OPP"
            )
            last_10_processed = pd.concat(
                [last_10_processed, opponent_dummies_10], axis=1
            )

            # Compute average stats
            numeric_cols = last_10_processed.select_dtypes(include=["number"]).columns
            avg_stats_pred = last_10_processed[numeric_cols].mean().to_frame().T

            # Ensure all necessary columns exist for prediction
            for col in X_target.columns:
                if col not in avg_stats_pred.columns:
                    avg_stats_pred[col] = 0  # fill missing columns

            avg_stats_pred = avg_stats_pred[
                X_target.columns
            ]  # match training set order

            # Predict
            prediction_from_avg = model_target.predict(avg_stats_pred)[0]
            st.markdown(
                f"<h2 style='color: #50fa7b;'>Predicted {metric_label}: {prediction_from_avg:.2f}</h2>",
                unsafe_allow_html=True,
            )

            # ===== SECTION 2: MODEL EVALUATION =====
            st.write("---")
            st.subheader(f"Model Evaluation")

            # Create features and target for this metric
            X, y = make_features_and_target(recent_games, target_col=target_col)

            # Train model using cached function
            model, X_train, X_test, y_train, y_test = train_model(X, y)

            scores = cross_val_score(
                model, X, y, cv=CROSS_VAL_FOLDS, scoring="neg_mean_absolute_error"
            )
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)

            st.write(f"Cross-validated MAE scores: {(-scores).round(2).tolist()}")
            st.write("Average MAE:", round(-scores.mean(), 2))
            st.write(f"Test MAE: {round(mae, 2)}")

            # ===== SECTION 3: PERFORMANCE AGAINST HIGH IMPACT TEAMS =====
            importances = model.feature_importances_
            features = X.columns
            impact_teams = [
                feature
                for i, feature in enumerate(features)
                if "OPP_" in feature and importances[i] > 0
            ]

            avg_stats_list = []
            teams_clean = []

            st.write(f"#### Performance Against High Impact Teams")
            for team in impact_teams:
                team_abbr = team.replace("OPP_", "")
                slug = ABBR_REMAP.get(team_abbr, team_abbr.lower())
                logo_url = f"https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png"

                team_games = recent_games[recent_games[team] == 1]

                if team_games.empty:
                    continue  # Skip this team if no games

                avg = team_games[target_col].mean()
                all_values = team_games[target_col].tolist()
                num_games = len(team_games)

                # Append only if valid
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

            # ===== SECTION 4: VISUALIZATIONS =====
            errors = y_test - y_pred

            st.write("---")
            st.subheader(f"Visualizations")
            plot_feature_importance(model, features)
            plot_actual_vs_predicted(y_test, y_pred, player_name, mae)
            plot_residuals(errors)
            plot_points_vs_teams(teams_clean, avg_stats_list, player_name, metric_label)
            plot_shap(model, X)


if __name__ == "__main__":
    main()
