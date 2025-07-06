import gradio as gr
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
import random
import io # Needed for handling plot images
import base64 # Needed for handling plot images

# --- Constants (can keep these) ---
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
    "WAS": "https://a.espncdn.com/i/teamlogos/nba/500/wsh.png"
}

ABBR_REMAP = {
    "ATL": "atl",   # Atlanta Hawks
    "BKN": "bkn",   # Brooklyn Nets
    "BOS": "bos",   # Boston Celtics
    "CHA": "cha",   # Charlotte Hornets
    "CHI": "chi",   # Chicago Bulls
    "CLE": "cle",   # Cleveland Cavaliers
    "DAL": "dal",   # Dallas Mavericks
    "DEN": "den",   # Denver Nuggets
    "DET": "det",   # Detroit Pistons
    "GSW": "gs",    # Golden State Warriors
    "HOU": "hou",   # Houston Rockets
    "IND": "ind",   # Indiana Pacers
    "LAC": "lac",   # LA Clippers
    "LAL": "lal",   # Los Angeles Lakers
    "MEM": "mem",   # Memphis Grizzlies
    "MIA": "mia",   # Miami Heat
    "MIL": "mil",   # Milwaukee Bucks
    "MIN": "min",   # Minnesota Timberwolves
    "NOP": "no",    # New Orleans Pelicans
    "NYK": "ny",    # New York Knicks
    "OKC": "okc",   # Oklahoma City Thunder
    "ORL": "orl",   # Orlando Magic
    "PHI": "phi",   # Philadelphia 76ers
    "PHX": "phx",   # Phoenix Suns
    "POR": "por",   # Portland Trail Blazers
    "SAC": "sac",   # Sacramento Kings
    "SAS": "sas",   # San Antonio Spurs
    "TOR": "tor",   # Toronto Raptors
    "UTA": "utah",  # Utah Jazz
    "WAS": "wsh"    # Washington Wizards
}

# --- Utility Functions (Adapted for Gradio - return figures, dataframes, or strings) ---

def plot_to_image(fig):
    """Converts a Matplotlib figure to a BytesIO object suitable for Gradio Image."""
    if fig is None: # Handle cases where plot might not be generated
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100) # Save with tight bounding box
    plt.close(fig) # Close the figure to free up memory
    buf.seek(0)
    return buf.getvalue() # Gradio Image component can take bytes

def plot_feature_importance(model, features, sort=True, title='Feature Importance', figsize=(10, 6)): # Smaller figsize for Gradio
    importances = model.feature_importances_
    if sort:
        sorted_idx = np.argsort(importances)
        features = np.array(features)[sorted_idx][-15:] # Limit to top 15 for cleaner plot
        importances = importances[sorted_idx][-15:]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(features, importances, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_test, y_pred, player_name, mae, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_test.values, label='Actual Points', marker='o')
    ax.plot(y_pred, label='Predicted Points', marker='x')
    ax.set_title(f'{player_name.title()}: Actual vs Predicted Points')
    ax.set_xlabel('Test Game Index')
    ax.set_ylabel('Points')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_residuals(errors, figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(errors)), errors)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_title('Prediction Errors (Residuals)')
    ax.set_xlabel('Test Game Index')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_points_vs_teams(teams_clean, avg_points, player_name, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(teams_clean, avg_points)
    ax.set_title(f'{player_name.title()} Performance vs. Impact Teams')
    ax.set_xlabel('Teams')
    ax.set_ylabel('Average Points')
    plt.tight_layout()
    return fig

def plot_shap(model, X, figsize=(10, 6)):
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        fig = plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, X, max_display=15, show=False)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"SHAP plot failed: {e}") # Use print for logs in Gradio backend
        return None # Return None if SHAP fails

def safe_divide(numerator, denominator):
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    mask = denominator == 0
    result = pd.Series(0.0, index=numerator.index)
    result[~mask] = numerator[~mask] / denominator[~mask]
    return result

def is_home_game(matchup):
    return 1 if ' vs ' in matchup else 0

def format_date_with_suffix(date_obj):
    day = date_obj.day
    if 11 <= day <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return date_obj.strftime(f"%B {day}{suffix}, %Y")

# --- Caching (functools.lru_cache is Python's built-in equivalent for simple cases) ---
from functools import lru_cache

@lru_cache(maxsize=128) # Cache results of this function
def _get_player_data(player_name_raw): # Renamed to avoid confusion with Gradio's input name
    player_list = players.find_players_by_full_name(player_name_raw)
    if player_list:
        return player_list[0]
    return None

@lru_cache(maxsize=128)
def _fetch_player_game_logs(player_id, seasons_tuple): # Use tuple for cacheability
    dfs = []
    seasons = list(seasons_tuple) # Convert back to list for iteration
    for season in seasons:
        time.sleep(1.0) # Reduced sleep a bit as Gradio might be more resilient or for quicker demos
        try:
            logs = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season, 
                season_type_all_star='Regular Season'
            ).get_data_frames()[0]
            if not logs.empty:
                dfs.append(logs)
        except Exception as e:
            print(f"Warning: Could not fetch data for {season} season: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame() # Return empty DataFrame if no logs

@lru_cache(maxsize=128)
def _preprocess_game_data(all_games_hash): # Pass a hash of DataFrame for cacheability
    all_games = _all_games_cache.get(all_games_hash) # Retrieve actual DataFrame from a temporary cache
    if all_games is None or all_games.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()
        
    recent_games = all_games.copy()

    recent_games['FG%'] = safe_divide(recent_games['FGM'], recent_games['FGA'])
    recent_games['FG3%'] = safe_divide(recent_games['FG3M'], recent_games['FG3A'])
    recent_games['FT%'] = safe_divide(recent_games['FTM'], recent_games['FTA'])
    recent_games['FGA_5game_avg'] = recent_games['FGA'].shift(1).rolling(window=5, min_periods=1).mean()

    recent_games['OPPONENT'] = recent_games['MATCHUP'].apply(lambda x: x.split()[-1])
    opponent_dummies = pd.get_dummies(recent_games['OPPONENT'], prefix='OPP')
    recent_games = pd.concat([recent_games, opponent_dummies], axis=1)
    recent_games.drop(columns=['OPPONENT'], inplace=True)
    recent_games['HOME/AWAY'] = recent_games['MATCHUP'].apply(is_home_game)

    base_cols = [   'GAME_DATE', 'PTS', 'MIN', 'FGA', 'FGA_5game_avg', 'FGM', 'FG%',
                    'FG3A', 'FG3M', 'FG3%', 'FTA', 'FTM', 'FT%', 'AST', 'REB',
                    'PLUS_MINUS', 'HOME/AWAY']
    
    for col in base_cols:
        if col not in recent_games.columns:
            recent_games[col] = 0

    actual_opponent_cols = [col for col in opponent_dummies.columns if col in recent_games.columns]
    recent_games = recent_games[base_cols + actual_opponent_cols]
    recent_games.dropna(subset=['FGA_5game_avg'], inplace=True) 

    if recent_games.empty:
        return pd.DataFrame(), pd.Series(), pd.DataFrame()

    recent_games = recent_games.sample(frac=1, random_state=42).reset_index(drop=True)
    X = recent_games.drop(columns=['PTS', 'GAME_DATE'])
    y = recent_games['PTS']

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return X, y, recent_games

@lru_cache(maxsize=4) # Cache the trained model for few recent player data shapes
def _train_prediction_model(X_train_hash, y_train_hash): # Pass hashes for cacheability
    X_train = _X_train_cache.get(X_train_hash)
    y_train = _y_train_cache.get(y_train_hash)
    if X_train is None or y_train is None:
        return None

    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    return model

# Global temporary caches to store DataFrames for lru_cache hashing strategy
_all_games_cache = {}
_X_train_cache = {}
_y_train_cache = {}

def dataframe_to_hash(df):
    """Helper to convert DataFrame to a hashable format for lru_cache."""
    if df.empty:
        return "empty"
    # Convert to JSON string and hash. More robust for large DFs.
    # For small DFs, df.values.tobytes() or df.to_json().encode() might be too slow or large.
    # A simple hash based on shape and some values might be sufficient for cache key purposes
    return hash(df.shape) + hash(df.sum().sum()) # A simple, fast hash for cache key

# --- Main Gradio Prediction Function ---

def predict_player_points(player_name: str):
    """
    Main function to process player data and make predictions,
    returning all necessary outputs for Gradio.
    """
    if not player_name:
        return ("Please enter a player name.", "", "", None, None, None, None, "", "")

    player_name = player_name.strip()
    
    # Reset caches for a new player search to avoid stale data (lru_cache handles identical inputs)
    # This explicit clearing might not be ideal with lru_cache by default for every run
    # For Gradio, lru_cache works well based on function arguments, so no need to clear manually here
    
    player_info_display = ""
    error_message = ""
    player_headshot_url = ""
    total_games_message = ""
    
    model_evaluation_text = ""
    impact_teams_html = ""
    last_10_games_table_html = ""
    next_game_prediction_text = ""

    plot_importance_img = None
    plot_actual_vs_predicted_img = None
    plot_residuals_img = None
    plot_points_vs_teams_img = None
    plot_shap_img = None

    try:
        player_info = _get_player_data(player_name)

        if player_info is None:
            error_message = f"Player '{player_name}' not found. Please check the spelling."
            return (error_message, "", "", None, None, None, None, "", "")
        
        player_id = player_info['id']
        player_info_display = f"**{player_info['full_name']}** (ID: {player_id})"
        player_headshot_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

        seasons = ('2023-24', '2024-25') # Must be tuple for lru_cache
        all_games = _fetch_player_game_logs(player_id, seasons)
        
        # Add all_games to global cache for preprocessing
        all_games_hash = dataframe_to_hash(all_games)
        _all_games_cache[all_games_hash] = all_games

        if all_games.empty: 
            error_message = "No game data found for this player in the specified seasons. This might be due to no games played yet or an API issue."
            return (error_message, "", "", None, None, None, None, "", "")

        total_games_message = f"Total games combined from {len(seasons)} seasons: {len(all_games)}"

        X, y, recent_games_for_opponent_analysis = _preprocess_game_data(all_games_hash)
        
        if X.empty or y.empty or len(X) < 2:
            error_message = "Not enough game data after preprocessing to train the prediction model (requires at least 2 games)."
            return (error_message, "", "", None, None, None, None, "", "")

        # Store X_train, y_train in global caches for model training
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train_hash = dataframe_to_hash(X_train_temp)
        y_train_hash = dataframe_to_hash(y_train_temp)
        _X_train_cache[X_train_hash] = X_train_temp
        _y_train_cache[y_train_hash] = y_train_temp

        model = _train_prediction_model(X_train_hash, y_train_hash)
        if model is None:
            error_message = "Failed to train prediction model."
            return (error_message, "", "", None, None, None, None, "", "")

        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        model_evaluation_text = f"""
        **Model Evaluation:**
        - Cross-validated MAE scores: {(-scores).round(2).tolist()}
        - Average MAE: {round(-scores.mean(), 2)}
        - Test MAE: {round(mae, 2)}
        """
        
        # --- Visualizations ---
        plot_importance_img = plot_to_image(plot_feature_importance(model, X.columns))
        plot_actual_vs_predicted_img = plot_to_image(plot_actual_vs_predicted(y_test, y_pred, player_name, mae))
        plot_residuals_img = plot_to_image(plot_residuals(y_test - y_pred))

        # Performance vs. Opponents
        avg_points_for_plot = []
        teams_clean_for_plot = []
        
        # Iterate through all opponent columns (which are in X's columns)
        for team_col in [col for col in X.columns if 'OPP_' in col]:
            team_abbr = team_col.replace("OPP_", "")
            
            # Ensure team_col exists in the analysis dataframe (recent_games_for_opponent_analysis)
            if team_col not in recent_games_for_opponent_analysis.columns:
                continue
            
            team_games = recent_games_for_opponent_analysis[recent_games_for_opponent_analysis[team_col] == 1]
            if team_games.empty:
                continue  

            avg = team_games['PTS'].mean()
            all_points_list = team_games['PTS'].tolist()
            num_games = len(team_games)

            avg_points_for_plot.append(avg)
            teams_clean_for_plot.append(team_abbr)
        
        if teams_clean_for_plot and avg_points_for_plot:
            plot_points_vs_teams_img = plot_to_image(plot_points_vs_teams(teams_clean_for_plot, avg_points_for_plot, player_name))
        
        plot_shap_img = plot_to_image(plot_shap(model, X))

        # --- Last 10 Games Table ---
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
        current_year = pd.Timestamp.now().year
        current_year_games = all_games[all_games['GAME_DATE'].dt.year == current_year].copy()
        games_sorted = current_year_games.sort_values(by='GAME_DATE', ascending=False).reset_index(drop=True)

        last_10_df = games_sorted[['GAME_DATE', 'MATCHUP', 'PTS']].head(10).copy()
        if not last_10_df.empty:
            last_10_df['GAME_DATE'] = last_10_df['GAME_DATE'].apply(format_date_with_suffix)
            last_10_df.index = range(1, len(last_10_df) + 1)
            last_10_games_table_html = last_10_df.to_html(classes="table table-dark table-striped", index=True)
            last_10_games_table_html = f"<h3>Last 10 Games in {current_year}</h3>" + last_10_games_table_html
        else:
            last_10_games_table_html = f"<h3>Last 10 Games in {current_year}</h3><p>No games found for {current_year} to display the last 10 games.</p>"
        
        # --- Next Game Prediction ---
        last_10_for_prediction = games_sorted.head(10).copy()
        if not last_10_for_prediction.empty:
            last_10_for_prediction['FG%'] = safe_divide(last_10_for_prediction['FGM'], last_10_for_prediction['FGA'])
            last_10_for_prediction['FG3%'] = safe_divide(last_10_for_prediction['FG3M'], last_10_for_prediction['FG3A'])
            last_10_for_prediction['FT%'] = safe_divide(last_10_for_prediction['FTM'], last_10_for_prediction['FTA'])
            last_10_for_prediction['FGA_5game_avg'] = last_10_for_prediction['FGA'].rolling(window=5, min_periods=1).mean()
            last_10_for_prediction['HOME/AWAY'] = last_10_for_prediction['MATCHUP'].apply(is_home_game)
            last_10_for_prediction['OPPONENT'] = last_10_for_prediction['MATCHUP'].apply(lambda x: x.split()[-1])
            opponent_dummies_10 = pd.get_dummies(last_10_for_prediction['OPPONENT'], prefix='OPP')
            last_10_for_prediction = pd.concat([last_10_for_prediction, opponent_dummies_10], axis=1)

            numeric_cols = last_10_for_prediction.select_dtypes(include=['number']).columns
            avg_stats = last_10_for_prediction[numeric_cols].mean().to_frame().T

            for col in X.columns:
                if col not in avg_stats.columns:
                    avg_stats[col] = 0   
            avg_stats = avg_stats[X.columns]   

            next_pred = model.predict(avg_stats)[0]
            next_game_prediction_text = f"**Next Game Prediction (Based on Last 10 Games Avg):** {next_pred:.2f} points"
        else:
            next_game_prediction_text = "Not enough recent games to make a prediction for the next game."

        return (
            "", # Clear error message
            player_info_display,
            player_headshot_url, # For Gradio Image component URL
            total_games_message,
            model_evaluation_text,
            plot_importance_img,
            plot_actual_vs_predicted_img,
            plot_residuals_img,
            plot_points_vs_teams_img,
            plot_shap_img,
            last_10_games_table_html,
            next_game_prediction_text,
            impact_teams_html # Placeholder, not currently used with individual team displays
        )

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"Error during prediction: {e}")
        return (error_message, "", "", None, None, None, None, None, None, None, "", "", "")

# --- Define the Gradio Interface ---

# Outputs for the Gradio Interface
output_components = [
    gr.Textbox(label="Error Message", visible=True), # General error messages
    gr.Markdown(label="Player Info"),
    gr.Image(label="Player Headshot", width=190, height=260), # Gradio can directly take URL or bytes for image
    gr.Markdown(label="Game Data Summary"),
    gr.Markdown(label="Model Evaluation"),
    gr.Image(label="Feature Importance Plot"),
    gr.Image(label="Actual vs. Predicted Points Plot"),
    gr.Image(label="Prediction Errors (Residuals) Plot"),
    gr.Image(label="Performance Against Opponent Teams Plot"),
    gr.Image(label="SHAP Feature Importance Summary Plot"),
    gr.HTML(label="Last 10 Games Data"), # Use HTML component for pandas.to_html()
    gr.Markdown(label="Next Game Prediction"),
    gr.HTML(label="Performance vs. Impact Teams (Details)") # Placeholder
]

# Create the Gradio Interface
iface = gr.Interface(
    fn=predict_player_points,
    inputs=gr.Textbox(label="Enter NBA Player Name (e.g. LeBron James):", placeholder="LeBron James"),
    outputs=output_components,
    title="NBA Player Points Predictor (Gradio)",
    description="Enter an NBA player's full name to get point predictions and performance insights using historical data.",
    live=False, # Set to False for model prediction apps, true for interactive ones
    allow_flagging="never",
    theme="soft" # Choose a nice theme, like "soft", "huggingface", "base"
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch() # This starts the Gradio web server