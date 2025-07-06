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
import random

st.set_page_config(layout="wide")

st.markdown("<h1 style='color: white;'>NBA Player Points Predictor</h1>", unsafe_allow_html=True)



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



def plot_feature_importance(model, features, sort=True, title='Feature Importance', figsize=(10,12)):
        importances = model.feature_importances_

        if sort:
            sorted_idx = np.argsort(importances)
            features = np.array(features)[sorted_idx][-20:]
            importances = importances[sorted_idx][-20:]

        fig, ax = plt.subplots(figsize=figsize)
        st.subheader("Feature Importance")
        bars = ax.barh(features, importances, color='skyblue')

        for i, (feature, importance, bar) in enumerate(zip(features, importances, bars)):
            if importance > 0.1:
                  ax.text(
                    importance - 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{importance:.4f}",
                    va='center',
                    ha='right',
                    color='white',
                    fontsize=12,
                    fontweight='bold'
            )
            else:
                # Put label outside bar on the right
                ax.text(
                    importance + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f"{importance:.4f}",
                    va='center',
                    ha='left',
                    fontsize=12
                )
                    
            
        ax.set_title(title)
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)

def plot_actual_vs_predicted(y_test, y_pred, player_name, mae):
    fig, ax = plt.subplots(figsize=(10, 6))
    st.subheader("Actual vs. Predicted Points")
    ax.plot(y_test.values, label='Actual Points', marker='o')
    ax.plot(y_pred, label='Predicted Points', marker='x')
    ax.set_title(f'{player_name.title()}: Actual vs Predicted Points')
    ax.set_xlabel('Test Game Index')
    ax.set_ylabel('Points')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    actual_vs_pred = pd.DataFrame({
        'Actual Points': y_test.reset_index(drop=True).round(2).astype(str),
        'Predicted Points': pd.Series(y_pred).round(2).astype(str)
    })
    
    actual_vs_pred.index = [f"Game {i+1}" for i in range(len(actual_vs_pred))]
    
    st.subheader("Actual vs Predicted Games")
    st.table(actual_vs_pred)

    st.write('Mean Absolute Error:', round(mae, 2))
    
def plot_residuals(errors):
    fig, ax = plt.subplots(figsize=(10, 4))
    st.write('---')
    st.subheader("Prediction Errors")
    ax.bar(range(len(errors)), errors)
    ax.axhline(y=0, color='black', linestyle='--')
    ax.set_title('Prediction Errors (Residuals)')
    ax.set_xlabel('Test Game Index')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    



def plot_points_vs_teams(teams_clean, avg_points, player_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    st.subheader("Performance Against High Impact Teams")
    ax.bar(teams_clean, avg_points)
    ax.set_title(f'{player_name.title()} Performance vs. Impact Teams')
    ax.set_xlabel('Teams')
    ax.set_ylabel('Average Points')
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
        return 1 if ' vs ' in matchup else 0

def main():
    
    if 'show_success' not in st.session_state:
        st.session_state.show_success = False

    # Input
    player_name = st.text_input("Enter NBA Player Name (e.g. LeBron James):", key='player_input')

    

    if player_name != st.session_state.get('last_input'):
        st.session_state.show_success = False
        st.session_state.last_input = player_name

    


    if not player_name:
            return
    
    with st.spinner(f'Running prediction for {player_name.title()}...'):
         time.sleep(random.randint(2,4))
         
    player_list = players.find_players_by_full_name(player_name)

    if not player_list:
        st.error("Player not found.")
        return
    else:
        
        st.session_state.show_success = True

    if st.session_state.show_success:
         st.markdown("<span style='color: #50fa7b; font-size: 18px;'>Player found successfully</span>", unsafe_allow_html=True)

    
    st.write("✅ Checkpoint: pulled player data")

    
    
    
    
    player_info = player_list[0]
    player_id = player_info['id']

    image_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{player_id}.png"

    st.markdown(f"""
                
    <img src="{image_url}" alt="{player_info['full_name']}" style="
            width: 280px; 
            border-radius: 10px; 
            position: absolute; 
            right: 0; 
            top: 0;
        " />
    <div style="
            display: inline-block;
            background-color: #111111; 
            padding: 20px 20px; 
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
            color: white;
            font-size: 22px;
            line-height: 1.6;
            margin-right: 190px;  /* Reserve space for image */
        ">
        <strong>{player_info['full_name']}<span style="margin-left: 16px;">(ID: {player_id})</span></strong>
    </div>
    """, unsafe_allow_html=True)
    # Fetch and concatenate player game logs from NBA API for given seasons.
    seasons = ['2023-24', '2024-25']
    dfs = []
    for season in seasons:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season, 
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        dfs.append(logs)


    all_games = pd.concat(dfs, ignore_index=True)


    st.write(f'<br>Total games combined from {len(seasons)} seasons: {len(all_games)}', unsafe_allow_html=True)

    
    # Create features and preprocess raw game data
    recent_games = all_games.copy()

    def safe_divide(numerator, denominator):
        numerator = numerator.astype(float)
        denominator = denominator.astype(float)
        mask = denominator == 0
        result = pd.Series(0.0, index=numerator.index)
        result[~mask] = numerator[~mask] / denominator[~mask]
        return result

    recent_games['FG%'] = safe_divide(recent_games['FGM'], recent_games['FGA'])
    recent_games['FG3%'] = safe_divide(recent_games['FG3M'], recent_games['FG3A'])
    recent_games['FT%'] = safe_divide(recent_games['FTM'], recent_games['FTA'])

    recent_games['FGA_5game_avg'] = recent_games['FGA'].shift(1).rolling(window=5, min_periods=1).mean()


    recent_games['OPPONENT'] = recent_games['MATCHUP'].apply(lambda x: x.split()[-1])
    opponent_dummies = pd.get_dummies(recent_games['OPPONENT'], prefix='OPP')

    recent_games = pd.concat([recent_games, opponent_dummies], axis=1)

    recent_games.drop(columns=['OPPONENT'], inplace=True)

    recent_games['HOME/AWAY'] = recent_games['MATCHUP'].apply(is_home_game)
    # Home = 1, Away = 0

    base_cols = [   'GAME_DATE', 
                    'PTS', 
                    'MIN', 
                    'FGA',
                    'FGA_5game_avg',
                    'FGM', 
                    'FG%',
                    'FG3A',
                    'FG3M',
                    'FG3%',
                    'FTA',
                    'FTM',
                    'FT%',
                    'AST', 
                    'REB',
                    'PLUS_MINUS',
                    'HOME/AWAY']

    recent_games = recent_games[base_cols + list(opponent_dummies.columns)]


    recent_games = recent_games.sample(frac=1, random_state=42).reset_index(drop=True)



    X = recent_games.drop(columns=['PTS', 'GAME_DATE'])
    y = recent_games['PTS']

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    st.write("✅ Checkpoint: created training set")
    
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)


    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    

    st.write("### Model Evaluation")
    st.write(f"Cross-validated MAE scores: {(-scores).round(2).tolist()}")
    st.write('Average MAE:', round(-scores.mean(), 2))
    st.write(f"Test MAE: {round(mae, 2)}")
    

    importances = model.feature_importances_
    features = X.columns
    impact_teams = [feature for i, feature in enumerate(features) if 'OPP_' in feature and importances[i] > 0]
    
    avg_points = []
    teams_clean = []

    for team in impact_teams:
        team_abbr = team.replace("OPP_", "")
        slug = ABBR_REMAP.get(team_abbr, team_abbr.lower())
        logo_url = f"https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png"

        team_games = recent_games[recent_games[team] == 1]

        if team_games.empty:
            continue  # Skip this team if no games

        avg = team_games['PTS'].mean()
        all_points = team_games['PTS'].tolist()
        num_games = len(team_games)

        # Append only if valid
        avg_points.append(avg)
        teams_clean.append(team_abbr)

        col1, col2 = st.columns([1, 6])
        with col1:
            st.image(logo_url, width=80)
        with col2:
            st.markdown(
                f'<div style="display:flex; align-items:center; height:85px; font-size:20px;">'
                f'<span>{num_games} games, avg points = {avg:.2f}, {all_points}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    

    teams_clean = [team.replace('OPP_', '') for team in impact_teams]
    errors = y_test - y_pred

    st.write('---')
    st.write('# Visualizations')
    plot_feature_importance(model, features)
    plot_actual_vs_predicted(y_test, y_pred, player_name, mae)
    plot_residuals(errors)
    plot_points_vs_teams(teams_clean, avg_points, player_name)
    plot_shap(model, X)

    st.write("✅ Checkpoint: predicted output")
    
    # Convert GAME_DATE to actual datetime format
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])

    # Filter games from 2025 only
    games_2025 = all_games[all_games['GAME_DATE'].dt.year == 2025].copy()

    # Sort in descending order to get most recent games
    games_2025_sorted = games_2025.sort_values(by='GAME_DATE', ascending=False).reset_index(drop=True)

    def format_date_with_suffix(date_obj):
        day = date_obj.day
        if 11 <= day <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        return date_obj.strftime(f"%B {day}{suffix}, %Y")

    # Format GAME_DATE for display
    last_10_debug = games_2025_sorted[['GAME_DATE', 'MATCHUP', 'PTS']].head(10).copy()
    last_10_debug['GAME_DATE'] = last_10_debug['GAME_DATE'].apply(format_date_with_suffix)

    last_10_debug.index = range(1, len(last_10_debug) + 1)
    
    # Show formatted table
    st.subheader("Last 10 Games in 2025")
    st.dataframe(last_10_debug)
   

    # Define last 10 for prediction
    last_10 = games_2025_sorted.head(10).copy()

    # Preprocess last_10 same as model input
    last_10['FG%'] = safe_divide(last_10['FGM'], last_10['FGA'])
    last_10['FG3%'] = safe_divide(last_10['FG3M'], last_10['FG3A'])
    last_10['FT%'] = safe_divide(last_10['FTM'], last_10['FTA'])
    last_10['FGA_5game_avg'] = last_10['FGA'].rolling(window=5, min_periods=1).mean()
    last_10['HOME/AWAY'] = last_10['MATCHUP'].apply(is_home_game)
    last_10['OPPONENT'] = last_10['MATCHUP'].apply(lambda x: x.split()[-1])
    opponent_dummies_10 = pd.get_dummies(last_10['OPPONENT'], prefix='OPP')
    last_10 = pd.concat([last_10, opponent_dummies_10], axis=1)

    # Compute average stats
    numeric_cols = last_10.select_dtypes(include=['number']).columns
    avg_stats = last_10[numeric_cols].mean().to_frame().T

    # Ensure all necessary columns exist for prediction
    for col in X.columns:
        if col not in avg_stats.columns:
            avg_stats[col] = 0  # fill missing columns

    avg_stats = avg_stats[X.columns]  # match training set order

    # Predict
    prediction_from_avg = model.predict(avg_stats)[0]
    st.subheader("Next Game Prediction (Based on Last 10 Games Avg)")
    st.markdown(f"<h4 style='color: #F2F4FF;'>Predicted Points: {prediction_from_avg:.2f}</h4>", unsafe_allow_html=True)

    
            
if __name__ == "__main__":
     main()
            

            

        
            
            
        
        

            






