# Prediction Engine

## What This Does

The prediction engine is the core of the NBA Stats Predictor. It takes a player's game history, builds a machine learning model, and predicts how many points, rebounds, or assists they will score in their next game.

## Why It Matters

Predicting player performance helps fantasy sports players make lineup decisions and helps basketball fans understand what drives a player's stats. The model shows not just a number, but why it made that prediction.

## How to Use It

1. Open the app and search for a player.
2. The app automatically fetches game data from the last 2 NBA seasons.
3. Click the **Points**, **Rebounds**, or **Assists** tab.
4. The "Next Game Prediction" section shows the predicted value.
5. Scroll down to see model accuracy, team breakdowns, and charts.

## How It Works

The prediction engine has four stages.

### Stage 1: Data Collection

The app calls the NBA API to get game logs. It fetches data from the current season and the previous season. Each game log includes stats like points, rebounds, assists, field goal attempts, free throws, and the opponent.

**Code location:** `src/api.py` - `fetch_and_combine_game_logs()`

### Stage 2: Feature Engineering

Raw game data is turned into features the model can learn from. The app creates:

- **Shooting percentages** - Field goal %, three-point %, and free throw %. Calculated by dividing makes by attempts. If the player had zero attempts, the percentage is set to zero instead of causing an error.
- **Rolling averages** - A 5-game rolling average of field goal attempts. This captures recent shooting volume trends.
- **Opponent encoding** - Each opponent team gets its own column (for example, `OPP_BOS` for Boston). The value is 1 if the player faced that team, 0 otherwise. This lets the model learn that a player scores more against certain teams.
- **Home/Away flag** - Set to 1 for home games and 0 for away games.

**Code location:** `src/data_processing.py` - `preprocess_game_data()`

### Stage 3: Model Training

The app trains an XGBoost regression model. XGBoost is a machine learning method that builds many small decision trees. Each tree corrects the mistakes of the previous ones.

The training process:
1. Split the data: 70% for training, 30% for testing.
2. Train the XGBoost model on the training set.
3. Evaluate on the test set using Mean Absolute Error (MAE). MAE is the average difference between predicted and actual values. Lower is better.
4. Run 5-fold cross-validation. This trains and tests the model 5 times on different data splits to check consistency.

**Code location:** `src/data_processing.py` - `train_model()`

### Stage 4: Next Game Prediction

To predict the next game, the app:
1. Takes the player's last 10 games.
2. Computes the same features (shooting %, rolling averages, opponent, home/away).
3. Averages these features into a single input row.
4. Feeds this input to the trained model.
5. The model outputs a predicted stat value.

**Code location:** `src/data_processing.py` - `_prepare_prediction_input()`

## Visualizations

Each prediction tab shows five charts:

| Chart | What It Shows |
|-------|---------------|
| Feature Importance | Which stats matter most for the prediction. Taller bars mean more influence. |
| Actual vs. Predicted | A line chart comparing the model's predictions to real game results on the test set. |
| Prediction Errors | A bar chart of how far off each prediction was. Bars above zero mean the model predicted too low. Bars below zero mean too high. |
| Team Performance | A bar chart of the player's average stat against teams that strongly influence the model. |
| SHAP Summary | Shows how each feature pushes the prediction up or down for every game in the dataset. |

**Code location:** `src/plots.py`

## Configuration Options

These settings in `src/constants.py` control the prediction engine:

| Option | Default | What It Controls |
|--------|---------|-----------------|
| `NUM_SEASONS` | 2 | Number of past seasons to include in training data |
| `LAST_N_GAMES` | 10 | Recent games used to build the prediction input |
| `ROLLING_WINDOW_SIZE` | 5 | Games included in the rolling average calculation |
| `TEST_SIZE` | 0.3 | Fraction of data reserved for testing (30%) |
| `RANDOM_STATE` | 42 | Fixed random seed so results are the same every run |
| `CROSS_VAL_FOLDS` | 5 | Number of cross-validation rounds |

## Common Issues

### "No game data available for recent seasons"
**What you see:** An error message instead of predictions.
**Why it happens:** The player has no recorded games in the last 2 seasons. This can happen for retired players or players who missed entire seasons.
**How to fix it:** Try a different player who has played recently.

### "Player not found"
**What you see:** An error message after entering a name.
**Why it happens:** The name did not match any player in the NBA API database. This can happen with misspellings or nicknames.
**How to fix it:** Use the player's full official name (for example, "LeBron James" not "LBJ").

### SHAP plot shows a warning
**What you see:** "SHAP plot failed" warning instead of the SHAP chart.
**Why it happens:** The SHAP library sometimes cannot explain certain model configurations.
**How to fix it:** This is a known limitation. The other four charts still work normally.

### Predictions seem far off
**What you see:** The predicted value is very different from what the player actually scored.
**Why it happens:** Machine learning models are estimates, not guarantees. Player performance varies game to game due to injuries, rest, opponent strength, and other factors the model does not capture.
**How to fix it:** Check the MAE score. An MAE of 5 means predictions are off by about 5 points on average. This is typical for NBA scoring prediction.
