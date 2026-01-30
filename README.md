# NBA Player Stats Predictor

A web application that predicts NBA player performance using machine learning.

**Live Demo:** [www.nbastatmaster.site](https://www.nbastatmaster.site)

---

## What This Project Does

Enter any NBA player's name and the app predicts their points, rebounds, and assists for the next game. It uses real game data from the last 2 NBA seasons and trains an XGBoost machine learning model for each stat.

The app also shows which features matter most for each prediction, how the player performs against specific teams, and how accurate the model is on past games.

---

## Features

- **Player Search** - Find any NBA player by name.
- **Last 10 Games** - View recent game stats (points, rebounds, assists).
- **Next Game Prediction** - Get a projected stat total based on recent performance.
- **Model Evaluation** - See Mean Absolute Error (MAE) and cross-validation scores.
- **Visualizations:**
  - Feature importance chart
  - Actual vs. predicted performance plot
  - Prediction error residuals
  - Performance against high-impact teams
  - SHAP feature importance summary

---

## Getting Started

### Requirements

- Python 3.10 or higher
- pip (Python package manager)
- An internet connection (the app fetches live NBA data)

### Installation

1. Clone the repository and navigate to the project folder:
   ```bash
   cd nba-stats-predictor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:
   ```bash
   streamlit run app.py
   ```

5. Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## How to Use

1. Type an NBA player's name in the text box (for example, "LeBron James").
2. Wait for the app to load game data and run predictions.
3. View the **Last 10 Games** table showing recent performance.
4. Switch between the **Points**, **Rebounds**, and **Assists** tabs to see predictions for each stat.

Each tab shows:
- **Next Game Prediction** - The predicted stat value based on the player's last 10 games.
- **Model Evaluation** - How accurate the model is, including error scores.
- **Performance Against Teams** - How the player performs against opponents that strongly influence predictions.
- **Visualizations** - Charts showing feature importance, prediction accuracy, errors, and SHAP values.

---

## Project Structure

```
nba-stats-predictor/
├── app.py               # Entry point - starts the Streamlit app
├── src/                 # Source package
│   ├── constants.py     # Configuration values (seasons, model settings)
│   ├── api.py           # Fetches player info and game logs from the NBA API
│   ├── data_processing.py # Cleans data, creates features, trains the model
│   ├── plots.py         # Creates all charts and visualizations
│   ├── ui_components.py # Builds the Streamlit user interface
│   └── config/
│       └── team_mappings.py # Team abbreviations and logo URLs
├── tests/               # Unit and integration tests (93 tests)
│   ├── conftest.py      # Shared test setup
│   ├── test_api_functions.py
│   ├── test_api_integration.py
│   ├── test_data_processing.py
│   ├── test_display_functions.py
│   ├── test_model_training.py
│   ├── test_plots.py
│   ├── test_ui_components.py
│   ├── test_utils.py
│   └── test_webapp.py   # Playwright end-to-end tests (9 tests)
├── docs/                # Documentation
│   └── features/
│       └── prediction-engine.md
├── requirements.txt     # Runtime dependencies
├── requirements-dev.txt # Test dependencies
└── pytest.ini           # Test configuration
```

---

## Configuration

All configuration values are in `src/constants.py`:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `NUM_SEASONS` | 2 | How many past seasons of game data to use |
| `LAST_N_GAMES` | 10 | How many recent games to display and average |
| `ROLLING_WINDOW_SIZE` | 5 | Window size for rolling average features |
| `TEST_SIZE` | 0.3 | Fraction of data held out for testing (30%) |
| `RANDOM_STATE` | 42 | Random seed for reproducible results |
| `CROSS_VAL_FOLDS` | 5 | Number of cross-validation rounds |
| `TOP_N_FEATURES` | 20 | Features shown in the importance chart |

---

## Running Tests

Install test dependencies:
```bash
pip install -r requirements-dev.txt
```

Run unit and integration tests:
```bash
pytest tests/ --ignore=tests/test_webapp.py
```

Run with coverage report:
```bash
pytest tests/ --ignore=tests/test_webapp.py --cov=. --cov-report=term-missing
```

Run the end-to-end webapp tests (requires Playwright):
```bash
pip install playwright
playwright install chromium
python scripts/with_server.py \
  --server "streamlit run app.py --server.headless true" \
  --port 8501 \
  -- python tests/test_webapp.py
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| [Streamlit](https://streamlit.io/) | Web interface |
| [nba_api](https://github.com/swar/nba_api) | NBA player and game data |
| [XGBoost](https://xgboost.readthedocs.io/) | Machine learning model |
| [scikit-learn](https://scikit-learn.org/) | Train/test splitting, evaluation metrics |
| [SHAP](https://shap.readthedocs.io/) | Explains which features drive predictions |
| [Matplotlib](https://matplotlib.org/) | Charts and plots |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |

---

## Future Enhancements

- **More stats** - Add prediction models for steals and blocks.
- **Team analytics** - Predict team-level outcomes and matchup results.
- **Advanced metrics** - Use usage rate, true shooting percentage, and other advanced stats.

---

## Contact

- **Live App:** [www.nbastatmaster.site](https://www.nbastatmaster.site)
- **LinkedIn:** [Saadiq Horton](https://www.linkedin.com/in/saadiq-horton-367a94260/)
- **GitHub:** [saadiqhorton](https://github.com/saadiqhorton)
