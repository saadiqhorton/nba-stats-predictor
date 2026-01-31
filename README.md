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
├── app.py                # Entry point - starts the Streamlit app
├── Dockerfile            # Container image for the Streamlit app
├── docker-compose.yml    # Multi-container setup (Nginx + 3 app instances)
├── nginx/                # Load balancer configuration
│   ├── Dockerfile        # Nginx container image
│   └── nginx.conf        # Routing, rate limiting, and security settings
├── scripts/              # Operations scripts
│   ├── scale.sh          # Scale backend instances (1-8)
│   └── reload_nginx.sh   # Reload Nginx without downtime
├── src/                  # Source package
│   ├── constants.py      # Configuration values (seasons, model settings)
│   ├── api.py            # Fetches player info and game logs from the NBA API
│   ├── data_processing.py # Cleans data, creates features, trains the model
│   ├── plots.py          # Creates all charts and visualizations
│   ├── ui_components.py  # Builds the Streamlit user interface
│   └── config/
│       └── team_mappings.py # Team abbreviations and logo URLs
├── tests/                # Unit and integration tests (93+ tests)
│   ├── conftest.py       # Shared test setup
│   ├── test_api_functions.py
│   ├── test_api_integration.py
│   ├── test_data_processing.py
│   ├── test_display_functions.py
│   ├── test_model_training.py
│   ├── test_plots.py
│   ├── test_ui_components.py
│   ├── test_utils.py
│   ├── test_webapp.py    # Playwright end-to-end tests (9 tests)
│   └── test_load_balancer.sh # Load balancer integration tests (16 tests)
├── docs/                 # Documentation
│   └── features/
│       ├── prediction-engine.md
│       └── load-balancer.md
├── requirements.txt      # Runtime dependencies
├── requirements-dev.txt  # Test dependencies
└── pytest.ini            # Test configuration
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

## Docker Deployment (Load Balanced)

The app can run behind a load balancer with multiple backend instances. This increases the number of users the app can handle at the same time and keeps the app running even if one instance crashes.

### Requirements

- Docker and Docker Compose v2
- At least 2 GB of free RAM (for 3 app instances + the load balancer)

### Quick Start

1. Build and start all containers:
   ```bash
   docker compose up -d --build
   ```

2. Open the app at `http://localhost:8088`.

3. Check that all containers are healthy:
   ```bash
   docker compose ps
   ```

### How It Works

The Docker setup runs 4 containers:

| Container | Role |
|-----------|------|
| `nba-nginx` | Receives all incoming requests and forwards them to one of the app instances |
| `nba-app-1` | Streamlit app instance 1 |
| `nba-app-2` | Streamlit app instance 2 |
| `nba-app-3` | Streamlit app instance 3 |

Nginx uses IP-based routing. All requests from the same user go to the same app instance. This is needed because the app stores session data in memory.

### Scaling Up or Down

Use the scaling script to change the number of app instances (1 to 8):

```bash
./scripts/scale.sh 5       # Scale to 5 instances
docker compose up -d --build
```

To reload the Nginx configuration without restarting:

```bash
./scripts/reload_nginx.sh
```

### Security

The load balancer adds these protections:

- **Rate limiting** - Each user is limited to 10 requests per second with a burst allowance of 20.
- **Security headers** - X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Content-Security-Policy, and Referrer-Policy are set on all responses.
- **Network isolation** - Only the Nginx port (8088) is exposed. The app instances are not directly reachable from outside Docker.
- **Version hiding** - The Nginx version number is not shown in response headers.

### Health Checks

Each container has a health check that runs every 30 seconds:

- **Nginx**: Checks `http://localhost/nginx-health`
- **App instances**: Checks `http://localhost:8501/_stcore/health`

Nginx waits for all app instances to be healthy before it starts accepting traffic.

### Stopping

```bash
docker compose down
```

---

## Future Enhancements

- **More stats** - Add prediction models for steals and blocks.
- **Team analytics** - Predict team-level outcomes and matchup results.
- **Advanced metrics** - Use usage rate, true shooting percentage, and other advanced stats.
- **Auto-scaling** - Automatically add or remove app instances based on traffic.
- **Redis session store** - Share session data between instances for true failover.

---

## Contact

- **Live App:** [www.nbastatmaster.site](https://www.nbastatmaster.site)
- **LinkedIn:** [Saadiq Horton](https://www.linkedin.com/in/saadiq-horton-367a94260/)
- **GitHub:** [saadiqhorton](https://github.com/saadiqhorton)
