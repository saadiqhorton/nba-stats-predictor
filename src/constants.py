"""Configuration constants for the NBA Stats Predictor."""

# Data fetching
NUM_SEASONS = 2  # Number of past seasons to fetch for training
LAST_N_GAMES = 10  # Number of recent games to display and use for prediction

# Feature engineering
ROLLING_WINDOW_SIZE = 5  # Window size for rolling averages

# Model training
TEST_SIZE = 0.3  # Proportion of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility
CROSS_VAL_FOLDS = 5  # Number of folds for cross-validation

# Visualization
TOP_N_FEATURES = 20  # Number of top features to display in importance plot
FEATURE_IMPORTANCE_THRESHOLD = 0.1  # Threshold for displaying feature importance labels

# UI
SPINNER_DELAY = 2  # Seconds to display loading spinner
PLAYER_IMAGE_WIDTH = 280  # Width of player headshot image in pixels
TEAM_LOGO_WIDTH = 80  # Width of team logo in pixels

# Target variables for prediction tabs
TARGETS = [
    ("PTS", "Points"),
    ("REB", "Rebounds"),
    ("AST", "Assists"),
]
