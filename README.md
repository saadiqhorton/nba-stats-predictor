# 🏀 NBA Player Points Predictor

### Predict. Analyze. Dominate.

---

## Project Overview

The **NBA Player Points Predictor** is an interactive web application designed to analyze and predict NBA player performance. Leveraging historical game log data, this tool offers insights into player scoring trends, model accuracy, and the factors influencing points scored. Built with Python and Streamlit, it provides a user-friendly interface for basketball enthusiasts, fantasy sports players, and data analytics curious minds.

---

## Features

* **Player Search:** Easily find any NBA player by name.
* **Historical Game Data:** Fetches and displays combined game logs for recent seasons.
* **Machine Learning Prediction:** Utilizes a powerful **XGBoost Regressor** model to predict player points based on various in-game statistics.
* **Model Evaluation:** Provides key metrics like Mean Absolute Error (MAE) and cross-validation scores to assess model performance.
* **Interactive Visualizations:**
    * **Feature Importance:** Understand which game statistics contribute most to point predictions.
    * **Actual vs. Predicted:** Visualize the model's accuracy against actual game scores.
    * **Prediction Residuals:** Analyze the errors in the model's predictions.
    * **Opponent Performance:** Explore a player's average points scored against different teams.
    * **SHAP Explanations:** Gain deeper insights into individual prediction contributions (if enabled and works reliably).
* **Recent Games Overview:** Displays a table of the player's last 10 games from the current season.
* **Next Game Prediction:** Offers a projected point total for the player's next game based on average recent performance.

---

## Technical Stack

* **Python:** Core programming language.
* **Streamlit:** For building the interactive web application interface.
* **`nba_api`:** To fetch real-time NBA player game log data.
* **`pandas` & `numpy`:** For data manipulation and numerical operations.
* **`scikit-learn`:** For machine learning utilities (train-test split, metrics).
* **`xgboost`:** The powerful gradient boosting model used for point prediction.
* **`matplotlib`:** For generating static data visualizations.
* **`shap`:** For model interpretability (SHAP values).

---

## Future Enhancements

Some plans in the near future to expand the capabilities of the program:

* **Expanded Prediction Metrics:** Incorporate prediction models for other key player statistics, such as:
    * **Rebounds:** Predicting total rebounds per game.
    * **Assists:** Predicting total assists per game.
    * **Steals & Blocks:** Predicting defensive contributions.
* **Team-Level Analytics:** Introduce features for analyzing and predicting team performance, including:
    * Team offensive/defensive ratings.
    * Win/loss prediction based on team matchups.
    * Visualizations of team trends over time.
* **Advanced Player Metrics:** Integrate more sophisticated analytical stats (e.g., Usage Rate, True Shooting Percentage) into the prediction models.
* **Improved User Interface:** Further refine the Streamlit UI/UX for even greater clarity and aesthetic appeal.

---

## How to Run This App Locally

Follow these steps to get a local copy of the NBA Player Points Predictor up and running on your machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/saadiqhorton/nba-stats-predictor.git](https://github.com/saadiqhorton/nba-stats-predictor.git)
    cd nba-stats-predictor
    ```
2.  **Create a virtual environment (recommended):**
    This isolates the project's dependencies from other Python projects.
    ```bash
    python -m venv venv
    # On Windows, activate with:
    .\venv\Scripts\activate
    # On macOS/Linux, activate with:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    Ensure you have `pip` up-to-date, then install all required libraries.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run nba_app.py
    ```
5.  The app will automatically open in your default web browser (usually at `http://localhost:8501`).

---

## Contact

Feel free to reach out if you have any questions or feedback!

* **LinkedIn:** [https://www.linkedin.com/in/saadiq-horton-367a94260/]
* **GitHub:** [https://github.com/saadiqhorton]

---
