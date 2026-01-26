# 🏀 NBA Player Stats Predictor

### Predict. Analyze. Dominate.

---

## 🌐 Live Demo

Try out the live web app here: **[www.nbastatmaster.site](https://www.nbastatmaster.site)**  

---

## Project Overview

The **NBA Player Stats Predictor** is an interactive web application designed to analyze and predict NBA player performance. Leveraging historical game log data, this tool offers insights into player trends, model accuracy, and the factors influencing outcomes. Built with Python and Streamlit, it provides a user-friendly interface for basketball enthusiasts, fantasy sports players, and data analytics curious minds.

---

## Features

* **Player Search:** Easily find any NBA player by name.
* **Historical Game Data:** Fetches and displays combined game logs for recent seasons.
* **Machine Learning Prediction:** Utilizes **XGBoost Regressor** models to predict player **points, rebounds, and assists** based on in-game statistics.
* **Model Evaluation:** Provides key metrics like Mean Absolute Error (MAE) and cross-validation scores to assess model performance.
* **Interactive Visualizations:**
    * **Feature Importance:** Understand which game statistics contribute most to predictions.
    * **Actual vs. Predicted:** Visualize the model's accuracy against actual game scores.
    * **Prediction Residuals:** Analyze the errors in the model's predictions.
    * **Opponent Performance:** Explore a player's average performance against different teams.
    * **SHAP Explanations:** Gain deeper insights into individual prediction contributions (if enabled and works reliably).
* **Recent Games Overview:** Displays a table of the player's last 10 games from the current season (points, rebounds, assists).
* **Next Game Prediction:** Offers a projected stat total for the player's next game (per stat tab) based on average recent performance.

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
    * **Steals & Blocks:** Predicting defensive contributions.
* **Team-Level Analytics:** Introduce features for analyzing and predicting team performance, including:
    * Team offensive/defensive ratings.
    * Win/loss prediction based on team matchups.
    * Visualizations of team trends over time.
* **Advanced Player Metrics:** Integrate more sophisticated analytical stats (e.g., Usage Rate, True Shooting Percentage) into the prediction models.
* **Improved User Interface:** Further refine the Streamlit UI/UX for even greater clarity and aesthetic appeal.

---

## Contact

Feel free to reach out if you have any questions or feedback!

* **Live App:** [www.nbastatmaster.site](https://www.nbastatmaster.site)  
* **LinkedIn:** [https://www.linkedin.com/in/saadiq-horton-367a94260/](https://www.linkedin.com/in/saadiq-horton-367a94260/)  
* **GitHub:** [https://github.com/saadiqhorton](https://github.com/saadiqhorton)  

---
