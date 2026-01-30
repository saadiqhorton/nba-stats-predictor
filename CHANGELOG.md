# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.1.0] - 2026-01-29

### Added
- Modular code structure: split monolithic `app.py` (630 lines) into 6 focused modules
- `constants.py` - Centralized configuration values
- `api.py` - NBA API interaction functions
- `data_processing.py` - Feature engineering and model training
- `plots.py` - All chart and visualization functions
- `ui_components.py` - Streamlit UI rendering components
- 23 new unit tests for plots, UI components, and API integration
- Playwright end-to-end webapp tests (9 tests)
- `conftest.py` with shared test fixtures and Streamlit cache clearing
- `pytest.ini` with test configuration and coverage settings

### Changed
- `app.py` reduced from 630 lines to 74 lines (entry point only)
- All modules now under 200 lines each
- Player name HTML rendering now uses `html.escape()` for security

### Fixed
- XSS vulnerability in player name display (unescaped HTML)
- Streamlit cache interference between unit tests
- SHAP error handling now catches `TypeError` for invalid models
- Removed unused imports across all files

### Security
- Added `html.escape()` on all player names before HTML embedding
- No hardcoded secrets or credentials in source code

## [1.0.0] - 2026-01-27

### Added
- Player search by name using the NBA API
- Game log fetching for the last 2 seasons
- XGBoost prediction models for points, rebounds, and assists
- Feature engineering: shooting percentages, rolling averages, opponent encoding, home/away
- Model evaluation with MAE and cross-validation
- Feature importance visualization
- Actual vs. predicted performance chart
- Prediction error residuals chart
- Performance against high-impact teams chart
- SHAP feature importance summary
- Last 10 games display table
- Tabbed interface for points, rebounds, and assists predictions
- Team logo display for opponent breakdown
- Streamlit caching for API calls and model training
