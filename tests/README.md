# Test Suite for NBA Stats Predictor

## Overview

Comprehensive test coverage for the NBA Stats Predictor application, including unit tests, integration tests, and end-to-end webapp tests.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures, sys.path setup, cache clearing
├── test_utils.py            # is_home_game, get_current_season, get_recent_seasons
├── test_data_processing.py  # preprocess_game_data, make_features_and_target
├── test_api_functions.py    # find_player, fetch_player_game_logs (mocked)
├── test_api_integration.py  # fetch_and_combine_game_logs, KeyError branch
├── test_model_training.py   # train_model, splits, reproducibility
├── test_display_functions.py# _format_date_with_suffix, safe_divide, _prepare_prediction_input
├── test_plots.py            # All plotting functions + SHAP error handling
├── test_ui_components.py    # UI rendering, XSS escaping, predictions
└── test_webapp.py           # Playwright end-to-end webapp tests
```

## Test Counts

| File | Tests | What It Covers |
|------|------:|----------------|
| `test_utils.py` | 11 | Home/away detection, season calculation |
| `test_data_processing.py` | 13 | Feature engineering, encoding, zero division |
| `test_api_functions.py` | 14 | Player search, game log fetching, error handling |
| `test_api_integration.py` | 5 | Multi-season fetching, KeyError branch |
| `test_model_training.py` | 10 | Model fitting, splits, reproducibility, leakage |
| `test_display_functions.py` | 12 | Date formatting, safe division, prediction input |
| `test_plots.py` | 9 | All 6 plot functions, SHAP failure handling |
| `test_ui_components.py` | 9 | Player header, game tables, predictions, XSS |
| `test_webapp.py` | 9 | Full app via Playwright (run separately) |
| **Total** | **92+9** | **93 unit + 9 e2e** |

## Running Tests

### Unit and Integration Tests

```bash
# Run all tests (excludes webapp tests)
pytest tests/ --ignore=tests/test_webapp.py

# Run with coverage
pytest tests/ --ignore=tests/test_webapp.py --cov=. --cov-report=term-missing

# Run specific file
pytest tests/test_data_processing.py

# Run by marker
pytest -m unit
```

### End-to-End Webapp Tests

```bash
pip install playwright
playwright install chromium
python scripts/with_server.py \
  --server "streamlit run app.py --server.headless true" \
  --port 8501 \
  -- python tests/test_webapp.py
```

## Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow-running tests

## Coverage

Current coverage: **83%** overall (all source modules at 99-100% except `app.py` entry point, which is covered by webapp e2e tests).
