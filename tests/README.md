# Test Suite for NBA Stats Predictor

## Overview

This test suite provides comprehensive coverage for the NBA Stats Predictor application, ensuring code quality and reliability.

## Test Structure

```
tests/
├── __init__.py                  # Test package initialization
├── test_utils.py                # Tests for utility functions
├── test_data_processing.py      # Tests for data preprocessing
├── test_api_functions.py        # Tests for API interactions (mocked)
└── test_model_training.py       # Tests for ML model training
```

## Test Coverage

### 1. **test_utils.py** - Utility Functions
- `is_home_game()` - Home/away game detection
- `get_current_season()` - Season calculation logic
- `get_recent_seasons()` - Season list generation

**Total Tests**: 12

### 2. **test_data_processing.py** - Data Processing
- Feature engineering (FG%, FG3%, FT%)
- Rolling averages calculation
- One-hot encoding for opponents
- Home/away encoding
- Zero division handling
- Data shuffling with reproducibility

**Total Tests**: 12

### 3. **test_api_functions.py** - API Functions (Mocked)
- Player search (`find_player()`)
- Game log fetching (`fetch_player_game_logs()`)
- Error handling for API failures
- Empty response handling

**Total Tests**: 9

### 4. **test_model_training.py** - Model Training
- Model fitting and predictions
- Train/test split validation
- Feature importances
- Reproducibility checks
- Data leakage prevention
- Different dataset sizes

**Total Tests**: 12

**Total Test Cases**: 45+

## Running Tests

### Setup

1. **Create virtual environment** (if not already created):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=. --cov-report=html
```

### Run Specific Test Files

```bash
# Run only utility tests
pytest tests/test_utils.py

# Run only data processing tests
pytest tests/test_data_processing.py

# Run only API tests
pytest tests/test_api_functions.py

# Run only model tests
pytest tests/test_model_training.py
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow
```

### Run Specific Test

```bash
# Run a specific test function
pytest tests/test_utils.py::TestIsHomeGame::test_home_game_vs

# Run a specific test class
pytest tests/test_utils.py::TestIsHomeGame
```

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests (slower)
- `@pytest.mark.slow` - Slow-running tests

## Coverage Report

After running tests with coverage, view the HTML report:

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# Open report in browser (Linux/Mac)
open htmlcov/index.html

# Or on Windows
start htmlcov/index.html
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. Add to your `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Writing New Tests

### Test Structure

```python
import pytest

class TestYourFunction:
    """Tests for your_function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {"key": "value"}
    
    @pytest.mark.unit
    def test_basic_functionality(self, sample_data):
        """Test basic functionality."""
        result = your_function(sample_data)
        assert result == expected_value
    
    @pytest.mark.unit
    def test_edge_case(self):
        """Test edge case handling."""
        result = your_function(edge_case_input)
        assert result is not None
```

### Best Practices

1. **Use descriptive test names** - Test names should describe what they test
2. **One assertion per test** - Keep tests focused (when possible)
3. **Use fixtures** - Share common setup code
4. **Mock external dependencies** - Don't make real API calls
5. **Test edge cases** - Empty inputs, zero values, large datasets
6. **Check for errors** - Test error handling paths

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running tests from the project root:

```bash
cd /path/to/nba-stats-predictor
pytest
```

### Streamlit Cache Warnings

Tests may show Streamlit cache warnings. These are expected and can be ignored during testing.

### Mock Failures

If mocked tests fail, ensure you're patching the correct import path:

```python
# Correct: patch where it's used
@patch('app.players.find_players_by_full_name')

# Incorrect: patch where it's defined
@patch('nba_api.stats.static.players.find_players_by_full_name')
```

## Next Steps

1. **Increase coverage** - Aim for 80%+ code coverage
2. **Add integration tests** - Test full workflows
3. **Add performance tests** - Test caching effectiveness
4. **Add UI tests** - Test Streamlit components (using selenium)
5. **Set up CI/CD** - Automate testing on every commit

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-mock documentation](https://pytest-mock.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
