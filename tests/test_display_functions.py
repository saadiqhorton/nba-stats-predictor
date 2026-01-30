"""Tests for display and helper functions extracted from main()."""

import pytest
import pandas as pd

from src.data_processing import (
    _format_date_with_suffix,
    _prepare_prediction_input,
    safe_divide,
)
from src.plots import _add_importance_label


class TestFormatDateWithSuffix:
    """Tests for _format_date_with_suffix function."""

    @pytest.mark.unit
    def test_first_day(self):
        """Test 1st suffix."""
        date = pd.Timestamp(year=2024, month=1, day=1)
        result = _format_date_with_suffix(date)
        assert "1st" in result
        assert "January" in result
        assert "2024" in result

    @pytest.mark.unit
    def test_second_day(self):
        """Test 2nd suffix."""
        date = pd.Timestamp(year=2024, month=2, day=2)
        result = _format_date_with_suffix(date)
        assert "2nd" in result

    @pytest.mark.unit
    def test_third_day(self):
        """Test 3rd suffix."""
        date = pd.Timestamp(year=2024, month=3, day=3)
        result = _format_date_with_suffix(date)
        assert "3rd" in result

    @pytest.mark.unit
    def test_fourth_day(self):
        """Test 4th suffix (th)."""
        date = pd.Timestamp(year=2024, month=4, day=4)
        result = _format_date_with_suffix(date)
        assert "4th" in result

    @pytest.mark.unit
    def test_eleventh_day(self):
        """Test 11th suffix (exception to normal rules)."""
        date = pd.Timestamp(year=2024, month=5, day=11)
        result = _format_date_with_suffix(date)
        assert "11th" in result

    @pytest.mark.unit
    def test_twelfth_day(self):
        """Test 12th suffix (exception to normal rules)."""
        date = pd.Timestamp(year=2024, month=6, day=12)
        result = _format_date_with_suffix(date)
        assert "12th" in result

    @pytest.mark.unit
    def test_thirteenth_day(self):
        """Test 13th suffix (exception to normal rules)."""
        date = pd.Timestamp(year=2024, month=7, day=13)
        result = _format_date_with_suffix(date)
        assert "13th" in result

    @pytest.mark.unit
    def test_twenty_first_day(self):
        """Test 21st suffix."""
        date = pd.Timestamp(year=2024, month=8, day=21)
        result = _format_date_with_suffix(date)
        assert "21st" in result

    @pytest.mark.unit
    def test_twenty_second_day(self):
        """Test 22nd suffix."""
        date = pd.Timestamp(year=2024, month=9, day=22)
        result = _format_date_with_suffix(date)
        assert "22nd" in result

    @pytest.mark.unit
    def test_thirty_first_day(self):
        """Test 31st suffix."""
        date = pd.Timestamp(year=2024, month=10, day=31)
        result = _format_date_with_suffix(date)
        assert "31st" in result


class TestSafeDivide:
    """Tests for safe_divide function."""

    @pytest.mark.unit
    def test_normal_division(self):
        """Test normal division case."""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 4.0, 5.0])
        result = safe_divide(numerator, denominator)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.unit
    def test_division_by_zero(self):
        """Test that division by zero returns 0."""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([2.0, 0.0, 5.0])
        result = safe_divide(numerator, denominator)
        assert result.iloc[0] == 5.0
        assert result.iloc[1] == 0.0  # Zero division returns 0
        assert result.iloc[2] == 6.0

    @pytest.mark.unit
    def test_all_zeros_denominator(self):
        """Test all zeros in denominator."""
        numerator = pd.Series([10.0, 20.0, 30.0])
        denominator = pd.Series([0.0, 0.0, 0.0])
        result = safe_divide(numerator, denominator)
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.unit
    def test_integer_input(self):
        """Test that integer inputs are converted to float."""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 4, 5])
        result = safe_divide(numerator, denominator)
        assert result.dtype == float

    @pytest.mark.unit
    def test_empty_series(self):
        """Test with empty series."""
        numerator = pd.Series([], dtype=float)
        denominator = pd.Series([], dtype=float)
        result = safe_divide(numerator, denominator)
        assert len(result) == 0


class TestPreparePreparationInput:
    """Tests for _prepare_prediction_input function."""

    @pytest.fixture
    def sample_last_10(self):
        """Create sample last 10 games dataframe."""
        return pd.DataFrame(
            {
                "FGM": [5.0, 6.0, 7.0],
                "FGA": [10.0, 12.0, 14.0],
                "FG3M": [2.0, 3.0, 2.0],
                "FG3A": [5.0, 6.0, 4.0],
                "FTM": [3.0, 4.0, 2.0],
                "FTA": [4.0, 5.0, 3.0],
                "MATCHUP": ["LAL vs BOS", "LAL @ GSW", "LAL vs MIA"],
            }
        )

    @pytest.fixture
    def sample_X_target(self):
        """Create sample X_target dataframe with expected columns."""
        return pd.DataFrame(
            {
                "FG%": [0.5],
                "FG3%": [0.4],
                "FT%": [0.75],
                "FGA_5game_avg": [12.0],
                "HOME/AWAY": [1],
                "OPP_BOS": [0],
                "OPP_GSW": [0],
                "OPP_MIA": [0],
            }
        )

    @pytest.mark.unit
    def test_returns_dataframe(self, sample_last_10, sample_X_target):
        """Test that function returns a DataFrame."""
        result = _prepare_prediction_input(sample_last_10, sample_X_target)
        assert isinstance(result, pd.DataFrame)

    @pytest.mark.unit
    def test_matches_target_columns(self, sample_last_10, sample_X_target):
        """Test that output columns match X_target columns."""
        result = _prepare_prediction_input(sample_last_10, sample_X_target)
        assert list(result.columns) == list(sample_X_target.columns)

    @pytest.mark.unit
    def test_single_row_output(self, sample_last_10, sample_X_target):
        """Test that output has single row (averaged stats)."""
        result = _prepare_prediction_input(sample_last_10, sample_X_target)
        assert len(result) == 1

    @pytest.mark.unit
    def test_missing_columns_filled_with_zero(self, sample_last_10):
        """Test that missing columns in prediction data are filled with 0."""
        # X_target with a column not in last_10
        X_target = pd.DataFrame(
            {
                "FG%": [0.5],
                "MISSING_COL": [1.0],
            }
        )
        result = _prepare_prediction_input(sample_last_10, X_target)
        assert "MISSING_COL" in result.columns
        assert result["MISSING_COL"].iloc[0] == 0


class TestAddImportanceLabel:
    """Tests for _add_importance_label function."""

    @pytest.fixture
    def mock_ax(self):
        """Create a mock axes object."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        return ax

    @pytest.fixture
    def mock_bar(self, mock_ax):
        """Create a mock bar object."""
        bars = mock_ax.barh([0], [0.15])
        return bars[0]

    @pytest.mark.unit
    def test_high_importance_label_placement(self, mock_ax, mock_bar):
        """Test label placement for high importance values."""
        # Should place label inside bar (left of value)
        _add_importance_label(mock_ax, 0.15, mock_bar)
        # Check that text was added to axes
        assert len(mock_ax.texts) == 1

    @pytest.mark.unit
    def test_low_importance_label_placement(self, mock_ax, mock_bar):
        """Test label placement for low importance values."""
        _add_importance_label(mock_ax, 0.05, mock_bar)
        # Check that text was added to axes
        assert len(mock_ax.texts) == 1

    @pytest.mark.unit
    def test_custom_threshold(self, mock_ax, mock_bar):
        """Test with custom threshold value."""
        # With threshold=0.2, value 0.15 should be treated as low
        _add_importance_label(mock_ax, 0.15, mock_bar, threshold=0.2)
        assert len(mock_ax.texts) == 1
