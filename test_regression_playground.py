"""Unit tests for regression playground helper functions.

Tests the core data transformation and statistical functions used in the
interactive regression playground.
"""

import numpy as np
import pytest


# ============================================================================
# Helper Functions (extracted from regression_playground.py for testing)
# ============================================================================

def apply_transform(x, transform_type):
    """Apply transformation to predictor values."""
    if transform_type == "None":
        return x
    elif transform_type == "Log":
        min_val = np.min(x)
        offset = abs(min_val) + 1 if min_val <= 0 else 0
        return np.log(x + offset)
    elif transform_type == "Square Root":
        min_val = np.min(x)
        offset = abs(min_val) if min_val < 0 else 0
        return np.sqrt(x + offset)
    elif transform_type == "Squared":
        return x ** 2
    elif transform_type == "Standardize":
        return (x - np.mean(x)) / np.std(x)
    return x


def apply_binning(x, bin_option):
    """Apply binning to discretize continuous predictor into categories.

    Args:
        x: numpy array of continuous values
        bin_option: string like "None", "2 bins (median)", "3 bins (tertiles)", etc.

    Returns:
        numpy array of integer bin labels (0, 1, 2, ...) or original x if "None"
    """
    if bin_option == "None":
        return x

    # Parse number of bins from option string
    n_bins = int(bin_option.split()[0])

    # Calculate quantile boundaries
    quantiles = np.linspace(0, 100, n_bins + 1)
    boundaries = np.percentile(x, quantiles)

    # Assign bin labels (0 to n_bins-1)
    bin_labels = np.digitize(x, boundaries[1:-1], right=False)

    return bin_labels.astype(float)


# ============================================================================
# Tests for apply_transform
# ============================================================================

class TestApplyTransform:
    """Tests for the apply_transform function."""

    def test_none_transform_returns_original(self):
        """None transform should return the original array unchanged."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = apply_transform(x, "None")
        np.testing.assert_array_equal(result, x)

    def test_log_transform_positive_values(self):
        """Log transform on positive values should apply natural log."""
        x = np.array([1.0, np.e, np.e**2])
        result = apply_transform(x, "Log")
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_log_transform_with_negative_values(self):
        """Log transform with negative values should add offset."""
        x = np.array([-1.0, 0.0, 1.0])
        result = apply_transform(x, "Log")
        # Offset should be abs(-1) + 1 = 2, so we're taking log of [1, 2, 3]
        expected = np.log(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_log_transform_with_zero(self):
        """Log transform with zero should add offset to avoid log(0)."""
        x = np.array([0.0, 1.0, 2.0])
        result = apply_transform(x, "Log")
        # Offset should be abs(0) + 1 = 1, so we're taking log of [1, 2, 3]
        expected = np.log(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_sqrt_transform_positive_values(self):
        """Square root transform on positive values."""
        x = np.array([0.0, 1.0, 4.0, 9.0])
        result = apply_transform(x, "Square Root")
        expected = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sqrt_transform_with_negative_values(self):
        """Square root transform with negative values should add offset."""
        x = np.array([-4.0, 0.0, 5.0])
        result = apply_transform(x, "Square Root")
        # Offset should be abs(-4) = 4, so we're taking sqrt of [0, 4, 9]
        expected = np.array([0.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_squared_transform(self):
        """Squared transform should square all values."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = apply_transform(x, "Squared")
        expected = np.array([4.0, 1.0, 0.0, 1.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_standardize_transform(self):
        """Standardize transform should produce mean=0, std=1."""
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = apply_transform(x, "Standardize")

        # Check mean is approximately 0
        assert abs(np.mean(result)) < 1e-10

        # Check std is approximately 1
        assert abs(np.std(result) - 1.0) < 1e-10

    def test_standardize_preserves_relative_order(self):
        """Standardize should preserve the relative order of values."""
        x = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = apply_transform(x, "Standardize")

        # Check that argmax and argmin are preserved
        assert np.argmax(result) == np.argmax(x)
        assert np.argmin(result) == np.argmin(x)

    def test_unknown_transform_returns_original(self):
        """Unknown transform type should return original array."""
        x = np.array([1.0, 2.0, 3.0])
        result = apply_transform(x, "UnknownTransform")
        np.testing.assert_array_equal(result, x)


# ============================================================================
# Tests for apply_binning
# ============================================================================

class TestApplyBinning:
    """Tests for the apply_binning function."""

    def test_none_binning_returns_original(self):
        """None binning should return the original array unchanged."""
        x = np.array([1.5, 2.3, 3.7, 4.1, 5.9])
        result = apply_binning(x, "None")
        np.testing.assert_array_equal(result, x)

    def test_two_bins_median_split(self):
        """2 bins should split at median into 0s and 1s."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = apply_binning(x, "2 bins (median)")

        # Should have only 0s and 1s
        unique_values = np.unique(result)
        assert len(unique_values) == 2
        assert 0 in unique_values
        assert 1 in unique_values

        # Lower half should be 0, upper half should be 1
        assert result[0] == 0  # 1.0 is in lower half
        assert result[-1] == 1  # 10.0 is in upper half

    def test_three_bins_tertiles(self):
        """3 bins should split into tertiles (0, 1, 2)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        result = apply_binning(x, "3 bins (tertiles)")

        # Should have 0, 1, and 2
        unique_values = np.unique(result)
        assert len(unique_values) == 3
        assert set(unique_values) == {0, 1, 2}

    def test_four_bins_quartiles(self):
        """4 bins should split into quartiles (0, 1, 2, 3)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        result = apply_binning(x, "4 bins (quartiles)")

        # Should have 0, 1, 2, and 3
        unique_values = np.unique(result)
        assert len(unique_values) == 4
        assert set(unique_values) == {0, 1, 2, 3}

    def test_five_bins_quintiles(self):
        """5 bins should split into quintiles (0, 1, 2, 3, 4)."""
        x = np.linspace(0, 100, 100)
        result = apply_binning(x, "5 bins (quintiles)")

        # Should have 0, 1, 2, 3, and 4
        unique_values = np.unique(result)
        assert len(unique_values) == 5
        assert set(unique_values) == {0, 1, 2, 3, 4}

    def test_binning_returns_float_type(self):
        """Binning should return float type for compatibility with regression."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = apply_binning(x, "2 bins (median)")
        assert result.dtype == np.float64

    def test_binning_preserves_order(self):
        """Higher values should get higher bin numbers."""
        x = np.array([1.0, 100.0])
        result = apply_binning(x, "2 bins (median)")
        assert result[0] < result[1]

    def test_binning_with_ties(self):
        """Binning should handle tied values correctly."""
        x = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        result = apply_binning(x, "2 bins (median)")

        # All 1.0s should be in same bin, all 5.0s should be in same bin
        assert result[0] == result[1] == result[2]
        assert result[3] == result[4] == result[5]

    def test_binning_approximately_equal_sizes(self):
        """Bins should have approximately equal sizes."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        result = apply_binning(x, "4 bins (quartiles)")

        # Each bin should have roughly 250 observations (25%)
        for bin_val in [0, 1, 2, 3]:
            count = np.sum(result == bin_val)
            # Allow 10% tolerance
            assert 200 < count < 300, f"Bin {bin_val} has {count} observations"


# ============================================================================
# Tests for Statistical Properties
# ============================================================================

class TestStatisticalProperties:
    """Tests for statistical properties of the regression setup."""

    def test_ols_unbiased_estimator(self):
        """OLS should be an unbiased estimator (mean bias ≈ 0 over many samples)."""
        import statsmodels.api as sm

        np.random.seed(42)
        true_intercept = 2.0
        true_slope = 1.5
        n_simulations = 100
        n_samples = 100

        estimated_intercepts = []
        estimated_slopes = []

        for _ in range(n_simulations):
            x = np.random.normal(0, 1, n_samples)
            noise = np.random.normal(0, 1, n_samples)
            y = true_intercept + true_slope * x + noise

            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            estimated_intercepts.append(model.params[0])
            estimated_slopes.append(model.params[1])

        # Mean of estimates should be close to true values
        mean_intercept = np.mean(estimated_intercepts)
        mean_slope = np.mean(estimated_slopes)

        assert abs(mean_intercept - true_intercept) < 0.2, \
            f"Intercept bias too large: {mean_intercept - true_intercept}"
        assert abs(mean_slope - true_slope) < 0.2, \
            f"Slope bias too large: {mean_slope - true_slope}"

    def test_confidence_interval_coverage(self):
        """95% CI should contain true value ~95% of the time."""
        import statsmodels.api as sm

        np.random.seed(42)
        true_slope = 1.5
        n_simulations = 200
        n_samples = 50

        coverage_count = 0

        for _ in range(n_simulations):
            x = np.random.normal(0, 1, n_samples)
            noise = np.random.normal(0, 1, n_samples)
            y = 2.0 + true_slope * x + noise

            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            ci = model.conf_int(alpha=0.05)
            ci_lower, ci_upper = ci[1, 0], ci[1, 1]

            if ci_lower <= true_slope <= ci_upper:
                coverage_count += 1

        coverage_rate = coverage_count / n_simulations

        # Should be close to 95% (allow some margin for simulation variance)
        assert 0.85 < coverage_rate < 1.0, \
            f"Coverage rate {coverage_rate:.2%} is outside expected range"

    def test_residuals_approximately_normal(self):
        """Residuals should be approximately normal when errors are normal."""
        import statsmodels.api as sm
        from scipy import stats

        np.random.seed(42)
        n = 200

        x = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 1, n)  # Normal errors
        y = 2.0 + 1.5 * x + noise

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        # Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(model.resid)

        # With normal errors, residuals should not strongly reject normality
        assert p_value > 0.01, \
            f"Residuals significantly non-normal (p={p_value:.4f})"

    def test_r_squared_bounds(self):
        """R² should always be between 0 and 1 for OLS regression."""
        import statsmodels.api as sm

        np.random.seed(42)
        n = 100

        x = np.random.normal(0, 1, n)
        noise = np.random.normal(0, 1, n)
        y = 2.0 + 1.5 * x + noise

        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()

        assert 0 <= model.rsquared <= 1, \
            f"R² = {model.rsquared} is out of bounds [0, 1]"

    def test_higher_noise_lower_r_squared(self):
        """Higher noise should result in lower R²."""
        import statsmodels.api as sm

        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)

        # Low noise
        y_low_noise = 2.0 + 1.5 * x + np.random.normal(0, 0.5, n)
        X = sm.add_constant(x)
        model_low = sm.OLS(y_low_noise, X).fit()

        # High noise
        y_high_noise = 2.0 + 1.5 * x + np.random.normal(0, 5.0, n)
        model_high = sm.OLS(y_high_noise, X).fit()

        assert model_low.rsquared > model_high.rsquared, \
            "Higher noise should result in lower R²"


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_transform_single_value(self):
        """Transforms should handle single-value arrays."""
        x = np.array([5.0])

        # These should not raise errors
        apply_transform(x, "None")
        apply_transform(x, "Log")
        apply_transform(x, "Square Root")
        apply_transform(x, "Squared")
        # Standardize will have std=0, but should still work
        result = apply_transform(x, "Standardize")
        assert np.isnan(result[0]) or result[0] == 0  # nan or 0 is acceptable

    def test_binning_single_value(self):
        """Binning should handle single-value arrays."""
        x = np.array([5.0])
        result = apply_binning(x, "2 bins (median)")
        assert len(result) == 1

    def test_transform_large_array(self):
        """Transforms should handle large arrays efficiently."""
        x = np.random.normal(0, 1, 100000)

        for transform in ["None", "Log", "Square Root", "Squared", "Standardize"]:
            result = apply_transform(x, transform)
            assert len(result) == len(x)

    def test_binning_large_array(self):
        """Binning should handle large arrays efficiently."""
        x = np.random.normal(0, 1, 100000)

        for bin_option in ["2 bins (median)", "3 bins (tertiles)", "4 bins (quartiles)", "5 bins (quintiles)"]:
            result = apply_binning(x, bin_option)
            assert len(result) == len(x)

    def test_transform_all_same_values(self):
        """Transforms should handle arrays where all values are the same."""
        x = np.array([3.0, 3.0, 3.0, 3.0])

        result_none = apply_transform(x, "None")
        np.testing.assert_array_equal(result_none, x)

        result_squared = apply_transform(x, "Squared")
        np.testing.assert_array_equal(result_squared, np.array([9.0, 9.0, 9.0, 9.0]))

    def test_binning_all_same_values(self):
        """Binning should handle arrays where all values are the same."""
        x = np.array([3.0, 3.0, 3.0, 3.0])
        result = apply_binning(x, "2 bins (median)")
        # All values should be in the same bin
        assert len(np.unique(result)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
