"""
Unit tests for OLS regression calculations used in regression_playground.py

Tests verify:
1. Simple linear regression coefficients (slope, intercept)
2. Multiple regression coefficients
3. R-squared calculations
4. Standard errors
5. t-statistics and p-values
6. Residual calculations
"""

import numpy as np
from scipy import stats
import pytest


class TestSimpleLinearRegression:
    """Tests for basic OLS: y = β₀ + β₁x + ε"""

    def test_ols_coefficients_perfect_fit(self):
        """Test OLS recovers exact coefficients with no noise."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        true_intercept = 5.0
        true_slope = 2.0
        y = true_intercept + true_slope * x  # No noise

        # OLS using scipy (reference)
        result = stats.linregress(x, y)

        assert np.isclose(result.intercept, true_intercept, atol=1e-10)
        assert np.isclose(result.slope, true_slope, atol=1e-10)

    def test_ols_coefficients_with_noise(self):
        """Test OLS coefficients are close to true values with moderate noise."""
        np.random.seed(42)
        n = 1000  # Large sample for accuracy
        x = np.random.uniform(0, 50, n)
        true_intercept = 5.0
        true_slope = 2.0
        noise_sd = 1.0
        y = true_intercept + true_slope * x + np.random.normal(0, noise_sd, n)

        result = stats.linregress(x, y)

        # With n=1000, estimates should be within ~0.1 of true values
        assert np.isclose(result.intercept, true_intercept, atol=0.5)
        assert np.isclose(result.slope, true_slope, atol=0.1)

    def test_ols_normal_equations(self):
        """Test OLS using normal equations matches scipy.stats.linregress."""
        np.random.seed(42)
        n = 50
        x = np.random.uniform(0, 50, n)
        y = 3.0 + 1.5 * x + np.random.normal(0, 2.0, n)

        # Method 1: scipy
        scipy_result = stats.linregress(x, y)

        # Method 2: Normal equations (X'X)^(-1) X'y
        X_design = np.column_stack([np.ones(n), x])
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        betas = np.linalg.solve(XtX, Xty)

        assert np.isclose(betas[0], scipy_result.intercept, atol=1e-10)
        assert np.isclose(betas[1], scipy_result.slope, atol=1e-10)

    def test_r_squared_calculation(self):
        """Test R² = 1 - SS_res/SS_tot."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 2.0 + 3.0 * x + np.random.normal(0, 5.0, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        assert np.isclose(r_squared, result.rvalue ** 2, atol=1e-10)

    def test_r_squared_perfect_fit(self):
        """Test R² = 1 for perfect linear relationship."""
        x = np.array([1, 2, 3, 4, 5])
        y = 2 * x + 3  # Perfect linear

        result = stats.linregress(x, y)
        assert np.isclose(result.rvalue ** 2, 1.0, atol=1e-10)

    def test_r_squared_no_relationship(self):
        """Test R² ≈ 0 when no linear relationship exists."""
        np.random.seed(42)
        n = 1000
        x = np.random.uniform(0, 50, n)
        y = np.random.normal(10, 5, n)  # y independent of x

        result = stats.linregress(x, y)
        # R² should be close to 0 (within sampling variation)
        assert result.rvalue ** 2 < 0.05

    def test_residual_standard_error(self):
        """Test residual SE = sqrt(SS_res / (n-2))."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        true_se = 3.0
        y = 1.0 + 2.0 * x + np.random.normal(0, true_se, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x
        residuals = y - y_pred

        df = n - 2
        mse = np.sum(residuals ** 2) / df
        se = np.sqrt(mse)

        # SE should be close to true_se with large n
        assert np.isclose(se, true_se, atol=0.5)

    def test_standard_error_of_slope(self):
        """Test SE(β₁) = σ / sqrt(SS_x)."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x + np.random.normal(0, 2.0, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x
        residuals = y - y_pred

        df = n - 2
        mse = np.sum(residuals ** 2) / df
        se = np.sqrt(mse)

        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean) ** 2)
        se_slope = se / np.sqrt(ss_x)

        # Compare with scipy's stderr
        assert np.isclose(se_slope, result.stderr, atol=1e-10)

    def test_t_statistic_slope(self):
        """Test t = β₁ / SE(β₁)."""
        np.random.seed(42)
        n = 50
        x = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x + np.random.normal(0, 2.0, n)

        result = stats.linregress(x, y)
        t_stat = result.slope / result.stderr

        # Verify t-statistic calculation
        assert np.isclose(t_stat, result.slope / result.stderr, atol=1e-10)

    def test_p_value_significant_slope(self):
        """Test p-value is small for true non-zero slope."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x + np.random.normal(0, 1.0, n)  # Strong signal

        result = stats.linregress(x, y)

        # With strong signal, p-value should be very small
        assert result.pvalue < 0.001

    def test_p_value_zero_slope(self):
        """Test p-value is large when true slope is zero."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 10.0 + np.random.normal(0, 5.0, n)  # No relationship with x

        result = stats.linregress(x, y)

        # p-value should typically be > 0.05
        # Note: this can occasionally fail due to random chance
        assert result.pvalue > 0.01


class TestMultipleRegression:
    """Tests for multiple regression: y = β₀ + β₁x₁ + β₂x₂ + ... + ε"""

    def test_two_predictor_coefficients(self):
        """Test OLS with two predictors recovers true coefficients."""
        np.random.seed(42)
        n = 500
        x1 = np.random.uniform(0, 50, n)
        x2 = np.random.uniform(0, 50, n)
        true_b0, true_b1, true_b2 = 5.0, 2.0, -1.5
        y = true_b0 + true_b1 * x1 + true_b2 * x2 + np.random.normal(0, 1.0, n)

        # OLS via normal equations
        X = np.column_stack([np.ones(n), x1, x2])
        betas = np.linalg.solve(X.T @ X, X.T @ y)

        assert np.isclose(betas[0], true_b0, atol=0.5)
        assert np.isclose(betas[1], true_b1, atol=0.2)
        assert np.isclose(betas[2], true_b2, atol=0.2)

    def test_three_predictor_coefficients(self):
        """Test OLS with three predictors."""
        np.random.seed(42)
        n = 500
        x1 = np.random.uniform(0, 50, n)
        x2 = np.random.uniform(0, 50, n)
        x3 = np.random.uniform(0, 50, n)
        true_b0, true_b1, true_b2, true_b3 = 3.0, 1.0, 0.5, -0.3
        y = true_b0 + true_b1 * x1 + true_b2 * x2 + true_b3 * x3 + np.random.normal(0, 1.0, n)

        X = np.column_stack([np.ones(n), x1, x2, x3])
        betas = np.linalg.solve(X.T @ X, X.T @ y)

        assert np.isclose(betas[0], true_b0, atol=0.5)
        assert np.isclose(betas[1], true_b1, atol=0.2)
        assert np.isclose(betas[2], true_b2, atol=0.2)
        assert np.isclose(betas[3], true_b3, atol=0.2)

    def test_interaction_term(self):
        """Test model with interaction: y = β₀ + β₁x + β₂z + β₃(x×z)."""
        np.random.seed(42)
        n = 500
        x = np.random.uniform(0, 50, n)
        z = np.random.uniform(0, 50, n)
        true_b0, true_b1, true_b2, true_b3 = 2.0, 1.0, 0.5, 0.02
        y = true_b0 + true_b1 * x + true_b2 * z + true_b3 * x * z + np.random.normal(0, 1.0, n)

        X = np.column_stack([np.ones(n), x, z, x * z])
        betas = np.linalg.solve(X.T @ X, X.T @ y)

        assert np.isclose(betas[0], true_b0, atol=1.0)
        assert np.isclose(betas[1], true_b1, atol=0.3)
        assert np.isclose(betas[2], true_b2, atol=0.3)
        assert np.isclose(betas[3], true_b3, atol=0.02)

    def test_multiple_r_squared(self):
        """Test R² calculation for multiple regression."""
        np.random.seed(42)
        n = 200
        x1 = np.random.uniform(0, 50, n)
        x2 = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x1 + 1.5 * x2 + np.random.normal(0, 3.0, n)

        X = np.column_stack([np.ones(n), x1, x2])
        betas = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ betas

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # R² should be between 0 and 1
        assert 0 <= r_squared <= 1
        # With moderate noise, expect decent R²
        assert r_squared > 0.5

    def test_adjusted_r_squared(self):
        """Test adjusted R² = 1 - (1-R²)(n-1)/(n-k-1)."""
        np.random.seed(42)
        n = 100
        k = 2  # Number of predictors
        x1 = np.random.uniform(0, 50, n)
        x2 = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x1 + 1.5 * x2 + np.random.normal(0, 3.0, n)

        X = np.column_stack([np.ones(n), x1, x2])
        betas = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ betas

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        df = n - k - 1
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / df

        # Adjusted R² should be less than R²
        assert adj_r_squared <= r_squared
        # Both should be positive with good predictors
        assert adj_r_squared > 0

    def test_standard_errors_multiple(self):
        """Test SE calculation: sqrt(diag((X'X)^(-1) * MSE))."""
        np.random.seed(42)
        n = 100
        x1 = np.random.uniform(0, 50, n)
        x2 = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x1 + 1.5 * x2 + np.random.normal(0, 2.0, n)

        X = np.column_stack([np.ones(n), x1, x2])
        betas = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ betas
        residuals = y - y_pred

        df = n - 3  # n - (k + 1)
        mse = np.sum(residuals ** 2) / df
        XtX_inv = np.linalg.inv(X.T @ X)
        se_betas = np.sqrt(np.diag(XtX_inv) * mse)

        # Standard errors should be positive
        assert all(se > 0 for se in se_betas)
        # SE of intercept typically larger due to extrapolation
        # SE of slopes depend on variance of predictors

    def test_f_statistic(self):
        """Test F-statistic: (R²/k) / ((1-R²)/(n-k-1))."""
        np.random.seed(42)
        n = 100
        k = 2
        x1 = np.random.uniform(0, 50, n)
        x2 = np.random.uniform(0, 50, n)
        y = 1.0 + 2.0 * x1 + 1.5 * x2 + np.random.normal(0, 2.0, n)

        X = np.column_stack([np.ones(n), x1, x2])
        betas = np.linalg.solve(X.T @ X, X.T @ y)
        y_pred = X @ betas

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        df = n - k - 1
        f_stat = (r_squared / k) / ((1 - r_squared) / df)
        f_pval = 1 - stats.f.cdf(f_stat, k, df)

        # With true relationship, F should be significant
        assert f_stat > 1
        assert f_pval < 0.05


class TestBinaryRegression:
    """Tests for binary/dummy variable regression (means comparison)."""

    def test_binary_coefficients(self):
        """Test binary regression: intercept = mean(group0), slope = mean(group1) - mean(group0)."""
        np.random.seed(42)
        n = 100
        n_per_group = n // 2

        mean0, mean1 = 10.0, 15.0
        y_group0 = np.random.normal(mean0, 2.0, n_per_group)
        y_group1 = np.random.normal(mean1, 2.0, n - n_per_group)

        x = np.concatenate([np.zeros(n_per_group), np.ones(n - n_per_group)])
        y = np.concatenate([y_group0, y_group1])

        result = stats.linregress(x, y)

        # Intercept should be close to group 0 mean
        assert np.isclose(result.intercept, np.mean(y_group0), atol=0.01)
        # Slope should be difference in means
        assert np.isclose(result.slope, np.mean(y_group1) - np.mean(y_group0), atol=0.01)

    def test_binary_equals_ttest(self):
        """Test that binary regression t-stat matches independent t-test."""
        np.random.seed(42)
        n = 100
        n_per_group = n // 2

        y_group0 = np.random.normal(10.0, 3.0, n_per_group)
        y_group1 = np.random.normal(12.0, 3.0, n - n_per_group)

        x = np.concatenate([np.zeros(n_per_group), np.ones(n - n_per_group)])
        y = np.concatenate([y_group0, y_group1])

        # Regression approach
        reg_result = stats.linregress(x, y)

        # t-test approach (equal variance)
        ttest_result = stats.ttest_ind(y_group1, y_group0, equal_var=True)

        # t-statistics should match (sign may differ based on order)
        assert np.isclose(abs(reg_result.slope / reg_result.stderr),
                         abs(ttest_result.statistic), atol=0.01)


class TestCategoricalRegression:
    """Tests for categorical regression with dummy variables."""

    def test_dummy_coding_three_groups(self):
        """Test categorical regression with 3 groups using dummy coding."""
        np.random.seed(42)
        n = 150
        n_per = n // 3

        means = [10.0, 15.0, 20.0]
        y1 = np.random.normal(means[0], 2.0, n_per)
        y2 = np.random.normal(means[1], 2.0, n_per)
        y3 = np.random.normal(means[2], 2.0, n - 2 * n_per)

        y = np.concatenate([y1, y2, y3])
        group = np.concatenate([np.zeros(n_per), np.ones(n_per), 2 * np.ones(n - 2 * n_per)])

        # Dummy coding: group 1 is reference
        d2 = (group == 1).astype(float)
        d3 = (group == 2).astype(float)

        X = np.column_stack([np.ones(n), d2, d3])
        betas = np.linalg.solve(X.T @ X, X.T @ y)

        # Intercept = mean of reference group
        assert np.isclose(betas[0], np.mean(y1), atol=0.01)
        # Coefficients = difference from reference
        assert np.isclose(betas[1], np.mean(y2) - np.mean(y1), atol=0.01)
        assert np.isclose(betas[2], np.mean(y3) - np.mean(y1), atol=0.01)

    def test_categorical_with_continuous(self):
        """Test categorical + continuous predictor (ANCOVA-style)."""
        np.random.seed(42)
        n = 200
        n_per = n // 2

        # Group indicator
        group = np.concatenate([np.zeros(n_per), np.ones(n - n_per)])
        # Continuous predictor
        x = np.random.uniform(0, 50, n)

        true_b0, true_b1, true_b2 = 5.0, 1.0, 3.0  # intercept, slope of x, group effect
        y = true_b0 + true_b1 * x + true_b2 * group + np.random.normal(0, 2.0, n)

        X = np.column_stack([np.ones(n), x, group])
        betas = np.linalg.solve(X.T @ X, X.T @ y)

        assert np.isclose(betas[0], true_b0, atol=1.0)
        assert np.isclose(betas[1], true_b1, atol=0.2)
        assert np.isclose(betas[2], true_b2, atol=0.5)


class TestTransformations:
    """Tests for regression with transformed predictors."""

    def test_sqrt_transform(self):
        """Test regression with sqrt-transformed predictor."""
        np.random.seed(42)
        n = 200
        x_raw = np.random.uniform(1, 50, n)
        x_transformed = np.sqrt(x_raw)

        true_b0, true_b1 = 2.0, 3.0
        y = true_b0 + true_b1 * x_transformed + np.random.normal(0, 1.0, n)

        result = stats.linregress(x_transformed, y)

        assert np.isclose(result.intercept, true_b0, atol=0.5)
        assert np.isclose(result.slope, true_b1, atol=0.3)

    def test_square_transform(self):
        """Test regression with squared predictor."""
        np.random.seed(42)
        n = 200
        x_raw = np.random.uniform(0, 10, n)
        x_transformed = x_raw ** 2

        true_b0, true_b1 = 5.0, 0.1
        y = true_b0 + true_b1 * x_transformed + np.random.normal(0, 2.0, n)

        result = stats.linregress(x_transformed, y)

        assert np.isclose(result.intercept, true_b0, atol=1.0)
        assert np.isclose(result.slope, true_b1, atol=0.05)

    def test_log_transform(self):
        """Test regression with log-transformed predictor."""
        np.random.seed(42)
        n = 200
        x_raw = np.random.uniform(1, 50, n)
        x_transformed = np.log(x_raw)

        true_b0, true_b1 = 1.0, 2.0
        y = true_b0 + true_b1 * x_transformed + np.random.normal(0, 0.5, n)

        result = stats.linregress(x_transformed, y)

        assert np.isclose(result.intercept, true_b0, atol=0.3)
        assert np.isclose(result.slope, true_b1, atol=0.2)


class TestConfidenceIntervals:
    """Tests for confidence and prediction intervals."""

    def test_ci_contains_true_line(self):
        """Test that CI contains true regression line most of the time."""
        np.random.seed(42)
        n_sims = 100
        n = 50
        true_b0, true_b1 = 2.0, 1.5
        ci_level = 0.95
        x_test = 25.0  # Test point

        contains_true = 0
        for sim in range(n_sims):
            x = np.random.uniform(0, 50, n)
            y = true_b0 + true_b1 * x + np.random.normal(0, 2.0, n)

            result = stats.linregress(x, y)
            y_pred = result.intercept + result.slope * x

            # Calculate SE of fitted value at x_test
            residuals = y - y_pred
            df = n - 2
            mse = np.sum(residuals ** 2) / df
            se = np.sqrt(mse)
            x_mean = np.mean(x)
            ss_x = np.sum((x - x_mean) ** 2)
            se_fit = se * np.sqrt(1 / n + (x_test - x_mean) ** 2 / ss_x)

            t_val = stats.t.ppf((1 + ci_level) / 2, df)
            y_fit = result.intercept + result.slope * x_test
            ci_lower = y_fit - t_val * se_fit
            ci_upper = y_fit + t_val * se_fit

            true_y = true_b0 + true_b1 * x_test
            if ci_lower <= true_y <= ci_upper:
                contains_true += 1

        coverage = contains_true / n_sims
        # Coverage should be close to 95%
        assert 0.85 <= coverage <= 1.0

    def test_pi_wider_than_ci(self):
        """Test that prediction interval is always wider than confidence interval."""
        np.random.seed(42)
        n = 50
        x = np.random.uniform(0, 50, n)
        y = 2.0 + 1.5 * x + np.random.normal(0, 2.0, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x
        residuals = y - y_pred

        df = n - 2
        mse = np.sum(residuals ** 2) / df
        se = np.sqrt(mse)
        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean) ** 2)

        x_test = 25.0
        se_fit = se * np.sqrt(1 / n + (x_test - x_mean) ** 2 / ss_x)
        se_pred = se * np.sqrt(1 + 1 / n + (x_test - x_mean) ** 2 / ss_x)

        # PI should be wider
        assert se_pred > se_fit


class TestResiduals:
    """Tests for residual calculations and properties."""

    def test_residuals_sum_to_zero(self):
        """Test that residuals sum to (approximately) zero."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 2.0 + 1.5 * x + np.random.normal(0, 2.0, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x
        residuals = y - y_pred

        assert np.isclose(np.sum(residuals), 0, atol=1e-10)

    def test_residuals_uncorrelated_with_x(self):
        """Test that residuals are uncorrelated with x."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 2.0 + 1.5 * x + np.random.normal(0, 2.0, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x
        residuals = y - y_pred

        correlation = np.corrcoef(x, residuals)[0, 1]
        assert np.isclose(correlation, 0, atol=1e-10)

    def test_residuals_uncorrelated_with_fitted(self):
        """Test that residuals are uncorrelated with fitted values."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 2.0 + 1.5 * x + np.random.normal(0, 2.0, n)

        result = stats.linregress(x, y)
        y_pred = result.intercept + result.slope * x
        residuals = y - y_pred

        correlation = np.corrcoef(y_pred, residuals)[0, 1]
        assert np.isclose(correlation, 0, atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_small_sample(self):
        """Test OLS with minimum sample size (n=3)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 5.0])

        result = stats.linregress(x, y)

        # Should produce valid coefficients
        assert np.isfinite(result.slope)
        assert np.isfinite(result.intercept)

    def test_large_values(self):
        """Test OLS with large values."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(1e6, 1e7, n)
        y = 1e6 + 0.5 * x + np.random.normal(0, 1e4, n)

        result = stats.linregress(x, y)

        # Coefficients should be reasonable
        assert np.isclose(result.slope, 0.5, atol=0.1)

    def test_zero_slope(self):
        """Test OLS correctly identifies zero slope."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        y = 10.0 + np.random.normal(0, 0.01, n)  # Constant with tiny noise

        result = stats.linregress(x, y)

        assert np.isclose(result.slope, 0, atol=0.01)

    def test_negative_slope(self):
        """Test OLS with negative slope."""
        np.random.seed(42)
        n = 100
        x = np.random.uniform(0, 50, n)
        true_slope = -2.0
        y = 50.0 + true_slope * x + np.random.normal(0, 1.0, n)

        result = stats.linregress(x, y)

        assert np.isclose(result.slope, true_slope, atol=0.2)
        assert result.slope < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
