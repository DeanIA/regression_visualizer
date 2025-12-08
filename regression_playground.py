import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from scipy import stats
    return go, mo, np, stats


@app.cell
def _(mo):
    mo.md("""# Regression Visualizer

Based on the material in 'Regression and Other Stories' by Andrew Gelman and Jennifer Hill. Created to prepare for the final exam in Dr. Ravi Shroff's Intermediate Quantitative Methods course at the Applied Statistics Depratment, NYU. Sharing is caring, MIT License etc. *Created by Dean Issacharoff let me know if you find any bugs.*

[GitHub Repository](https://github.com/DeanIA/regression_visualizer)
""")
    return


@app.cell
def _(mo):
    mo.md("### Model Type")
    return


@app.cell
def _(mo):
    model_type = mo.ui.radio(
        options=[
            "Basic Linear Regression",
            "Binary (Means Comparison)",
            "Binned (Categorized)",
            "Multivariable: Categorical",
            "Multivariable: Continuous",
            "Logistic Regression",
        ],
        value="Basic Linear Regression",
    )
    n_bins_slider = mo.ui.slider(
        start=2, stop=10, step=1, value=3, label="Number of Bins"
    )
    x_transform = mo.ui.dropdown(
        options=["None (x)", "Center (x - mean)", "Standardize (z-score)", "Min-Max (0-1)",
                 "Square Root (√x)", "Square (x²)", "Log (ln x)", "Constant (no x)"],
        value="None (x)",
        label="X Transformation:"
    )
    return model_type, n_bins_slider, x_transform


@app.cell
def _(mo):
    # Categorical variable controls (for Multivariable: With Categorical)
    group_var_name = mo.ui.text(value="Group", label="Categorical Variable:", placeholder="e.g., Treatment")
    group0_name_multi = mo.ui.text(value="Control", label="Group 0:", placeholder="e.g., Control")
    group1_name_multi = mo.ui.text(value="Treatment", label="Group 1:", placeholder="e.g., Treatment")

    # Interaction checkboxes (separate for each mode)
    add_interaction = mo.ui.checkbox(value=False, label="Add Interaction (different slopes per group)")
    add_interaction_cont = mo.ui.checkbox(value=False, label="Add Interaction (x × z)")
    add_continuous3 = mo.ui.checkbox(value=False, label="Add 3rd Predictor (w)")

    # Second continuous predictor (used in "With Continuous" mode)
    continuous2_name = mo.ui.text(value="z", label="2nd Predictor Name:", placeholder="e.g., Age")
    continuous2_hold = mo.ui.slider(start=0, stop=50, step=1, value=25, label="Hold 2nd predictor at:", show_value=True)

    # Third continuous predictor
    continuous3_name = mo.ui.text(value="w", label="3rd Predictor Name:", placeholder="e.g., Income")
    continuous3_hold = mo.ui.slider(start=0, stop=50, step=1, value=25, label="Hold 3rd predictor at:", show_value=True)

    # Distribution type toggle
    dist_type = mo.ui.dropdown(options=["Uniform", "Normal"], value="Uniform", label="Distribution:")

    # Predictor range controls (uniform distribution bounds) - text inputs for narrow width
    x_min = mo.ui.text(value="0", label="X min:")
    x_max = mo.ui.text(value="50", label="X max:")
    z_min = mo.ui.text(value="0", label="Z min:")
    z_max = mo.ui.text(value="50", label="Z max:")
    w_min = mo.ui.text(value="0", label="W min:")
    w_max = mo.ui.text(value="50", label="W max:")

    # Predictor distribution controls (normal distribution) - text inputs for narrow width
    x_mean = mo.ui.text(value="25", label="X μ:")
    x_sd = mo.ui.text(value="10", label="X σ:")
    z_mean = mo.ui.text(value="25", label="Z μ:")
    z_sd = mo.ui.text(value="10", label="Z σ:")
    w_mean = mo.ui.text(value="25", label="W μ:")
    w_sd = mo.ui.text(value="10", label="W σ:")

    # Coefficients for multiple regression
    beta_group = mo.ui.slider(start=0, stop=50, step=0.1, value=2.0, label="Group Effect (β)", show_value=True)
    beta_interaction = mo.ui.slider(start=0, stop=50, step=0.1, value=0.5, label="Interaction (β)", show_value=True)
    beta_interaction_cont = mo.ui.slider(start=0, stop=1, step=0.01, value=0.05, label="x×z Interaction (β)", show_value=True)
    beta_continuous2 = mo.ui.slider(start=0, stop=50, step=0.1, value=0.5, label="2nd Predictor Effect (β)", show_value=True)
    beta_continuous3 = mo.ui.slider(start=0, stop=50, step=0.1, value=0.3, label="3rd Predictor Effect (β)", show_value=True)

    return add_continuous3, add_interaction, add_interaction_cont, beta_continuous2, beta_continuous3, beta_group, beta_interaction, beta_interaction_cont, continuous2_hold, continuous2_name, continuous3_hold, continuous3_name, dist_type, group0_name_multi, group1_name_multi, group_var_name, w_max, w_mean, w_min, w_sd, x_max, x_mean, x_min, x_sd, z_max, z_mean, z_min, z_sd


@app.cell
def _(mo, model_type, n_bins_slider):
    is_binned = model_type.value == "Binned (Categorized)"
    is_continuous = model_type.value == "Basic Linear Regression"
    is_multiple = model_type.value.startswith("Multivariable:")
    has_grouping = model_type.value == "Multivariable: Categorical"
    is_logistic = model_type.value == "Logistic Regression"

    # Show appropriate controls based on model type
    items = [model_type]
    if is_binned:
        items.append(n_bins_slider)
    _display = mo.vstack(items, gap=1)
    _display
    return has_grouping, is_binned, is_continuous, is_logistic, is_multiple


@app.cell
def _(mo):
    mo.md("### Variable Names")
    return


@app.cell
def _(mo, model_type):
    is_binary = model_type.value == "Binary (Means Comparison)"

    x_name = mo.ui.text(
        value="Group" if is_binary else "x",
        label="Predictor (x):" if not is_binary else "Group Variable:",
        placeholder="e.g., Treatment" if is_binary else "e.g., Hours Studied"
    )
    y_name = mo.ui.text(value="y", label="Outcome (y):", placeholder="e.g., Test Score")

    # Names for binary groups
    group0_name = mo.ui.text(value="Control", label="Group 0 Name:", placeholder="e.g., Control")
    group1_name = mo.ui.text(value="Treatment", label="Group 1 Name:", placeholder="e.g., Treatment")

    return group0_name, group1_name, is_binary, x_name, y_name


@app.cell
def _(add_continuous3, continuous2_name, continuous3_name, group0_name, group0_name_multi, group1_name, group1_name_multi, group_var_name, has_grouping, is_binary, is_multiple, mo, x_name, y_name):
    # Build display based on model type - only variable names, no distribution controls
    if is_multiple and has_grouping:
        # Categorical variable mode
        _rows = [
            mo.hstack([x_name, y_name], justify="start", gap=4),
            mo.hstack([group_var_name, group0_name_multi, group1_name_multi], justify="start", gap=4),
        ]
        if add_continuous3.value:
            _rows.append(continuous3_name)
        _display = mo.vstack(_rows, gap=2)
    elif is_multiple:
        # Continuous predictors mode: x + z + optional w
        _rows = [
            mo.hstack([x_name, y_name], justify="start", gap=4),
            mo.hstack([continuous2_name, add_continuous3], justify="start", gap=4),
        ]
        if add_continuous3.value:
            _rows.append(continuous3_name)
        _display = mo.vstack(_rows, gap=2)
    elif is_binary:
        _display = mo.hstack([y_name, group0_name, group1_name], justify="start", gap=4)
    else:
        # Basic Linear Regression or Binned mode
        _display = mo.hstack([x_name, y_name], justify="start", gap=4)
    _display
    return


@app.cell
def _(mo):
    mo.md("### Data Parameters")
    return


@app.cell
def _(mo):
    n_points_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50, label="Number of Points", show_value=True
    )
    noise_slider = mo.ui.slider(
        start=0, stop=50, step=0.5, value=1.0, label="Error SD (σ)", show_value=True
    )
    seed_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=42, label="Random Seed", show_value=True
    )
    return n_points_slider, noise_slider, seed_slider


@app.cell
def _(add_continuous3, dist_type, has_grouping, is_binary, is_logistic, is_multiple, mo, n_points_slider, noise_slider, seed_slider, w_max, w_mean, w_min, w_sd, x_max, x_mean, x_min, x_sd, z_max, z_mean, z_min, z_sd):
    _rows = [
        mo.hstack([n_points_slider, seed_slider], justify="start", gap=4),
    ]

    # Distribution toggle and predictor parameters (all modes except binary)
    if not is_binary:
        _rows.append(dist_type)
        _is_normal = dist_type.value == "Normal"
        if is_multiple and not has_grouping:
            # Continuous mode: x + z + optional w - always show min/max
            _rows.append(mo.hstack([x_min, x_max, z_min, z_max], justify="start", gap=4))
            # Show μ/σ only for normal distribution
            if _is_normal:
                _rows.append(mo.hstack([x_mean, x_sd, z_mean, z_sd], justify="start", gap=4))
            if add_continuous3.value:
                _rows.append(mo.hstack([w_min, w_max], justify="start", gap=4))
                if _is_normal:
                    _rows.append(mo.hstack([w_mean, w_sd], justify="start", gap=4))
        else:
            # Basic, Binned, Logistic, or Categorical mode: just x (+ optional w for categorical)
            _rows.append(mo.hstack([x_min, x_max], justify="start", gap=4))
            if _is_normal:
                _rows.append(mo.hstack([x_mean, x_sd], justify="start", gap=4))
            if is_multiple and has_grouping and add_continuous3.value:
                _rows.append(mo.hstack([w_min, w_max], justify="start", gap=4))
                if _is_normal:
                    _rows.append(mo.hstack([w_mean, w_sd], justify="start", gap=4))

    # Error SD (not applicable for logistic regression)
    if not is_logistic:
        _rows.append(noise_slider)

    mo.vstack(_rows, gap=2)
    return


@app.cell
def _(mo):
    mo.md("### Line Parameters")
    return


@app.cell
def _(is_binary, is_logistic, mo):
    if is_logistic:
        _slope_label = "Log-odds change per unit x (β₁)"
        _intercept_label = "Log-odds at x=0 (β₀)"
    elif is_binary:
        _slope_label = "Group Difference (β₁)"
        _intercept_label = "Group 0 Mean (β₀)"
    else:
        _slope_label = "Slope (β₁)"
        _intercept_label = "Intercept (β₀)"

    # For logistic regression with x in [0, 50]:
    # - β₁ (slope): ±0.25 covers gentle to steep sigmoids
    #   - β₁ = 0.12 gives full S-curve (5% to 95%) across x range
    #   - β₁ = 0.25 gives very steep transition
    # - β₀ (intercept): -6 to +6 covers P(x=0) from ~0.2% to ~99.8%
    #   - β₀ = -3 with β₁ = 0.12 gives nice centered S-curve
    _slope_start = -0.25 if is_logistic else -25
    _slope_stop = 0.25 if is_logistic else 25
    _slope_step = 0.01 if is_logistic else 0.1
    _slope_default = 0.12 if is_logistic else 1.0
    slope_slider = mo.ui.slider(
        start=_slope_start, stop=_slope_stop, step=_slope_step, value=_slope_default,
        label=_slope_label,
        show_value=True
    )
    # For logistic, intercept is log-odds which can be negative
    _intercept_start = -6 if is_logistic else 0
    _intercept_stop = 6 if is_logistic else 50
    _intercept_step = 0.1 if is_logistic else 0.5
    _intercept_default = -3.0 if is_logistic else 0.0
    intercept_slider = mo.ui.slider(
        start=_intercept_start, stop=_intercept_stop, step=_intercept_step, value=_intercept_default,
        label=_intercept_label,
        show_value=True
    )
    # Threshold slider for logistic regression classification
    threshold_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.5,
        label="Classification Threshold",
        show_value=True
    )
    return intercept_slider, slope_slider, threshold_slider


@app.cell
def _(add_continuous3, add_interaction, add_interaction_cont, beta_continuous2, beta_continuous3, beta_group, beta_interaction, beta_interaction_cont, continuous2_hold, continuous3_hold, has_grouping, intercept_slider, is_binary, is_continuous, is_logistic, is_multiple, mo, slope_slider, threshold_slider, x_transform):
    # Build display based on model type
    if is_multiple and has_grouping:
        # Categorical variable mode: x + Group + optional interaction + optional 3rd predictor
        _rows = [
            mo.hstack([intercept_slider, slope_slider, beta_group], justify="start", gap=4),
            mo.hstack([add_interaction, beta_interaction], justify="start", gap=4) if add_interaction.value else add_interaction,
        ]
        if add_continuous3.value:
            _rows.append(mo.hstack([add_continuous3, beta_continuous3, continuous3_hold], justify="start", gap=4))
        else:
            _rows.append(add_continuous3)
        _display = mo.vstack(_rows, gap=2)
    elif is_multiple:
        # Continuous predictors mode: x + z + optional interaction + optional w
        _rows = [
            mo.hstack([intercept_slider, slope_slider, beta_continuous2], justify="start", gap=4),
            mo.hstack([continuous2_hold], justify="start", gap=4),
            mo.hstack([add_interaction_cont, beta_interaction_cont], justify="start", gap=4) if add_interaction_cont.value else add_interaction_cont,
        ]
        if add_continuous3.value:
            _rows.append(mo.hstack([beta_continuous3, continuous3_hold], justify="start", gap=4))
        _display = mo.vstack(_rows, gap=2)
    elif is_binary:
        _display = mo.vstack([
            mo.hstack([intercept_slider, slope_slider], justify="start", gap=4),
            mo.md(f"*Group 0 Mean = β₀ = {intercept_slider.value:.1f}, Group 1 Mean = β₀ + β₁ = {intercept_slider.value + slope_slider.value:.1f}*")
        ])
    elif is_continuous:
        _display = mo.vstack([
            mo.hstack([slope_slider, intercept_slider], justify="start", gap=4),
            x_transform
        ], gap=2)
    elif is_logistic:
        _display = mo.vstack([
            mo.hstack([slope_slider, intercept_slider], justify="start", gap=4),
            threshold_slider
        ], gap=2)
    else:
        _display = mo.hstack([slope_slider, intercept_slider], justify="start", gap=4)
    _display
    return


@app.cell
def _(mo):
    mo.md("### Display Options")
    return


@app.cell
def _(mo):
    show_sd = mo.ui.checkbox(value=False, label="Show Standard Deviation (SD) — spread of individual data points around line")
    show_ci = mo.ui.checkbox(value=False, label="Show Confidence Interval (CI) — uncertainty about TRUE regression line")
    show_pi = mo.ui.checkbox(value=False, label="Show Prediction Interval (PI) — range for NEW individual observation")
    sd_multiplier = mo.ui.slider(start=1, stop=3, step=0.5, value=1, label="± SD")
    ci_level = mo.ui.slider(start=0.80, stop=0.99, step=0.01, value=0.95, label="CI Level")
    pi_level = mo.ui.slider(start=0.80, stop=0.99, step=0.01, value=0.95, label="PI Level")
    return ci_level, pi_level, sd_multiplier, show_ci, show_pi, show_sd


@app.cell
def _(ci_level, is_logistic, mo, pi_level, sd_multiplier, show_ci, show_pi, show_sd):
    _items = []

    if not is_logistic:
        _items.append(show_sd)
        if show_sd.value:
            _items.append(sd_multiplier)

    _items.append(show_ci)
    if show_ci.value:
        _items.append(ci_level)

    if not is_logistic:
        _items.append(show_pi)
        if show_pi.value:
            _items.append(pi_level)

    _display = mo.vstack(_items, gap=0)
    _display
    return


@app.cell
def _(add_continuous3, add_interaction, add_interaction_cont, beta_continuous2, beta_continuous3, beta_group, beta_interaction, beta_interaction_cont, continuous2_hold, continuous2_name, continuous3_hold, continuous3_name, dist_type, group0_name, group0_name_multi, group1_name, group1_name_multi, group_var_name, has_grouping, intercept_slider, is_binary, is_binned, is_logistic, is_multiple, mo, n_bins_slider, np, slope_slider, threshold_slider, w_max, w_mean, w_min, w_sd, x_max, x_mean, x_min, x_sd, x_name, x_transform, y_name, z_max, z_mean, z_min, z_sd):
    x_label = "x" if is_binary else (x_name.value or "x")
    y_label = y_name.value or "y"
    slope = slope_slider.value
    intercept = intercept_slider.value
    g0_label = group0_name.value or "Group 0"
    g1_label = group1_name.value or "Group 1"
    transform = x_transform.value

    # Multiple regression specific labels
    grp_var_label = group_var_name.value or "Group"
    g0_multi_label = group0_name_multi.value or "Control"
    g1_multi_label = group1_name_multi.value or "Treatment"
    z_label = continuous2_name.value or "z"
    w_label = continuous3_name.value or "w"
    b_group = beta_group.value
    b_interaction = beta_interaction.value
    b_interaction_cont = beta_interaction_cont.value
    b_cont2 = beta_continuous2.value
    b_cont3 = beta_continuous3.value
    z_hold = continuous2_hold.value
    w_hold = continuous3_hold.value
    has_interaction = add_interaction.value
    has_interaction_cont = add_interaction_cont.value
    has_cont3 = add_continuous3.value

    # Distribution type and predictor parameters
    is_uniform = dist_type.value == "Uniform"

    # Parse text inputs to floats
    _x_min_val = float(x_min.value) if x_min.value else 0
    _x_max_val = float(x_max.value) if x_max.value else 50
    _z_min_val = float(z_min.value) if z_min.value else 0
    _z_max_val = float(z_max.value) if z_max.value else 50
    _w_min_val = float(w_min.value) if w_min.value else 0
    _w_max_val = float(w_max.value) if w_max.value else 50
    _x_mean_val = float(x_mean.value) if x_mean.value else 25
    _x_sd_val = float(x_sd.value) if x_sd.value else 10
    _z_mean_val = float(z_mean.value) if z_mean.value else 25
    _z_sd_val = float(z_sd.value) if z_sd.value else 10
    _w_mean_val = float(w_mean.value) if w_mean.value else 25
    _w_sd_val = float(w_sd.value) if w_sd.value else 10

    # Predictor ranges (for uniform) or mean/sd (for normal)
    x_lo = _x_min_val
    x_hi = _x_max_val
    z_lo = _z_min_val
    z_hi = _z_max_val
    w_lo = _w_min_val
    w_hi = _w_max_val
    x_mu = _x_mean_val
    x_sigma = _x_sd_val
    z_mu = _z_mean_val
    z_sigma = _z_sd_val
    w_mu = _w_mean_val
    w_sigma = _w_sd_val

    # Get transformed x label for equation
    if transform == "Center (x - mean)":
        x_term = f"({x_label} - x̄)"
    elif transform == "Standardize (z-score)":
        x_term = f"z_{x_label}"
    elif transform == "Min-Max (0-1)":
        x_term = f"{x_label}_scaled"
    elif transform == "Square Root (√x)":
        x_term = f"√{x_label}"
    elif transform == "Square (x²)":
        x_term = f"{x_label}²"
    elif transform == "Log (ln x)":
        x_term = f"ln({x_label})"
    elif transform == "Constant (no x)":
        x_term = None
    else:
        x_term = x_label

    if is_multiple:
        if has_grouping:
            # Grouping variable mode: y = β₀ + β₁x + β₂Group + β₃(x×Group) + β₄w
            eq_parts = [f"{intercept:.1f}", f"{slope:.1f}×{x_term or x_label}", f"{b_group:.1f}×{grp_var_label}"]
            if has_interaction:
                eq_parts.append(f"{b_interaction:.1f}×({x_term or x_label}×{grp_var_label})")
            if has_cont3:
                eq_parts.append(f"{b_cont3:.1f}×{w_label}")
            equation = f"{y_label} = " + " + ".join(eq_parts)
            equation_subtitle = f"where {grp_var_label} = 0 for {g0_multi_label}, 1 for {g1_multi_label}"
        else:
            # Continuous predictors mode: y = β₀ + β₁x + β₂z + β₃(x×z) + β₄w
            eq_parts = [f"{intercept:.1f}", f"{slope:.1f}×{x_term or x_label}", f"{b_cont2:.1f}×{z_label}"]
            if has_interaction_cont:
                eq_parts.append(f"{b_interaction_cont:.2f}×({x_term or x_label}×{z_label})")
            if has_cont3:
                eq_parts.append(f"{b_cont3:.1f}×{w_label}")
            equation = f"{y_label} = " + " + ".join(eq_parts)
            equation_subtitle = None
    elif is_binary:
        equation = f"{y_label} = {intercept:.1f} + {slope:.1f} × {x_label}"
        equation_subtitle = f"where {x_label} = 0 for {g0_label}, 1 for {g1_label}"
    elif is_logistic:
        _odds_ratio = np.exp(slope)
        _prob_at_zero = 1 / (1 + np.exp(-intercept))
        _threshold = threshold_slider.value
        # Calculate x value where P(Y=1) = threshold
        # P = 1/(1+exp(-z)) = threshold → z = log(threshold/(1-threshold))
        # β₀ + β₁*x = z → x = (z - β₀) / β₁
        if 0 < _threshold < 1 and slope != 0:
            _z_threshold = np.log(_threshold / (1 - _threshold))
            _x_decision = (_z_threshold - intercept) / slope
            _decision_text = f"Predict {y_label}=1 when {x_label} {'>' if slope > 0 else '<'} **{_x_decision:.1f}**"
        else:
            _decision_text = ""
        equation = f"log(odds) = **{intercept:.2f}** + **{slope:.2f}** × {x_label}"
        equation_subtitle = f"OR = **{_odds_ratio:.3f}** — {_decision_text} (threshold={_threshold})"
    elif is_binned:
        # Binned/categorical model: y = β₀ + β₁(Bin2) + β₂(Bin3) + ...
        _n_bins = n_bins_slider.value
        _eq_parts = ["β₀"]
        _bin_explanations = []
        for _i in range(2, _n_bins + 1):
            _eq_parts.append(f"β{_i-1}(Bin{_i})")
            _bin_explanations.append(f"Bin{_i}=1 if in bin {_i}, 0 otherwise")
        equation = f"{y_label} = " + " + ".join(_eq_parts)
        equation_subtitle = f"Bin1 is reference; " + "; ".join(_bin_explanations)
    elif x_term is None:
        equation = f"{y_label} = {intercept:.1f}"
        equation_subtitle = "constant model, no predictor"
    else:
        equation = f"{y_label} = {intercept:.1f} + {slope:.1f} × {x_term}"
        equation_subtitle = None

    # Build equation output with title and optional subtitle
    if equation_subtitle:
        equation_output = mo.md(f"**Model Equation:** {equation} *({equation_subtitle})*")
    else:
        equation_output = mo.md(f"**Model Equation:** {equation}")
    return b_cont2, b_cont3, b_group, b_interaction, b_interaction_cont, equation_output, g0_label, g0_multi_label, g1_label, g1_multi_label, grp_var_label, has_cont3, has_interaction, has_interaction_cont, intercept, is_uniform, slope, transform, w_hi, w_hold, w_label, w_lo, x_hi, x_label, x_lo, x_mu, x_sigma, x_term, y_label, z_hi, z_hold, z_label, z_lo


@app.cell
def _(b_cont2, b_cont3, b_group, b_interaction, b_interaction_cont, g0_label, g0_multi_label, g1_label, g1_multi_label, grp_var_label, has_cont3, has_grouping, has_interaction, has_interaction_cont, intercept, is_binary, is_logistic, is_multiple, mo, np, slope, transform, w_hold, w_label, x_label, x_term, y_label, z_hold, z_label):
    if is_logistic:
        # Logistic regression interpretation with odds ratio
        _prob_at_zero = 1 / (1 + np.exp(-intercept))
        _odds_ratio = np.exp(slope)

        intercept_interp = f"When **{x_label}**=0, the log-odds of {y_label}=1 is **{intercept:.2f}**, corresponding to a probability of **{_prob_at_zero:.3f}**."

        if slope > 0:
            direction = "increase"
            odds_direction = "multiply"
        elif slope < 0:
            direction = "decrease"
            odds_direction = "divide"
        else:
            direction = "no change in"
            odds_direction = "unchanged"

        if slope != 0:
            slope_interp = f"For each 1-unit increase in **{x_label}**, the log-odds {direction} by **{abs(slope):.2f}**. **Odds ratio = {_odds_ratio:.3f}**: the odds {odds_direction} by this factor."
        else:
            slope_interp = f"**{x_label}** has no effect on the probability of {y_label}=1 (odds ratio = 1)."

        coef_text = f"""### Coefficient Interpretation (Logistic Regression)

**Intercept (β₀ = {intercept:.2f}):** {intercept_interp}

**Slope (β₁ = {slope:.2f}):** {slope_interp}

**Odds Ratio Interpretation:**
- OR > 1: higher {x_label} → higher odds of {y_label}=1
- OR < 1: higher {x_label} → lower odds of {y_label}=1
- OR = 1: {x_label} has no effect
"""
    elif is_multiple:
        if has_grouping:
            # Grouping variable mode - RAOS-style interpretation
            # Intercept interpretation
            intercept_interp = f"For **{g0_multi_label}** (when {grp_var_label}=0), when **{x_label}**=0"
            if has_cont3:
                intercept_interp += f" and **{w_label}**=0"
            intercept_interp += f", the predicted **{y_label}** is **{intercept:.1f}**."

            # Slope (β₁) interpretation - continuous x effect
            if slope > 0:
                direction = "higher"
            elif slope < 0:
                direction = "lower"
            else:
                direction = "unchanged"

            slope_interp = f"Comparing two observations with the same **{grp_var_label}**"
            if has_cont3:
                slope_interp += f" and same **{w_label}**"
            if slope != 0:
                slope_interp += f", a 1-unit difference in **{x_term or x_label}** corresponds to a **{abs(slope):.1f}**-unit {direction} **{y_label}**."
            else:
                slope_interp += f", **{x_label}** has no effect on **{y_label}**."

            # Group effect (β₂) interpretation
            if b_group > 0:
                grp_direction = "higher"
            elif b_group < 0:
                grp_direction = "lower"
            else:
                grp_direction = "same"

            group_interp = f"Comparing **{g1_multi_label}** to **{g0_multi_label}** at the same **{x_label}**"
            if has_cont3:
                group_interp += f" and same **{w_label}**"
            if not has_interaction:
                if b_group != 0:
                    group_interp += f", **{g1_multi_label}** has **{abs(b_group):.1f}** units {grp_direction} **{y_label}**."
                else:
                    group_interp += f", there is no difference between groups."
            else:
                group_interp += f", **{g1_multi_label}** has **{abs(b_group):.1f}** units {grp_direction} **{y_label}** *when {x_label}=0*."

            coef_text = f"""### Coefficient Interpretation (Multivariable, RAOS Style)

**Intercept (β₀ = {intercept:.1f}):** {intercept_interp}

**Slope of {x_term or x_label} (β₁ = {slope:.1f}):** {slope_interp}

**{grp_var_label} Effect (β₂ = {b_group:.1f}):** {group_interp}
"""
            # Interaction interpretation (β₃)
            if has_interaction:
                if b_interaction > 0:
                    int_direction = "stronger"
                elif b_interaction < 0:
                    int_direction = "weaker"
                else:
                    int_direction = "the same"

                if b_interaction != 0:
                    interaction_interp = f"The effect of **{x_term or x_label}** on **{y_label}** is **{abs(b_interaction):.1f}** units {int_direction} for **{g1_multi_label}** than for **{g0_multi_label}**. (Slope for {g0_multi_label}: {slope:.1f}, Slope for {g1_multi_label}: {slope + b_interaction:.1f})"
                else:
                    interaction_interp = f"The effect of **{x_label}** is the same for both groups (parallel lines)."

                coef_text += f"""
**Interaction (β₃ = {b_interaction:.1f}):** {interaction_interp}
"""

            # Third continuous predictor interpretation
            if has_cont3:
                if b_cont3 > 0:
                    w_direction = "higher"
                elif b_cont3 < 0:
                    w_direction = "lower"
                else:
                    w_direction = "unchanged"

                w_interp = f"Holding **{x_label}** and **{grp_var_label}** constant, a 1-unit increase in **{w_label}** corresponds to **{abs(b_cont3):.1f}** units {w_direction} **{y_label}**. *(Currently held at {w_label}={w_hold:.1f} for visualization)*"

                coef_text += f"""
**{w_label} Effect (β = {b_cont3:.1f}):** {w_interp}
"""
        else:
            # Continuous predictors mode: y = β₀ + β₁x + β₂z + β₃(x×z) + β₄w
            # Intercept interpretation
            intercept_interp = f"When **{x_label}**=0 and **{z_label}**=0"
            if has_cont3:
                intercept_interp += f" and **{w_label}**=0"
            intercept_interp += f", the predicted **{y_label}** is **{intercept:.1f}**."

            # Slope (β₁) interpretation
            if slope > 0:
                x_direction = "higher"
            elif slope < 0:
                x_direction = "lower"
            else:
                x_direction = "unchanged"

            slope_interp = f"Holding **{z_label}**"
            if has_cont3:
                slope_interp += f" and **{w_label}**"
            slope_interp += " constant"
            if has_interaction_cont:
                slope_interp += f" (when {z_label}=0)"
            if slope != 0:
                slope_interp += f", a 1-unit increase in **{x_term or x_label}** corresponds to a **{abs(slope):.1f}**-unit {x_direction} **{y_label}**."
            else:
                slope_interp += f", **{x_label}** has no effect on **{y_label}**."

            # Second predictor (β₂) interpretation
            if b_cont2 > 0:
                z_direction = "higher"
            elif b_cont2 < 0:
                z_direction = "lower"
            else:
                z_direction = "unchanged"

            z_interp = f"Holding **{x_label}**"
            if has_cont3:
                z_interp += f" and **{w_label}**"
            z_interp += " constant"
            if has_interaction_cont:
                z_interp += f" (when {x_label}=0)"
            if b_cont2 != 0:
                z_interp += f", a 1-unit increase in **{z_label}** corresponds to a **{abs(b_cont2):.1f}**-unit {z_direction} **{y_label}**. *(Currently held at {z_label}={z_hold:.1f} for visualization)*"
            else:
                z_interp += f", **{z_label}** has no effect on **{y_label}**."

            coef_text = f"""### Coefficient Interpretation (Continuous Predictors, RAOS Style)

**Intercept (β₀ = {intercept:.1f}):** {intercept_interp}

**Slope of {x_term or x_label} (β₁ = {slope:.1f}):** {slope_interp}

**{z_label} Effect (β₂ = {b_cont2:.1f}):** {z_interp}
"""

            # Interaction interpretation (β₃) for continuous mode
            if has_interaction_cont:
                if b_interaction_cont > 0:
                    int_direction = "stronger"
                elif b_interaction_cont < 0:
                    int_direction = "weaker"
                else:
                    int_direction = "unchanged"

                if b_interaction_cont != 0:
                    interaction_interp = f"The effect of **{x_term or x_label}** on **{y_label}** changes by **{abs(b_interaction_cont):.2f}** units for each 1-unit increase in **{z_label}**. (At {z_label}={z_hold:.1f}, the slope of {x_label} is {slope + b_interaction_cont * z_hold:.2f})"
                else:
                    interaction_interp = f"The effect of **{x_label}** is the same at all levels of **{z_label}** (no interaction)."

                coef_text += f"""
**{x_label}×{z_label} Interaction (β₃ = {b_interaction_cont:.2f}):** {interaction_interp}
"""

            # Third continuous predictor interpretation (w)
            if has_cont3:
                if b_cont3 > 0:
                    w_direction = "higher"
                elif b_cont3 < 0:
                    w_direction = "lower"
                else:
                    w_direction = "unchanged"

                w_interp = f"Holding **{x_label}** and **{z_label}** constant, a 1-unit increase in **{w_label}** corresponds to **{abs(b_cont3):.1f}** units {w_direction} **{y_label}**. *(Currently held at {w_label}={w_hold:.1f} for visualization)*"

                coef_text += f"""
**{w_label} Effect (β = {b_cont3:.1f}):** {w_interp}
"""

    elif is_binary:
        # Binary/dummy variable interpretation
        intercept_interp = f"The mean of **{y_label}** for **{g0_label}** (when {x_label}=0) is **{intercept:.1f}**."

        if slope > 0:
            direction = "higher"
        elif slope < 0:
            direction = "lower"
        else:
            direction = "the same as"

        if slope != 0:
            slope_interp = f"**{g1_label}** has a mean **{y_label}** that is **{abs(slope):.1f}** units {direction} than **{g0_label}**. (Mean = {intercept + slope:.1f})"
        else:
            slope_interp = f"**{g1_label}** and **{g0_label}** have the same mean **{y_label}** (no group difference)."

        coef_text = f"""### Coefficient Interpretation (Means Comparison)

**Intercept (β₀ = {intercept:.1f}):** {intercept_interp}

**Group Difference (β₁ = {slope:.1f}):** {slope_interp}
"""
    elif transform == "Constant (no x)":
        # Constant model - intercept only
        coef_text = f"""### Coefficient Interpretation (Constant Model)

**Intercept (β₀ = {intercept:.1f}):** The predicted value of **{y_label}** is always **{intercept:.1f}**, regardless of any predictor. This is simply the mean of {y_label}.
"""
    else:
        # Continuous variable interpretation with transformations
        if transform == "Center (x - mean)":
            intercept_interp = f"When **{x_label}** equals its mean (x̄), the predicted value of **{y_label}** is **{intercept:.1f}**. (Centering shifts the intercept to the mean of x.)"
            if slope != 0:
                slope_interp = f"For every 1-unit increase in **{x_label}**, **{y_label}** changes by **{slope:.1f}** units. (Same as uncentered; centering only changes the intercept.)"
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."
        elif transform == "Standardize (z-score)":
            intercept_interp = f"When **{x_label}** equals its mean (z=0), the predicted value of **{y_label}** is **{intercept:.1f}**."
            if slope != 0:
                slope_interp = f"For every **1 standard deviation** increase in **{x_label}**, **{y_label}** changes by **{slope:.1f}** units. (Standardized coefficient allows comparison across different scales.)"
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."
        elif transform == "Min-Max (0-1)":
            intercept_interp = f"When **{x_label}** is at its minimum (scaled=0), the predicted value of **{y_label}** is **{intercept:.1f}**."
            if slope != 0:
                slope_interp = f"Moving from the **minimum to maximum** of **{x_label}** (0→1 in scaled units), **{y_label}** changes by **{slope:.1f}** units total."
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."
        elif transform == "Square Root (√x)":
            intercept_interp = f"When **{x_label}** equals 0 (so √{x_label}=0), the predicted value of **{y_label}** is **{intercept:.1f}**."
            if slope != 0:
                slope_interp = f"For every 1-unit increase in **√{x_label}**, **{y_label}** changes by **{slope:.1f}** units. (Note: the effect on {y_label} per unit of raw {x_label} decreases as {x_label} increases.)"
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."
        elif transform == "Square (x²)":
            intercept_interp = f"When **{x_label}** equals 0, the predicted value of **{y_label}** is **{intercept:.1f}**."
            if slope != 0:
                slope_interp = f"For every 1-unit increase in **{x_label}²**, **{y_label}** changes by **{slope:.1f}** units. (Note: the effect on {y_label} per unit of raw {x_label} increases as {x_label} increases.)"
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."
        elif transform == "Log (ln x)":
            intercept_interp = f"When **ln({x_label})=0** (i.e., {x_label}=1), the predicted value of **{y_label}** is **{intercept:.1f}**."
            if slope != 0:
                slope_interp = f"For every 1-unit increase in **ln({x_label})** (i.e., {x_label} multiplied by e≈2.72), **{y_label}** changes by **{slope:.1f}** units. Alternatively: a 1% increase in {x_label} is associated with a **{slope/100:.3f}** unit change in {y_label}."
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."
        else:
            # No transformation
            intercept_interp = f"When **{x_label}** equals 0, the predicted value of **{y_label}** is **{intercept:.1f}**."
            if slope > 0:
                direction = "increase"
            elif slope < 0:
                direction = "decrease"
            else:
                direction = "no change"

            if slope != 0:
                slope_interp = f"For every 1-unit increase in **{x_label}**, **{y_label}** is expected to {direction} by **{abs(slope):.1f}** units."
            else:
                slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope is 0)."

        coef_text = f"""### Coefficient Interpretation

**Intercept (β₀ = {intercept:.1f}):** {intercept_interp}

**Slope (β₁ = {slope:.1f}):** {slope_interp}
"""
    return (coef_text,)


@app.cell
def _(b_cont2, b_cont3, b_group, b_interaction, b_interaction_cont, ci_level, g0_label, g0_multi_label, g1_label, g1_multi_label, go, grp_var_label, has_cont3, has_grouping, has_interaction, has_interaction_cont, intercept, is_binary, is_binned, is_logistic, is_multiple, is_uniform, mo, n_bins_slider, n_points_slider, noise_slider, np, pi_level, sd_multiplier, seed_slider, show_ci, show_pi, show_sd, slope, stats, threshold_slider, transform, w_hi, w_hold, w_label, w_lo, x_hi, x_label, x_lo, x_mu, x_sigma, x_term, y_label, z_hi, z_hold, z_label, z_lo):
    # Fixed axis ranges
    Y_MIN, Y_MAX = 0, 50

    np.random.seed(seed_slider.value)
    n = n_points_slider.value

    # Create figure
    fig = go.Figure()

    if is_multiple:
        # Multiple regression mode - generate data from uniform or normal distributions
        if is_uniform:
            x_data_raw = np.random.uniform(x_lo, x_hi, n)
        else:
            # Normal distribution clipped to min/max bounds
            x_data_raw = np.clip(np.random.normal(x_mu, x_sigma, n), x_lo, x_hi)
        # For log transform, ensure positive values
        if transform == "Log (ln x)":
            x_data_raw = np.abs(x_data_raw) + 0.1

        # Apply transformation to x for the model
        # Store statistics for scaling transforms (needed for line transform too)
        _x_mean_raw = np.mean(x_data_raw)
        _x_std_raw = np.std(x_data_raw)
        _x_min_raw = np.min(x_data_raw)
        _x_max_raw = np.max(x_data_raw)

        if transform == "Center (x - mean)":
            x_data_transformed = x_data_raw - _x_mean_raw
        elif transform == "Standardize (z-score)":
            x_data_transformed = (x_data_raw - _x_mean_raw) / _x_std_raw
        elif transform == "Min-Max (0-1)":
            x_data_transformed = (x_data_raw - _x_min_raw) / (_x_max_raw - _x_min_raw)
        elif transform == "Square Root (√x)":
            x_data_raw = np.abs(x_data_raw)  # Ensure non-negative for sqrt
            x_data_transformed = np.sqrt(x_data_raw)
        elif transform == "Square (x²)":
            x_data_transformed = x_data_raw ** 2
        elif transform == "Log (ln x)":
            x_data_transformed = np.log(x_data_raw)
        elif transform == "Constant (no x)":
            x_data_transformed = np.zeros_like(x_data_raw)
        else:
            x_data_transformed = x_data_raw

        # Transform x_line for regression curve (always use min/max bounds)
        X_MIN = x_lo
        X_MAX = x_hi
        if transform == "Log (ln x)" or transform == "Square Root (√x)":
            X_MIN = max(0.1, X_MIN)
        x_line = np.linspace(0, 50, 100)
        if transform == "Center (x - mean)":
            x_line_transformed = x_line - _x_mean_raw
        elif transform == "Standardize (z-score)":
            x_line_transformed = (x_line - _x_mean_raw) / _x_std_raw
        elif transform == "Min-Max (0-1)":
            x_line_transformed = (x_line - _x_min_raw) / (_x_max_raw - _x_min_raw)
        elif transform == "Square Root (√x)":
            x_line_transformed = np.sqrt(np.maximum(x_line, 0))
        elif transform == "Square (x²)":
            x_line_transformed = x_line ** 2
        elif transform == "Log (ln x)":
            x_line_transformed = np.log(np.maximum(x_line, 0.1))
        elif transform == "Constant (no x)":
            x_line_transformed = np.zeros_like(x_line)
        else:
            x_line_transformed = x_line

        if has_grouping:
            # Grouping variable mode: y = β₀ + β₁x + β₂Group + β₃(x×Group) + β₄w
            group_data = np.random.binomial(1, 0.5, n)  # Random group assignment

            # Generate third continuous predictor if enabled
            if has_cont3:
                if is_uniform:
                    w_data = np.random.uniform(w_lo, w_hi, n)
                else:
                    w_data = np.clip(np.random.normal(w_mu, w_sigma, n), w_lo, w_hi)
            else:
                w_data = np.zeros(n)

            # True DGP
            y_data = (intercept
                      + slope * x_data_transformed
                      + b_group * group_data
                      + (b_interaction * x_data_transformed * group_data if has_interaction else 0)
                      + (b_cont3 * w_data if has_cont3 else 0)
                      + np.random.normal(0, noise_slider.value, n))

            # Colors for groups
            group_colors = ["#636EFA", "#EF553B"]  # Blue for group 0, red for group 1
            group_labels_list = [g0_multi_label, g1_multi_label]

            # Plot points colored by group
            for g in [0, 1]:
                mask = group_data == g
                fig.add_trace(go.Scatter(
                    x=x_data_raw[mask],
                    y=y_data[mask],
                    mode="markers",
                    name=group_labels_list[g],
                    marker=dict(color=group_colors[g], size=8, opacity=0.6),
                ))

            # w held at specified value for visualization
            w_held = w_hold if has_cont3 else 0

            # Plot regression lines for each group
            for g in [0, 1]:
                if has_interaction:
                    y_line = (intercept
                              + slope * x_line_transformed
                              + b_group * g
                              + b_interaction * x_line_transformed * g
                              + b_cont3 * w_held)
                else:
                    y_line = (intercept
                              + slope * x_line_transformed
                              + b_group * g
                              + b_cont3 * w_held)

                line_name = f"{group_labels_list[g]} fit"
                if has_interaction:
                    line_name += f" (slope={slope + b_interaction * g:.1f})"

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=line_name,
                    line=dict(color=group_colors[g], width=3),
                ))

            # Layout for grouping mode
            x_axis_title = x_label
            if has_cont3:
                x_axis_title += f" ({w_label} held at {w_held:.1f})"

        else:
            # Continuous predictors mode: y = β₀ + β₁x + β₂z + β₃(x×z) + β₄w
            if is_uniform:
                z_data = np.random.uniform(z_lo, z_hi, n)
            else:
                z_data = np.clip(np.random.normal(z_mu, z_sigma, n), z_lo, z_hi)

            # Generate third continuous predictor if enabled
            if has_cont3:
                if is_uniform:
                    w_data = np.random.uniform(w_lo, w_hi, n)
                else:
                    w_data = np.clip(np.random.normal(w_mu, w_sigma, n), w_lo, w_hi)
            else:
                w_data = np.zeros(n)

            # True DGP
            y_data = (intercept
                      + slope * x_data_transformed
                      + b_cont2 * z_data
                      + (b_interaction_cont * x_data_transformed * z_data if has_interaction_cont else 0)
                      + (b_cont3 * w_data if has_cont3 else 0)
                      + np.random.normal(0, noise_slider.value, n))

            # z held at specified value for visualization
            z_held = z_hold
            w_held = w_hold if has_cont3 else 0

            # Discretize z into 3 categories (thirds for uniform, ±1 SD for normal)
            if is_uniform:
                z_range = z_hi - z_lo
                z_low_threshold = z_lo + z_range / 3
                z_high_threshold = z_lo + 2 * z_range / 3
            else:
                z_low_threshold = z_mu - z_sigma
                z_high_threshold = z_mu + z_sigma
            z_categories = np.where(z_data < z_low_threshold, 0,
                                    np.where(z_data > z_high_threshold, 2, 1))

            # Define colors and labels for z categories
            z_colors = ["#636EFA", "#00CC96", "#EF553B"]  # Blue, Green, Red
            z_cat_labels = [
                f"Low {z_label} (<{z_low_threshold:.0f})",
                f"Mid {z_label} ({z_low_threshold:.0f}-{z_high_threshold:.0f})",
                f"High {z_label} (>{z_high_threshold:.0f})"
            ]

            # Plot points by z category with discrete colors
            for cat_idx in range(3):
                mask = z_categories == cat_idx
                if np.sum(mask) > 0:
                    fig.add_trace(go.Scatter(
                        x=x_data_raw[mask],
                        y=y_data[mask],
                        mode="markers",
                        name=z_cat_labels[cat_idx],
                        marker=dict(
                            color=z_colors[cat_idx],
                            size=10,
                            opacity=0.7,
                        ),
                        text=[f"{z_label}={z:.1f}" for z in z_data[mask]],
                        hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<br>%{{text}}<extra>{z_cat_labels[cat_idx]}</extra>",
                    ))

            # Plot regression lines for each z category
            # Use representative z values: low/mid/high category centers
            if is_uniform:
                z_representative = [z_lo + z_range * 0.17, z_lo + z_range * 0.5, z_lo + z_range * 0.83]
            else:
                z_representative = [z_mu - 1.5 * z_sigma, z_mu, z_mu + 1.5 * z_sigma]
            w_held = w_hold if has_cont3 else 0

            for cat_idx in range(3):
                z_val = z_representative[cat_idx]
                y_line = (intercept
                          + slope * x_line_transformed
                          + b_cont2 * z_val
                          + (b_interaction_cont * x_line_transformed * z_val if has_interaction_cont else 0)
                          + (b_cont3 * w_held if has_cont3 else 0))

                line_name = f"{z_cat_labels[cat_idx]} line"

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=line_name,
                    line=dict(color=z_colors[cat_idx], width=3),
                ))

            # Layout for continuous mode
            x_axis_title = x_label
            if has_cont3:
                x_axis_title += f" ({w_label} held at {w_held:.1f})"

        # Calculate dynamic y-axis range based on data (always start at 0)
        fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Regression Plot",
            xaxis=dict(
                title=x_axis_title,
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )

        # Compute OLS for multiple regression
        # Build design matrix based on model structure
        if has_grouping:
            # Grouping mode: y = β₀ + β₁x + β₂Group + β₃(x×Group) + β₄w
            _predictors = [x_data_transformed, group_data]
            _pred_names = [x_label, "Group"]
            if has_interaction:
                _predictors.append(x_data_transformed * group_data)
                _pred_names.append(f"{x_label}×Group")
            if has_cont3:
                _predictors.append(w_data)
                _pred_names.append(w_label)
        else:
            # Continuous mode: y = β₀ + β₁x + β₂z + β₃(x×z) + β₄w
            _predictors = [x_data_transformed, z_data]
            _pred_names = [x_label, z_label]
            if has_interaction_cont:
                _predictors.append(x_data_transformed * z_data)
                _pred_names.append(f"{x_label}×{z_label}")
            if has_cont3:
                _predictors.append(w_data)
                _pred_names.append(w_label)

        # Build design matrix: [1, x1, x2, ...]
        _X_design = np.column_stack([np.ones(n)] + _predictors)
        _n_predictors = len(_predictors)

        # OLS using normal equations: (X'X)^(-1) X'y
        _XtX = _X_design.T @ _X_design
        _Xty = _X_design.T @ y_data
        _multi_betas = np.linalg.solve(_XtX, _Xty)

        _multi_intercept = _multi_betas[0]
        _multi_coefs = _multi_betas[1:]

        # OLS predictions and residuals
        _multi_y_pred = _X_design @ _multi_betas
        _multi_residuals = y_data - _multi_y_pred

        # Calculate statistics
        _multi_df = n - (_n_predictors + 1)  # n - (k + 1)
        _multi_mse = np.sum(_multi_residuals**2) / _multi_df if _multi_df > 0 else 0
        _multi_se = np.sqrt(_multi_mse)

        # Standard errors for coefficients: sqrt(diag((X'X)^(-1) * MSE))
        _XtX_inv = np.linalg.inv(_XtX)
        _multi_se_betas = np.sqrt(np.diag(_XtX_inv) * _multi_mse)

    elif is_binned:
        # Binned mode: Generate continuous data then bin it
        n_bins = n_bins_slider.value

        # Generate continuous x from uniform or normal distribution (clipped to bounds)
        if is_uniform:
            x_continuous = np.random.uniform(x_lo, x_hi, n)
        else:
            x_continuous = np.clip(np.random.normal(x_mu, x_sigma, n), x_lo, x_hi)
        y_data = intercept + slope * x_continuous + np.random.normal(0, noise_slider.value, n)

        # X range always uses min/max bounds
        X_MIN = x_lo
        X_MAX = x_hi

        # Create bin edges and assign bins
        bin_edges = np.linspace(X_MIN, X_MAX, n_bins + 1)
        bin_indices = np.digitize(x_continuous, bin_edges[1:-1])  # 0 to n_bins-1
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Create dummy variables for bins (Bin 1 = reference)
        # Design matrix: [1, D2, D3, ..., Dk] where Di = 1 if in bin i
        _X_design = np.column_stack([
            np.ones(n),  # Intercept
            *[(bin_indices == i).astype(float) for i in range(1, n_bins)]  # Dummies for bins 2 to k
        ])

        # OLS using normal equations: (X'X)^(-1) X'y
        _XtX = _X_design.T @ _X_design
        _Xty = _X_design.T @ y_data
        _ols_betas = np.linalg.solve(_XtX, _Xty)  # [intercept, beta_bin2, beta_bin3, ...]

        _ols_intercept = _ols_betas[0]  # Mean of reference group (Bin 1)
        _ols_bin_coefs = _ols_betas[1:]  # Differences from reference

        # OLS predictions and residuals
        _ols_y_pred = _X_design @ _ols_betas
        _ols_residuals = y_data - _ols_y_pred

        # Calculate statistics using OLS
        df = n - n_bins  # n - k (number of parameters = number of bins)
        mse = np.sum(_ols_residuals**2) / df
        se = np.sqrt(mse)

        # Standard errors for coefficients: sqrt(diag((X'X)^(-1) * MSE))
        _XtX_inv = np.linalg.inv(_XtX)
        _se_betas = np.sqrt(np.diag(_XtX_inv) * mse)

        # For compatibility
        _x_mean = np.mean(x_continuous)
        ss_x = np.sum((x_continuous - _x_mean)**2)

        t_val_ci = stats.t.ppf((1 + ci_level.value) / 2, df)
        t_val_pi = stats.t.ppf((1 + pi_level.value) / 2, df)

        # Colors for bins
        colors = [f"hsl({int(i * 360 / n_bins)}, 70%, 50%)" for i in range(n_bins)]

        # Bin labels
        bin_labels = [f"Bin {i+1}" for i in range(n_bins)]

        # Plot points by bin with jitter
        jitter = (X_MAX - X_MIN) / n_bins * 0.15
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                x_jittered = bin_centers[i] + np.random.uniform(-jitter, jitter, np.sum(mask))
                fig.add_trace(
                    go.Scatter(
                        x=x_jittered,
                        y=y_data[mask],
                        mode="markers",
                        name=bin_labels[i],
                        marker=dict(color=colors[i], size=8, opacity=0.6),
                        legendgroup=f"bin{i}",
                    )
                )

        # Calculate bin means
        bin_means = [np.mean(y_data[bin_indices == i]) if np.sum(bin_indices == i) > 0 else np.nan for i in range(n_bins)]
        bin_counts = [np.sum(bin_indices == i) for i in range(n_bins)]

        # Add horizontal step lines for each bin mean (the binned prediction)
        for i in range(n_bins):
            if not np.isnan(bin_means[i]):
                fig.add_trace(
                    go.Scatter(
                        x=[bin_edges[i], bin_edges[i+1]],
                        y=[bin_means[i], bin_means[i]],
                        mode="lines",
                        name="Bin Mean" if i == 0 else None,
                        showlegend=(i == 0),
                        line=dict(color="black", width=3),
                    )
                )

        # Add true regression line
        x_line = np.array([0, 50])
        y_line = intercept + slope * x_line
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="True Regression Line",
                line=dict(color="#EF553B", width=3, dash="dash"),
            )
        )

        # SD bands per bin (full width rectangles)
        if show_sd.value:
            sd_mult = sd_multiplier.value
            sd_pct = (stats.norm.cdf(sd_mult) - stats.norm.cdf(-sd_mult)) * 100
            for i in range(n_bins):
                if not np.isnan(bin_means[i]):
                    fig.add_shape(
                        type="rect",
                        x0=bin_edges[i],
                        x1=bin_edges[i+1],
                        y0=bin_means[i] - sd_mult * se,
                        y1=bin_means[i] + sd_mult * se,
                        fillcolor="rgba(0, 200, 0, 0.2)",
                        line=dict(width=0),
                    )
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="rgba(0, 200, 0, 0.3)"),
                name=f"±{sd_mult} SD ({sd_pct:.0f}% of data)"))

        # CI bands per bin (full width)
        if show_ci.value:
            for i in range(n_bins):
                if bin_counts[i] > 1 and not np.isnan(bin_means[i]):
                    se_mean = se / np.sqrt(bin_counts[i])
                    ci_half = t_val_ci * se_mean
                    fig.add_shape(
                        type="rect",
                        x0=bin_edges[i],
                        x1=bin_edges[i+1],
                        y0=bin_means[i] - ci_half,
                        y1=bin_means[i] + ci_half,
                        fillcolor="rgba(99, 110, 250, 0.3)",
                        line=dict(width=0),
                    )
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="rgba(99, 110, 250, 0.3)"),
                name=f"{int(ci_level.value*100)}% CI"))

        # Layout for binned
        fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Regression Plot",
            xaxis=dict(
                title=f"{x_label} (binned)",
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )

    elif is_logistic:
        # Logistic regression mode: Binary outcome (0/1)
        # Generate x from selected distribution
        if is_uniform:
            x_data = np.random.uniform(x_lo, x_hi, n)
        else:
            x_data = np.clip(np.random.normal(x_mu, x_sigma, n), x_lo, x_hi)

        # Compute linear predictor and probability
        _z = intercept + slope * x_data
        _prob = 1 / (1 + np.exp(-_z))

        # Generate binary outcomes from Bernoulli(prob)
        y_data = np.random.binomial(1, _prob)

        # Add jitter to y for visualization (0s and 1s would overlap)
        _jitter_amount = 0.03
        y_jittered = y_data + np.random.uniform(-_jitter_amount, _jitter_amount, n)

        # Classify based on threshold
        _pred_class = (_prob >= threshold_slider.value).astype(int)

        # Color by classification correctness: TP=green, TN=blue, FP=orange, FN=red
        _colors = []
        for _actual, _pred in zip(y_data, _pred_class):
            if _actual == 1 and _pred == 1:  # True Positive
                _colors.append("#2CA02C")  # green
            elif _actual == 0 and _pred == 0:  # True Negative
                _colors.append("#1F77B4")  # blue
            elif _actual == 0 and _pred == 1:  # False Positive
                _colors.append("#FF7F0E")  # orange
            else:  # False Negative (_actual == 1 and _pred == 0)
                _colors.append("#D62728")  # red

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_jittered,
                mode="markers",
                name="Data",
                marker=dict(color=_colors, size=8, opacity=0.7),
                showlegend=False,
            )
        )

        # Add legend entries for classification outcomes
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name="True Positive",
                marker=dict(color="#2CA02C", size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name="True Negative",
                marker=dict(color="#1F77B4", size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name="False Positive",
                marker=dict(color="#FF7F0E", size=8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                name="False Negative",
                marker=dict(color="#D62728", size=8),
            )
        )

        # MLE fitting for logistic regression
        x_line = np.linspace(0, 50, 200)
        from scipy.optimize import minimize

        def _neg_log_likelihood(betas, X, y):
            z = betas[0] + betas[1] * X
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)  # numerical stability
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        # Fit model using MLE
        _mle_result = minimize(_neg_log_likelihood, x0=[0, 0], args=(x_data, y_data), method='BFGS')
        _mle_intercept = _mle_result.x[0]
        _mle_slope = _mle_result.x[1]

        # Compute Hessian numerically for standard errors
        def _hessian_logistic(betas, X, y):
            z = betas[0] + betas[1] * X
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            w = p * (1 - p)
            # Hessian of negative log-likelihood
            H00 = np.sum(w)
            H01 = np.sum(w * X)
            H11 = np.sum(w * X**2)
            return np.array([[H00, H01], [H01, H11]])

        _hessian = _hessian_logistic(_mle_result.x, x_data, y_data)
        try:
            _cov_matrix = np.linalg.inv(_hessian)
            _se_intercept = np.sqrt(_cov_matrix[0, 0])
            _se_slope = np.sqrt(_cov_matrix[1, 1])
        except:
            _cov_matrix = None
            _se_intercept = np.nan
            _se_slope = np.nan

        # Compute deviances and AIC
        # Null model (intercept only)
        _null_result = minimize(lambda b: -np.sum(y_data * np.log(np.clip(1/(1+np.exp(-b[0])), 1e-10, 1-1e-10)) +
                                                   (1-y_data) * np.log(np.clip(1-1/(1+np.exp(-b[0])), 1e-10, 1-1e-10))),
                                 x0=[0], method='BFGS')
        _null_deviance = 2 * _neg_log_likelihood([_null_result.x[0], 0], x_data, y_data)
        _residual_deviance = 2 * _neg_log_likelihood(_mle_result.x, x_data, y_data)
        _aic = _residual_deviance + 2 * 2  # 2 parameters

        # McFadden's pseudo-R²
        _pseudo_r2 = 1 - (_residual_deviance / _null_deviance) if _null_deviance > 0 else 0

        # Plot MLE fitted curve
        _mle_z_line = _mle_intercept + _mle_slope * x_line
        _mle_prob_line = 1 / (1 + np.exp(-_mle_z_line))
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=_mle_prob_line,
                mode="lines",
                name="Fitted P(Y=1)",
                line=dict(color="black", width=3),
            )
        )

        # Add CI band if selected (uses delta method on log-odds scale)
        if show_ci.value and _cov_matrix is not None:
            z_crit = stats.norm.ppf((1 + ci_level.value) / 2)

            # Variance of linear predictor: Var(β₀ + β₁x) = Var(β₀) + 2x·Cov(β₀,β₁) + x²·Var(β₁)
            _var_z = np.array([
                _cov_matrix[0,0] + 2*x*_cov_matrix[0,1] + x**2*_cov_matrix[1,1]
                for x in x_line
            ])
            _se_z = np.sqrt(_var_z)

            # CI on log-odds scale
            _z_lower = _mle_z_line - z_crit * _se_z
            _z_upper = _mle_z_line + z_crit * _se_z

            # Transform to probability scale (ensures CI stays within [0,1])
            _prob_lower = 1 / (1 + np.exp(-_z_lower))
            _prob_upper = 1 / (1 + np.exp(-_z_upper))

            # Add CI band
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line, x_line[::-1]]),
                    y=np.concatenate([_prob_upper, _prob_lower[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,0,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{int(ci_level.value*100)}% CI',
                    showlegend=True,
                )
            )

        # Add threshold line
        fig.add_trace(
            go.Scatter(
                x=[0, 50],
                y=[threshold_slider.value, threshold_slider.value],
                mode="lines",
                name=f"Threshold ({threshold_slider.value:.2f})",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

        # Store for reg_stats
        _ols_intercept = _mle_intercept  # Using MLE as "OLS" for compatibility
        _ols_slope = _mle_slope
        _ols_y_pred = 1 / (1 + np.exp(-(_mle_intercept + _mle_slope * x_data)))
        _ols_residuals = y_data - _ols_y_pred  # Pearson residuals
        df = n - 2
        mse = np.sum(_ols_residuals**2) / df if df > 0 else 0
        se = np.sqrt(mse)
        ss_x = np.sum((x_data - np.mean(x_data))**2)
        _x_mean = np.mean(x_data)

        # Layout for logistic regression
        fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Logistic Regression Plot",
            xaxis=dict(
                title=x_label,
                range=[0, 50],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title=f"P({y_label}=1)",
                range=[-0.1, 1.1],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
                tickvals=[0, 0.25, 0.5, 0.75, 1],
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=80, l=50, r=50),
        )

    elif is_binary:
        # Binary mode: Generate two groups
        n_per_group = n // 2

        # Group 0 data
        y_group0 = intercept + np.random.normal(0, noise_slider.value, n_per_group)
        # Group 1 data
        y_group1 = (intercept + slope) + np.random.normal(0, noise_slider.value, n - n_per_group)

        # Combine for statistics
        x_data = np.concatenate([np.zeros(n_per_group), np.ones(n - n_per_group)])
        y_data = np.concatenate([y_group0, y_group1])

        # Run actual OLS regression on the data
        _ols_result = stats.linregress(x_data, y_data)
        _ols_slope = _ols_result.slope
        _ols_intercept = _ols_result.intercept

        # OLS predictions and residuals
        _ols_y_pred = _ols_intercept + _ols_slope * x_data
        _ols_residuals = y_data - _ols_y_pred

        # Calculate pooled statistics using OLS
        df = n - 2
        mse = np.sum(_ols_residuals**2) / df
        se = np.sqrt(mse)

        # Sum of squares for x (binary: 0s and 1s)
        _x_mean = np.mean(x_data)
        ss_x = np.sum((x_data - _x_mean)**2)

        # Standard errors for group means
        se_mean0 = se / np.sqrt(n_per_group)
        se_mean1 = se / np.sqrt(n - n_per_group)

        t_val_ci = stats.t.ppf((1 + ci_level.value) / 2, n - 2)
        t_val_pi = stats.t.ppf((1 + pi_level.value) / 2, n - 2)

        # Jitter for visualization
        jitter = 0.1
        x_group0_jittered = np.random.uniform(-jitter, jitter, n_per_group)
        x_group1_jittered = 1 + np.random.uniform(-jitter, jitter, n - n_per_group)

        # Add Group 0 points
        fig.add_trace(
            go.Scatter(
                x=x_group0_jittered,
                y=y_group0,
                mode="markers",
                name=g0_label,
                marker=dict(color="#636EFA", size=8, opacity=0.6),
            )
        )

        # Add Group 1 points
        fig.add_trace(
            go.Scatter(
                x=x_group1_jittered,
                y=y_group1,
                mode="markers",
                name=g1_label,
                marker=dict(color="#EF553B", size=8, opacity=0.6),
            )
        )

        # Add group means as larger markers
        mean0 = intercept
        mean1 = intercept + slope

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[mean0, mean1],
                mode="markers+lines",
                name="Group Means",
                marker=dict(color="black", size=15, symbol="diamond"),
                line=dict(color="black", width=2, dash="dash"),
            )
        )

        # CI error bars for means
        if show_ci.value:
            ci_half0 = t_val_ci * se_mean0
            ci_half1 = t_val_ci * se_mean1
            fig.add_trace(
                go.Scatter(
                    x=[0, 0, None, 1, 1],
                    y=[mean0 - ci_half0, mean0 + ci_half0, None, mean1 - ci_half1, mean1 + ci_half1],
                    mode="lines",
                    name=f"{int(ci_level.value*100)}% CI",
                    line=dict(color="rgba(99, 110, 250, 0.8)", width=4),
                )
            )

        # PI bands for each group
        if show_pi.value:
            pi_half = t_val_pi * se * np.sqrt(1 + 1/n_per_group)
            # Group 0 PI
            fig.add_shape(
                type="rect", x0=-0.3, x1=0.3, y0=mean0 - pi_half, y1=mean0 + pi_half,
                fillcolor="rgba(255, 165, 0, 0.2)", line=dict(width=0),
            )
            # Group 1 PI
            fig.add_shape(
                type="rect", x0=0.7, x1=1.3, y0=mean1 - pi_half, y1=mean1 + pi_half,
                fillcolor="rgba(255, 165, 0, 0.2)", line=dict(width=0),
            )
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="rgba(255, 165, 0, 0.3)"),
                name=f"{int(pi_level.value*100)}% PI"))

        # SD bands
        if show_sd.value:
            sd_mult = sd_multiplier.value
            sd_pct = (stats.norm.cdf(sd_mult) - stats.norm.cdf(-sd_mult)) * 100
            fig.add_shape(
                type="rect", x0=-0.3, x1=0.3, y0=mean0 - sd_mult*se, y1=mean0 + sd_mult*se,
                fillcolor="rgba(0, 200, 0, 0.2)", line=dict(width=0),
            )
            fig.add_shape(
                type="rect", x0=0.7, x1=1.3, y0=mean1 - sd_mult*se, y1=mean1 + sd_mult*se,
                fillcolor="rgba(0, 200, 0, 0.2)", line=dict(width=0),
            )
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="rgba(0, 200, 0, 0.3)"),
                name=f"±{sd_mult} SD ({sd_pct:.0f}% of data)"))

        # Layout for binary
        fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Regression Plot",
            xaxis=dict(
                title=x_label,
                range=[-0.5, 1.5],
                tickvals=[0, 1],
                ticktext=[g0_label, g1_label],
                zeroline=False,
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )
    else:
        # Continuous mode with transformations
        # Generate x from uniform or normal distribution (clipped to bounds)
        if is_uniform:
            x_data_raw = np.random.uniform(x_lo, x_hi, n)
        else:
            x_data_raw = np.clip(np.random.normal(x_mu, x_sigma, n), x_lo, x_hi)

        # For log transform, ensure positive values
        if transform == "Log (ln x)":
            x_data_raw = np.abs(x_data_raw) + 0.1
        # For sqrt transform, ensure non-negative
        elif transform == "Square Root (√x)":
            x_data_raw = np.abs(x_data_raw)

        # X range always uses min/max bounds
        X_MIN = x_lo
        X_MAX = x_hi
        if transform == "Log (ln x)" or transform == "Square Root (√x)":
            X_MIN = max(0.1, X_MIN)

        # Apply transformation to x for the model
        # Store statistics for scaling transforms
        _x_mean_raw = np.mean(x_data_raw)
        _x_std_raw = np.std(x_data_raw)
        _x_min_raw = np.min(x_data_raw)
        _x_max_raw = np.max(x_data_raw)

        if transform == "Center (x - mean)":
            x_data_transformed = x_data_raw - _x_mean_raw
        elif transform == "Standardize (z-score)":
            x_data_transformed = (x_data_raw - _x_mean_raw) / _x_std_raw
        elif transform == "Min-Max (0-1)":
            x_data_transformed = (x_data_raw - _x_min_raw) / (_x_max_raw - _x_min_raw)
        elif transform == "Square Root (√x)":
            x_data_transformed = np.sqrt(x_data_raw)
        elif transform == "Square (x²)":
            x_data_transformed = x_data_raw ** 2
        elif transform == "Log (ln x)":
            x_data_transformed = np.log(x_data_raw)
        elif transform == "Constant (no x)":
            x_data_transformed = np.zeros_like(x_data_raw)
        else:
            x_data_transformed = x_data_raw

        # Generate y using transformed x
        y_data = intercept + slope * x_data_transformed + np.random.normal(0, noise_slider.value, n)

        # For plotting the regression curve
        x_line_smooth = np.linspace(0, 50, 100)

        if transform == "Center (x - mean)":
            x_line_transformed = x_line_smooth - _x_mean_raw
        elif transform == "Standardize (z-score)":
            x_line_transformed = (x_line_smooth - _x_mean_raw) / _x_std_raw
        elif transform == "Min-Max (0-1)":
            x_line_transformed = (x_line_smooth - _x_min_raw) / (_x_max_raw - _x_min_raw)
        elif transform == "Square Root (√x)":
            x_line_transformed = np.sqrt(x_line_smooth)
        elif transform == "Square (x²)":
            x_line_transformed = x_line_smooth ** 2
        elif transform == "Log (ln x)":
            x_line_transformed = np.log(x_line_smooth)
        elif transform == "Constant (no x)":
            x_line_transformed = np.zeros_like(x_line_smooth)
        else:
            x_line_transformed = x_line_smooth

        y_line_smooth = intercept + slope * x_line_transformed

        # Calculate statistics for intervals (using transformed x)
        _x_mean = np.mean(x_data_transformed)

        # Run actual OLS regression on the data for summary statistics
        if transform == "Constant (no x)":
            # Intercept-only model
            _ols_intercept = np.mean(y_data)
            _ols_slope = 0.0
        else:
            # Use scipy's linregress for proper OLS
            _ols_result = stats.linregress(x_data_transformed, y_data)
            _ols_slope = _ols_result.slope
            _ols_intercept = _ols_result.intercept

        # Residuals from OLS fit (for proper statistics)
        _ols_y_pred = _ols_intercept + _ols_slope * x_data_transformed
        _ols_residuals = y_data - _ols_y_pred

        # Residuals from TRUE line (for visualization)
        y_pred = intercept + slope * x_data_transformed
        residuals = y_data - y_pred

        # Degrees of freedom depends on whether we have a slope
        df = n - 2 if transform != "Constant (no x)" else n - 1

        # Use OLS residuals for proper MSE/SE calculation
        mse = np.sum(_ols_residuals**2) / df
        se = np.sqrt(mse)

        # Sum of squares for transformed x
        ss_x = np.sum((x_data_transformed - _x_mean)**2)

        # Handle constant model (no x variation)
        if transform == "Constant (no x)" or ss_x == 0:
            se_fit = se / np.sqrt(n) * np.ones_like(x_line_smooth)
            se_pred = se * np.sqrt(1 + 1/n) * np.ones_like(x_line_smooth)
        else:
            # Standard error of the fitted values (for CI)
            se_fit = se * np.sqrt(1/n + (x_line_transformed - _x_mean)**2 / ss_x)
            # Standard error for prediction (for PI)
            se_pred = se * np.sqrt(1 + 1/n + (x_line_transformed - _x_mean)**2 / ss_x)

        # t-values for user-selected confidence levels
        t_val_ci = stats.t.ppf((1 + ci_level.value) / 2, df)
        t_val_pi = stats.t.ppf((1 + pi_level.value) / 2, df)

        # Add prediction interval (PI)
        if show_pi.value:
            pi_upper = y_line_smooth + t_val_pi * se_pred
            pi_lower = y_line_smooth - t_val_pi * se_pred
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line_smooth, x_line_smooth[::-1]]),
                    y=np.concatenate([pi_upper, pi_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255, 165, 0, 0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{int(pi_level.value*100)}% Prediction Interval",
                    hoverinfo="skip",
                )
            )

        # Add confidence interval (CI)
        if show_ci.value:
            ci_upper = y_line_smooth + t_val_ci * se_fit
            ci_lower = y_line_smooth - t_val_ci * se_fit
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line_smooth, x_line_smooth[::-1]]),
                    y=np.concatenate([ci_upper, ci_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(99, 110, 250, 0.3)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{int(ci_level.value*100)}% Confidence Interval",
                    hoverinfo="skip",
                )
            )

        # Add standard deviation band
        if show_sd.value:
            sd_mult = sd_multiplier.value
            sd_pct = (stats.norm.cdf(sd_mult) - stats.norm.cdf(-sd_mult)) * 100
            sd_upper = y_line_smooth + sd_mult * se
            sd_lower = y_line_smooth - sd_mult * se
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line_smooth, x_line_smooth[::-1]]),
                    y=np.concatenate([sd_upper, sd_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0, 200, 0, 0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"±{sd_mult} SD ({sd_pct:.0f}% of data)",
                    hoverinfo="skip",
                )
            )

        # Add scatter points (plot against raw x)
        fig.add_trace(
            go.Scatter(
                x=x_data_raw,
                y=y_data,
                mode="markers",
                name="Data Points",
                marker=dict(color="#636EFA", size=8, opacity=0.7),
            )
        )

        # Add regression line/curve
        fig.add_trace(
            go.Scatter(
                x=x_line_smooth,
                y=y_line_smooth,
                mode="lines",
                name="Regression Line" if transform == "None (x)" else "Regression Curve",
                line=dict(color="#EF553B", width=3),
            )
        )

        # Layout for continuous
        # x-axis title based on transformation
        if x_term is None:
            x_axis_title = x_label + " (not used in model)"
        else:
            x_axis_title = x_label

        fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Regression Plot",
            xaxis=dict(
                title=x_axis_title,
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.99,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )

    # Check if we should show 4D warning (continuous mode with 3rd predictor)
    if is_multiple and not has_grouping and has_cont3:
        plot_output = mo.md(f"""
### Cannot display graph with 3+ continuous predictors

With **{x_label}**, **{z_label}**, and **{w_label}** as predictors, the regression relationship exists in 4-dimensional space and cannot be represented in a 2D plot.

**The model equation and coefficient interpretations above are still valid.**

To visualize a subset of the relationship, disable the 3rd predictor checkbox.
""")
    else:
        plot_output = mo.ui.plotly(fig, config={"scrollZoom": False, "displayModeBar": False})

    # Return actual regression statistics for R summary
    # Available for all modes now
    if not is_multiple:
        reg_stats = {
            "n": n,
            "df": df,
            "mse": mse,
            "se": se,
            "ss_x": ss_x,
            "x_mean": _x_mean,
            "y_data": y_data,
            "ols_y_pred": _ols_y_pred,
            "ols_intercept": _ols_intercept,
            "ols_residuals": _ols_residuals,
            "is_binary": is_binary,
            "is_binned": is_binned,
            "is_multiple": False,
        }
        if is_logistic:
            reg_stats["x_data"] = x_data
            reg_stats["ols_slope"] = _ols_slope
            reg_stats["is_logistic"] = True
            reg_stats["mle_intercept"] = _mle_intercept
            reg_stats["mle_slope"] = _mle_slope
            reg_stats["se_intercept"] = _se_intercept
            reg_stats["se_slope"] = _se_slope
            reg_stats["null_deviance"] = _null_deviance
            reg_stats["residual_deviance"] = _residual_deviance
            reg_stats["aic"] = _aic
            reg_stats["pseudo_r2"] = _pseudo_r2
        elif is_binary:
            reg_stats["x_data"] = x_data
            reg_stats["ols_slope"] = _ols_slope
            reg_stats["g0_label"] = g0_label
            reg_stats["g1_label"] = g1_label
        elif is_binned:
            reg_stats["x_data"] = x_continuous
            reg_stats["bin_labels"] = bin_labels
            reg_stats["bin_coefs"] = _ols_bin_coefs
            reg_stats["se_betas"] = _se_betas
            reg_stats["n_bins"] = n_bins
        else:
            reg_stats["x_data"] = x_data_transformed
            reg_stats["ols_slope"] = _ols_slope
    else:
        # Multiple regression mode
        reg_stats = {
            "n": n,
            "df": _multi_df,
            "mse": _multi_mse,
            "se": _multi_se,
            "y_data": y_data,
            "ols_y_pred": _multi_y_pred,
            "ols_intercept": _multi_intercept,
            "ols_residuals": _multi_residuals,
            "is_binary": False,
            "is_binned": False,
            "is_multiple": True,
            "pred_names": _pred_names,
            "coefs": _multi_coefs,
            "se_betas": _multi_se_betas,
            "n_predictors": _n_predictors,
            "has_grouping": has_grouping,
        }
        # Add grouping data for colored residuals plot
        if has_grouping:
            reg_stats["group_data"] = group_data
            reg_stats["g0_label"] = g0_multi_label
            reg_stats["g1_label"] = g1_multi_label
    return plot_output, reg_stats


@app.cell
def _(coef_text, equation_output, go, intercept, mo, np, plot_output, reg_stats, slope, stats, transform, x_label, y_label):
    # Only show regression summary for basic linear regression
    if reg_stats is None:
        _r_summary = """### Regression Summary
*Summary statistics only available for Basic Linear Regression mode.*
"""
        # Use the original coef_text (which uses TRUE/slider values) for non-basic modes
        _coef_text_final = coef_text
        _resid_plot = None
        _is_logistic = False
    else:
        # Extract actual OLS regression results from the plotting cell
        _n = reg_stats["n"]
        _df = reg_stats["df"]
        _se = reg_stats["se"]
        _y_data = reg_stats["y_data"]
        _ols_y_pred = reg_stats["ols_y_pred"]
        _ols_intercept = reg_stats["ols_intercept"]
        _is_binned = reg_stats.get("is_binned", False)
        _is_multiple = reg_stats.get("is_multiple", False)
        _is_logistic = reg_stats.get("is_logistic", False)

        # Significance stars helper
        def _sig_stars(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            elif p < 0.1: return "."
            else: return ""

        # R² from actual OLS fit
        _ss_res = np.sum((_y_data - _ols_y_pred)**2)
        _ss_tot = np.sum((_y_data - np.mean(_y_data))**2)
        _r_squared = 1 - _ss_res / _ss_tot if _ss_tot > 0 else 0
        _r_squared = max(0, _r_squared)
        _adj_r_squared = 1 - (1 - _r_squared) * (_n - 1) / _df if _df > 0 else 0

        if _is_multiple:
            # Multiple regression mode - R-style output
            _pred_names = reg_stats["pred_names"]
            _coefs = reg_stats["coefs"]
            _se_betas = reg_stats["se_betas"]
            _n_predictors = reg_stats["n_predictors"]

            # t-values and p-values for intercept
            _t_intercept = _ols_intercept / _se_betas[0] if _se_betas[0] > 0 else 0
            _p_intercept = 2 * (1 - stats.t.cdf(abs(_t_intercept), _df)) if _df > 0 else 1

            # t-values and p-values for each predictor
            _t_preds = []
            _p_preds = []
            for _i in range(_n_predictors):
                _t_val = _coefs[_i] / _se_betas[_i + 1] if _se_betas[_i + 1] > 0 else 0
                _p_val = 2 * (1 - stats.t.cdf(abs(_t_val), _df)) if _df > 0 else 1
                _t_preds.append(_t_val)
                _p_preds.append(_p_val)

            # F-statistic for multiple regression (k predictors)
            if _df > 0 and _r_squared < 1:
                _f_stat = (_r_squared / _n_predictors) / ((1 - _r_squared) / _df)
                _f_pval = 1 - stats.f.cdf(_f_stat, _n_predictors, _df)
            else:
                _f_stat = 0
                _f_pval = 1

            # Build R-style coefficient table
            _coef_lines = [f"(Intercept)   {_ols_intercept:8.3f} {_se_betas[0]:7.3f} {_t_intercept:7.2f}  {_p_intercept:.2e} {_sig_stars(_p_intercept)}"]
            for _i in range(_n_predictors):
                _label = _pred_names[_i]
                _coef_lines.append(f"{_label:13s} {_coefs[_i]:8.3f} {_se_betas[_i + 1]:7.3f} {_t_preds[_i]:7.2f}  {_p_preds[_i]:.2e} {_sig_stars(_p_preds[_i])}")

            _coef_table = "\n".join(_coef_lines)
            _r_summary = f"""### Regression Summary (OLS on simulated data)
```
Coefficients:
              Estimate Std.Err t value  Pr(>|t|)
{_coef_table}
---
Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1

Residual SE: {_se:.3f} on {_df} df
R-squared: {_r_squared:.4f}, Adj R²: {_adj_r_squared:.4f}
F-statistic: {_f_stat:.2f} on {_n_predictors} and {_df} DF
p-value: {_f_pval:.2e}
```
"""
            # Use the original coef_text for multiple regression interpretation
            _coef_text_final = coef_text

            # Model fit section
            _model_fit = f"""
**Model Fit:**
- **Residual SE (σ̂) = {_se:.3f}** — the standard deviation of residuals
- **R² = {_r_squared:.3f}** — {_r_squared*100:.1f}% of variance in {y_label} is explained

*(True DGP: β₀ = {intercept:.1f}, β₁ = {slope:.1f})*
"""

        elif _is_logistic:
            # Logistic regression - MLE summary with R glm() style output
            _mle_intercept = reg_stats["mle_intercept"]
            _mle_slope = reg_stats["mle_slope"]
            _se_intercept = reg_stats["se_intercept"]
            _se_slope = reg_stats["se_slope"]
            _null_deviance = reg_stats["null_deviance"]
            _residual_deviance = reg_stats["residual_deviance"]
            _aic = reg_stats["aic"]
            _pseudo_r2 = reg_stats["pseudo_r2"]

            # z-values (using normal distribution, not t)
            _z_intercept = _mle_intercept / _se_intercept if _se_intercept > 0 and not np.isnan(_se_intercept) else 0
            _z_slope = _mle_slope / _se_slope if _se_slope > 0 and not np.isnan(_se_slope) else 0

            # p-values (two-tailed, normal distribution)
            _p_intercept = 2 * (1 - stats.norm.cdf(abs(_z_intercept)))
            _p_slope = 2 * (1 - stats.norm.cdf(abs(_z_slope)))

            # Odds ratio
            _odds_ratio = np.exp(_mle_slope)

            _r_summary = f"""### Regression Summary (MLE on simulated data)
```
Coefficients:
              Estimate Std.Err z value  Pr(>|z|)
(Intercept)   {_mle_intercept:8.3f} {_se_intercept:7.3f} {_z_intercept:7.2f}  {_p_intercept:.2e} {_sig_stars(_p_intercept)}
{x_label:13s} {_mle_slope:8.3f} {_se_slope:7.3f} {_z_slope:7.2f}  {_p_slope:.2e} {_sig_stars(_p_slope)}
---
Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1

(Dispersion parameter for binomial family taken to be 1)

Null deviance: {_null_deviance:.2f} on {_n - 1} df
Residual deviance: {_residual_deviance:.2f} on {_n - 2} df
AIC: {_aic:.2f}

Odds Ratio for {x_label}: {_odds_ratio:.3f}
McFadden's Pseudo-R²: {_pseudo_r2:.4f}
```
"""
            # Logistic coefficient interpretation using MLE estimates
            _prob_at_zero = 1 / (1 + np.exp(-_mle_intercept))

            if _mle_slope > 0:
                _direction = "increase"
                _odds_direction = "multiply"
            elif _mle_slope < 0:
                _direction = "decrease"
                _odds_direction = "divide"
            else:
                _direction = "no change in"
                _odds_direction = "unchanged"

            _coef_text_final = f"""### Coefficient Interpretation (MLE Estimates)

**Intercept (β₀ = {_mle_intercept:.3f}):**
When **{x_label}**=0, the log-odds of {y_label}=1 is **{_mle_intercept:.3f}**, corresponding to a probability of **{_prob_at_zero:.3f}**.

**Slope (β₁ = {_mle_slope:.3f}):**
For each 1-unit increase in **{x_label}**, the log-odds {_direction} by **{abs(_mle_slope):.3f}**.
**Odds ratio = {_odds_ratio:.3f}**: the odds {_odds_direction} by this factor.

**Odds Ratio Interpretation:**
- OR > 1: higher {x_label} → higher odds of {y_label}=1
- OR < 1: higher {x_label} → lower odds of {y_label}=1
- OR = 1: {x_label} has no effect

**Model Fit:**
- **Pseudo-R² = {_pseudo_r2:.3f}** — McFadden's measure of explained variation
- **AIC = {_aic:.1f}** — lower is better for model comparison

*(True DGP: β₀ = {intercept:.1f}, β₁ = {slope:.1f})*
"""

        elif _is_binned:
            # Binned mode with categorical dummies - R-style output
            _bin_labels = reg_stats["bin_labels"]
            _bin_coefs = reg_stats["bin_coefs"]
            _se_betas = reg_stats["se_betas"]
            _n_bins = reg_stats["n_bins"]

            # t-values and p-values for all coefficients
            _t_intercept = _ols_intercept / _se_betas[0] if _se_betas[0] > 0 else 0
            _p_intercept = 2 * (1 - stats.t.cdf(abs(_t_intercept), _df)) if _df > 0 else 1

            _t_bins = []
            _p_bins = []
            for _i in range(_n_bins - 1):
                _t_val = _bin_coefs[_i] / _se_betas[_i + 1] if _se_betas[_i + 1] > 0 else 0
                _p_val = 2 * (1 - stats.t.cdf(abs(_t_val), _df)) if _df > 0 else 1
                _t_bins.append(_t_val)
                _p_bins.append(_p_val)

            # F-statistic for binned model (k-1 predictors)
            _df_model = _n_bins - 1  # Number of dummy variables
            if _df > 0 and _r_squared < 1:
                _f_stat = (_r_squared / _df_model) / ((1 - _r_squared) / _df)
                _f_pval = 1 - stats.f.cdf(_f_stat, _df_model, _df)
            else:
                _f_stat = 0
                _f_pval = 1

            # Build R-style coefficient table
            _coef_lines = [f"(Intercept)   {_ols_intercept:8.3f} {_se_betas[0]:7.3f} {_t_intercept:7.2f}  {_p_intercept:.2e} {_sig_stars(_p_intercept)}"]
            for _i in range(_n_bins - 1):
                # Label like "xBin 2", "xBin 3", etc.
                _label = f"{x_label}{_bin_labels[_i + 1]}"
                _coef_lines.append(f"{_label:13s} {_bin_coefs[_i]:8.3f} {_se_betas[_i + 1]:7.3f} {_t_bins[_i]:7.2f}  {_p_bins[_i]:.2e} {_sig_stars(_p_bins[_i])}")

            _coef_table = "\n".join(_coef_lines)
            _r_summary = f"""### Regression Summary (OLS on simulated data)
```
Coefficients:
              Estimate Std.Err t value  Pr(>|t|)
{_coef_table}
---
Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1
Reference: {_bin_labels[0]} (baseline)

Residual SE: {_se:.3f} on {_df} df
R-squared: {_r_squared:.4f}, Adj R²: {_adj_r_squared:.4f}
F-statistic: {_f_stat:.2f} on {_df_model} and {_df} DF
p-value: {_f_pval:.2e}
```
"""
            # Set placeholders for non-binned variables (won't be used in binned interpretation)
            _ols_slope = 0
            _se_intercept = _se_betas[0]
            _se_slope = 0
        else:
            # Non-binned, non-multiple modes: extract slope and stats
            _ols_slope = reg_stats["ols_slope"]
            _ss_x = reg_stats["ss_x"]
            _x_mean = reg_stats["x_mean"]

            # Calculate standard errors from actual OLS
            if transform == "Constant (no x)" or _ss_x == 0:
                _se_intercept = _se / np.sqrt(_n)
                _se_slope = 0
            else:
                _se_intercept = _se * np.sqrt(1/_n + _x_mean**2 / _ss_x)
                _se_slope = _se / np.sqrt(_ss_x)

            # t-values using OLS ESTIMATED coefficients
            _t_intercept = _ols_intercept / _se_intercept if _se_intercept > 0 else 0
            _t_slope = _ols_slope / _se_slope if _se_slope > 0 else 0

            # p-values (two-tailed)
            _p_intercept = 2 * (1 - stats.t.cdf(abs(_t_intercept), _df)) if _df > 0 else 1
            _p_slope = 2 * (1 - stats.t.cdf(abs(_t_slope), _df)) if _df > 0 and _se_slope > 0 else 1

            # F-statistic
            if transform != "Constant (no x)" and _df > 0 and _r_squared < 1:
                _f_stat = (_r_squared / 1) / ((1 - _r_squared) / _df)
                _f_pval = 1 - stats.f.cdf(_f_stat, 1, _df)
            else:
                _f_stat = 0
                _f_pval = 1

            # Use appropriate label for predictor in summary table
            _predictor_label = x_label if not reg_stats.get("is_binary", False) else "Group"
            _r_summary = f"""### Regression Summary (OLS on simulated data)
```
Coefficients:
              Estimate Std.Err t value  Pr(>|t|)
(Intercept)   {_ols_intercept:8.3f} {_se_intercept:7.3f} {_t_intercept:7.2f}  {_p_intercept:.2e} {_sig_stars(_p_intercept)}
{_predictor_label:13s} {_ols_slope:8.3f} {_se_slope:7.3f} {_t_slope:7.2f}  {_p_slope:.2e} {_sig_stars(_p_slope)}
---
Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1

Residual SE: {_se:.3f} on {_df} df
R-squared: {_r_squared:.4f}, Adj R²: {_adj_r_squared:.4f}
F-statistic: {_f_stat:.2f} on 1 and {_df} DF
p-value: {_f_pval:.2e}
```
"""

        # Generate coefficient interpretation using OLS ESTIMATED values (not slider TRUE values)
        # Only for non-multiple, non-logistic regression modes (they have their own interpretation above)
        if not _is_multiple and not _is_logistic:
            # Model fit section with descriptions (same for all transforms)
            _model_fit = f"""
**Model Fit:**
- **Residual SE (σ̂) = {_se:.3f}** — the standard deviation of residuals; on average, predictions are off by about this much
- **R² = {_r_squared:.3f}** — {_r_squared*100:.1f}% of the variance in {y_label} is explained by the model

*(True DGP: β₀ = {intercept:.1f}, β₁ = {slope:.1f})*
"""

            if reg_stats.get("is_binary", False):
                # Binary mode - means comparison interpretation (check first, ignores transform)
                _g0_label = reg_stats["g0_label"]
                _g1_label = reg_stats["g1_label"]
                _intercept_interp = f"The estimated mean of **{y_label}** for **{_g0_label}** is **{_ols_intercept:.2f}**."

                if _ols_slope > 0:
                    _direction = "higher"
                elif _ols_slope < 0:
                    _direction = "lower"
                else:
                    _direction = "the same as"

                if _ols_slope != 0:
                    _slope_interp = f"**{_g1_label}** has a mean **{y_label}** that is **{abs(_ols_slope):.2f}** units {_direction} than **{_g0_label}**. (Estimated mean for {_g1_label} = {_ols_intercept + _ols_slope:.2f})"
                else:
                    _slope_interp = f"**{_g1_label}** and **{_g0_label}** have approximately the same mean **{y_label}** (no group difference)."

                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Group Difference (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero; |t| > 2 suggests significance*
{_model_fit}"""
            elif reg_stats.get("is_binned", False):
                # Binned mode - categorical dummy interpretation (check before transforms)
                _bin_labels = reg_stats["bin_labels"]
                _bin_coefs = reg_stats["bin_coefs"]
                _se_betas = reg_stats["se_betas"]
                _n_bins = reg_stats["n_bins"]

                _intercept_interp = f"The estimated mean of **{y_label}** for **{_bin_labels[0]}** (reference group) is **{_ols_intercept:.2f}**."

                # Build interpretation for each bin coefficient
                _bin_interps = []
                for _bi in range(_n_bins - 1):
                    _coef = _bin_coefs[_bi]
                    _se_bin = _se_betas[_bi + 1]
                    _t_bin = _coef / _se_bin if _se_bin > 0 else 0
                    _mean_bin = _ols_intercept + _coef

                    if _coef > 0:
                        _dir = "higher"
                    elif _coef < 0:
                        _dir = "lower"
                    else:
                        _dir = "the same as"

                    _bin_interps.append(f"- **{_bin_labels[_bi + 1]}**: mean = **{_mean_bin:.2f}** ({abs(_coef):.2f} units {_dir} than reference, t={_t_bin:.2f})")

                _bins_text = "\n".join(_bin_interps)

                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Bin Effects (relative to {_bin_labels[0]}):**
{_bins_text}

- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Constant (no x)":
                # Constant model - intercept only
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
The predicted value of **{y_label}** is always **{_ols_intercept:.2f}**, regardless of any predictor. This is simply the mean of {y_label}.
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Center (x - mean)":
                _intercept_interp = f"When **{x_label}** equals its mean (x̄), the predicted value of **{y_label}** is **{_ols_intercept:.2f}**. (Centering shifts the intercept to the mean of x.)"
                if _ols_slope > 0:
                    _direction = "increases"
                elif _ols_slope < 0:
                    _direction = "decreases"
                else:
                    _direction = "stays the same"
                _slope_interp = f"For every 1-unit increase in **{x_label}**, **{y_label}** {_direction} by **{abs(_ols_slope):.2f}** units. (Same as uncentered; centering only changes the intercept.)"
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Standardize (z-score)":
                _intercept_interp = f"When **{x_label}** equals its mean (z=0), the predicted value of **{y_label}** is **{_ols_intercept:.2f}**."
                if _ols_slope > 0:
                    _direction = "increases"
                elif _ols_slope < 0:
                    _direction = "decreases"
                else:
                    _direction = "stays the same"
                _slope_interp = f"For every **1 standard deviation** increase in **{x_label}**, **{y_label}** {_direction} by **{abs(_ols_slope):.2f}** units. (Standardized coefficient allows comparison across different scales.)"
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Min-Max (0-1)":
                _intercept_interp = f"When **{x_label}** is at its minimum (scaled=0), the predicted value of **{y_label}** is **{_ols_intercept:.2f}**."
                if _ols_slope > 0:
                    _direction = "increases"
                elif _ols_slope < 0:
                    _direction = "decreases"
                else:
                    _direction = "stays the same"
                _slope_interp = f"Moving from the **minimum to maximum** of **{x_label}** (0→1 in scaled units), **{y_label}** {_direction} by **{abs(_ols_slope):.2f}** units total."
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Square Root (√x)":
                _intercept_interp = f"When **{x_label}** equals 0 (so √{x_label}=0), the predicted value of **{y_label}** is **{_ols_intercept:.2f}**."
                if _ols_slope > 0:
                    _direction = "increases"
                elif _ols_slope < 0:
                    _direction = "decreases"
                else:
                    _direction = "stays the same"
                _slope_interp = f"For every 1-unit increase in **√{x_label}**, **{y_label}** {_direction} by **{abs(_ols_slope):.2f}** units."
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Square (x²)":
                _intercept_interp = f"When **{x_label}** equals 0, the predicted value of **{y_label}** is **{_ols_intercept:.2f}**."
                if _ols_slope > 0:
                    _direction = "increases"
                elif _ols_slope < 0:
                    _direction = "decreases"
                else:
                    _direction = "stays the same"
                _slope_interp = f"For every 1-unit increase in **{x_label}²**, **{y_label}** {_direction} by **{abs(_ols_slope):.2f}** units."
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            elif transform == "Log (ln x)":
                _intercept_interp = f"When **ln({x_label})=0** (i.e., {x_label}=1), the predicted value of **{y_label}** is **{_ols_intercept:.2f}**."
                if _ols_slope > 0:
                    _direction = "increases"
                elif _ols_slope < 0:
                    _direction = "decreases"
                else:
                    _direction = "stays the same"
                _slope_interp = f"For every 1-unit increase in **ln({x_label})**, **{y_label}** {_direction} by **{abs(_ols_slope):.2f}** units. A 1% increase in {x_label} corresponds to ~**{_ols_slope/100:.4f}** units change in {y_label}."
                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""
            else:
                # No transformation - standard linear regression
                _intercept_interp = f"When **{x_label}** equals 0, the predicted value of **{y_label}** is **{_ols_intercept:.2f}**."
                if _ols_slope > 0:
                    _direction = "increase"
                elif _ols_slope < 0:
                    _direction = "decrease"
                else:
                    _direction = "no change"

                if _ols_slope != 0:
                    _slope_interp = f"For every 1-unit increase in **{x_label}**, **{y_label}** is expected to {_direction} by **{abs(_ols_slope):.2f}** units."
                else:
                    _slope_interp = f"Changes in **{x_label}** have no effect on **{y_label}** (slope ≈ 0)."

                _coef_text_final = f"""### Coefficient Interpretation (OLS Estimates)

**Intercept (β₀ = {_ols_intercept:.2f}):**
{_intercept_interp}

**Slope (β₁ = {_ols_slope:.2f}):**
{_slope_interp}
- *SE: standard error of the estimate; smaller = more precise*
- *t = β/SE: how many SEs the coefficient is from zero*
{_model_fit}"""

    # Create Residuals vs Fitted plot using OLS values (calibration plot for logistic)
    if _is_logistic:
        # Calibration plot for logistic regression
        _pred_probs = reg_stats["ols_y_pred"]  # MLE predicted probabilities
        _y_binary = reg_stats["y_data"]  # Actual 0/1 outcomes

        # Bin predictions into ~10 groups
        _n_bins = 10
        _bin_edges = np.linspace(0, 1, _n_bins + 1)
        _bin_centers = []
        _observed_rates = []
        _bin_counts = []

        for _i in range(_n_bins):
            _mask = (_pred_probs >= _bin_edges[_i]) & (_pred_probs < _bin_edges[_i+1])
            if _i == _n_bins - 1:  # Include right edge for last bin
                _mask = (_pred_probs >= _bin_edges[_i]) & (_pred_probs <= _bin_edges[_i+1])
            if np.sum(_mask) > 0:
                _bin_centers.append(np.mean(_pred_probs[_mask]))
                _observed_rates.append(np.mean(_y_binary[_mask]))
                _bin_counts.append(np.sum(_mask))

        _calib_fig = go.Figure()

        # Perfect calibration line (diagonal)
        _calib_fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="gray", dash="dash", width=2),
                name="Perfect Calibration",
            )
        )

        # Actual calibration points (sized by count)
        _calib_fig.add_trace(
            go.Scatter(
                x=_bin_centers,
                y=_observed_rates,
                mode="markers",
                marker=dict(
                    color="#636EFA",
                    size=[max(10, min(25, c/4)) for c in _bin_counts],
                    opacity=0.8
                ),
                name="Observed Rate",
            )
        )

        _calib_fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Calibration Plot",
            xaxis=dict(
                title="Predicted Probability",
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title="Observed Rate",
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                gridcolor="lightgray",
                fixedrange=True,
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )
        _resid_plot = mo.ui.plotly(_calib_fig, config={"scrollZoom": False, "displayModeBar": False})
    else:
        _ols_residuals = reg_stats["ols_residuals"]
        _resid_fig = go.Figure()

        # Check if we have grouping data for colored residuals
        if reg_stats.get("is_multiple") and reg_stats.get("has_grouping") and "group_data" in reg_stats:
            _group_data = reg_stats["group_data"]
            _g0_label = reg_stats["g0_label"]
            _g1_label = reg_stats["g1_label"]
            # Add separate traces for each group with different colors
            _mask_g0 = _group_data == 0
            _mask_g1 = _group_data == 1
            _resid_fig.add_trace(
                go.Scatter(
                    x=_ols_y_pred[_mask_g0],
                    y=_ols_residuals[_mask_g0],
                    mode="markers",
                    marker=dict(color="#636EFA", size=8, opacity=0.7),
                    name=_g0_label,
                )
            )
            _resid_fig.add_trace(
                go.Scatter(
                    x=_ols_y_pred[_mask_g1],
                    y=_ols_residuals[_mask_g1],
                    mode="markers",
                    marker=dict(color="#EF553B", size=8, opacity=0.7),
                    name=_g1_label,
                )
            )
        elif reg_stats.get("is_binary"):
            # Binary mode also has grouping
            _g0_label = reg_stats["g0_label"]
            _g1_label = reg_stats["g1_label"]
            _x_data = reg_stats.get("x_data")
            if _x_data is not None:
                _mask_g0 = _x_data == 0
                _mask_g1 = _x_data == 1
                _resid_fig.add_trace(
                    go.Scatter(
                        x=_ols_y_pred[_mask_g0],
                        y=_ols_residuals[_mask_g0],
                        mode="markers",
                        marker=dict(color="#636EFA", size=8, opacity=0.7),
                        name=_g0_label,
                    )
                )
                _resid_fig.add_trace(
                    go.Scatter(
                        x=_ols_y_pred[_mask_g1],
                        y=_ols_residuals[_mask_g1],
                        mode="markers",
                        marker=dict(color="#EF553B", size=8, opacity=0.7),
                        name=_g1_label,
                    )
                )
            else:
                _resid_fig.add_trace(
                    go.Scatter(
                        x=_ols_y_pred,
                        y=_ols_residuals,
                        mode="markers",
                        marker=dict(color="#636EFA", size=8, opacity=0.7),
                        name="Residuals",
                    )
                )
        else:
            # Default single-color residuals
            _resid_fig.add_trace(
                go.Scatter(
                    x=_ols_y_pred,
                    y=_ols_residuals,
                    mode="markers",
                    marker=dict(color="#636EFA", size=8, opacity=0.7),
                    name="Residuals",
                )
            )
        # Add horizontal line at y=0
        _resid_fig.add_hline(y=0, line_dash="dash", line_color="red", line_width=2)
        # Fixed axis ranges for residual plot
        _resid_fig.update_layout(
            template="plotly_white",
            width=600,
            height=600,
            dragmode=False,
            title="Residuals vs Fitted",
            xaxis=dict(
                title="Fitted Values (ŷ)",
                range=[0, 50],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            yaxis=dict(
                title="Residuals (y - ŷ)",
                range=[-50, 50],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
                fixedrange=True,
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )
        _resid_plot = mo.ui.plotly(_resid_fig, config={"scrollZoom": False, "displayModeBar": False})

    # Create side-by-side layout for interpretation and summary
    _left_col = mo.md(_coef_text_final)
    _right_col = mo.md(_r_summary)
    _bottom_row = mo.hstack([_left_col, _right_col], widths=[1, 1], gap=4, align="start")

    # Create plots row with subtitles: regression plot | residuals/calibration plot (if available)
    if _resid_plot is not None:
        _reg_subtitle = mo.md("*Shows the relationship between predictor and outcome. Look for: linear pattern, spread of points around line.*")
        if _is_logistic:
            _resid_subtitle = mo.md("*Calibration plot: Points on the diagonal = well-calibrated. Above diagonal = underconfident (predicted too low). Below = overconfident (predicted too high).*")
        else:
            _resid_subtitle = mo.md("*Checks model assumptions. Look for: random scatter around zero (good), patterns/funnel shapes (bad — suggests non-linearity or heteroscedasticity).*")
        _reg_with_subtitle = mo.vstack([plot_output, _reg_subtitle], gap=1)
        _resid_with_subtitle = mo.vstack([_resid_plot, _resid_subtitle], gap=1)
        _plots_row = mo.hstack([_reg_with_subtitle, _resid_with_subtitle], widths=[1, 1], gap=2, justify="center")
    else:
        _plots_row = plot_output

    # Layout: Centered equation -> Plots -> Interpretation | Summary
    mo.vstack([
        equation_output,
        _plots_row,
        _bottom_row
    ], gap=2, align="stretch")
    return


if __name__ == "__main__":
    app.run()
