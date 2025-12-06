import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import statsmodels.api as sm
    return go, make_subplots, mo, np, pd, sm


@app.cell
def _(mo):
    mo.md("# Interactive Linear Regression Playground")
    return


@app.cell
def _(mo):
    mo.md("### Model Complexity")
    return


@app.cell
def _(mo):
    # Master toggle
    all_predictors = mo.ui.checkbox(value=False, label="Select All Predictors")

    # Predictor checkboxes
    include_x1 = mo.ui.checkbox(value=True, label="x₁")
    include_x2 = mo.ui.checkbox(value=False, label="x₂")
    include_x3 = mo.ui.checkbox(value=False, label="x₃")
    include_group = mo.ui.checkbox(value=False, label="Group")
    # Predictor × Group interactions
    include_x1_group = mo.ui.checkbox(value=False, label="x₁ × Group")
    include_x2_group = mo.ui.checkbox(value=False, label="x₂ × Group")
    include_x3_group = mo.ui.checkbox(value=False, label="x₃ × Group")
    # Predictor × Predictor interactions
    include_x1_x2 = mo.ui.checkbox(value=False, label="x₁ × x₂")
    include_x1_x3 = mo.ui.checkbox(value=False, label="x₁ × x₃")
    include_x2_x3 = mo.ui.checkbox(value=False, label="x₂ × x₃")
    return (
        all_predictors,
        include_group,
        include_x1,
        include_x1_group,
        include_x1_x2,
        include_x1_x3,
        include_x2,
        include_x2_group,
        include_x2_x3,
        include_x3,
        include_x3_group,
    )


@app.cell
def _(
    all_predictors,
    include_group,
    include_x1,
    include_x1_group,
    include_x1_x2,
    include_x1_x3,
    include_x2,
    include_x2_group,
    include_x2_x3,
    include_x3,
    include_x3_group,
    mo,
):
    predictor_tip = mo.md("Continuous predictor variable included in the regression model")
    group_tip = mo.md("Binary categorical variable (0 or 1) that splits data into two groups.")
    group_interaction_tip = mo.md("Tests whether the effect of this predictor differs between groups (moderation).")
    predictor_interaction_tip = mo.md("Tests whether the effect of one predictor depends on another.")

    mo.hstack(
        [
            mo.vstack([
                mo.md("**Options:**"),
                all_predictors
            ], justify="start"),
            mo.vstack([
                mo.md("**Predictors:**"),
                mo.hstack([include_x1, include_x2, include_x3, mo.accordion({"ℹ️": predictor_tip})], align="center"),
            ], justify="start"),
            mo.vstack([
                mo.md("**Group:**"),
                mo.hstack([include_group, mo.accordion({"ℹ️": group_tip})], align="center"),
            ], justify="start"),
            mo.vstack([
                mo.md("**Predictor × Group:**"),
                mo.hstack([include_x1_group, include_x2_group, include_x3_group, mo.accordion({"ℹ️": group_interaction_tip})], align="center"),
            ], justify="start"),
            mo.vstack([
                mo.md("**Predictor × Predictor:**"),
                mo.hstack([include_x1_x2, include_x1_x3, include_x2_x3, mo.accordion({"ℹ️": predictor_interaction_tip})], align="center"),
            ], justify="start"),
        ],
        justify="start",
        gap=4,
    )
    return group_interaction_tip, group_tip, predictor_interaction_tip, predictor_tip


@app.cell
def _(mo):
    mo.md("### Predictor Names")
    return


@app.cell
def _(mo):
    x1_name = mo.ui.text(value="x₁", label="x₁ name:", placeholder="e.g., Age")
    x2_name = mo.ui.text(value="x₂", label="x₂ name:", placeholder="e.g., Income")
    x3_name = mo.ui.text(value="x₃", label="x₃ name:", placeholder="e.g., Education")
    group_name = mo.ui.text(value="Group", label="Group name:", placeholder="e.g., Treatment")
    y_name = mo.ui.text(value="y", label="y name:", placeholder="e.g., Salary")
    return group_name, x1_name, x2_name, x3_name, y_name


@app.cell
def _(
    all_predictors,
    group_name,
    include_group,
    include_x1,
    include_x2,
    include_x3,
    mo,
    x1_name,
    x2_name,
    x3_name,
    y_name,
):
    # Logic to determine active inputs based on individual checks OR Select All
    show_x1 = include_x1.value or all_predictors.value
    show_x2 = include_x2.value or all_predictors.value
    show_x3 = include_x3.value or all_predictors.value
    show_group = include_group.value or all_predictors.value

    name_inputs = [y_name]  # Always show y name
    if show_x1:
        name_inputs.append(x1_name)
    if show_x2:
        name_inputs.append(x2_name)
    if show_x3:
        name_inputs.append(x3_name)
    if show_group:
        name_inputs.append(group_name)
    mo.hstack(name_inputs, justify="start", gap=2)
    return name_inputs, show_group, show_x1, show_x2, show_x3


@app.cell
def _(mo):
    mo.md("### Sample & Noise Parameters")
    return


@app.cell
def _(mo):
    sample_size_slider = mo.ui.slider(
        start=20, stop=500, step=10, value=100, label="Sample Size (n)"
    )
    noise_slider = mo.ui.slider(
        start=0.1, stop=5, step=0.1, value=1.0, label="Noise (σ)"
    )

    sample_size_def = mo.md("**Sample Size (n):** Number of observations. Larger samples give more precise estimates.")
    noise_def = mo.md("**Noise (σ):** Standard deviation of error term ε. Higher = more scatter around regression line.")
    return noise_def, noise_slider, sample_size_def, sample_size_slider


@app.cell
def _(mo, noise_def, noise_slider, sample_size_def, sample_size_slider):
    mo.hstack(
        [
            mo.hstack([sample_size_slider, mo.accordion({"ℹ️": sample_size_def})], align="center"),
            mo.hstack([noise_slider, mo.accordion({"ℹ️": noise_def})], align="center"),
        ],
        justify="start",
        gap=4,
    )
    return


@app.cell
def _(mo):
    mo.md("""### Predictor Distributions & Transformations
*Mean = center of distribution | SD = spread | Transform: Log, Sqrt, Squared, Standardize | Bins = discretize into categories*""")
    return


@app.cell
def _(mo):
    # Distribution sliders
    x1_mean_slider = mo.ui.slider(start=-5, stop=5, step=0.5, value=0.0, label="x₁ Mean")
    x1_sd_slider = mo.ui.slider(start=0.5, stop=5, step=0.25, value=1.0, label="x₁ SD")
    x2_mean_slider = mo.ui.slider(start=-5, stop=5, step=0.5, value=0.0, label="x₂ Mean")
    x2_sd_slider = mo.ui.slider(start=0.5, stop=5, step=0.25, value=1.0, label="x₂ SD")
    x3_mean_slider = mo.ui.slider(start=-5, stop=5, step=0.5, value=0.0, label="x₃ Mean")
    x3_sd_slider = mo.ui.slider(start=0.5, stop=5, step=0.25, value=1.0, label="x₃ SD")

    # Transformation dropdowns
    transform_options = ["None", "Log", "Square Root", "Squared", "Standardize"]
    x1_transform = mo.ui.dropdown(options=transform_options, value="None", label="x₁ Transform")
    x2_transform = mo.ui.dropdown(options=transform_options, value="None", label="x₂ Transform")
    x3_transform = mo.ui.dropdown(options=transform_options, value="None", label="x₃ Transform")

    # Binning options
    bin_options = ["None", "2 bins (median)", "3 bins (tertiles)", "4 bins (quartiles)", "5 bins (quintiles)"]
    x1_bins = mo.ui.dropdown(options=bin_options, value="None", label="x₁ Bins")
    x2_bins = mo.ui.dropdown(options=bin_options, value="None", label="x₂ Bins")
    x3_bins = mo.ui.dropdown(options=bin_options, value="None", label="x₃ Bins")

    return (
        bin_options,
        transform_options,
        x1_bins,
        x1_mean_slider,
        x1_sd_slider,
        x1_transform,
        x2_bins,
        x2_mean_slider,
        x2_sd_slider,
        x2_transform,
        x3_bins,
        x3_mean_slider,
        x3_sd_slider,
        x3_transform,
    )


@app.cell
def _(
    mo,
    show_x1,
    show_x2,
    show_x3,
    x1_bins,
    x1_mean_slider,
    x1_sd_slider,
    x1_transform,
    x2_bins,
    x2_mean_slider,
    x2_sd_slider,
    x2_transform,
    x3_bins,
    x3_mean_slider,
    x3_sd_slider,
    x3_transform,
):
    dist_rows = []
    if show_x1:
        dist_rows.append(
            mo.hstack(
                [x1_mean_slider, x1_sd_slider, x1_transform, x1_bins],
                justify="start",
                align="center",
                gap=2,
            )
        )
    if show_x2:
        dist_rows.append(
            mo.hstack(
                [x2_mean_slider, x2_sd_slider, x2_transform, x2_bins],
                justify="start",
                align="center",
                gap=2,
            )
        )
    if show_x3:
        dist_rows.append(
            mo.hstack(
                [x3_mean_slider, x3_sd_slider, x3_transform, x3_bins],
                justify="start",
                align="center",
                gap=2,
            )
        )
    if dist_rows:
        mo.vstack(dist_rows, justify="start")
    else:
        mo.md("*Enable predictors above*")
    return (dist_rows,)


@app.cell
def _(mo):
    mo.md("### True Coefficients")
    return


@app.cell
def _(mo):
    intercept_slider = mo.ui.slider(
        start=-10, stop=10, step=0.5, value=2.0, label="Intercept (β₀)"
    )
    b1_slider = mo.ui.slider(start=-5, stop=5, step=0.25, value=1.5, label="β₁")
    b2_slider = mo.ui.slider(start=-5, stop=5, step=0.25, value=1.0, label="β₂")
    b3_slider = mo.ui.slider(start=-5, stop=5, step=0.25, value=0.5, label="β₃")
    b_group_slider = mo.ui.slider(start=-5, stop=5, step=0.25, value=1.0, label="β_group")
    b1_group_slider = mo.ui.slider(start=-3, stop=3, step=0.25, value=0.5, label="β₁×group")
    b2_group_slider = mo.ui.slider(start=-3, stop=3, step=0.25, value=0.3, label="β₂×group")
    b3_group_slider = mo.ui.slider(start=-3, stop=3, step=0.25, value=0.2, label="β₃×group")
    # Predictor × Predictor interaction sliders
    b_x1_x2_slider = mo.ui.slider(start=-3, stop=3, step=0.25, value=0.4, label="β₁₂ (x₁×x₂)")
    b_x1_x3_slider = mo.ui.slider(start=-3, stop=3, step=0.25, value=0.3, label="β₁₃ (x₁×x₃)")
    b_x2_x3_slider = mo.ui.slider(start=-3, stop=3, step=0.25, value=0.2, label="β₂₃ (x₂×x₃)")

    # Definitions
    intercept_def = mo.md("**Intercept (β₀):** Predicted y when all predictors=0 (and Group=0). Where the line crosses y-axis.")
    slope_def = mo.md("**Slope (βᵢ):** Change in y per one-unit increase in predictor, holding others constant. Positive=upward, negative=downward.")
    group_effect_def = mo.md("**Group Effect:** Difference in intercept between Group 1 and Group 0. Shifts line up/down for Group 1.")
    group_interaction_def = mo.md("**Predictor × Group:** How much the slope differs between groups. Zero=parallel lines (no interaction).")
    predictor_interaction_def = mo.md("**Predictor × Predictor:** How the effect of one predictor changes as another predictor changes. Creates curved response surfaces.")
    return (
        b1_group_slider,
        b1_slider,
        b2_group_slider,
        b2_slider,
        b3_group_slider,
        b3_slider,
        b_group_slider,
        b_x1_x2_slider,
        b_x1_x3_slider,
        b_x2_x3_slider,
        group_effect_def,
        group_interaction_def,
        intercept_def,
        intercept_slider,
        predictor_interaction_def,
        slope_def,
    )


@app.cell
def _(
    all_predictors,
    b1_group_slider,
    b1_slider,
    b2_group_slider,
    b2_slider,
    b3_group_slider,
    b3_slider,
    b_group_slider,
    b_x1_x2_slider,
    b_x1_x3_slider,
    b_x2_x3_slider,
    group_effect_def,
    group_interaction_def,
    include_group,
    include_x1,
    include_x1_group,
    include_x1_x2,
    include_x1_x3,
    include_x2,
    include_x2_group,
    include_x2_x3,
    include_x3,
    include_x3_group,
    intercept_def,
    intercept_slider,
    mo,
    predictor_interaction_def,
    slope_def,
):
    # Logic to resolve "Select All"
    s_x1 = include_x1.value or all_predictors.value
    s_x2 = include_x2.value or all_predictors.value
    s_x3 = include_x3.value or all_predictors.value
    s_g = include_group.value or all_predictors.value
    s_x1g = include_x1_group.value or all_predictors.value
    s_x2g = include_x2_group.value or all_predictors.value
    s_x3g = include_x3_group.value or all_predictors.value
    s_x1x2 = include_x1_x2.value or all_predictors.value
    s_x1x3 = include_x1_x3.value or all_predictors.value
    s_x2x3 = include_x2_x3.value or all_predictors.value

    coef_rows = []
    defs_needed = set()

    coef_rows.append(intercept_slider)
    defs_needed.add("intercept")

    if s_x1:
        coef_rows.append(b1_slider)
        defs_needed.add("slope")
    if s_x2:
        coef_rows.append(b2_slider)
        defs_needed.add("slope")
    if s_x3:
        coef_rows.append(b3_slider)
        defs_needed.add("slope")
    if s_g:
        coef_rows.append(b_group_slider)
        defs_needed.add("group")
    if s_x1g:
        coef_rows.append(b1_group_slider)
        defs_needed.add("group_interaction")
    if s_x2g:
        coef_rows.append(b2_group_slider)
        defs_needed.add("group_interaction")
    if s_x3g:
        coef_rows.append(b3_group_slider)
        defs_needed.add("group_interaction")
    if s_x1x2:
        coef_rows.append(b_x1_x2_slider)
        defs_needed.add("predictor_interaction")
    if s_x1x3:
        coef_rows.append(b_x1_x3_slider)
        defs_needed.add("predictor_interaction")
    if s_x2x3:
        coef_rows.append(b_x2_x3_slider)
        defs_needed.add("predictor_interaction")

    # Add relevant definitions
    def_accordions = []
    if "intercept" in defs_needed:
        def_accordions.append(mo.accordion({"ℹ️ Intercept": intercept_def}))
    if "slope" in defs_needed:
        def_accordions.append(mo.accordion({"ℹ️ Slope": slope_def}))
    if "group" in defs_needed:
        def_accordions.append(mo.accordion({"ℹ️ Group": group_effect_def}))
    if "group_interaction" in defs_needed:
        def_accordions.append(mo.accordion({"ℹ️ Pred×Group": group_interaction_def}))
    if "predictor_interaction" in defs_needed:
        def_accordions.append(mo.accordion({"ℹ️ Pred×Pred": predictor_interaction_def}))

    mo.vstack([
        mo.vstack(coef_rows, justify="start"),
        mo.hstack(def_accordions, justify="start", gap=1),
    ], justify="start")
    return (
        coef_rows,
        def_accordions,
        defs_needed,
        s_g,
        s_x1,
        s_x1g,
        s_x1x2,
        s_x1x3,
        s_x2,
        s_x2g,
        s_x2x3,
        s_x3,
        s_x3g,
    )


@app.cell
def _(
    group_name,
    mo,
    s_g,
    s_x1,
    s_x1g,
    s_x1x2,
    s_x1x3,
    s_x2,
    s_x2g,
    s_x2x3,
    s_x3,
    s_x3g,
    x1_name,
    x2_name,
    x3_name,
    y_name,
):
    # Dynamic model equation with custom names
    n1 = x1_name.value or "x₁"
    n2 = x2_name.value or "x₂"
    n3 = x3_name.value or "x₃"
    ng = group_name.value or "Group"
    ny = y_name.value or "y"

    terms = ["β₀"]
    if s_x1:
        terms.append(f"β₁·{n1}")
    if s_x2:
        terms.append(f"β₂·{n2}")
    if s_x3:
        terms.append(f"β₃·{n3}")
    if s_g:
        terms.append(f"β_g·{ng}")
    if s_x1g:
        terms.append(f"β₁g·({n1}·{ng})")
    if s_x2g:
        terms.append(f"β₂g·({n2}·{ng})")
    if s_x3g:
        terms.append(f"β₃g·({n3}·{ng})")
    if s_x1x2:
        terms.append(f"β₁₂·({n1}·{n2})")
    if s_x1x3:
        terms.append(f"β₁₃·({n1}·{n3})")
    if s_x2x3:
        terms.append(f"β₂₃·({n2}·{n3})")
    terms.append("ε")

    model_eq = f"**Model:** {ny} = " + " + ".join(terms)
    mo.md(model_eq)
    return model_eq, n1, n2, n3, ng, ny, terms


@app.cell
def _(np):
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
        """Apply binning to discretize continuous predictor into categories."""
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

    return (apply_binning, apply_transform)


@app.cell
def _(
    apply_binning,
    apply_transform,
    b1_group_slider,
    b1_slider,
    b2_group_slider,
    b2_slider,
    b3_group_slider,
    b3_slider,
    b_group_slider,
    b_x1_x2_slider,
    b_x1_x3_slider,
    b_x2_x3_slider,
    intercept_slider,
    noise_slider,
    np,
    pd,
    s_g,
    s_x1,
    s_x1g,
    s_x1x2,
    s_x1x3,
    s_x2,
    s_x2g,
    s_x2x3,
    s_x3,
    s_x3g,
    sample_size_slider,
    x1_bins,
    x1_mean_slider,
    x1_sd_slider,
    x1_transform,
    x2_bins,
    x2_mean_slider,
    x2_sd_slider,
    x2_transform,
    x3_bins,
    x3_mean_slider,
    x3_sd_slider,
    x3_transform,
):
    np.random.seed(42)

    n = sample_size_slider.value
    noise_std = noise_slider.value

    # True coefficients
    true_intercept = intercept_slider.value
    true_b1 = b1_slider.value if s_x1 else 0.0
    true_b2 = b2_slider.value if s_x2 else 0.0
    true_b3 = b3_slider.value if s_x3 else 0.0
    true_b_group = b_group_slider.value if s_g else 0.0
    true_b1_group = b1_group_slider.value if s_x1g else 0.0
    true_b2_group = b2_group_slider.value if s_x2g else 0.0
    true_b3_group = b3_group_slider.value if s_x3g else 0.0
    # Predictor × Predictor interactions
    true_b_x1_x2 = b_x1_x2_slider.value if s_x1x2 else 0.0
    true_b_x1_x3 = b_x1_x3_slider.value if s_x1x3 else 0.0
    true_b_x2_x3 = b_x2_x3_slider.value if s_x2x3 else 0.0

    # Generate raw predictors
    x1_raw = np.random.normal(x1_mean_slider.value, x1_sd_slider.value, n)
    x2_raw = np.random.normal(x2_mean_slider.value, x2_sd_slider.value, n)
    x3_raw = np.random.normal(x3_mean_slider.value, x3_sd_slider.value, n)

    # Apply transformations
    x1_transformed = apply_transform(x1_raw, x1_transform.value)
    x2_transformed = apply_transform(x2_raw, x2_transform.value)
    x3_transformed = apply_transform(x3_raw, x3_transform.value)

    # Apply binning (after transformation)
    x1 = apply_binning(x1_transformed, x1_bins.value)
    x2 = apply_binning(x2_transformed, x2_bins.value)
    x3 = apply_binning(x3_transformed, x3_bins.value)

    # Generate group if needed
    has_groups = s_g or s_x1g or s_x2g or s_x3g
    group = np.random.binomial(1, 0.5, n) if has_groups else np.zeros(n, dtype=int)

    # Generate y
    y = (
        true_intercept
        + true_b1 * x1
        + true_b2 * x2
        + true_b3 * x3
        + true_b_group * group
        + true_b1_group * x1 * group
        + true_b2_group * x2 * group
        + true_b3_group * x3 * group
        + true_b_x1_x2 * x1 * x2
        + true_b_x1_x3 * x1 * x3
        + true_b_x2_x3 * x2 * x3
        + np.random.normal(0, noise_std, n)
    )

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "group": group,
        "group_label": ["Group 0" if g == 0 else "Group 1" for g in group],
        "y": y,
    })
    df["x1_group"] = df["x1"] * df["group"]
    df["x2_group"] = df["x2"] * df["group"]
    df["x3_group"] = df["x3"] * df["group"]
    # Predictor × Predictor interaction columns
    df["x1_x2"] = df["x1"] * df["x2"]
    df["x1_x3"] = df["x1"] * df["x3"]
    df["x2_x3"] = df["x2"] * df["x3"]

    true_coefs = {
        "const": true_intercept,
        "x1": true_b1,
        "x2": true_b2,
        "x3": true_b3,
        "group": true_b_group,
        "x1_group": true_b1_group,
        "x2_group": true_b2_group,
        "x3_group": true_b3_group,
        "x1_x2": true_b_x1_x2,
        "x1_x3": true_b_x1_x3,
        "x2_x3": true_b_x2_x3,
    }
    return (
        df,
        group,
        has_groups,
        n,
        noise_std,
        true_b1,
        true_b1_group,
        true_b2,
        true_b2_group,
        true_b3,
        true_b3_group,
        true_b_group,
        true_b_x1_x2,
        true_b_x1_x3,
        true_b_x2_x3,
        true_coefs,
        true_intercept,
        x1,
        x1_raw,
        x2,
        x2_raw,
        x3,
        x3_raw,
        y,
    )


@app.cell
def _(
    df,
    pd,
    s_g,
    s_x1,
    s_x1g,
    s_x1x2,
    s_x1x3,
    s_x2,
    s_x2g,
    s_x2x3,
    s_x3,
    s_x3g,
    sm,
):
    # Build model columns
    cols = []
    if s_x1:
        cols.append("x1")
    if s_x2:
        cols.append("x2")
    if s_x3:
        cols.append("x3")
    if s_g:
        cols.append("group")
    if s_x1g:
        cols.append("x1_group")
    if s_x2g:
        cols.append("x2_group")
    if s_x3g:
        cols.append("x3_group")
    if s_x1x2:
        cols.append("x1_x2")
    if s_x1x3:
        cols.append("x1_x3")
    if s_x2x3:
        cols.append("x2_x3")

    if cols:
        X = sm.add_constant(df[cols])
    else:
        X = sm.add_constant(pd.DataFrame(index=df.index))

    model = sm.OLS(df["y"], X).fit()

    df_results = df.copy()
    df_results["fitted"] = model.fittedvalues
    df_results["residuals"] = model.resid
    return X, cols, df_results, model


@app.cell
def _(
    X,
    df_results,
    go,
    has_groups,
    make_subplots,
    mo,
    model,
    n1,
    n2,
    n3,
    ng,
    np,
    ny,
    s_g,
    s_x1,
    s_x2,
    s_x3,
):
    # Determine which x to plot
    # Logic: Prioritize Continuous (x1, x2, x3), then Group, then nothing.
    if s_x1:
        plot_x_col = "x1"
        plot_x_label = n1
        plot_is_categorical = False
    elif s_x2:
        plot_x_col = "x2"
        plot_x_label = n2
        plot_is_categorical = False
    elif s_x3:
        plot_x_col = "x3"
        plot_x_label = n3
        plot_is_categorical = False
    elif s_g:
        # Only group variable enabled
        plot_x_col = "group"
        plot_x_label = ng
        plot_is_categorical = True
    else:
        plot_x_col = None
        plot_x_label = ""
        plot_is_categorical = False

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Scatter Plot ({plot_x_label} vs {ny})" if plot_x_col else "No predictors",
            "Residual Plot"
        ),
        horizontal_spacing=0.1,
    )

    colors = {"Group 0": "#636EFA", "Group 1": "#EF553B"}

    if plot_x_col:
        if plot_is_categorical:
            # === SPECIFIC LOGIC FOR BINARY PREDICTOR ONLY ===
            
            # 1. Plot Jittered Raw Data
            jitter_amount = 0.05
            for grp_val, grp_label in [(0, "Group 0"), (1, "Group 1")]:
                subset = df_results[df_results["group"] == grp_val]
                jittered_x = subset[plot_x_col] + np.random.uniform(-jitter_amount, jitter_amount, len(subset))
                fig.add_trace(
                    go.Scatter(
                        x=jittered_x,
                        y=subset["y"],
                        mode="markers",
                        name=grp_label,
                        marker=dict(color=colors[grp_label], opacity=0.4, size=8),
                        legendgroup=grp_label,
                    ),
                    row=1, col=1,
                )

            # 2. Calculate Means for the Regression Line
            mean_0 = df_results[df_results["group"] == 0]["y"].mean()
            mean_1 = df_results[df_results["group"] == 1]["y"].mean()

            # 3. Plot the Regression Line (connecting the means)
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[mean_0, mean_1],
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="black", width=3, dash="solid"),
                    showlegend=True,
                ),
                row=1, col=1,
            )

            # 4. Plot the Means as distinct points
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[mean_0, mean_1],
                    mode="markers",
                    name="Group Means",
                    marker=dict(color="black", size=12, symbol="diamond"),
                    showlegend=False,
                ),
                row=1, col=1,
            )
            
            # Residuals for categorical
            for grp_val, grp_label in [(0, "Group 0"), (1, "Group 1")]:
                subset = df_results[df_results["group"] == grp_val]
                fig.add_trace(
                    go.Scatter(
                        x=subset["fitted"],
                        y=subset["residuals"],
                        mode="markers",
                        name=grp_label,
                        marker=dict(color=colors[grp_label], opacity=0.6),
                        legendgroup=grp_label,
                        showlegend=False,
                    ),
                    row=1, col=2,
                )
                
        elif has_groups:
            # === CONTINUOUS X + GROUPS ===
            for grp_label in ["Group 0", "Group 1"]:
                subset = df_results[df_results["group_label"] == grp_label]
                fig.add_trace(
                    go.Scatter(
                        x=subset[plot_x_col],
                        y=subset["y"],
                        mode="markers",
                        name=grp_label,
                        marker=dict(color=colors[grp_label], opacity=0.6),
                        legendgroup=grp_label,
                    ),
                    row=1, col=1,
                )
            
            # Residuals
            for grp_label in ["Group 0", "Group 1"]:
                subset = df_results[df_results["group_label"] == grp_label]
                fig.add_trace(
                    go.Scatter(
                        x=subset["fitted"],
                        y=subset["residuals"],
                        mode="markers",
                        name=grp_label,
                        marker=dict(color=colors[grp_label], opacity=0.6),
                        legendgroup=grp_label,
                        showlegend=False,
                    ),
                    row=1, col=2,
                )
        else:
            # === SIMPLE CONTINUOUS PREDICTOR ===
            fig.add_trace(
                go.Scatter(
                    x=df_results[plot_x_col],
                    y=df_results["y"],
                    mode="markers",
                    name="Data",
                    marker=dict(color=colors["Group 0"], opacity=0.6),
                ),
                row=1, col=1,
            )
            
            # Trend line logic for simple regression
            if s_x1 and not s_x2 and not s_x3:
                sorted_df = df_results.sort_values(plot_x_col)
                # Prediction Intervals
                predictions = model.get_prediction(X.loc[sorted_df.index])
                pred_summary = predictions.summary_frame(alpha=0.05)
                
                # PI bands
                fig.add_trace(go.Scatter(
                    x=np.concatenate([sorted_df[plot_x_col], sorted_df[plot_x_col][::-1]]),
                    y=np.concatenate([pred_summary["obs_ci_upper"], pred_summary["obs_ci_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(99, 110, 250, 0.15)", line=dict(color="rgba(0,0,0,0)"),
                    name="95% Prediction Interval", showlegend=True, hoverinfo="skip"
                ), row=1, col=1)

                # CI bands
                fig.add_trace(go.Scatter(
                    x=np.concatenate([sorted_df[plot_x_col], sorted_df[plot_x_col][::-1]]),
                    y=np.concatenate([pred_summary["mean_ci_upper"], pred_summary["mean_ci_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(99, 110, 250, 0.3)", line=dict(color="rgba(0,0,0,0)"),
                    name="95% Confidence Interval", showlegend=True, hoverinfo="skip"
                ), row=1, col=1)

                # Fitted line
                fig.add_trace(
                    go.Scatter(
                        x=sorted_df[plot_x_col],
                        y=sorted_df["fitted"],
                        mode="lines",
                        name="Fitted Line",
                        line=dict(color=colors["Group 0"], width=3),
                        showlegend=True,
                    ),
                    row=1, col=1,
                )

            # Residuals
            fig.add_trace(
                go.Scatter(
                    x=df_results["fitted"],
                    y=df_results["residuals"],
                    mode="markers",
                    name="Residuals",
                    marker=dict(color=colors["Group 0"], opacity=0.6),
                    showlegend=False,
                ),
                row=1, col=2,
            )

    else:
         # Fallback: Plot Y vs Index
        fig.add_trace(go.Scatter(y=df_results["y"], mode="markers", name="Data (y vs index)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_results["fitted"], y=df_results["residuals"], mode="markers", name="Residuals"), row=1, col=2)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Axis Updates
    fig.update_xaxes(title_text=plot_x_label if plot_x_col else "", row=1, col=1)
    if plot_is_categorical:
        fig.update_xaxes(tickvals=[0, 1], ticktext=["Group 0", "Group 1"], row=1, col=1)
        
    fig.update_yaxes(title_text=ny, row=1, col=1)
    fig.update_xaxes(title_text="Fitted Values", row=1, col=2)
    fig.update_yaxes(title_text="Residuals", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    mo.ui.plotly(fig)
    return (
        colors,
        fig,
        jitter_amount,
        mean_0,
        mean_1,
        plot_is_categorical,
        plot_x_col,
        plot_x_label,
    )


@app.cell
def _(mo, model, n1, n2, n3, ng, ny):
    def format_pvalue(p):
        if p < 0.001:
            return "< 0.001"
        return f"{p:.3f}"

    def interpret_coefficients(model, n1, n2, n3, ng, ny):
        """Generates plain English interpretations of regression coefficients."""
        params = model.params
        pvalues = model.pvalues
        interpretations = []

        # Intercept
        b0 = params["const"]
        interpretations.append(
            f"**Intercept:** When all predictors are 0 (and for {ng} 0), the predicted **{ny}** is **{b0:.2f}**."
        )

        # Map variable names to custom labels
        var_map = {
            "x1": n1, "x2": n2, "x3": n3, "group": ng,
            "x1_group": f"{n1} × {ng}", "x2_group": f"{n2} × {ng}", "x3_group": f"{n3} × {ng}",
            "x1_x2": f"{n1} × {n2}", "x1_x3": f"{n1} × {n3}", "x2_x3": f"{n2} × {n3}"
        }

        for var, label in var_map.items():
            if var in params:
                coef = params[var]
                pval = pvalues[var]
                sig_text = "(statistically significant)" if pval < 0.05 else "(not significant)"
                
                if "group" in var and "x" not in var: # Binary group effect
                    direction = "higher" if coef > 0 else "lower"
                    interpretations.append(
                        f"**{label}:** Being in Group 1 is associated with a **{abs(coef):.2f}** {direction} **{ny}** compared to Group 0 {sig_text}."
                    )
                elif "x" in var and "group" not in var and "_" not in var: # Continuous main effect
                    direction = "increase" if coef > 0 else "decrease"
                    interpretations.append(
                        f"**{label}:** A 1-unit increase in {label} is associated with a **{abs(coef):.2f}** unit {direction} in **{ny}** {sig_text}."
                    )
                else: # Interactions
                    interpretations.append(
                        f"**{label}:** The interaction effect is **{coef:.2f}**. This adjusts the slope depending on the other variable {sig_text}."
                    )
        
        return interpretations

    # Create Custom HTML Table
    # Check if table has data (might be empty if model invalid, though statsmodels usually handles it)
    if len(model.summary().tables) > 1:
        results_html = list(model.summary().tables[1].data)
        rows = results_html[1:]
        
        table_rows = ""
        for row in rows:
            var_name = row[0]
            coef = float(row[1].strip())
            std_err = row[2]
            t_stat = row[3]
            p_val_raw = float(row[4].strip())
            ci_lower = row[5]
            ci_upper = row[6]
            
            # Style row based on significance
            row_style = "font-weight: bold; color: #2e7d32; background-color: #e8f5e9;" if p_val_raw < 0.05 else ""
            
            table_rows += f"""
            <tr style="{row_style if p_val_raw < 0.05 else ''}">
                <td style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">{var_name}</td>
                <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{coef:.4f}</td>
                <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{std_err}</td>
                <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{t_stat}</td>
                <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">{format_pvalue(p_val_raw)}</td>
                <td style="text-align: right; padding: 8px; border-bottom: 1px solid #ddd;">[{ci_lower}, {ci_upper}]</td>
            </tr>
            """

        custom_table = f"""
        <div style="font-family: sans-serif; margin-bottom: 20px;">
            <h3 style="margin-bottom: 10px;">Regression Results (R² = {model.rsquared:.3f})</h3>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <thead>
                    <tr style="background-color: #f5f5f5; border-bottom: 2px solid #ccc;">
                        <th style="text-align: left; padding: 10px;">Variable</th>
                        <th style="text-align: right; padding: 10px;">Coef (β)</th>
                        <th style="text-align: right; padding: 10px;">Std Err</th>
                        <th style="text-align: right; padding: 10px;">t</th>
                        <th style="text-align: right; padding: 10px;">P>|t|</th>
                        <th style="text-align: right; padding: 10px;">[0.025, 0.975]</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            <p style="font-size: 12px; color: gray; margin-top: 5px;">* Rows in <span style="color: #2e7d32; font-weight: bold;">green</span> are statistically significant (p < 0.05).</p>
        </div>
        """

        # Get plain English interpretations
        plain_english = interpret_coefficients(model, n1, n2, n3, ng, ny)
        
        # Render Layout
        mo.vstack([
            mo.Html(custom_table),
            mo.accordion({
                "🗣️ Coefficient Interpretation (Plain English)": mo.vstack([
                    mo.md(line) for line in plain_english
                ])
            })
        ])
    else:
        mo.md("No model results available.")
        
    return (
        custom_table,
        format_pvalue,
        interpret_coefficients,
        plain_english,
        results_html,
        rows,
        table_rows,
    )


if __name__ == "__main__":
    app.run()