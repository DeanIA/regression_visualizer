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
    mo.md("# Regression Visualizer")
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
        ],
        value="Basic Linear Regression",
    )
    n_bins_slider = mo.ui.slider(
        start=2, stop=10, step=1, value=3, label="Number of Bins"
    )
    x_transform = mo.ui.dropdown(
        options=["None (x)", "Square Root (√x)", "Square (x²)", "Log (ln x)", "Constant (no x)"],
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
    continuous2_hold = mo.ui.slider(start=0, stop=100, step=1, value=50, label="Hold 2nd predictor at:")

    # Third continuous predictor
    continuous3_name = mo.ui.text(value="w", label="3rd Predictor Name:", placeholder="e.g., Income")
    continuous3_hold = mo.ui.slider(start=0, stop=100, step=1, value=50, label="Hold 3rd predictor at:")

    # Predictor distribution controls (mean/SD like RAOS KidIQ - mom_iq mean~100, sd~15)
    x_mean = mo.ui.number(value=5, label="X Mean:", start=-1000, stop=1000, step=0.1)
    x_sd = mo.ui.number(value=2, label="X SD:", start=0.1, stop=100, step=0.1)
    z_mean = mo.ui.number(value=100, label="Z Mean:", start=-1000, stop=1000, step=0.1)
    z_sd = mo.ui.number(value=15, label="Z SD:", start=0.1, stop=100, step=0.1)
    w_mean = mo.ui.number(value=50, label="W Mean:", start=-1000, stop=1000, step=0.1)
    w_sd = mo.ui.number(value=10, label="W SD:", start=0.1, stop=100, step=0.1)

    # Coefficients for multiple regression
    beta_group = mo.ui.slider(start=-10, stop=10, step=0.1, value=2.0, label="Group Effect (β)")
    beta_interaction = mo.ui.slider(start=-5, stop=5, step=0.1, value=0.5, label="Interaction (β)")
    beta_interaction_cont = mo.ui.slider(start=-0.5, stop=0.5, step=0.01, value=0.05, label="x×z Interaction (β)")
    beta_continuous2 = mo.ui.slider(start=-5, stop=5, step=0.1, value=0.5, label="2nd Predictor Effect (β)")
    beta_continuous3 = mo.ui.slider(start=-5, stop=5, step=0.1, value=0.3, label="3rd Predictor Effect (β)")

    return add_continuous3, add_interaction, add_interaction_cont, beta_continuous2, beta_continuous3, beta_group, beta_interaction, beta_interaction_cont, continuous2_hold, continuous2_name, continuous3_hold, continuous3_name, group0_name_multi, group1_name_multi, group_var_name, w_mean, w_sd, x_mean, x_sd, z_mean, z_sd


@app.cell
def _(mo, model_type, n_bins_slider):
    is_binned = model_type.value == "Binned (Categorized)"
    is_continuous = model_type.value == "Basic Linear Regression"
    is_multiple = model_type.value.startswith("Multivariable:")
    has_grouping = model_type.value == "Multivariable: Categorical"

    # Show appropriate controls based on model type
    items = [model_type]
    if is_binned:
        items.append(n_bins_slider)
    _display = mo.vstack(items, gap=1)
    _display
    return has_grouping, is_binned, is_continuous, is_multiple


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
def _(add_continuous3, continuous2_name, continuous3_name, group0_name, group0_name_multi, group1_name, group1_name_multi, group_var_name, has_grouping, is_binary, is_multiple, mo, w_mean, w_sd, x_mean, x_name, x_sd, y_name, z_mean, z_sd):
    # Build display based on model type
    if is_multiple and has_grouping:
        # Categorical variable mode
        _rows = [
            mo.hstack([x_name, y_name], justify="start", gap=4),
            mo.hstack([group_var_name, group0_name_multi, group1_name_multi], justify="start", gap=4),
            mo.hstack([x_mean, x_sd], justify="start", gap=4),
        ]
        if add_continuous3.value:
            _rows.insert(2, mo.hstack([continuous3_name, w_mean, w_sd], justify="start", gap=4))
        _display = mo.vstack(_rows, gap=2)
    elif is_multiple:
        # Continuous predictors mode: x + z + optional w
        _rows = [
            mo.hstack([x_name, y_name], justify="start", gap=4),
            mo.hstack([continuous2_name, add_continuous3], justify="start", gap=4),
            mo.md("**Predictor Distributions** *(set mean/SD to match real data)*"),
            mo.hstack([x_mean, x_sd, z_mean, z_sd], justify="start", gap=4),
        ]
        if add_continuous3.value:
            _rows.insert(2, mo.hstack([continuous3_name], justify="start", gap=4))
            _rows.append(mo.hstack([w_mean, w_sd], justify="start", gap=4))
        _display = mo.vstack(_rows, gap=2)
    elif is_binary:
        _display = mo.hstack([y_name, group0_name, group1_name], justify="start", gap=4)
    else:
        _display = mo.hstack([x_name, y_name], justify="start", gap=4)
    _display
    return


@app.cell
def _(mo):
    mo.md("### Line Parameters")
    return


@app.cell
def _(is_binary, mo):
    slope_slider = mo.ui.slider(
        start=-5, stop=5, step=0.1, value=1.0,
        label="Slope (β₁)" if not is_binary else "Group Difference (β₁)"
    )
    intercept_slider = mo.ui.slider(
        start=-10, stop=10, step=0.5, value=0.0,
        label="Intercept (β₀)" if not is_binary else "Group 0 Mean (β₀)"
    )
    return intercept_slider, slope_slider


@app.cell
def _(add_continuous3, add_interaction, add_interaction_cont, beta_continuous2, beta_continuous3, beta_group, beta_interaction, beta_interaction_cont, continuous2_hold, continuous3_hold, has_grouping, intercept_slider, is_binary, is_continuous, is_multiple, mo, slope_slider, x_transform):
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
    else:
        _display = mo.hstack([slope_slider, intercept_slider], justify="start", gap=4)
    _display
    return


@app.cell
def _(mo):
    mo.md("### Data Parameters")
    return


@app.cell
def _(mo):
    n_points_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=50, label="Number of Points"
    )
    noise_slider = mo.ui.slider(
        start=0, stop=5, step=0.1, value=1.0, label="Error SD (σ)"
    )
    seed_slider = mo.ui.slider(
        start=1, stop=100, step=1, value=42, label="Random Seed"
    )
    return n_points_slider, noise_slider, seed_slider


@app.cell
def _(mo, n_points_slider, noise_slider, seed_slider):
    mo.hstack([n_points_slider, noise_slider, seed_slider], justify="start", gap=4)
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
def _(ci_level, mo, pi_level, sd_multiplier, show_ci, show_pi, show_sd):
    _items = []

    _items.append(show_sd)
    if show_sd.value:
        _items.append(sd_multiplier)

    _items.append(show_ci)
    if show_ci.value:
        _items.append(ci_level)

    _items.append(show_pi)
    if show_pi.value:
        _items.append(pi_level)

    _display = mo.vstack(_items, gap=0)
    _display
    return


@app.cell
def _(add_continuous3, add_interaction, add_interaction_cont, beta_continuous2, beta_continuous3, beta_group, beta_interaction, beta_interaction_cont, continuous2_hold, continuous2_name, continuous3_hold, continuous3_name, group0_name, group0_name_multi, group1_name, group1_name_multi, group_var_name, has_grouping, intercept_slider, is_binary, is_multiple, mo, slope_slider, w_mean, w_sd, x_mean, x_name, x_sd, x_transform, y_name, z_mean, z_sd):
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

    # Predictor distributions (mean/SD)
    x_mu = x_mean.value
    x_sigma = x_sd.value
    z_mu = z_mean.value
    z_sigma = z_sd.value
    w_mu = w_mean.value
    w_sigma = w_sd.value

    # Get transformed x label for equation
    if transform == "Square Root (√x)":
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
            equation = f"**{y_label} = " + " + ".join(eq_parts) + f"**  (where {grp_var_label} = 0 for {g0_multi_label}, 1 for {g1_multi_label})"
        else:
            # Continuous predictors mode: y = β₀ + β₁x + β₂z + β₃(x×z) + β₄w
            eq_parts = [f"{intercept:.1f}", f"{slope:.1f}×{x_term or x_label}", f"{b_cont2:.1f}×{z_label}"]
            if has_interaction_cont:
                eq_parts.append(f"{b_interaction_cont:.2f}×({x_term or x_label}×{z_label})")
            if has_cont3:
                eq_parts.append(f"{b_cont3:.1f}×{w_label}")
            equation = f"**{y_label} = " + " + ".join(eq_parts) + "**"
    elif is_binary:
        equation = f"**{y_label} = {intercept:.1f} + {slope:.1f} × {x_label}**  (where {x_label} = 0 for {g0_label}, 1 for {g1_label})"
    elif x_term is None:
        equation = f"**{y_label} = {intercept:.1f}**  (constant model, no predictor)"
    else:
        equation = f"**{y_label} = {intercept:.1f} + {slope:.1f} × {x_term}**"

    mo.md(f"### Model Equation: {equation}")
    return b_cont2, b_cont3, b_group, b_interaction, b_interaction_cont, g0_label, g0_multi_label, g1_label, g1_multi_label, grp_var_label, has_cont3, has_interaction, has_interaction_cont, intercept, slope, transform, w_hold, w_label, w_mu, w_sigma, x_label, x_mu, x_sigma, x_term, y_label, z_hold, z_label, z_mu, z_sigma


@app.cell
def _(b_cont2, b_cont3, b_group, b_interaction, b_interaction_cont, g0_label, g0_multi_label, g1_label, g1_multi_label, grp_var_label, has_cont3, has_grouping, has_interaction, has_interaction_cont, intercept, is_binary, is_multiple, mo, slope, transform, w_hold, w_label, x_label, x_term, y_label, z_hold, z_label):
    if is_multiple:
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
        if transform == "Square Root (√x)":
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
def _(b_cont2, b_cont3, b_group, b_interaction, b_interaction_cont, ci_level, g0_label, g0_multi_label, g1_label, g1_multi_label, go, grp_var_label, has_cont3, has_grouping, has_interaction, has_interaction_cont, intercept, is_binary, is_binned, is_multiple, mo, n_bins_slider, n_points_slider, noise_slider, np, pi_level, sd_multiplier, seed_slider, show_ci, show_pi, show_sd, slope, stats, transform, w_hold, w_label, w_mu, w_sigma, x_label, x_mu, x_sigma, x_term, y_label, z_hold, z_label, z_mu, z_sigma):
    # Fixed axis ranges
    Y_MIN, Y_MAX = -15, 15

    np.random.seed(seed_slider.value)
    n = n_points_slider.value

    # Create figure
    fig = go.Figure()

    if is_multiple:
        # Multiple regression mode - generate data from normal distributions
        x_data_raw = np.random.normal(x_mu, x_sigma, n)
        # For log transform, ensure positive values
        if transform == "Log (ln x)":
            x_data_raw = np.abs(x_data_raw) + 0.1

        # Apply transformation to x for the model
        if transform == "Square Root (√x)":
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

        # Transform x_line for regression curve (use ±3 SD range)
        X_MIN = x_mu - 3 * x_sigma
        X_MAX = x_mu + 3 * x_sigma
        if transform == "Log (ln x)" or transform == "Square Root (√x)":
            X_MIN = max(0.1, X_MIN)
        x_line = np.linspace(X_MIN, X_MAX, 100)
        if transform == "Square Root (√x)":
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
                w_data = np.random.uniform(0, 10, n)
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
            z_data = np.random.normal(z_mu, z_sigma, n)

            # Generate third continuous predictor if enabled
            if has_cont3:
                w_data = np.random.normal(w_mu, w_sigma, n)
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

            # Discretize z into 3 categories: Low (< mean-1SD), Medium, High (> mean+1SD)
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
            # Use representative z values: low = mean-1.5SD, mid = mean, high = mean+1.5SD
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
        y_data_max = np.max(y_data)
        y_padding = y_data_max * 0.1
        y_range = [0, y_data_max + y_padding]

        fig.update_layout(
            template="plotly_white",
            height=600,
            xaxis=dict(
                title=x_axis_title,
                range=[X_MIN - (X_MAX - X_MIN) * 0.05, X_MAX + (X_MAX - X_MIN) * 0.05],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            yaxis=dict(
                title=y_label,
                range=y_range,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )

    elif is_binned:
        # Binned mode: Generate continuous data then bin it
        X_MIN, X_MAX = 0, 10
        n_bins = n_bins_slider.value

        # Generate continuous x and y
        x_continuous = np.random.uniform(X_MIN, X_MAX, n)
        y_data = intercept + slope * x_continuous + np.random.normal(0, noise_slider.value, n)

        # Create bin edges and assign bins
        bin_edges = np.linspace(X_MIN, X_MAX, n_bins + 1)
        bin_indices = np.digitize(x_continuous, bin_edges[1:-1])  # 0 to n_bins-1
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate statistics
        y_pred = intercept + slope * x_continuous
        residuals = y_data - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se = np.sqrt(mse)

        t_val_ci = stats.t.ppf((1 + ci_level.value) / 2, n - 2)
        t_val_pi = stats.t.ppf((1 + pi_level.value) / 2, n - 2)

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
        x_line = np.array([X_MIN, X_MAX])
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
            height=600,
            xaxis=dict(
                title=f"{x_label} (binned)",
                range=[X_MIN - 0.5, X_MAX + 0.5],
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                dtick=5,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=50, b=50, l=50, r=50),
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

        # Calculate pooled statistics
        y_pred = intercept + slope * x_data
        residuals = y_data - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se = np.sqrt(mse)

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
            height=600,
            xaxis=dict(
                title=x_label,
                range=[-0.5, 1.5],
                tickvals=[0, 1],
                ticktext=[g0_label, g1_label],
                zeroline=False,
                gridcolor="lightgray",
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                dtick=5,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=50, b=50, l=50, r=50),
        )
    else:
        # Continuous mode with transformations
        # For log transform, x must be positive
        if transform == "Log (ln x)":
            X_MIN, X_MAX = 0.1, 10
        else:
            X_MIN, X_MAX = 0, 10

        x_data_raw = np.random.uniform(X_MIN, X_MAX, n)

        # Apply transformation to x for the model
        if transform == "Square Root (√x)":
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
        x_line_smooth = np.linspace(X_MIN, X_MAX, 100)

        if transform == "Square Root (√x)":
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
            height=600,
            xaxis=dict(
                title=x_axis_title,
                range=[X_MIN, X_MAX],
                dtick=2,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            yaxis=dict(
                title=y_label,
                range=[Y_MIN, Y_MAX],
                dtick=5,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
                gridcolor="lightgray",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
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
        plot_output = mo.ui.plotly(fig)

    # Return actual regression statistics for R summary
    # Only available for basic linear regression mode
    if not is_binned and not is_binary and not is_multiple:
        reg_stats = {
            "n": n,
            "df": df,
            "mse": mse,
            "se": se,
            "ss_x": ss_x,
            "x_mean": _x_mean,
            "y_data": y_data,
            "ols_y_pred": _ols_y_pred,
            "ols_slope": _ols_slope,
            "ols_intercept": _ols_intercept,
            "x_data_transformed": x_data_transformed,
        }
    else:
        reg_stats = None
    return plot_output, reg_stats


@app.cell
def _(coef_text, mo, np, plot_output, reg_stats, stats, transform, x_label):
    # Only show regression summary for basic linear regression
    if reg_stats is None:
        _r_summary = """### Regression Summary
*Summary statistics only available for Basic Linear Regression mode.*
"""
    else:
        # Extract actual OLS regression results from the plotting cell
        _n = reg_stats["n"]
        _df = reg_stats["df"]
        _se = reg_stats["se"]
        _ss_x = reg_stats["ss_x"]
        _x_mean = reg_stats["x_mean"]
        _y_data = reg_stats["y_data"]
        _ols_y_pred = reg_stats["ols_y_pred"]
        _ols_slope = reg_stats["ols_slope"]
        _ols_intercept = reg_stats["ols_intercept"]

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

        # Significance stars
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

        # F-statistic
        if transform != "Constant (no x)" and _df > 0 and _r_squared < 1:
            _f_stat = (_r_squared / 1) / ((1 - _r_squared) / _df)
            _f_pval = 1 - stats.f.cdf(_f_stat, 1, _df)
        else:
            _f_stat = 0
            _f_pval = 1

        _r_summary = f"""### Regression Summary (OLS on simulated data)
```
Coefficients:
              Estimate Std.Err t value  Pr(>|t|)
(Intercept)   {_ols_intercept:8.3f} {_se_intercept:7.3f} {_t_intercept:7.2f}  {_p_intercept:.2e} {_sig_stars(_p_intercept)}
{x_label:13s} {_ols_slope:8.3f} {_se_slope:7.3f} {_t_slope:7.2f}  {_p_slope:.2e} {_sig_stars(_p_slope)}
---
Signif: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1

Residual SE: {_se:.3f} on {_df} df
R-squared: {_r_squared:.4f}, Adj R²: {_adj_r_squared:.4f}
F-statistic: {_f_stat:.2f} on 1 and {_df} DF
p-value: {_f_pval:.2e}
```
"""

    # Create side-by-side layout for interpretation and summary
    _left_col = mo.md(coef_text)
    _right_col = mo.md(_r_summary)

    _summary_row = mo.hstack([_left_col, _right_col], widths=[1, 1], gap=4, align="start")

    # Stack the summary and plot vertically
    mo.vstack([_summary_row, plot_output], gap=2)
    return


if __name__ == "__main__":
    app.run()
