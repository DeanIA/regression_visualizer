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
    mo.md("# Single Variable Linear Regression Visualizer")
    return


@app.cell
def _(mo):
    mo.md("### Model Type")
    return


@app.cell
def _(mo):
    model_type = mo.ui.radio(
        options=["Continuous (Regression)", "Binary (Means Comparison)", "Binned (Categorized)"],
        value="Continuous (Regression)",
        label="Predictor Type:"
    )
    n_bins_slider = mo.ui.slider(
        start=2, stop=10, step=1, value=3, label="Number of Bins"
    )
    return model_type, n_bins_slider


@app.cell
def _(mo, model_type, n_bins_slider):
    is_binned = model_type.value == "Binned (Categorized)"
    mo.hstack([model_type, n_bins_slider] if is_binned else [model_type], justify="start", gap=4)
    return (is_binned,)


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
def _(group0_name, group1_name, is_binary, mo, x_name, y_name):
    mo.hstack([x_name, y_name, group0_name, group1_name] if is_binary else [x_name, y_name], justify="start", gap=4)
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
def _(intercept_slider, is_binary, mo, slope_slider):
    mo.vstack([
        mo.hstack([intercept_slider, slope_slider], justify="start", gap=4),
        mo.md(f"*Group 0 Mean = β₀ = {intercept_slider.value:.1f}, Group 1 Mean = β₀ + β₁ = {intercept_slider.value + slope_slider.value:.1f}*") if is_binary else None
    ]) if is_binary else mo.hstack([slope_slider, intercept_slider], justify="start", gap=4)
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
    # Checkboxes for toggling display
    show_variance = mo.ui.checkbox(value=False, label="Show Variance")
    show_sd = mo.ui.checkbox(value=False, label="Show Standard Deviation (SD)")
    show_ci = mo.ui.checkbox(value=False, label="Show Confidence Interval (CI)")
    show_pi = mo.ui.checkbox(value=False, label="Show Prediction Interval (PI)")

    # Sliders for adjustable values
    sd_multiplier = mo.ui.slider(start=1, stop=3, step=0.5, value=1, label="SD Multiplier")
    ci_level = mo.ui.slider(start=0.80, stop=0.99, step=0.01, value=0.95, label="CI Level")
    pi_level = mo.ui.slider(start=0.80, stop=0.99, step=0.01, value=0.95, label="PI Level")

    return ci_level, pi_level, sd_multiplier, show_ci, show_pi, show_sd, show_variance


@app.cell
def _(ci_level, mo, pi_level, sd_multiplier, show_ci, show_pi, show_sd, show_variance):
    mo.vstack([
        mo.hstack([show_variance, mo.md(f"*Shows ±2 SD band around the regression line to visualize spread of residuals (Variance = SD²)*")], align="center", gap=2),
        mo.hstack([show_sd, sd_multiplier, mo.md(f"*Shows ±{sd_multiplier.value} SD band. SD measures average distance of points from the line.*")], align="center", gap=2),
        mo.hstack([show_ci, ci_level, mo.md(f"*{int(ci_level.value*100)}% CI: Range where the TRUE regression line likely falls. Narrower with more data.*")], align="center", gap=2),
        mo.hstack([show_pi, pi_level, mo.md(f"*{int(pi_level.value*100)}% PI: Range where a NEW individual observation would likely fall. Always wider than CI.*")], align="center", gap=2),
    ], gap=1)
    return


@app.cell
def _(group0_name, group1_name, intercept_slider, is_binary, mo, slope_slider, x_name, y_name):
    x_label = x_name.value or ("Group" if is_binary else "x")
    y_label = y_name.value or "y"
    slope = slope_slider.value
    intercept = intercept_slider.value
    g0_label = group0_name.value or "Group 0"
    g1_label = group1_name.value or "Group 1"

    if is_binary:
        equation = f"**{y_label} = {intercept:.1f} + {slope:.1f} × {x_label}**  (where {x_label} = 0 for {g0_label}, 1 for {g1_label})"
    else:
        equation = f"**{y_label} = {intercept:.1f} + {slope:.1f} × {x_label}**"

    mo.md(f"### Model Equation: {equation}")
    return g0_label, g1_label, intercept, slope, x_label, y_label


@app.cell
def _(g0_label, g1_label, intercept, is_binary, mo, slope, x_label, y_label):
    if is_binary:
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

        mo.md(f"""### Coefficient Interpretation (Means Comparison)

**Intercept (β₀ = {intercept:.1f}):** {intercept_interp}

**Group Difference (β₁ = {slope:.1f}):** {slope_interp}
""")
    else:
        # Continuous variable interpretation
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

        mo.md(f"""### Coefficient Interpretation

**Intercept (β₀ = {intercept:.1f}):** {intercept_interp}

**Slope (β₁ = {slope:.1f}):** {slope_interp}
""")
    return


@app.cell
def _(ci_level, g0_label, g1_label, go, intercept, is_binary, is_binned, mo, n_bins_slider, n_points_slider, noise_slider, np, pi_level, sd_multiplier, seed_slider, show_ci, show_pi, show_sd, show_variance, slope, stats, x_label, y_label):
    # Fixed axis ranges
    Y_MIN, Y_MAX = -15, 15

    np.random.seed(seed_slider.value)
    n = n_points_slider.value

    # Create figure
    fig = go.Figure()

    if is_binned:
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
                name=f"±{sd_mult} SD"))

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
                name=f"±{sd_mult} SD"))

        # Variance bands
        if show_variance.value:
            fig.add_shape(
                type="rect", x0=-0.3, x1=0.3, y0=mean0 - 2*se, y1=mean0 + 2*se,
                fillcolor="rgba(128, 0, 128, 0.15)", line=dict(width=0),
            )
            fig.add_shape(
                type="rect", x0=0.7, x1=1.3, y0=mean1 - 2*se, y1=mean1 + 2*se,
                fillcolor="rgba(128, 0, 128, 0.15)", line=dict(width=0),
            )
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(size=10, color="rgba(128, 0, 128, 0.2)"),
                name=f"±2 SD (Var={mse:.2f})"))

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
        # Continuous mode (original code)
        X_MIN, X_MAX = 0, 10
        x_data = np.random.uniform(X_MIN, X_MAX, n)
        y_data = intercept + slope * x_data + np.random.normal(0, noise_slider.value, n)

        # Calculate statistics for intervals
        x_mean = np.mean(x_data)
        x_line_smooth = np.linspace(X_MIN, X_MAX, 100)
        y_line_smooth = intercept + slope * x_line_smooth

        # Residuals and variance estimation
        y_pred = intercept + slope * x_data
        residuals = y_data - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se = np.sqrt(mse)

        # Sum of squares for x
        ss_x = np.sum((x_data - x_mean)**2)

        # Standard error of the fitted values (for CI)
        se_fit = se * np.sqrt(1/n + (x_line_smooth - x_mean)**2 / ss_x)

        # Standard error for prediction (for PI)
        se_pred = se * np.sqrt(1 + 1/n + (x_line_smooth - x_mean)**2 / ss_x)

        # t-values for user-selected confidence levels
        t_val_ci = stats.t.ppf((1 + ci_level.value) / 2, n - 2)
        t_val_pi = stats.t.ppf((1 + pi_level.value) / 2, n - 2)

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
            sd_upper = y_line_smooth + sd_mult * se
            sd_lower = y_line_smooth - sd_mult * se
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line_smooth, x_line_smooth[::-1]]),
                    y=np.concatenate([sd_upper, sd_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0, 200, 0, 0.2)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"±{sd_mult} SD (σ = {se:.2f})",
                    hoverinfo="skip",
                )
            )

        # Add variance visualization
        if show_variance.value:
            var_upper = y_line_smooth + 2 * se
            var_lower = y_line_smooth - 2 * se
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x_line_smooth, x_line_smooth[::-1]]),
                    y=np.concatenate([var_upper, var_lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(128, 0, 128, 0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"±2 SD (Var = {mse:.2f})",
                    hoverinfo="skip",
                )
            )

        # Add scatter points
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                name="Data Points",
                marker=dict(color="#636EFA", size=8, opacity=0.7),
            )
        )

        # Add regression line
        x_line = np.array([X_MIN, X_MAX])
        y_line = intercept + slope * x_line

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regression Line",
                line=dict(color="#EF553B", width=3),
            )
        )

        # Layout for continuous
        fig.update_layout(
            template="plotly_white",
            height=600,
            xaxis=dict(
                title=x_label,
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

    mo.ui.plotly(fig)
    return


if __name__ == "__main__":
    app.run()
