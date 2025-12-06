import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    return go, mo, np


@app.cell
def _(mo):
    mo.md("# Single Variable Linear Regression Visualizer")
    return


@app.cell
def _(mo):
    mo.md("### Variable Names")
    return


@app.cell
def _(mo):
    x_name = mo.ui.text(value="x", label="Predictor (x):", placeholder="e.g., Hours Studied")
    y_name = mo.ui.text(value="y", label="Outcome (y):", placeholder="e.g., Test Score")
    return x_name, y_name


@app.cell
def _(mo, x_name, y_name):
    mo.hstack([x_name, y_name], justify="start", gap=4)
    return


@app.cell
def _(mo):
    mo.md("### Line Parameters")
    return


@app.cell
def _(mo):
    slope_slider = mo.ui.slider(
        start=-5, stop=5, step=0.1, value=1.0, label="Slope (β₁)"
    )
    intercept_slider = mo.ui.slider(
        start=-10, stop=10, step=0.5, value=0.0, label="Intercept (β₀)"
    )
    return intercept_slider, slope_slider


@app.cell
def _(intercept_slider, mo, slope_slider):
    mo.hstack([slope_slider, intercept_slider], justify="start", gap=4)
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
        start=0, stop=5, step=0.1, value=1.0, label="Noise (σ)"
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
def _(intercept_slider, mo, slope_slider, x_name, y_name):
    x_label = x_name.value or "x"
    y_label = y_name.value or "y"
    slope = slope_slider.value
    intercept = intercept_slider.value

    equation = f"**{y_label} = {intercept:.1f} + {slope:.1f} × {x_label}**"
    mo.md(f"### Model Equation: {equation}")
    return intercept, slope, x_label, y_label


@app.cell
def _(intercept, mo, slope, x_label, y_label):
    # Intercept interpretation
    intercept_interp = f"When **{x_label}** equals 0, the predicted value of **{y_label}** is **{intercept:.1f}**."

    # Slope interpretation
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
def _(ci_level, go, intercept, mo, n_points_slider, noise_slider, np, pi_level, sd_multiplier, seed_slider, show_ci, show_pi, show_sd, show_variance, slope, x_label, y_label):
    from scipy import stats

    # Fixed axis ranges (positive x only)
    X_MIN, X_MAX = 0, 10
    Y_MIN, Y_MAX = -15, 15

    # Generate data points
    np.random.seed(seed_slider.value)
    n = n_points_slider.value
    x_data = np.random.uniform(X_MIN, X_MAX, n)

    # Generate y with noise
    y_data = intercept + slope * x_data + np.random.normal(0, noise_slider.value, n)

    # Calculate statistics for intervals
    x_mean = np.mean(x_data)
    x_line_smooth = np.linspace(X_MIN, X_MAX, 100)
    y_line_smooth = intercept + slope * x_line_smooth

    # Residuals and variance estimation
    y_pred = intercept + slope * x_data
    residuals = y_data - y_pred
    mse = np.sum(residuals**2) / (n - 2)  # Mean squared error
    se = np.sqrt(mse)  # Standard error of residuals

    # Sum of squares for x
    ss_x = np.sum((x_data - x_mean)**2)

    # Standard error of the fitted values (for CI)
    se_fit = se * np.sqrt(1/n + (x_line_smooth - x_mean)**2 / ss_x)

    # Standard error for prediction (for PI)
    se_pred = se * np.sqrt(1 + 1/n + (x_line_smooth - x_mean)**2 / ss_x)

    # t-values for user-selected confidence levels
    t_val_ci = stats.t.ppf((1 + ci_level.value) / 2, n - 2)
    t_val_pi = stats.t.ppf((1 + pi_level.value) / 2, n - 2)

    # Create figure
    fig = go.Figure()

    # Add prediction interval (PI) - widest band, add first so it's behind
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

    # Add standard deviation band (user-selected multiplier)
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

    # Add variance visualization (2 SD band to show spread)
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

    # Add regression line (spanning the full fixed range)
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

    # Fixed layout with constant axis ranges
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
