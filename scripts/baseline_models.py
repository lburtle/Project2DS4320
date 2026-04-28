"""
Baseline Models — Random Forest + Linear Regression
Tabular forecasting of temp_mean_c using lag features.
Designed to run alongside the TFT as a comparison baseline.

Add to your analysis pipeline by importing and calling run_baseline_models(df)
after engineer_features(df) has been called.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET    = "temp_mean_c"
LAG_DAYS  = [1, 7, 14, 30, 90, 365]   # lag features to engineer
TEST_DAYS = 90                          # hold-out window (matches TFT horizon)


# ─────────────────────────────────────────────────────────────
# 1. Build lag-feature tabular dataset
# ─────────────────────────────────────────────────────────────
def build_tabular(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the time series into a flat tabular dataset by adding
    lag features for the target variable. Each row represents one
    location-day, with past values of temp_mean_c as predictors.
    """
    feature_cols = [
        "temp_max_c", "temp_min_c",
        "precipitation_mm", "snowfall_cm",
        "wind_max_kmh", "sunshine_sec", "solar_radiation_mj",
        "day_of_year_sin", "day_of_year_cos", "year_normalized",
        "latitude", "longitude",
    ]

    dfs = []
    for loc, group in df.groupby("location"):
        group = group.sort_values("time_idx").copy()

        # Add lag features
        for lag in LAG_DAYS:
            group[f"target_lag_{lag}"] = group[TARGET].shift(lag)

        # Add rolling statistics (7-day and 30-day)
        group["rolling_mean_7"]  = group[TARGET].shift(1).rolling(7).mean()
        group["rolling_mean_30"] = group[TARGET].shift(1).rolling(30).mean()
        group["rolling_std_7"]   = group[TARGET].shift(1).rolling(7).std()

        dfs.append(group)

    tabular = pd.concat(dfs).dropna().reset_index(drop=True)

    lag_cols     = [f"target_lag_{l}" for l in LAG_DAYS]
    rolling_cols = ["rolling_mean_7", "rolling_mean_30", "rolling_std_7"]
    all_features = feature_cols + lag_cols + rolling_cols

    # Keep only columns that exist
    all_features = [c for c in all_features if c in tabular.columns]

    return tabular, all_features


# ─────────────────────────────────────────────────────────────
# 2. Train/test split (last 90 days per location = test)
# ─────────────────────────────────────────────────────────────
def split_data(tabular: pd.DataFrame, feature_cols: list):
    max_idx = tabular["time_idx"].max()
    cutoff  = max_idx - TEST_DAYS

    train = tabular[tabular["time_idx"] <= cutoff]
    test  = tabular[tabular["time_idx"] >  cutoff]

    X_train = train[feature_cols].values
    y_train = train[TARGET].values
    X_test  = test[feature_cols].values
    y_test  = test[TARGET].values

    return X_train, y_train, X_test, y_test, train, test


# ─────────────────────────────────────────────────────────────
# 3. Train models
# ─────────────────────────────────────────────────────────────
def train_models(X_train, y_train):
    # Linear Regression (simplest baseline)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators = 200,
        max_depth    = 12,
        min_samples_leaf = 5,
        n_jobs       = -1,
        random_state = 42,
    )
    rf.fit(X_train, y_train)

    return lr, rf, scaler


# ─────────────────────────────────────────────────────────────
# 4. Evaluate & print metrics
# ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<25}  MAE={mae:.3f}°C   RMSE={rmse:.3f}°C   R²={r2:.4f}")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ─────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────
def plot_predictions(test_df, lr_preds, rf_preds, feature_cols):
    """Actual vs predicted for each location — RF and LR side by side."""
    locations = sorted(test_df["location"].unique())
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    for i, loc in enumerate(locations):
        ax   = axes[i]
        mask = test_df["location"] == loc
        dates   = test_df.loc[mask, "timestamp"].values
        actuals = test_df.loc[mask, TARGET].values
        rf_p    = rf_preds[mask]
        lr_p    = lr_preds[mask]

        ax.plot(dates, actuals, label="Actual",           color="#2E75B6", linewidth=1.5)
        ax.plot(dates, rf_p,   label="Random Forest",     color="#E84545", linewidth=1.5, linestyle="--")
        ax.plot(dates, lr_p,   label="Linear Regression", color="#55A868", linewidth=1.5, linestyle=":")

        ax.set_title(loc, fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Temp (°C)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Baseline Models — Actual vs Predicted (90-day test set)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("images/baseline_forecasts.png", dpi=150, bbox_inches="tight")
    print("Saved images/baseline_forecasts.png")


def plot_feature_importance(rf, feature_cols):
    """Random Forest feature importances — analogous to TFT attention weights."""
    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2E75B6" if "lag" in c or "rolling" in c else "#55A868" for c in importances.index]
    ax.barh(importances.index, importances.values, color=colors)
    ax.set_title("Random Forest — Feature Importances", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.grid(axis="x", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#2E75B6", label="Lag / rolling features"),
              Patch(color="#55A868", label="Raw weather features")]
    ax.legend(handles=legend, fontsize=9)

    plt.tight_layout()
    plt.savefig("images/rf_feature_importance.png", dpi=150, bbox_inches="tight")
    print("Saved images/rf_feature_importance.png")


def plot_metrics_comparison(results: list):
    """Bar chart comparing MAE and RMSE across models."""
    res_df = pd.DataFrame(results)
    x      = np.arange(len(res_df))
    width  = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, metric, color in zip(axes, ["MAE", "RMSE", "R2"],
                                  ["#2E75B6", "#E84545", "#55A868"]):
        bars = ax.bar(x, res_df[metric], width=0.5, color=color, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(res_df["model"], fontsize=9)
        ax.set_title(metric, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, res_df[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison — 90-day Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/model_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved images/model_comparison.png")


# ─────────────────────────────────────────────────────────────
# Entry point — call this from analysis.py run()
# ─────────────────────────────────────────────────────────────
def run_baseline_models(df: pd.DataFrame):
    print("\n── Baseline Models ───────────────────────────────────")

    # Build tabular dataset with lag features
    print("Building tabular dataset with lag features...")
    tabular, feature_cols = build_tabular(df)
    print(f"  Features: {len(feature_cols)}  |  Rows: {len(tabular):,}")

    # Split
    X_train, y_train, X_test, y_test, train_df, test_df = split_data(tabular, feature_cols)
    print(f"  Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

    # Train
    print("Training models...")
    lr, rf, scaler = train_models(X_train, y_train)

    # Predict
    X_test_scaled = scaler.transform(X_test)
    lr_preds = lr.predict(X_test_scaled)
    rf_preds = rf.predict(X_test)

    # Evaluate
    print("\nTest set metrics (90-day hold-out):")
    results = [
        evaluate("Linear Regression",  y_test, lr_preds),
        evaluate("Random Forest",       y_test, rf_preds),
    ]

    # Plots
    print("\nGenerating baseline plots...")
    plot_predictions(test_df, lr_preds, rf_preds, feature_cols)
    plot_feature_importance(rf, feature_cols)
    plot_metrics_comparison(results)

    return rf, lr, results