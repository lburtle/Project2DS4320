"""
Temporal Fusion Transformer — Virginia Climate Forecasting
Uses pytorch-forecasting's TFT implementation to forecast weather variables
and analyze climate trends from 15 years of Open-Meteo ERA5 data.

"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
from pymongo import MongoClient
from dotenv import load_dotenv

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
load_dotenv(override=True)

# ── Config ────────────────────────────────────────────────────
TARGET         = "temp_mean_c"        # primary forecast target
MAX_ENCODER    = 365                  # look back 1 year
MAX_PREDICTION = 90                   # forecast 90 days ahead
BATCH_SIZE     = 128
MAX_EPOCHS     = 50
LEARNING_RATE  = 3e-3

# All continuous covariates the model can use as inputs
TIME_VARYING_KNOWN = []               # no known-future features in this dataset
TIME_VARYING_UNKNOWN = [
    "temp_max_c", "temp_min_c", "temp_mean_c",
    "precipitation_mm", "rain_mm", "snowfall_cm",
    "wind_max_kmh", "wind_gust_kmh",
    "evapotranspiration_mm", "precip_hours",
    "sunshine_sec", "solar_radiation_mj",
    # engineered features (added below)
    "day_of_year_sin", "day_of_year_cos",  # cyclical encoding of seasonality
    "year_normalized",                      # linear trend signal
]
STATIC_CATEGORICALS = ["location"]
STATIC_REALS        = ["latitude", "longitude"]

# Load data from MongoDB
def load_from_mongo() -> pd.DataFrame:
    print("Loading data from MongoDB...")
    client = MongoClient(
        os.environ["MONGOHOST"],
        username=os.environ["MONGOUSER"],
        password=os.environ["MONGOPASS"],
    )
    db = client["weather_db"]
    cursor = db["weather"].find({}, {"_id": 0, "wind_direction_deg": 0})
    df = pd.DataFrame(list(cursor))
    client.close()

    # Flatten metadata
    df["location"]  = df["metadata"].apply(lambda x: x["location"])
    df["region"]    = df["metadata"].apply(lambda x: x["region"])
    df["latitude"]  = df["metadata"].apply(lambda x: x["latitude"])
    df["longitude"] = df["metadata"].apply(lambda x: x["longitude"])
    df = df.drop(columns=["metadata"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} records across {df['location'].nunique()} locations.")
    return df

# Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Cyclical day-of-year encoding — lets the model learn seasonality
    # without treating Dec 31 and Jan 1 as maximally distant
    doy = df["timestamp"].dt.day_of_year
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Normalized year — gives the model a linear trend signal
    # 0.0 = Jan 2010, 1.0 = Dec 2024
    year_min = df["timestamp"].dt.year.min()
    year_max = df["timestamp"].dt.year.max()
    df["year_normalized"] = (df["timestamp"].dt.year - year_min) / (year_max - year_min)

    # Integer time index per group — required by pytorch-forecasting
    df["time_idx"] = (
        df.groupby("location")["timestamp"]
        .transform(lambda s: (s - s.min()).dt.days)
        .astype(int)
    )

    # Fill nulls — three-pass strategy to handle edge cases:
    # 1. Per-location ffill/bfill (handles interior + boundary NaNs within a group)
    # 2. Global ffill/bfill (handles NaNs at the very start of a group with no prior value)
    # 3. Fill any remaining with column median (last resort for fully-null columns)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    df[num_cols] = (
        df.groupby("location")[num_cols]
        .transform(lambda x: x.ffill().bfill())
    )
    df[num_cols] = df[num_cols].ffill().bfill()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Sanity check — surface any remaining NaNs before they hit the model
    remaining_nans = df[num_cols].isna().sum()
    remaining_nans = remaining_nans[remaining_nans > 0]
    if not remaining_nans.empty:
        print(f"  WARNING: NaNs still present after fill:\n{remaining_nans}")
    else:
        print("  All NaNs resolved.")

    return df

# Build TimeSeriesDataSet
def build_datasets(df: pd.DataFrame):
    # Train on everything except last 90 days
    max_time_idx = df["time_idx"].max()
    cutoff        = max_time_idx - MAX_PREDICTION

    train_df = df[df["time_idx"] <= cutoff]
    val_df   = df.copy()   # val uses full range (TFT handles windowing internally)

    # Hard null check — diagnose exactly what's still null in train_df
    null_summary = train_df.isnull().sum()
    null_summary = null_summary[null_summary > 0]
    if not null_summary.empty:
        print("Nulls in train_df before TimeSeriesDataSet:")
        print(null_summary)
        # Force fill on train_df directly as a last resort
        num_cols = train_df.select_dtypes(include="number").columns.tolist()
        train_df = train_df.copy()  # avoid SettingWithCopyWarning
        train_df[num_cols] = train_df[num_cols].ffill().bfill().fillna(0)
        val_df = val_df.copy()
        num_cols = val_df.select_dtypes(include="number").columns.tolist()
        val_df[num_cols] = val_df[num_cols].ffill().bfill().fillna(0)
        print("Force-filled remaining nulls in train_df.")

    training = TimeSeriesDataSet(
        train_df,
        time_idx                  = "time_idx",
        target                    = TARGET,
        group_ids                 = ["location"],
        min_encoder_length        = MAX_ENCODER // 2,
        max_encoder_length        = MAX_ENCODER,
        min_prediction_length     = 1,
        max_prediction_length     = MAX_PREDICTION,
        static_categoricals       = STATIC_CATEGORICALS,
        static_reals              = STATIC_REALS,
        time_varying_known_reals  = TIME_VARYING_KNOWN,
        time_varying_unknown_reals= [c for c in TIME_VARYING_UNKNOWN if c in df.columns],
        target_normalizer         = GroupNormalizer(groups=["location"], transformation="softplus"),
        add_relative_time_idx     = True,    # adds position encoding
        add_target_scales         = True,    # adds mean/std of target as static reals
        add_encoder_length        = True,
        allow_missing_timesteps   = False,
    )

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
    val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    return training, train_loader, val_loader

# Build & Train TFT
def build_tft(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training,
        learning_rate          = LEARNING_RATE,
        hidden_size            = 64,        # embedding dimension
        attention_head_size    = 4,         # multi-head attention
        dropout                = 0.1,
        hidden_continuous_size = 32,        # continuous variable processing
        loss                   = QuantileLoss(),   # probabilistic output
        log_interval           = 10,
        reduce_on_plateau_patience = 4,
    )

# Train TFT
def train(tft, train_loader, val_loader):
    early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min", verbose=True)
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min",
                                  dirpath="checkpoints/", filename="tft-best")

    trainer = pl.Trainer(
        max_epochs        = MAX_EPOCHS,
        accelerator       = "auto",      # uses GPU if available, else CPU
        gradient_clip_val = 0.1,
        callbacks         = [early_stop, checkpoint],
        enable_progress_bar = True,
        logger            = False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Load best checkpoint
    best = TemporalFusionTransformer.load_from_checkpoint(checkpoint.best_model_path)
    return best

# Analysis & Visualization
def plot_forecasts(tft, val_loader, training, df):
    """Plot actual vs predicted for each location."""
    raw_preds, index = tft.predict(val_loader, mode="raw", return_index=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    locations = df["location"].unique()

    for i, loc in enumerate(sorted(locations)):
        ax = axes[i]
        loc_mask = index["location"] == loc
        if not loc_mask.any():
            continue

        # Median prediction (quantile 0.5)
        pred_median = raw_preds.prediction[loc_mask].median(dim=0).values.squeeze().numpy()

        # Get corresponding actuals from df
        loc_df = df[df["location"] == loc].sort_values("time_idx")
        actuals = loc_df[TARGET].values[-MAX_PREDICTION:]
        dates   = loc_df["timestamp"].values[-MAX_PREDICTION:]

        ax.plot(dates, actuals,    label="Actual",    color="#2E75B6", linewidth=1.5)
        ax.plot(dates, pred_median[:len(actuals)],
                label="Forecast (p50)", color="#E84545", linewidth=1.5, linestyle="--")

        # Prediction interval (p10–p90)
        p10 = raw_preds.prediction[loc_mask, :, 0].squeeze().numpy()
        p90 = raw_preds.prediction[loc_mask, :, -1].squeeze().numpy()
        ax.fill_between(dates, p10[:len(actuals)], p90[:len(actuals)],
                        alpha=0.15, color="#E84545", label="p10–p90 interval")

        ax.set_title(loc, fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean Temp (°C)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("TFT Forecast vs Actual — Mean Temperature (90-day horizon)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("images/tft_forecasts.png", dpi=150, bbox_inches="tight")
    print("Saved tft_forecasts.png")


def plot_attention(tft, val_loader):
    """Plot encoder attention weights — shows which past timesteps the model attends to."""
    interpretation = tft.interpret_output(
        tft.predict(val_loader, mode="raw", return_index=True)[0],
        reduction="sum"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Variable importance
    var_imp = interpretation["encoder_variables"]
    vars_   = list(var_imp.index)
    vals    = var_imp.values
    axes[0].barh(vars_, vals, color="#2E75B6")
    axes[0].set_title("Encoder Variable Importance", fontweight="bold")
    axes[0].set_xlabel("Attention Weight")
    axes[0].grid(axis="x", alpha=0.3)

    # Attention over time (how far back the model looks)
    attn = interpretation["encoder_length_histogram"]
    axes[1].bar(range(len(attn)), attn.numpy(), color="#55A868")
    axes[1].set_title("Attention by Encoder Position\n(rightmost = most recent)", fontweight="bold")
    axes[1].set_xlabel("Encoder Position (days ago)")
    axes[1].set_ylabel("Attention Weight")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/tft_attention.png", dpi=150, bbox_inches="tight")
    print("Saved tft_attention.png")


def plot_warming_trend(df):
    """
    Simple annual mean temperature trend per location.
    Complements TFT with an interpretable long-range view.
    """
    annual = (
        df.groupby(["location", df["timestamp"].dt.year])["temp_mean_c"]
        .mean()
        .reset_index()
        .rename(columns={"timestamp": "year"})
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors

    for i, loc in enumerate(sorted(annual["location"].unique())):
        sub = annual[annual["location"] == loc]
        ax.plot(sub["year"], sub["temp_mean_c"], label=loc,
                color=colors[i % len(colors)], linewidth=1.8, marker="o", markersize=3)

        # Linear trend line
        z = np.polyfit(sub["year"], sub["temp_mean_c"], 1)
        p = np.poly1d(z)
        ax.plot(sub["year"], p(sub["year"]), "--", color=colors[i % len(colors)],
                alpha=0.5, linewidth=1)
        rate = z[0] * 10  # °C per decade
        print(f"  {loc:<22} warming rate: {rate:+.3f}°C/decade")

    ax.set_title("Annual Mean Temperature Trends — Virginia (2010–2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Mean Temperature (°C)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/warming_trends.png", dpi=150, bbox_inches="tight")
    print("Saved warming_trends.png")


# Main function

def run():
    # 1. Load & prepare
    df = load_from_mongo()
    df = engineer_features(df)

    # 2. Warming trend plot (no model needed — good sanity check first)
    print("\nWarming rates per location (linear regression on annual means):")
    plot_warming_trend(df)

    

    # 3. Build datasets
    print("\nBuilding TimeSeriesDataSet...")
    training, train_loader, val_loader = build_datasets(df)
    print(f"  Training samples: {len(training)}")

    # 4. Build & train TFT
    print("\nTraining Temporal Fusion Transformer...")
    tft = build_tft(training)
    print(f"  Model parameters: {sum(p.numel() for p in tft.parameters()):,}")
    tft = train(tft, train_loader, val_loader)

    # 5. Visualize results
    print("\nGenerating forecast plots...")
    plot_forecasts(tft, val_loader, training, df)

    print("\nGenerating attention analysis...")
    plot_attention(tft, val_loader)

    print("\nDone. Output files:")
    print("  warming_trends.png  — annual trend lines per location")
    print("  tft_forecasts.png   — 90-day forecast vs actual per location")
    print("  tft_attention.png   — variable importance + temporal attention weights")
    return df