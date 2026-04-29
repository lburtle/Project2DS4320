import requests
import pandas as pd

def get_historical_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
        ],
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return pd.DataFrame(data["daily"])

# Example: Boston from 1940 to 2024
df = get_historical_weather(
    lat=42.36,
    lon=-71.06,
    start_date="1940-01-01",
    end_date="2024-12-31"
)
df["time"] = pd.to_datetime(df["time"])

import matplotlib.pyplot as plt
import pandas as pd

# Resample to annual mean to smooth out seasonality
df["year"] = df["time"].dt.year
annual = df.groupby("year")["temperature_2m_mean"].mean()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Raw daily temperatures
axes[0].plot(df["time"], df["temperature_2m_mean"], alpha=0.3, color="steelblue", linewidth=0.5)
axes[0].set_title("Daily Mean Temperature Over Time")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Temperature (°C)")

# Plot 2: Annual mean with trend line
axes[1].plot(annual.index, annual.values, color="steelblue", linewidth=1.5, label="Annual Mean")

# Add a trend line
import numpy as np
z = np.polyfit(annual.index, annual.values, 1)
p = np.poly1d(z)
axes[1].plot(annual.index, p(annual.index), color="red", linewidth=2, linestyle="--", label=f"Trend ({z[0]:.3f}°C/yr)")

axes[1].set_title("Annual Mean Temperature with Trend")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()

plt.tight_layout()
plt.savefig("../images/temperature_trend.png", dpi=150)

print(f"Warming trend: {z[0]:.4f}°C per year ({z[0]*10:.3f}°C per decade)")