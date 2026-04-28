from pymongo import MongoClient
import pymongo
from dotenv import load_dotenv
import os
import json
import requests
from datetime import datetime, timezone
import time

load_dotenv(override=True)


## Test connection

def get_mongo_client():
    client = MongoClient(
        os.environ["MONGOHOST"],
        username=os.environ["MONGOUSER"],
        password=os.environ["MONGOPASS"],
    )
    return client

## Weather data ingest
LOCATIONS = [
    {"name": "Richmond",       "lat": 37.5407,  "lon": -77.4360,  "region": "Central"},
    {"name": "Virginia Beach", "lat": 36.8529,  "lon": -75.9780,  "region": "Coastal"},
    {"name": "Charlottesville","lat": 38.0293,  "lon": -78.4767,  "region": "Piedmont"},
    {"name": "Roanoke",        "lat": 37.2710,  "lon": -79.9414,  "region": "Southwest"},
    {"name": "Harrisonburg",   "lat": 38.4496,  "lon": -78.8689,  "region": "Shenandoah"},
    {"name": "Northern VA",    "lat": 38.8048,  "lon": -77.0469,  "region": "Northern"},
    {"name": "Norfolk",        "lat": 36.8508,  "lon": -76.2859,  "region": "Coastal"},
    {"name": "Bristol",        "lat": 36.5951,  "lon": -82.1887,  "region": "Appalachian"},
]

START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Daily variables to pull from Open-Meteo
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "et0_fao_evapotranspiration",
    "precipitation_hours",
    "sunshine_duration",
    "shortwave_radiation_sum",
]

# Field-level fill strategies for None values returned by Open-Meteo.
# "zero"   — physical absence (no rain, no snow, no sunshine)
# "interp" — sensor/model gap; use linear interpolation across the series
FILL_STRATEGY = {
    "temp_max_c":            "interp",
    "temp_min_c":            "interp",
    "temp_mean_c":           "interp",
    "precipitation_mm":      "zero",
    "rain_mm":               "zero",
    "snowfall_cm":           "zero",
    "wind_max_kmh":          "interp",
    "wind_gust_kmh":         "interp",
    "wind_direction_deg":    "interp",
    "evapotranspiration_mm": "interp",
    "precip_hours":          "zero",
    "sunshine_sec":          "zero",
    "solar_radiation_mj":    "interp",
}


def clean_docs(docs: list[dict]) -> tuple[list[dict], dict]:
    """
    Clean None values from a list of documents for a single location.
    Returns cleaned docs and a summary of how many nulls were filled per field.
    Uses linear interpolation for sensor gaps and zero-fill for accumulation fields.
    """
    import pandas as pd

    # Pull each field into a Series for vectorised filling
    field_map = {
        "temp_max_c":            "temperature_2m_max",
        "temp_min_c":            "temperature_2m_min",
        "temp_mean_c":           "temperature_2m_mean",
        "precipitation_mm":      "precipitation_sum",
        "rain_mm":               "rain_sum",
        "snowfall_cm":           "snowfall_sum",
        "wind_max_kmh":          "wind_speed_10m_max",
        "wind_gust_kmh":         "wind_gusts_10m_max",
        "wind_direction_deg":    "wind_direction_10m_dominant",
        "evapotranspiration_mm": "et0_fao_evapotranspiration",
        "precip_hours":          "precipitation_hours",
        "sunshine_sec":          "sunshine_duration",
        "solar_radiation_mj":    "shortwave_radiation_sum",
    }

    fill_report = {}

    for doc_field, strategy in FILL_STRATEGY.items():
        values = [doc.get(doc_field) for doc in docs]
        null_count = sum(1 for v in values if v is None)

        if null_count == 0:
            fill_report[doc_field] = 0
            continue

        fill_report[doc_field] = null_count
        s = pd.Series(values, dtype=float)

        if strategy == "zero":
            s = s.fillna(0.0)
        elif strategy == "interp":
            # Linear interpolation between known values, then
            # ffill/bfill for any gaps at the edges of the series
            s = s.interpolate(method="linear").ffill().bfill()

        for i, doc in enumerate(docs):
            doc[doc_field] = round(float(s.iloc[i]), 4)

    return docs, fill_report


def fetch_weather(location: dict) -> list[dict]:
    """Fetch daily weather from Open-Meteo and return list of MongoDB documents."""
    params = {
        "latitude":   location["lat"],
        "longitude":  location["lon"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "daily":      DAILY_VARS,
        "timezone":   "America/New_York",
    }

    # Exponential backoff: wait 30s, 60s, 120s before giving up
    max_retries = 4
    wait = 30
    for attempt in range(max_retries):
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        if resp.status_code == 429:
            if attempt < max_retries - 1:
                print(f"  Rate limited. Waiting {wait}s before retry {attempt + 2}/{max_retries}...")
                time.sleep(wait)
                wait *= 2
                continue
            else:
                resp.raise_for_status()
        else:
            resp.raise_for_status()
            break

    data = resp.json()
    daily = data["daily"]
    dates = daily["time"]

    docs = []
    for i, date_str in enumerate(dates):
        doc = {
            "timestamp": datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc),
            "metadata": {
                "location":  location["name"],
                "region":    location["region"],
                "latitude":  location["lat"],
                "longitude": location["lon"],
            },
            "temp_max_c":            daily["temperature_2m_max"][i],
            "temp_min_c":            daily["temperature_2m_min"][i],
            "temp_mean_c":           daily["temperature_2m_mean"][i],
            "precipitation_mm":      daily["precipitation_sum"][i],
            "rain_mm":               daily["rain_sum"][i],
            "snowfall_cm":           daily["snowfall_sum"][i],
            "wind_max_kmh":          daily["wind_speed_10m_max"][i],
            "wind_gust_kmh":         daily["wind_gusts_10m_max"][i],
            "wind_direction_deg":    daily["wind_direction_10m_dominant"][i],
            "evapotranspiration_mm": daily["et0_fao_evapotranspiration"][i],
            "precip_hours":          daily["precipitation_hours"][i],
            "sunshine_sec":          daily["sunshine_duration"][i],
            "solar_radiation_mj":    daily["shortwave_radiation_sum"][i],
        }
        docs.append(doc)

    return docs


def setup_timeseries_collection(db) -> pymongo.collection.Collection:
    existing = db.list_collection_names()

    if "weather" not in existing:
        db.create_collection(
            "weather",
            timeseries={
                "timeField":   "timestamp",
                "metaField":   "metadata",
                "granularity": "hours",
            },
        )
        print("  Created time series collection: 'weather'")
    else:
        print("  Collection 'weather' already exists, skipping creation.")

    return db["weather"]


def run():
    client = get_mongo_client()
    db = client["weather_db"]

    try:
        client.admin.command('ping')
        print("Connected successfully!")
    except Exception as e:
        print(f"Connection failed: {e}")

    collection = setup_timeseries_collection(db)

    # ── Ingest each location ─────────────────────────────────
    total_inserted = 0

    for loc in LOCATIONS:
        print(f"Fetching {loc['name']} ({loc['region']})...")

        existing_count = collection.count_documents({"metadata.location": loc["name"]})
        if existing_count > 0:
            print(f"  Already have {existing_count:,} docs for {loc['name']}, skipping.")
            continue

        try:
            docs = fetch_weather(loc)

            if docs:
                # Clean nulls before inserting
                docs, fill_report = clean_docs(docs)
                filled_fields = {k: v for k, v in fill_report.items() if v > 0}
                if filled_fields:
                    print(f"  Nulls filled: { {k: v for k, v in filled_fields.items()} }")
                else:
                    print(f"  No nulls found.")

                result = collection.insert_many(docs, ordered=False)
                inserted = len(result.inserted_ids)
                total_inserted += inserted
                print(f"  Inserted {inserted:,} daily records.")

        except Exception as e:
            print(f"  ERROR for {loc['name']}: {e}")

        time.sleep(30)

    print(f"\nDone. Total records inserted: {total_inserted:,}")

    # ── Verification ─────────────────────────────────────────
    print("\n── Verification ──────────────────────────────────────")
    print(f"  Total docs in collection: {collection.count_documents({}):,}")

    pipeline = [
        {"$group": {
            "_id": "$metadata.location",
            "count": {"$sum": 1},
            "earliest": {"$min": "$timestamp"},
            "latest":   {"$max": "$timestamp"},
        }},
        {"$sort": {"_id": 1}}
    ]
    for r in collection.aggregate(pipeline):
        print(f"  {r['_id']:<20} {r['count']:>5} days  "
              f"{r['earliest'].date()} → {r['latest'].date()}")

    weather = db["weather"]

    # ── Collection-level summary ──────────────────────────────
    total_docs = weather.count_documents({})
    print(f"Total documents: {total_docs:,}")

    pipeline_range = [
        {"$group": {
            "_id": None,
            "earliest": {"$min": "$timestamp"},
            "latest":   {"$max": "$timestamp"},
        }}
    ]
    r = list(weather.aggregate(pipeline_range))[0]
    print(f"Date range: {r['earliest'].date()} → {r['latest'].date()}")

    print("\nPer-location doc counts:")
    pipeline_loc = [
        {"$group": {"_id": "$metadata.location", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    for r in weather.aggregate(pipeline_loc):
        print(f"  {r['_id']:<22} {r['count']:>6} days")

    # ── Numerical feature stats ───────────────────────────────
    numerical_fields = [
        "temp_max_c", "temp_min_c", "temp_mean_c",
        "precipitation_mm", "rain_mm", "snowfall_cm",
        "wind_max_kmh", "wind_gust_kmh", "wind_direction_deg",
        "evapotranspiration_mm", "precip_hours",
        "sunshine_sec", "solar_radiation_mj",
    ]

    print("\nNumerical feature stats:")
    stats_out = {}
    for field in numerical_fields:
        pipeline_stats = [
            {"$match": {field: {"$ne": None}}},
            {"$group": {
                "_id": None,
                "min":   {"$min": f"${field}"},
                "max":   {"$max": f"${field}"},
                "mean":  {"$avg": f"${field}"},
                "count": {"$sum": 1},
            }}
        ]
        res = list(weather.aggregate(pipeline_stats))
        if res:
            s = res[0]
            null_count = total_docs - s["count"]
            null_pct = null_count / total_docs * 100
            stats_out[field] = {
                "min": round(s["min"], 3),
                "max": round(s["max"], 3),
                "mean": round(s["mean"], 3),
                "null_count": null_count,
                "null_pct": round(null_pct, 2),
            }
            print(f"  {field:<28} min={s['min']:>8.2f}  max={s['max']:>8.2f}  "
                  f"mean={s['mean']:>8.2f}  nulls={null_pct:.1f}%")

    import sys
    sample = weather.find_one()
    approx_bytes = sys.getsizeof(str(sample))
    print(f"\nApprox doc size (str repr): {approx_bytes} bytes")

    try:
        coll_stats = db.command("collStats", "weather")
        size_mb = coll_stats.get("size", 0) / (1024 * 1024)
        storage_mb = coll_stats.get("storageSize", 0) / (1024 * 1024)
        print(f"Collection logical size: {size_mb:.2f} MB")
        print(f"Collection storage size: {storage_mb:.2f} MB")
    except Exception as e:
        print(f"collStats unavailable: {e}")

    with open("weather_stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)
    print("\nSaved weather_stats.json")
    client.close()