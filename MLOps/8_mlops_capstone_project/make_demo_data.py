from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def make_taxi_like_data(
    n: int,
    seed: int,
    drift: bool = False,
    bad_data: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pickup_datetime = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 60 * 24 * 60, size=n),
        unit="m",
    )

    trip_distance = rng.gamma(shape=2.0, scale=2.0, size=n)
    passenger_count = rng.integers(1, 5, size=n)
    fare_amount = 3.0 + trip_distance * rng.normal(3.2, 0.4, size=n)
    PULocationID = rng.integers(1, 80, size=n)
    DOLocationID = rng.integers(1, 80, size=n)

    if drift:
        trip_distance = trip_distance * 1.8
        fare_amount = fare_amount * 1.4
        PULocationID = rng.integers(60, 120, size=n)
        DOLocationID = rng.integers(60, 120, size=n)

    hour = pickup_datetime.hour
    weekend = pickup_datetime.dayofweek >= 5

    tip_amount = (
        0.15 * fare_amount
        + 0.35 * trip_distance
        + 0.8 * weekend.astype(float)
        + 0.03 * hour
        + rng.normal(0, 1.0 if not drift else 3.0, size=n)
    )

    tip_amount = np.clip(tip_amount, 0, None)

    dropoff_datetime = pickup_datetime + pd.to_timedelta(
        np.maximum(3, trip_distance * rng.normal(4.0, 0.8, size=n)),
        unit="m",
    )

    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pickup_datetime,
            "lpep_dropoff_datetime": dropoff_datetime,
            "PULocationID": PULocationID,
            "DOLocationID": DOLocationID,
            "passenger_count": passenger_count,
            "trip_distance": trip_distance,
            "fare_amount": fare_amount,
            "tip_amount": tip_amount,
        }
    )

    if bad_data:
        df.loc[:10, "trip_distance"] = -5
        df.loc[11:20, "fare_amount"] = -10
        df.loc[21:30, "lpep_dropoff_datetime"] = df.loc[21:30, "lpep_pickup_datetime"] - pd.Timedelta(minutes=5)

    return df


def main() -> None:
    datasets = {
        "reference.parquet": make_taxi_like_data(n=4000, seed=1),
        "batch_baseline.parquet": make_taxi_like_data(n=1200, seed=2),
        "batch_drift.parquet": make_taxi_like_data(n=1200, seed=3, drift=True),
        "batch_failure.parquet": make_taxi_like_data(n=1200, seed=4, bad_data=True),
    }

    for filename, df in datasets.items():
        path = DATA_DIR / filename
        df.to_parquet(path, index=False)
        print(f"Wrote {path} shape={df.shape}")


if __name__ == "__main__":
    main()