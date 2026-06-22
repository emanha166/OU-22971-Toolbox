# Ray Capstone Project: TLC-backed Per-Zone Recommendations Under Skew

## Project goal

This project implements a Ray-based distributed replay system for NYC TLC Green Taxi pickup demand.

For each active pickup zone and each 15-minute replay tick, the system outputs one recommendation:

* `NEED`: the zone currently looks unusually busy.
* `OK`: the zone looks normal.

The goal is not advanced forecasting. The goal is to demonstrate distributed-system behavior with Ray: actors, remote tasks, blocking execution, asynchronous execution, skew, bounded concurrency, fallback behavior, and idempotent actor writes.

## Data

The project uses two adjacent NYC TLC Green Taxi parquet files:

* Reference month: `green_tripdata_2024-01.parquet`
* Replay month: `green_tripdata_2024-02.parquet`

The reference month is used to select active pickup zones and build baselines.
The replay month is used to simulate time-ordered demand.

Raw TLC files are not committed to GitHub. They should be placed locally under:

```text
data/raw/
```

## Prepare step

The prepare step validates the input months, selects active pickup zones, builds reference baselines, aggregates replay demand into 15-minute ticks, and writes prepared runtime assets.

Command used:

```powershell
python .\main.py prepare --reference-parquet .\data\raw\green_tripdata_2024-01.parquet --replay-parquet .\data\raw\green_tripdata_2024-02.parquet --output-dir .\prepared --n-zones 20 --tick-minutes 15 --seed 7
```

For the local Docker demo, I used a smaller deterministic subset of 8 active zones in `prepared_small/` due to local memory limits. The architecture and behavior are unchanged.

## Docker setup

Build the Docker image:

```powershell
docker build -t ray-capstone:local .
```

The image installs Python 3.12, Ray 2.39.0, pandas, pyarrow, and numpy.

## Run commands

### Blocking baseline

```powershell
docker run --rm --shm-size=1g -v "${PWD}:/app" -w /app ray-capstone:local python main.py run --prepared-dir prepared_small --output-dir out_blocking_docker --mode blocking --max-ticks 8 --slow-zone-fraction 0.25 --slow-zone-sleep-s 0.5
```

Blocking mode waits for all zone results before finalizing each tick.

### Async controller

```powershell
docker run --rm --shm-size=2g -v "${PWD}:/app" -w /app ray-capstone:local python main.py run --prepared-dir prepared_small --output-dir out_async_docker --mode async --max-ticks 8 --max-inflight-zones 4 --tick-timeout-s 1.0 --completion-fraction 0.75 --slow-zone-fraction 0.25 --slow-zone-sleep-s 0.5 --fallback-policy always_previous
```

Async mode allows scoring tasks to report directly to their `ZoneActor`.
The driver finalizes each tick using a partial-readiness policy.

### Stress test

```powershell
docker run --rm --shm-size=2g -v "${PWD}:/app" -w /app ray-capstone:local python main.py run --prepared-dir prepared_small --output-dir out_stress_docker --mode stress --max-ticks 8 --max-inflight-zones 4
```

Stress mode uses harsher skew settings to test whether the system continues to make progress under load.

## Decision rule

Each scoring task receives a minimal per-zone snapshot.

The scoring function compares the current replay demand against:

* the zone baseline from the reference month
* recent local demand history

If current demand is elevated, the recommendation is `NEED`.
Otherwise, the recommendation is `OK`.

The scoring task is deterministic from its input snapshot.

## Partial-readiness policy

The async controller finalizes each tick when either:

1. enough zones have completed, or
2. the tick timeout has elapsed.

For this demo:

```text
completion_fraction = 0.75
tick_timeout_s = 1.0
```

With 8 active zones:

```text
8 * 0.75 = 6 zones
```

So the async controller can finalize a tick once 6 zones are ready.

## Fallback policy

The fallback policy used in the async run is:

```text
always_previous
```

If a zone does not report before finalization, the actor reuses the previous accepted decision for that zone.
If there is no previous decision, the first fallback is `OK`.

## Output artifacts

Each run produces:

```text
run_config.json
metrics.csv
latency_log.json
tick_summary.json
actor_counters.json
decisions.csv
```

The main output folders are:

```text
out_blocking_docker/
out_async_docker/
out_stress_docker/
```

## Results summary

### Blocking

Blocking mode waits for all 8 zones before closing each tick.

Observed behavior:

* no fallbacks
* no late reports
* each zone has 8 accepted decisions
* tick latency is affected by the slowest zone

### Async

Async mode finalizes ticks using completion fraction or timeout.

Observed behavior:

* some fallbacks
* some late reports
* each zone still has 8 accepted decisions
* the system continues even when some zones are slow

### Stress

Stress mode increases skew.

Observed behavior:

* most ticks finalize by timeout
* more fallbacks and late reports
* each zone still has 8 accepted decisions
* the system preserves deterministic output semantics under skew

## Notes

On the local Windows environment, Ray import and initialization were unstable.
To make the project reproducible, the final runs were executed through Docker using the included Dockerfile.

Ray warnings about `/dev/shm` inside Docker were performance warnings and did not prevent successful artifact generation.

