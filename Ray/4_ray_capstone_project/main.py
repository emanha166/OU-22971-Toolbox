from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_N_ZONES = 20
DEFAULT_TICK_MINUTES = 15
DEFAULT_SEED = 7
DEFAULT_MAX_TICKS = 32
DEFAULT_FALLBACK_POLICY = "always_previous"


# -----------------------------
# Small helpers
# -----------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_column(df: pd.DataFrame, candidates: Iterable[str], *, label: str) -> str:
    """Find a dataframe column by case-insensitive candidate names."""
    by_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(
        f"Could not find {label} column. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def _pickup_datetime_col(df: pd.DataFrame) -> str:
    return _find_column(
        df,
        [
            "lpep_pickup_datetime",
            "tpep_pickup_datetime",
            "pickup_datetime",
            "pickup_time",
            "datetime",
        ],
        label="pickup datetime",
    )


def _pickup_zone_col(df: pd.DataFrame) -> str:
    return _find_column(
        df,
        [
            "PULocationID",
            "pu_location_id",
            "pulocationid",
            "pickup_location_id",
            "pickup_zone_id",
            "zone_id",
            "LocationID",
        ],
        label="pickup zone/location id",
    )


def _month_period(df: pd.DataFrame, datetime_col: str) -> pd.Period:
    s = pd.to_datetime(df[datetime_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Column {datetime_col!r} contains no valid datetimes")
    # Use the modal month because some TLC files may contain a tiny number of boundary rows.
    return s.dt.to_period("M").value_counts().sort_values(ascending=False).index[0]


def _validate_adjacent_months(reference_df: pd.DataFrame, replay_df: pd.DataFrame) -> Tuple[str, str, pd.Period, pd.Period]:
    ref_dt_col = _pickup_datetime_col(reference_df)
    rep_dt_col = _pickup_datetime_col(replay_df)

    ref_month = _month_period(reference_df, ref_dt_col)
    rep_month = _month_period(replay_df, rep_dt_col)

    if ref_month.year != rep_month.year:
        raise ValueError(
            f"Expected adjacent months from the same year. Got {ref_month} and {rep_month}."
        )
    if ref_month + 1 != rep_month:
        raise ValueError(
            f"Expected replay month to be immediately after reference month. Got {ref_month} and {rep_month}."
        )
    return ref_dt_col, rep_dt_col, ref_month, rep_month


def _month_tick_range(month: pd.Period, tick_minutes: int) -> pd.DatetimeIndex:
    start = month.to_timestamp()
    end = (month + 1).to_timestamp()
    return pd.date_range(start=start, end=end, freq=f"{tick_minutes}min", inclusive="left")


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_complete_zone_tick_grid(
    zones: List[int],
    ticks: pd.DatetimeIndex,
    tick_minutes: int,
) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [zones, ticks], names=["zone_id", "tick_start"]
    ).to_frame(index=False)
    grid["tick_id"] = ((grid["tick_start"] - ticks[0]) / pd.Timedelta(minutes=tick_minutes)).astype(int)
    grid["hour_of_day"] = grid["tick_start"].dt.hour.astype(int)
    grid["day_of_week"] = grid["tick_start"].dt.dayofweek.astype(int)
    return grid


def _aggregate_counts(
    df: pd.DataFrame,
    dt_col: str,
    zone_col: str,
    active_zones: List[int],
    tick_minutes: int,
) -> pd.DataFrame:
    work = df[[dt_col, zone_col]].copy()
    work[dt_col] = pd.to_datetime(work[dt_col], errors="coerce")
    work = work.dropna(subset=[dt_col, zone_col])
    work["zone_id"] = work[zone_col].astype(int)
    work = work[work["zone_id"].isin(active_zones)]
    work["tick_start"] = work[dt_col].dt.floor(f"{tick_minutes}min")
    counts = (
        work.groupby(["zone_id", "tick_start"], as_index=False)
        .size()
        .rename(columns={"size": "demand_count"})
    )
    counts["demand_count"] = counts["demand_count"].astype(int)
    return counts[["zone_id", "tick_start", "demand_count"]]


def _select_slow_zones(zones: List[int], fraction: float, seed: int) -> set[int]:
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0 or not zones:
        return set()
    n_slow = max(1, int(round(len(zones) * fraction)))
    rng = np.random.default_rng(seed)
    return set(int(z) for z in rng.choice(np.array(zones), size=n_slow, replace=False))


def _selected_tick_ids(metadata: Dict[str, Any], args: argparse.Namespace) -> List[int]:
    start = int(getattr(args, "start_tick_id", 0))
    max_ticks = int(getattr(args, "max_ticks", DEFAULT_MAX_TICKS))
    total = int(metadata["n_replay_ticks"])
    end = min(total, start + max_ticks)
    return list(range(start, end))


def _config_dict(args: argparse.Namespace, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
        if k not in {"handler"}
    }
    if extra:
        payload.update(extra)
    return payload


def _score_rule(snapshot: Dict[str, Any]) -> str:
    """A deliberately simple and deterministic recommendation rule."""
    observed = int(snapshot.get("demand_count", 0))
    baseline = float(snapshot.get("baseline_count", 0.0))
    recent_mean = float(snapshot.get("recent_mean", 0.0))

    # Elevated if current demand clearly exceeds both historical and recent local context.
    # The +2 guard prevents tiny baseline zones from becoming NEED due to one random trip.
    threshold = max(1.5 * baseline, recent_mean + 2.0, baseline + 2.0)
    return "NEED" if observed > 0 and observed >= threshold else "OK"


def _score_zone_local(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    sleep_s = float(snapshot.get("score_sleep_s", 0.0) or 0.0)
    if sleep_s > 0:
        time.sleep(sleep_s)

    decision = _score_rule(snapshot)
    ended = time.perf_counter()
    return {
        "zone_id": int(snapshot["zone_id"]),
        "tick_id": int(snapshot["tick_id"]),
        "decision": decision,
        "task_latency_s": ended - started,
        "score_sleep_s": sleep_s,
        "observed_count": int(snapshot.get("demand_count", 0)),
        "baseline_count": float(snapshot.get("baseline_count", 0.0)),
        "recent_mean": float(snapshot.get("recent_mean", 0.0)),
    }


# -----------------------------
# Ray runtime pieces
# -----------------------------
@ray.remote
class ZoneActor:
    def __init__(self, zone_id: int, zone_data_path: str):
        self.zone_id = int(zone_id)
        self.zone_data_path = zone_data_path
        self.data = pd.read_parquet(zone_data_path).sort_values("tick_id").reset_index(drop=True)
        self.by_tick: Dict[int, Dict[str, Any]] = {
            int(row.tick_id): row._asdict() for row in self.data.itertuples(index=False)
        }

        self.active_tick_id: Optional[int] = None
        self.last_tick_id: Optional[int] = None
        self.last_decision: Optional[str] = None
        self.recent_demand: List[int] = []

        # One durable accepted outcome per tick.
        self.decisions: Dict[int, Dict[str, Any]] = {}
        # Reported but not necessarily finalized async results.
        self.reports: Dict[int, Dict[str, Any]] = {}

        self.duplicate_reports = 0
        self.late_reports = 0
        self.duplicate_writes = 0
        self.fallbacks = 0

    def start_tick(self, tick_id: int) -> Dict[str, Any]:
        tick_id = int(tick_id)
        self.active_tick_id = tick_id
        return {"zone_id": self.zone_id, "tick_id": tick_id, "active": True}

    def next_snapshot(self, tick_id: int) -> Dict[str, Any]:
        tick_id = int(tick_id)
        if self.active_tick_id != tick_id:
            self.active_tick_id = tick_id

        row = self.by_tick.get(tick_id, {})
        demand_count = int(row.get("demand_count", 0) or 0)
        baseline_count = float(row.get("baseline_count", 0.0) or 0.0)
        recent_mean = float(np.mean(self.recent_demand[-4:])) if self.recent_demand else 0.0

        return {
            "zone_id": self.zone_id,
            "tick_id": tick_id,
            "tick_start": str(row.get("tick_start", "")),
            "demand_count": demand_count,
            "baseline_count": baseline_count,
            "hour_of_day": int(row.get("hour_of_day", 0) or 0),
            "day_of_week": int(row.get("day_of_week", 0) or 0),
            "recent_mean": recent_mean,
            "last_decision": self.last_decision,
        }

    def _commit_decision(
        self,
        tick_id: int,
        decision: str,
        used_fallback: bool,
        source: str,
        task_latency_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        tick_id = int(tick_id)
        decision = str(decision)

        if tick_id in self.decisions:
            # Idempotent: do not mutate state again for a duplicate write.
            self.duplicate_writes += 1
            existing = dict(self.decisions[tick_id])
            existing["write_status"] = "duplicate_ignored"
            return existing

        row = self.by_tick.get(tick_id, {})
        demand_count = int(row.get("demand_count", 0) or 0)
        baseline_count = float(row.get("baseline_count", 0.0) or 0.0)
        tick_start = row.get("tick_start", "")
        if isinstance(tick_start, pd.Timestamp):
            tick_start = tick_start.isoformat()

        accepted = {
            "zone_id": self.zone_id,
            "tick_id": tick_id,
            "tick_start": str(tick_start),
            "decision": decision,
            "used_fallback": bool(used_fallback),
            "source": source,
            "demand_count": demand_count,
            "baseline_count": baseline_count,
            "task_latency_s": task_latency_s,
            "write_status": "accepted",
        }

        self.decisions[tick_id] = accepted
        self.recent_demand.append(demand_count)
        self.last_tick_id = tick_id
        self.last_decision = decision
        if used_fallback:
            self.fallbacks += 1
        if self.active_tick_id == tick_id:
            self.active_tick_id = None
        return dict(accepted)

    def write_decision(
        self,
        tick_id: int,
        decision: str,
        used_fallback: bool = False,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Blocking mode calls this after the controller has accepted the task result.
        task_latency_s = None if result is None else result.get("task_latency_s")
        return self._commit_decision(
            tick_id=tick_id,
            decision=decision,
            used_fallback=used_fallback,
            source="blocking_controller" if not used_fallback else "fallback",
            task_latency_s=task_latency_s,
        )

    def report_decision(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Async scoring tasks call this directly. The actor decides whether this report is on time.
        tick_id = int(result["tick_id"])
        zone_id = int(result["zone_id"])
        if zone_id != self.zone_id:
            return {"accepted": False, "reason": "wrong_zone", "zone_id": self.zone_id, "tick_id": tick_id}

        if tick_id in self.decisions:
            self.late_reports += 1
            return {"accepted": False, "reason": "closed_tick", "zone_id": self.zone_id, "tick_id": tick_id}

        if self.active_tick_id != tick_id:
            self.late_reports += 1
            return {"accepted": False, "reason": "inactive_tick", "zone_id": self.zone_id, "tick_id": tick_id}

        if tick_id in self.reports:
            self.duplicate_reports += 1
            return {"accepted": False, "reason": "duplicate_report", "zone_id": self.zone_id, "tick_id": tick_id}

        self.reports[tick_id] = dict(result)
        return {"accepted": True, "reason": "reported", "zone_id": self.zone_id, "tick_id": tick_id}

    def status(self, tick_id: int) -> Dict[str, Any]:
        tick_id = int(tick_id)
        return {
            "zone_id": self.zone_id,
            "tick_id": tick_id,
            "active": self.active_tick_id == tick_id,
            "has_report": tick_id in self.reports,
            "finalized": tick_id in self.decisions,
        }

    def finalize_tick(self, tick_id: int, fallback_policy: str = DEFAULT_FALLBACK_POLICY) -> Dict[str, Any]:
        tick_id = int(tick_id)
        if tick_id in self.decisions:
            existing = dict(self.decisions[tick_id])
            existing["write_status"] = "already_finalized"
            return existing

        if tick_id in self.reports:
            report = self.reports[tick_id]
            return self._commit_decision(
                tick_id=tick_id,
                decision=report["decision"],
                used_fallback=False,
                source="async_report",
                task_latency_s=report.get("task_latency_s"),
            )

        policy = (fallback_policy or DEFAULT_FALLBACK_POLICY).lower()
        if policy in {"always_previous", "previous_else_ok"}:
            decision = self.last_decision or "OK"
        elif policy == "always_ok":
            decision = "OK"
        else:
            raise ValueError(f"Unsupported fallback_policy={fallback_policy!r}")

        return self._commit_decision(
            tick_id=tick_id,
            decision=decision,
            used_fallback=True,
            source=f"fallback:{policy}",
            task_latency_s=None,
        )

    def get_decisions(self) -> List[Dict[str, Any]]:
        return [dict(self.decisions[k]) for k in sorted(self.decisions)]

    def get_counters(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "duplicate_reports": self.duplicate_reports,
            "late_reports": self.late_reports,
            "duplicate_writes": self.duplicate_writes,
            "fallbacks": self.fallbacks,
            "accepted_decisions": len(self.decisions),
        }


@ray.remote(max_retries=1)
def score_zone(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return _score_zone_local(snapshot)


@ray.remote(max_retries=1)
def score_zone_and_report(snapshot: Dict[str, Any], zone_actor: ray.actor.ActorHandle) -> Dict[str, Any]:
    result = _score_zone_local(snapshot)
    ack = ray.get(zone_actor.report_decision.remote(result))
    result["report_ack"] = ack
    return result


# -----------------------------
# Prepare step
# -----------------------------
def prepare_assets(
    reference_parquet: Path,
    replay_parquet: Path,
    output_dir: Path,
    n_zones: int = DEFAULT_N_ZONES,
    tick_minutes: int = DEFAULT_TICK_MINUTES,
    seed: int = DEFAULT_SEED,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zones_dir = output_dir / "zones"
    if zones_dir.exists():
        shutil.rmtree(zones_dir)
    zones_dir.mkdir(parents=True, exist_ok=True)

    reference_df = pd.read_parquet(reference_parquet)
    replay_df = pd.read_parquet(replay_parquet)

    ref_dt_col, rep_dt_col, ref_month, rep_month = _validate_adjacent_months(reference_df, replay_df)
    ref_zone_col = _pickup_zone_col(reference_df)
    rep_zone_col = _pickup_zone_col(replay_df)

    ref_work = reference_df[[ref_dt_col, ref_zone_col]].copy()
    ref_work[ref_dt_col] = pd.to_datetime(ref_work[ref_dt_col], errors="coerce")
    ref_work = ref_work.dropna(subset=[ref_dt_col, ref_zone_col])
    ref_work["zone_id"] = ref_work[ref_zone_col].astype(int)

    zone_counts = (
        ref_work.groupby("zone_id", as_index=False)
        .size()
        .rename(columns={"size": "reference_pickups"})
        .sort_values(["reference_pickups", "zone_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    active_zones = [int(z) for z in zone_counts.head(int(n_zones))["zone_id"].tolist()]
    if not active_zones:
        raise ValueError("No active zones were selected from the reference month")

    ref_ticks = _month_tick_range(ref_month, tick_minutes)
    rep_ticks = _month_tick_range(rep_month, tick_minutes)

    ref_grid = _make_complete_zone_tick_grid(active_zones, ref_ticks, tick_minutes)
    ref_counts = _aggregate_counts(reference_df, ref_dt_col, ref_zone_col, active_zones, tick_minutes)
    ref_full = ref_grid.merge(ref_counts, on=["zone_id", "tick_start"], how="left")
    ref_full["demand_count"] = ref_full["demand_count"].fillna(0).astype(int)

    baselines = (
        ref_full.groupby(["zone_id", "hour_of_day", "day_of_week"], as_index=False)["demand_count"]
        .mean()
        .rename(columns={"demand_count": "baseline_count"})
    )
    zone_default_baselines = (
        ref_full.groupby("zone_id", as_index=False)["demand_count"]
        .mean()
        .rename(columns={"demand_count": "zone_default_baseline_count"})
    )

    replay_grid = _make_complete_zone_tick_grid(active_zones, rep_ticks, tick_minutes)
    replay_counts = _aggregate_counts(replay_df, rep_dt_col, rep_zone_col, active_zones, tick_minutes)
    replay_full = replay_grid.merge(replay_counts, on=["zone_id", "tick_start"], how="left")
    replay_full["demand_count"] = replay_full["demand_count"].fillna(0).astype(int)
    replay_full = replay_full.merge(baselines, on=["zone_id", "hour_of_day", "day_of_week"], how="left")
    replay_full = replay_full.merge(zone_default_baselines, on="zone_id", how="left")
    replay_full["baseline_count"] = replay_full["baseline_count"].fillna(
        replay_full["zone_default_baseline_count"]
    )
    replay_full["baseline_count"] = replay_full["baseline_count"].fillna(0.0).astype(float)
    replay_full = replay_full.drop(columns=["zone_default_baseline_count"])

    _safe_to_parquet(baselines, output_dir / "reference_baselines.parquet")
    _safe_to_parquet(replay_full, output_dir / "replay_ticks.parquet")
    zone_counts.to_csv(output_dir / "reference_zone_counts.csv", index=False)

    for zone_id, part in replay_full.groupby("zone_id", sort=True):
        _safe_to_parquet(part, zones_dir / f"zone_{int(zone_id)}.parquet")

    # Required pandas cross-check on a deterministic sample window.
    sample_start = rep_ticks[min(4, len(rep_ticks) - 1)]
    sample_end_index = min(len(rep_ticks), 4 + 24)
    sample_end = rep_ticks[sample_end_index] if sample_end_index < len(rep_ticks) else rep_ticks[-1] + pd.Timedelta(minutes=tick_minutes)

    direct = replay_df[[rep_dt_col, rep_zone_col]].copy()
    direct[rep_dt_col] = pd.to_datetime(direct[rep_dt_col], errors="coerce")
    direct = direct.dropna(subset=[rep_dt_col, rep_zone_col])
    direct["zone_id"] = direct[rep_zone_col].astype(int)
    direct = direct[
        direct["zone_id"].isin(active_zones)
        & (direct[rep_dt_col] >= sample_start)
        & (direct[rep_dt_col] < sample_end)
    ].copy()
    direct["tick_start"] = direct[rep_dt_col].dt.floor(f"{tick_minutes}min")
    direct_counts = (
        direct.groupby(["zone_id", "tick_start"], as_index=False)
        .size()
        .rename(columns={"size": "direct_count"})
    )
    prepared_sample = replay_full[
        (replay_full["tick_start"] >= sample_start) & (replay_full["tick_start"] < sample_end)
    ][["zone_id", "tick_start", "demand_count"]]
    merged = prepared_sample.merge(direct_counts, on=["zone_id", "tick_start"], how="left")
    merged["direct_count"] = merged["direct_count"].fillna(0).astype(int)
    crosscheck = {
        "sample_start": sample_start.isoformat(),
        "sample_end": sample_end.isoformat(),
        "prepared_total": int(merged["demand_count"].sum()),
        "direct_total": int(merged["direct_count"].sum()),
        "row_mismatches": int((merged["demand_count"] != merged["direct_count"]).sum()),
    }
    crosscheck["ok"] = bool(
        crosscheck["prepared_total"] == crosscheck["direct_total"]
        and crosscheck["row_mismatches"] == 0
    )

    metadata = {
        "reference_parquet": str(reference_parquet),
        "replay_parquet": str(replay_parquet),
        "reference_month": str(ref_month),
        "replay_month": str(rep_month),
        "pickup_datetime_columns": {"reference": ref_dt_col, "replay": rep_dt_col},
        "pickup_zone_columns": {"reference": ref_zone_col, "replay": rep_zone_col},
        "active_zones": active_zones,
        "n_zones": len(active_zones),
        "tick_minutes": int(tick_minutes),
        "seed": int(seed),
        "n_reference_ticks": len(ref_ticks),
        "n_replay_ticks": len(rep_ticks),
        "crosscheck": crosscheck,
    }
    _write_json(output_dir / "active_zones.json", active_zones)
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "crosscheck.json", crosscheck)

    if not crosscheck["ok"]:
        raise ValueError(f"Prepared replay cross-check failed: {crosscheck}")


# -----------------------------
# Run helpers
# -----------------------------
def _load_runtime(prepared_dir: Path) -> Tuple[Dict[str, Any], List[int], Dict[int, ray.actor.ActorHandle]]:
    metadata = _read_json(prepared_dir / "metadata.json")
    zones = [int(z) for z in metadata["active_zones"]]
    actors = {
        z: ZoneActor.remote(z, str(prepared_dir / "zones" / f"zone_{z}.parquet"))
        for z in zones
    }
    return metadata, zones, actors


def _add_skew(snapshot: Dict[str, Any], slow_zones: set[int], args: argparse.Namespace) -> Dict[str, Any]:
    snapshot = dict(snapshot)
    zone_id = int(snapshot["zone_id"])
    is_slow = zone_id in slow_zones
    snapshot["is_slow_zone"] = is_slow
    snapshot["score_sleep_s"] = float(args.slow_zone_sleep_s) if is_slow else 0.0
    return snapshot


def _summarize_tick_results(
    tick_id: int,
    mode: str,
    tick_latency_s: float,
    results: List[Dict[str, Any]],
    finalizations: List[Dict[str, Any]],
    completed_before_finalization: int,
) -> Dict[str, Any]:
    latencies = [float(r.get("task_latency_s", 0.0) or 0.0) for r in results]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    max_latency = float(np.max(latencies)) if latencies else 0.0
    fallback_count = int(sum(1 for f in finalizations if f.get("used_fallback")))
    return {
        "tick_id": int(tick_id),
        "mode": mode,
        "tick_latency_s": float(tick_latency_s),
        "mean_zone_latency_s": mean_latency,
        "max_zone_latency_s": max_latency,
        "max_mean_latency_ratio": float(max_latency / mean_latency) if mean_latency > 0 else 0.0,
        "completed_before_finalization": int(completed_before_finalization),
        "fallback_count": fallback_count,
        "zones_finalized": len(finalizations),
    }


def _write_run_artifacts(
    output_dir: Path,
    mode: str,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    slow_zones: set[int],
    tick_summaries: List[Dict[str, Any]],
    latency_log: List[Dict[str, Any]],
    actors: Dict[int, ray.actor.ActorHandle],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    decisions_by_zone = ray.get([actor.get_decisions.remote() for actor in actors.values()])
    decisions = [item for zone_items in decisions_by_zone for item in zone_items]
    decisions_df = pd.DataFrame(decisions).sort_values(["tick_id", "zone_id"]) if decisions else pd.DataFrame()
    decisions_df.to_csv(output_dir / "decisions.csv", index=False)

    metrics_df = pd.DataFrame(tick_summaries)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    counters = ray.get([actor.get_counters.remote() for actor in actors.values()])
    _write_json(output_dir / "actor_counters.json", counters)
    _write_json(output_dir / "latency_log.json", latency_log)
    _write_json(output_dir / "tick_summary.json", tick_summaries)
    _write_json(
        output_dir / "run_config.json",
        _config_dict(
            args,
            extra={
                "mode": mode,
                "slow_zones": sorted(int(z) for z in slow_zones),
                "prepared_metadata": metadata,
            },
        ),
    )


# -----------------------------
# Run modes
# -----------------------------
def run_blocking(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    metadata, zones, actors = _load_runtime(prepared_dir)
    tick_ids = _selected_tick_ids(metadata, args)
    slow_zones = _select_slow_zones(zones, args.slow_zone_fraction, args.seed)

    tick_summaries: List[Dict[str, Any]] = []
    latency_log: List[Dict[str, Any]] = []

    for tick_id in tick_ids:
        tick_started = time.perf_counter()
        ray.get([actors[z].start_tick.remote(tick_id) for z in zones])

        snapshots = ray.get([actors[z].next_snapshot.remote(tick_id) for z in zones])
        snapshots = [_add_skew(s, slow_zones, args) for s in snapshots]

        result_refs = [score_zone.remote(s) for s in snapshots]
        results = ray.get(result_refs)
        latency_log.extend(results)

        finalizations = ray.get(
            [
                actors[int(r["zone_id"])].write_decision.remote(
                    int(r["tick_id"]), r["decision"], False, r
                )
                for r in results
            ]
        )

        tick_latency_s = time.perf_counter() - tick_started
        tick_summaries.append(
            _summarize_tick_results(
                tick_id=tick_id,
                mode="blocking",
                tick_latency_s=tick_latency_s,
                results=results,
                finalizations=finalizations,
                completed_before_finalization=len(results),
            )
        )

    _write_run_artifacts(output_dir, "blocking", args, metadata, slow_zones, tick_summaries, latency_log, actors)


def run_async(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    metadata, zones, actors = _load_runtime(prepared_dir)
    tick_ids = _selected_tick_ids(metadata, args)
    slow_zones = _select_slow_zones(zones, args.slow_zone_fraction, args.seed)

    tick_summaries: List[Dict[str, Any]] = []
    latency_log: List[Dict[str, Any]] = []

    max_inflight = max(1, int(args.max_inflight_zones))
    completion_target = int(math.ceil(len(zones) * float(args.completion_fraction)))
    poll_s = 0.05

    global_pending: List[ray.ObjectRef] = []
    ref_meta: Dict[ray.ObjectRef, Dict[str, Any]] = {}

    def drain_finished_refs(timeout: float = 0.0) -> None:
        nonlocal global_pending, latency_log
        if not global_pending:
            return
        done, remaining = ray.wait(global_pending, num_returns=len(global_pending), timeout=timeout)
        if done:
            latency_log.extend(ray.get(done))
            for ref in done:
                ref_meta.pop(ref, None)
        global_pending = remaining

    for tick_id in tick_ids:
        tick_started = time.perf_counter()
        ray.get([actors[z].start_tick.remote(tick_id) for z in zones])

        snapshots = ray.get([actors[z].next_snapshot.remote(tick_id) for z in zones])
        snapshot_by_zone = {int(s["zone_id"]): _add_skew(s, slow_zones, args) for s in snapshots}

        launch_queue = list(zones)
        tick_result_refs: set[ray.ObjectRef] = set()

        def launch_more_for_tick() -> None:
            while launch_queue and len(global_pending) < max_inflight:
                zone_id = int(launch_queue.pop(0))
                ref = score_zone_and_report.remote(snapshot_by_zone[zone_id], actors[zone_id])
                global_pending.append(ref)
                ref_meta[ref] = {"zone_id": zone_id, "tick_id": tick_id}
                tick_result_refs.add(ref)

        launch_more_for_tick()

        finalization_reason = "unknown"
        completed_before_finalization = 0

        while True:
            statuses = ray.get([actors[z].status.remote(tick_id) for z in zones])
            completed_before_finalization = int(sum(1 for s in statuses if s["has_report"] or s["finalized"]))
            elapsed = time.perf_counter() - tick_started

            if completed_before_finalization >= completion_target:
                finalization_reason = "completion_fraction"
                break
            if elapsed >= float(args.tick_timeout_s):
                finalization_reason = "timeout"
                break
            if not launch_queue and not any(ref_meta.get(ref, {}).get("tick_id") == tick_id for ref in global_pending):
                finalization_reason = "all_launched_finished"
                break

            if global_pending:
                done, remaining = ray.wait(global_pending, num_returns=1, timeout=poll_s)
                if done:
                    latency_log.extend(ray.get(done))
                    for ref in done:
                        ref_meta.pop(ref, None)
                global_pending = remaining
            else:
                time.sleep(poll_s)

            launch_more_for_tick()

        finalizations = ray.get(
            [actors[z].finalize_tick.remote(tick_id, args.fallback_policy) for z in zones]
        )
        tick_latency_s = time.perf_counter() - tick_started

        # Attach reason to the tick summary.
        summary = _summarize_tick_results(
            tick_id=tick_id,
            mode="async",
            tick_latency_s=tick_latency_s,
            results=[r for r in latency_log if int(r.get("tick_id", -1)) == tick_id],
            finalizations=finalizations,
            completed_before_finalization=completed_before_finalization,
        )
        summary["finalization_reason"] = finalization_reason
        summary["completion_target"] = completion_target
        summary["max_inflight_zones"] = max_inflight
        tick_summaries.append(summary)

        # Do not wait for late tasks here. They may still report and will be ignored by actors.
        # But remove any already-completed old refs so the bounded queue does not fill with done refs.
        drain_finished_refs(timeout=0.0)

    # After the replay window is done, give late tasks a bounded opportunity to finish so late-report
    # counters become visible in artifacts. This does not affect tick finalization latency.
    final_drain_deadline = time.perf_counter() + max(1.0, float(args.slow_zone_sleep_s) * 2.0)
    while global_pending and time.perf_counter() < final_drain_deadline:
        drain_finished_refs(timeout=0.1)

    _write_run_artifacts(output_dir, "async", args, metadata, slow_zones, tick_summaries, latency_log, actors)


def run_stress(prepared_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    # Reuse async path with harsher skew. CLI values still win if they are already harsher.
    args = argparse.Namespace(**vars(args))
    args.slow_zone_fraction = max(float(args.slow_zone_fraction), 0.50)
    args.slow_zone_sleep_s = max(float(args.slow_zone_sleep_s), 2.00)
    args.tick_timeout_s = min(float(args.tick_timeout_s), 1.00)
    run_async(prepared_dir, output_dir, args)


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capstone starter for TLC-backed per-zone recommendations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--reference-parquet", type=Path, required=True)
    prepare.add_argument("--replay-parquet", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--n-zones", type=int, default=DEFAULT_N_ZONES)
    prepare.add_argument("--tick-minutes", type=int, default=DEFAULT_TICK_MINUTES)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.set_defaults(handler=handle_prepare)

    run = subparsers.add_parser("run")
    run.add_argument("--prepared-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--mode", choices=("blocking", "async", "stress"), required=True)
    run.add_argument("--max-inflight-zones", type=int, default=4)
    run.add_argument("--tick-timeout-s", type=float, default=2.0)
    run.add_argument("--completion-fraction", type=float, default=0.75)
    run.add_argument("--slow-zone-fraction", type=float, default=0.25)
    run.add_argument("--slow-zone-sleep-s", type=float, default=1.0)
    run.add_argument("--fallback-policy", default=DEFAULT_FALLBACK_POLICY)
    run.add_argument("--ray-address", default=None)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--start-tick-id", type=int, default=0)
    run.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    run.set_defaults(handler=handle_run)

    return parser


def handle_prepare(args: argparse.Namespace) -> None:
    prepare_assets(
        args.reference_parquet,
        args.replay_parquet,
        args.output_dir,
        n_zones=args.n_zones,
        tick_minutes=args.tick_minutes,
        seed=args.seed,
    )


def handle_run(args: argparse.Namespace) -> None:
    if not ray.is_initialized():
        if args.ray_address:
            ray.init(address=args.ray_address)
        else:
            ray.init(include_dashboard=False, num_cpus=2)

    try:
        if args.mode == "blocking":
            run_blocking(args.prepared_dir, args.output_dir, args)
        elif args.mode == "async":
            run_async(args.prepared_dir, args.output_dir, args)
        else:
            run_stress(args.prepared_dir, args.output_dir, args)
    finally:
        # Safe in local runs; on Ray Jobs this simply disconnects the driver.
        ray.shutdown()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
