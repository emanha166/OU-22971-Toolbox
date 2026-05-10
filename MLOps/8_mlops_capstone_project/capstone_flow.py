from pathlib import Path
import json

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from metaflow import FlowSpec, Parameter, step
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
from typing import Any

try:
    import nannyml as nml
except Exception:
    nml = None


EXPERIMENT_NAME = "8_capstone_green_taxi"


def build_features(df: pd.DataFrame):
    """
    Convert raw taxi-like data into a stable model-ready feature table.
    This function will later be reused for training, evaluation, and inference.
    """
    out = pd.DataFrame(index=df.index)

    pickup_dt = pd.to_datetime(df["lpep_pickup_datetime"], errors="coerce")

    out["pickup_hour"] = pickup_dt.dt.hour.fillna(0).astype(int)
    out["pickup_dayofweek"] = pickup_dt.dt.dayofweek.fillna(0).astype(int)
    out["pickup_month"] = pickup_dt.dt.month.fillna(0).astype(int)

    out["passenger_count"] = (
        pd.to_numeric(df["passenger_count"], errors="coerce")
        .fillna(1)
        .clip(lower=0, upper=8)
    )

    trip_distance = (
        pd.to_numeric(df["trip_distance"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=50)
    )

    fare_amount = (
        pd.to_numeric(df["fare_amount"], errors="coerce")
        .fillna(0)
        .clip(lower=0, upper=300)
    )

    out["trip_distance_log"] = np.log1p(trip_distance)
    out["fare_amount_log"] = np.log1p(fare_amount)

    out["PULocationID"] = (
        pd.to_numeric(df["PULocationID"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    out["DOLocationID"] = (
        pd.to_numeric(df["DOLocationID"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )

    y = pd.to_numeric(df["tip_amount"], errors="coerce").fillna(0)

    feature_spec = {
        "feature_names": list(out.columns),
        "dtypes": {col: str(dtype) for col, dtype in out.dtypes.items()},
        "target": "tip_amount",
        "n_features": int(out.shape[1]),
    }

    return out, y, feature_spec


class GreenTaxiCapstoneFlow(FlowSpec):
    reference_path = Parameter(
        "reference-path",
        default="data/reference.parquet",
        help="Path to the reference dataset.",
    )

    batch_path = Parameter(
        "batch-path",
        default="data/batch_baseline.parquet",
        help="Path to the new batch dataset.",
    )

    tracking_uri = Parameter(
        "tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking server URI.",
    )

    model_name = Parameter(
        "model-name",
        default="green_taxi_tip_model",
        help="Registered model name in MLflow Model Registry.",
    )

    invalid_row_tolerance = Parameter(
        "invalid-row-tolerance",
        default=0.01,
        help="Maximum allowed fraction of invalid rows before rejecting the batch.",
    )

    max_champion_rmse = Parameter(
        "max-champion-rmse",
        default=2.0,
        help="Maximum acceptable champion RMSE before retraining is recommended.",
    )

    min_champion_r2 = Parameter(
        "min-champion-r2",
        default=0.0,
        help="Minimum acceptable champion R2 before retraining is recommended.",
    )

    max_rmse_increase_pct = Parameter(
        "max-rmse-increase-pct",
        default=10.0,
        help="Maximum allowed percentage increase in RMSE before retraining is recommended.",
    )

    min_candidate_improvement = Parameter(
        "min-candidate-improvement",
        default=1.0,
        help="Minimum required candidate RMSE improvement percentage before promotion.",
    )

    candidate_training_path = Parameter(
        "candidate-training-path",
        default="",
        help="Optional additional dataset used to train the candidate model together with the reference dataset.",
    )

    @step
    def start(self):
        print("Starting Green Taxi MLOps capstone flow")
        print(f"Reference path: {self.reference_path}")
        print(f"Batch path: {self.batch_path}")
        print(f"MLflow tracking URI: {self.tracking_uri}")

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        self.next(self.load_data)

    @step
    def load_data(self):
        reference_path = Path(self.reference_path)
        batch_path = Path(self.batch_path)

        if not reference_path.exists():
            raise FileNotFoundError(f"Reference dataset not found: {reference_path}")

        if not batch_path.exists():
            raise FileNotFoundError(f"Batch dataset not found: {batch_path}")

        self.reference_df = pd.read_parquet(reference_path)
        self.batch_df = pd.read_parquet(batch_path)

        self.reference_rows = int(len(self.reference_df))
        self.batch_rows = int(len(self.batch_df))

        print(f"Loaded reference data: {self.reference_df.shape}")
        print(f"Loaded batch data: {self.batch_df.shape}")

        self.next(self.integrity_gate)

    @step
    def integrity_gate(self):
        required_columns = [
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "tip_amount",
        ]

        missing_columns = [
            col for col in required_columns if col not in self.batch_df.columns
        ]

        hard_failures = []
        integrity_warnings = []

        if missing_columns:
            hard_failures.append(f"Missing required columns: {missing_columns}")

        batch_rows = max(int(self.batch_rows), 1)
        negative_trip_distance_count = int(
            (self.batch_df["trip_distance"] < 0).sum()
            if "trip_distance" in self.batch_df.columns
            else 0
        )
        negative_fare_amount_count = int(
            (self.batch_df["fare_amount"] < 0).sum()
            if "fare_amount" in self.batch_df.columns
            else 0
        )
        negative_tip_amount_count = int(
            (self.batch_df["tip_amount"] < 0).sum()
            if "tip_amount" in self.batch_df.columns
            else 0
        )

        pickup_dt = pd.to_datetime(
            self.batch_df["lpep_pickup_datetime"], errors="coerce"
        ) if "lpep_pickup_datetime" in self.batch_df.columns else pd.Series([], dtype="datetime64[ns]")
        dropoff_dt = pd.to_datetime(
            self.batch_df["lpep_dropoff_datetime"], errors="coerce"
        ) if "lpep_dropoff_datetime" in self.batch_df.columns else pd.Series([], dtype="datetime64[ns]")
        invalid_time_mask = (
            dropoff_dt < pickup_dt
        ) | pickup_dt.isna() | dropoff_dt.isna()
        invalid_time_count = int(invalid_time_mask.sum())

        tolerance = float(self.invalid_row_tolerance)

        for count, label in [
            (negative_trip_distance_count, "negative trip_distance"),
            (negative_fare_amount_count, "negative fare_amount"),
            (negative_tip_amount_count, "negative tip_amount"),
            (invalid_time_count, "invalid pickup/dropoff datetime"),
        ]:
            ratio = count / batch_rows
            if count > 0:
                if ratio > tolerance:
                    hard_failures.append(
                        f"Found {count} rows with {label} ({ratio:.4f} ratio)"
                    )
                else:
                    integrity_warnings.append(
                        f"Found {count} rows with {label} ({ratio:.4f} ratio), within tolerance"
                    )

        self.negative_trip_distance_count = negative_trip_distance_count
        self.negative_fare_amount_count = negative_fare_amount_count
        self.negative_tip_amount_count = negative_tip_amount_count
        self.invalid_time_count = invalid_time_count

        self.integrity_warnings = integrity_warnings
        self.integrity_warning_count = int(len(integrity_warnings))
        self.integrity_warn = len(integrity_warnings) > 0

        self.hard_failures = hard_failures
        self.hard_failure_count = int(len(hard_failures))
        self.batch_rejected = self.hard_failure_count > 0
        self.integrity_status = "failed" if self.batch_rejected else "passed"

        if self.batch_rejected:
            print("Integrity gate failed")
            for reason in hard_failures:
                print(f"- {reason}")
            self.action = "reject_batch"
        else:
            print("Integrity gate passed")
            if self.integrity_warn:
                print("Integrity warnings:")
                for warning in integrity_warnings:
                    print(f"- {warning}")
            self.action = "continue"

        self.next(self.nannyml_soft_gate)

    @step
    def nannyml_soft_gate(self):
        print("Running NannyML-inspired soft monitoring gate")

        self.nannyml_available = nml is not None
        features_to_monitor = [
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "tip_amount",
        ]

        drift_scores = {}
        nannyml_warnings = []

        for feature in features_to_monitor:
            if feature not in self.reference_df.columns or feature not in self.batch_df.columns:
                continue

            reference_values = pd.to_numeric(
                self.reference_df[feature], errors="coerce"
            ).dropna()
            batch_values = pd.to_numeric(self.batch_df[feature], errors="coerce").dropna()

            if len(reference_values) < 2 or len(batch_values) < 2:
                continue

            reference_std = float(reference_values.std())
            mean_diff = float(abs(batch_values.mean() - reference_values.mean()))
            drift_score = mean_diff / max(reference_std, 1e-6)
            drift_scores[feature] = float(drift_score)

            if drift_score >= 1.0:
                nannyml_warnings.append(
                    f"Feature '{feature}' drift score {drift_score:.2f} indicates strong shift"
                )
            elif drift_score >= 0.5:
                nannyml_warnings.append(
                    f"Feature '{feature}' drift score {drift_score:.2f} indicates moderate shift"
                )

        self.nannyml_summary = {
            "nannyml_available": self.nannyml_available,
            "reference_rows": int(self.reference_rows),
            "batch_rows": int(self.batch_rows),
            "features_monitored": features_to_monitor,
            "drift_scores": drift_scores,
            "warnings": nannyml_warnings,
            "status": "warn" if nannyml_warnings else "ok",
        }

        self.nannyml_status = self.nannyml_summary["status"]
        self.nannyml_warn = len(nannyml_warnings) > 0
        self.nannyml_warning_count = int(len(nannyml_warnings))
        self.nannyml_warnings = nannyml_warnings
        self.nannyml_drift_scores = drift_scores

        if self.nannyml_warn:
            print("NannyML soft monitoring warnings:")
            for warning in nannyml_warnings:
                print(f"- {warning}")
        else:
            print("No NannyML soft monitoring warnings detected")

        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        if self.batch_rejected:
            print("Skipping feature engineering because batch was rejected")

            self.feature_engineering_status = "skipped"
            self.feature_spec = {}
            self.reference_feature_rows = 0
            self.batch_feature_rows = 0
            self.n_features = 0
            self.reference_rows_after_cleaning = int(self.reference_rows)
            self.batch_rows_after_cleaning = int(self.batch_rows)
            self.reference_rows_dropped = 0
            self.batch_rows_dropped = 0

        else:
            print("Running feature engineering")

            reference_pickup_dt = pd.to_datetime(
                self.reference_df["lpep_pickup_datetime"], errors="coerce"
            )
            reference_dropoff_dt = pd.to_datetime(
                self.reference_df["lpep_dropoff_datetime"], errors="coerce"
            )
            reference_valid_mask = (
                pd.to_numeric(self.reference_df["trip_distance"], errors="coerce") >= 0
            ) & (
                pd.to_numeric(self.reference_df["fare_amount"], errors="coerce") >= 0
            ) & (
                pd.to_numeric(self.reference_df["tip_amount"], errors="coerce") >= 0
            ) & (reference_dropoff_dt >= reference_pickup_dt)

            batch_pickup_dt = pd.to_datetime(
                self.batch_df["lpep_pickup_datetime"], errors="coerce"
            )
            batch_dropoff_dt = pd.to_datetime(
                self.batch_df["lpep_dropoff_datetime"], errors="coerce"
            )
            batch_valid_mask = (
                pd.to_numeric(self.batch_df["trip_distance"], errors="coerce") >= 0
            ) & (
                pd.to_numeric(self.batch_df["fare_amount"], errors="coerce") >= 0
            ) & (
                pd.to_numeric(self.batch_df["tip_amount"], errors="coerce") >= 0
            ) & (batch_dropoff_dt >= batch_pickup_dt)

            self.reference_df_clean = self.reference_df[reference_valid_mask].copy()
            self.batch_df_clean = self.batch_df[batch_valid_mask].copy()

            self.reference_rows_after_cleaning = int(len(self.reference_df_clean))
            self.batch_rows_after_cleaning = int(len(self.batch_df_clean))
            self.reference_rows_dropped = int(self.reference_rows - self.reference_rows_after_cleaning)
            self.batch_rows_dropped = int(self.batch_rows - self.batch_rows_after_cleaning)

            print(f"Reference rows after cleaning: {self.reference_rows_after_cleaning}")
            print(f"Batch rows after cleaning: {self.batch_rows_after_cleaning}")
            print(f"Reference rows dropped: {self.reference_rows_dropped}")
            print(f"Batch rows dropped: {self.batch_rows_dropped}")

            self.X_reference, self.y_reference, self.feature_spec = build_features(
                self.reference_df_clean
            )
            self.X_batch, self.y_batch, batch_feature_spec = build_features(
                self.batch_df_clean
            )

            reference_columns = list(self.X_reference.columns)
            batch_columns = list(self.X_batch.columns)

            if reference_columns != batch_columns:
                raise ValueError(
                    "Feature schema mismatch between reference and batch datasets"
                )

            self.feature_engineering_status = "completed"
            self.reference_feature_rows = int(len(self.X_reference))
            self.batch_feature_rows = int(len(self.X_batch))
            self.n_features = int(self.X_reference.shape[1])

            print(f"Reference feature table: {self.X_reference.shape}")
            print(f"Batch feature table: {self.X_batch.shape}")
            print(f"Feature columns: {reference_columns}")

        self.next(self.load_or_bootstrap_champion)

    @step
    def load_or_bootstrap_champion(self):
        if self.batch_rejected:
            print("Champion bootstrap is skipped because the batch was rejected.")
            self.champion_status = "skipped"
            self.champion_version = ""
            self.bootstrap_rmse = 0.0
            self.bootstrap_r2 = 0.0
        else:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = MlflowClient()
            try:
                champion = client.get_model_version_by_alias(self.model_name, "champion")
                print("Existing champion found.")
                self.champion_status = "loaded_existing"
                self.champion_version = str(champion.version)
                self.bootstrap_rmse = 0.0
                self.bootstrap_r2 = 0.0
            except Exception:
                print("No champion found. Bootstrapping initial champion model.")
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X_reference, self.y_reference, test_size=0.2, random_state=42
                )
                model = RandomForestRegressor(
                    n_estimators=80, random_state=42, min_samples_leaf=5, n_jobs=-1
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = math.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
                with mlflow.start_run(run_name="bootstrap_champion"):
                    mlflow.log_param("model_type", "RandomForestRegressor")
                    mlflow.log_param("n_estimators", 80)
                    mlflow.log_param("min_samples_leaf", 5)
                    mlflow.log_metric("bootstrap_rmse", rmse)
                    mlflow.log_metric("bootstrap_r2", r2)
                    model_info = mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name=self.model_name,
                        signature=infer_signature(X_train, model.predict(X_train)),
                        input_example=X_train.head(5),
                        await_registration_for=300,
                    )
                    version = getattr(model_info, "registered_model_version", None) or getattr(model_info, "model_version", None)
                    if version is None:
                        raise RuntimeError("Could not determine registered model version from MLflow model_info.")
                    version = str(version)
                    client.set_registered_model_alias(self.model_name, "champion", version)
                    client.set_model_version_tag(self.model_name, version, "role", "champion")
                    client.set_model_version_tag(self.model_name, version, "promotion_reason", "bootstrap")
                    client.set_model_version_tag(self.model_name, version, "validation_status", "approved")
                self.champion_status = "bootstrapped"
                self.champion_version = str(version)
                self.bootstrap_rmse = float(rmse)
                self.bootstrap_r2 = float(r2)

        self.next(self.evaluate_champion)

    @step
    def evaluate_champion(self):
        if self.batch_rejected:
            print("Skipping champion evaluation because batch was rejected.")
            self.champion_evaluation_status = "skipped"
            self.rmse_baseline = 0.0
            self.r2_baseline = 0.0
            self.rmse_champion = 0.0
            self.r2_champion = 0.0
            self.rmse_increase_pct = 0.0
        else:
            print("Evaluating champion on batch.")
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            model_uri = f"models:/{self.model_name}@champion"
            print(f"Loading champion model from: {model_uri}")
            try:
                champion_model = mlflow.pyfunc.load_model(model_uri)
                if champion_model is None:
                    raise ValueError("pyfunc returned None")
            except Exception as e:
                print(f"Warning: pyfunc load failed ({e}), trying sklearn load")
                try:
                    champion_model = mlflow.sklearn.load_model(model_uri)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load champion model from {model_uri}: pyfunc error {e}, sklearn error {e2}")
            y_pred_baseline = champion_model.predict(self.X_reference)
            self.rmse_baseline = float(math.sqrt(mean_squared_error(self.y_reference, y_pred_baseline)))
            self.r2_baseline = float(r2_score(self.y_reference, y_pred_baseline))
            y_pred_champion = champion_model.predict(self.X_batch)
            rmse_champion = math.sqrt(mean_squared_error(self.y_batch, y_pred_champion))
            r2_champion = r2_score(self.y_batch, y_pred_champion)
            if self.rmse_baseline > 0:
                self.rmse_increase_pct = float(((rmse_champion - self.rmse_baseline) / self.rmse_baseline) * 100)
            else:
                self.rmse_increase_pct = 0.0
            self.champion_evaluation_status = "completed"
            self.rmse_champion = float(rmse_champion)
            self.r2_champion = float(r2_champion)
            print(f"Baseline RMSE: {self.rmse_baseline}")
            print(f"Baseline R2: {self.r2_baseline}")
            print(f"Champion RMSE: {self.rmse_champion}")
            print(f"Champion R2: {self.r2_champion}")
            print(f"RMSE increase %: {self.rmse_increase_pct:.2f}%")

        self.next(self.decide_retrain)

    @step
    def decide_retrain(self):
        if self.batch_rejected:
            self.retrain_needed = False
            self.retrain_reason = "batch rejected by hard integrity gate"
            self.retrain_recommended = "false"
            self.promotion_recommended = "false"
        elif self.champion_evaluation_status != "completed":
            self.retrain_needed = False
            self.retrain_reason = "champion evaluation was not completed"
            self.retrain_recommended = "false"
            self.promotion_recommended = "false"
        else:
            reasons = []
            if self.rmse_increase_pct > float(self.max_rmse_increase_pct):
                reasons.append(
                    f"rmse_increase_pct {self.rmse_increase_pct:.2f}% exceeds threshold {float(self.max_rmse_increase_pct):.2f}%"
                )
            if self.rmse_champion > float(self.max_champion_rmse):
                reasons.append(
                    f"rmse_champion {self.rmse_champion:.4f} exceeds threshold {float(self.max_champion_rmse):.4f}"
                )
            if self.r2_champion < float(self.min_champion_r2):
                reasons.append(
                    f"r2_champion {self.r2_champion:.4f} is below threshold {float(self.min_champion_r2):.4f}"
                )
            if reasons:
                self.retrain_needed = True
                self.retrain_reason = "; ".join(reasons)
                self.retrain_recommended = "true"
            else:
                self.retrain_needed = False
                self.retrain_reason = "champion performance is within thresholds"
                self.retrain_recommended = "false"
            self.promotion_recommended = "false"

        print(f"Retrain needed: {self.retrain_needed}")
        print(f"Retrain reason: {self.retrain_reason}")

        self.next(self.retrain_candidate)

    @step
    def retrain_candidate(self):
        self.candidate_training_path_used = ""
        self.candidate_training_rows = 0
        self.candidate_extra_training_rows = 0

        if self.batch_rejected:
            print("Skipping candidate retraining because batch was rejected.")
            self.candidate_status = "skipped"
            self.candidate_version = ""
            self.rmse_candidate = 0.0
            self.r2_candidate = 0.0
            self.candidate_improvement_pct = 0.0
        elif not self.retrain_needed:
            print("Skipping candidate retraining because retrain_needed is False.")
            self.candidate_status = "not_needed"
            self.candidate_version = ""
            self.rmse_candidate = 0.0
            self.r2_candidate = 0.0
            self.candidate_improvement_pct = 0.0
        else:
            print("Retraining candidate model.")
            X_train = self.X_reference
            y_train = self.y_reference
            candidate_extra_training_rows = 0

            if self.candidate_training_path:
                print(f"Loading additional training data from: {self.candidate_training_path}")
                extra_df = pd.read_parquet(self.candidate_training_path)

                # Apply same cleaning rules as in feature_engineering
                extra_pickup_dt = pd.to_datetime(
                    extra_df["lpep_pickup_datetime"], errors="coerce"
                )
                extra_dropoff_dt = pd.to_datetime(
                    extra_df["lpep_dropoff_datetime"], errors="coerce"
                )
                extra_valid_mask = (
                    pd.to_numeric(extra_df["trip_distance"], errors="coerce") >= 0
                ) & (
                    pd.to_numeric(extra_df["fare_amount"], errors="coerce") >= 0
                ) & (
                    pd.to_numeric(extra_df["tip_amount"], errors="coerce") >= 0
                ) & (extra_dropoff_dt >= extra_pickup_dt)

                extra_df_clean = extra_df[extra_valid_mask].copy()
                X_extra, y_extra, _ = build_features(extra_df_clean)

                # Verify feature columns match
                if list(X_extra.columns) != list(self.X_reference.columns):
                    raise ValueError(
                        "Feature schema mismatch between reference and candidate training datasets"
                    )

                # Concatenate training data
                X_train = pd.concat([self.X_reference, X_extra], ignore_index=True)
                y_train = pd.concat([self.y_reference, y_extra], ignore_index=True)
                candidate_extra_training_rows = int(len(X_extra))

            candidate_training_rows = int(len(X_train))

            model = RandomForestRegressor(
                n_estimators=120, random_state=123, min_samples_leaf=3, n_jobs=-1
            )
            model.fit(X_train, y_train)
            y_pred_candidate = model.predict(self.X_batch)
            rmse_candidate = math.sqrt(mean_squared_error(self.y_batch, y_pred_candidate))
            r2_candidate = r2_score(self.y_batch, y_pred_candidate)
            if self.rmse_champion > 0:
                candidate_improvement_pct = ((self.rmse_champion - rmse_candidate) / self.rmse_champion) * 100
            else:
                candidate_improvement_pct = 0.0
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="train_candidate"):
                mlflow.log_param("model_type", "RandomForestRegressor")
                mlflow.log_param("role", "candidate")
                mlflow.log_param("n_estimators", 120)
                mlflow.log_param("min_samples_leaf", 3)
                mlflow.log_param("random_state", 123)
                mlflow.log_param("candidate_training_path_used", str(self.candidate_training_path) if self.candidate_training_path else "")
                mlflow.log_param("candidate_training_rows", candidate_training_rows)
                mlflow.log_param("candidate_extra_training_rows", candidate_extra_training_rows)
                mlflow.log_metric("rmse_candidate", rmse_candidate)
                mlflow.log_metric("r2_candidate", r2_candidate)
                mlflow.log_metric("candidate_improvement_pct", candidate_improvement_pct)
                mlflow.log_metric("rmse_champion", self.rmse_champion)
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.model_name,
                    signature=infer_signature(
                        self.X_reference.head(100), model.predict(self.X_reference.head(100))
                    ),
                    input_example=self.X_reference.head(5),
                    await_registration_for=300,
                )
                version = getattr(model_info, "registered_model_version", None) or getattr(model_info, "model_version", None)
                if version is None:
                    raise RuntimeError("Could not determine registered candidate model version from MLflow model_info.")
                version = str(version)
                client = MlflowClient()
                client.set_model_version_tag(self.model_name, version, "role", "candidate")
                client.set_model_version_tag(self.model_name, version, "validation_status", "pending")
                client.set_model_version_tag(self.model_name, version, "trained_on", str(self.reference_path))
                client.set_model_version_tag(self.model_name, version, "eval_batch_id", str(self.batch_path))
                client.set_model_version_tag(self.model_name, version, "decision_reason", self.retrain_reason)
            self.candidate_status = "trained"
            self.candidate_version = version
            self.rmse_candidate = float(rmse_candidate)
            self.r2_candidate = float(r2_candidate)
            self.candidate_improvement_pct = float(candidate_improvement_pct)
            self.candidate_training_path_used = str(self.candidate_training_path) if self.candidate_training_path else ""
            self.candidate_training_rows = candidate_training_rows
            self.candidate_extra_training_rows = candidate_extra_training_rows
            print(f"Candidate version: {self.candidate_version}")
            print(f"Candidate RMSE: {self.rmse_candidate}")
            print(f"Candidate R2: {self.r2_candidate}")
            print(f"Candidate improvement %: {self.candidate_improvement_pct:.2f}%")
            print(f"Candidate training path used: {self.candidate_training_path_used}")
            print(f"Candidate extra training rows: {self.candidate_extra_training_rows}")
            print(f"Candidate total training rows: {self.candidate_training_rows}")

        self.next(self.promotion_gate)

    @step
    def promotion_gate(self):
        if self.batch_rejected:
            print("Skipping promotion because batch was rejected.")
            self.promotion_recommended = "false"
            self.promotion_status = "skipped"
            self.promotion_reason = "batch rejected by hard integrity gate"
            self.promoted_version = ""
            self.previous_champion_version = self.champion_version
        elif self.candidate_status != "trained":
            print("Skipping promotion because no trained candidate is available.")
            self.promotion_recommended = "false"
            self.promotion_status = "not_applicable"
            self.promotion_reason = "no trained candidate available"
            self.promoted_version = ""
            self.previous_champion_version = self.champion_version
        else:
            print("Evaluating candidate promotion criteria.")
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(EXPERIMENT_NAME)
            client = MlflowClient(tracking_uri=self.tracking_uri)
            criteria_pass = True
            reasons = []

            if self.candidate_improvement_pct < float(self.min_candidate_improvement):
                criteria_pass = False
                reasons.append(
                    f"candidate_improvement_pct {self.candidate_improvement_pct:.2f}% is below threshold {float(self.min_candidate_improvement):.2f}%"
                )
            if self.hard_failure_count > 0:
                criteria_pass = False
                reasons.append("hard integrity failures exist")
            if self.batch_rejected:
                criteria_pass = False
                reasons.append("batch was rejected")

            if criteria_pass:
                if self.champion_version:
                    client.set_registered_model_alias(self.model_name, "previous_champion", self.champion_version)
                    client.set_model_version_tag(
                        self.model_name,
                        self.champion_version,
                        "role",
                        "previous_champion",
                    )
                    client.set_model_version_tag(
                        self.model_name,
                        self.champion_version,
                        "demoted_reason",
                        "candidate_promoted",
                    )
                client.set_registered_model_alias(self.model_name, "champion", self.candidate_version)
                client.set_model_version_tag(self.model_name, self.candidate_version, "role", "champion")
                client.set_model_version_tag(self.model_name, self.candidate_version, "validation_status", "approved")
                client.set_model_version_tag(self.model_name, self.candidate_version, "promotion_reason", "candidate passed promotion gate")
                old_champion_version = self.champion_version
                self.promotion_recommended = "true"
                self.promotion_status = "promoted"
                self.promotion_reason = "candidate improved RMSE enough to pass promotion gate"
                self.previous_champion_version = old_champion_version
                self.promoted_version = self.candidate_version
                self.champion_version = self.candidate_version
            else:
                client.set_model_version_tag(self.model_name, self.candidate_version, "validation_status", "rejected")
                client.set_model_version_tag(self.model_name, self.candidate_version, "decision_reason", "candidate did not pass promotion gate")
                self.promotion_recommended = "false"
                self.promotion_status = "rejected"
                self.promotion_reason = "; ".join(reasons)
                self.promoted_version = ""
                self.previous_champion_version = self.champion_version

        print(f"Promotion recommended: {self.promotion_recommended}")
        print(f"Promotion status: {self.promotion_status}")
        print(f"Promotion reason: {self.promotion_reason}")
        print(f"Previous champion version: {self.previous_champion_version}")
        print(f"Promoted version: {self.promoted_version}")

        self.next(self.batch_inference)

    @step
    def batch_inference(self):
        if self.batch_rejected:
            print("Skipping batch inference because batch was rejected.")
            self.inference_status = "skipped"
            self.prediction_rows = 0
            self.predictions_artifact_path = ""
            self.next(self.log_mlflow)
            return

        if self.feature_engineering_status != "completed":
            print("Skipping batch inference because feature engineering was not completed.")
            self.inference_status = "skipped"
            self.prediction_rows = 0
            self.predictions_artifact_path = ""
            self.next(self.log_mlflow)
            return

        print("Running offline batch inference.")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        model_uri = f"models:/{self.model_name}@champion"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            if model is None:
                raise ValueError("pyfunc returned None")
        except Exception as e:
            print(f"Warning: pyfunc load failed ({e}), trying sklearn load")
            model = mlflow.sklearn.load_model(model_uri)

        y_pred = model.predict(self.X_batch)
        predictions_df = pd.DataFrame(
            {
                "row_id": range(len(self.X_batch)),
                "y_true": list(self.y_batch),
                "y_pred": list(y_pred),
                "prediction_error": list(self.y_batch - y_pred),
                "absolute_error": list(abs(self.y_batch - y_pred)),
                "model_name": self.model_name,
                "champion_version": self.champion_version,
                "batch_path": str(self.batch_path),
            }
        )

        artifact_dir = Path("artifacts")
        artifact_dir.mkdir(exist_ok=True)
        predictions_path = artifact_dir / "predictions.parquet"
        predictions_df.to_parquet(predictions_path, index=False)

        self.inference_status = "completed"
        self.prediction_rows = int(len(predictions_df))
        self.predictions_artifact_path = str(predictions_path)

        print("Offline batch inference completed")
        print(f"Prediction rows: {self.prediction_rows}")
        print(f"Predictions artifact: {self.predictions_artifact_path}")

        self.next(self.log_mlflow)

    @step
    def log_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)

        candidate_training_path_used = getattr(self, "candidate_training_path_used", "")
        candidate_training_rows = getattr(self, "candidate_training_rows", 0)
        candidate_extra_training_rows = getattr(self, "candidate_extra_training_rows", 0)

        run_name = f"integrity_{self.integrity_status}"

        self.decision = {
            "stage": "monitoring_retrain_promotion_decision",
            "action": self.action,
            "batch_rejected": self.batch_rejected,
            "integrity_status": self.integrity_status,
            "feature_engineering_status": self.feature_engineering_status,
            "hard_failure_count": self.hard_failure_count,
            "hard_failures": self.hard_failures,
            "reference_path": str(self.reference_path),
            "batch_path": str(self.batch_path),
            "reference_rows": self.reference_rows,
            "batch_rows": self.batch_rows,
            "reference_feature_rows": self.reference_feature_rows,
            "batch_feature_rows": self.batch_feature_rows,
            "n_features": self.n_features,
            "model_name": self.model_name,
            "champion_status": self.champion_status,
            "champion_version": self.champion_version,
            "bootstrap_rmse": self.bootstrap_rmse,
            "bootstrap_r2": self.bootstrap_r2,
            "champion_evaluation_status": self.champion_evaluation_status,
            "rmse_champion": self.rmse_champion,
            "r2_champion": self.r2_champion,
            "invalid_row_tolerance": float(self.invalid_row_tolerance),
            "integrity_warn": self.integrity_warn,
            "integrity_warning_count": self.integrity_warning_count,
            "integrity_warnings": self.integrity_warnings,
            "soft_monitoring_status": self.nannyml_status,
            "soft_monitoring_warn": self.nannyml_warn,
            "soft_monitoring_warning_count": self.nannyml_warning_count,
            "soft_monitoring_warnings": self.nannyml_warnings,
            "soft_monitoring_drift_scores": self.nannyml_drift_scores,
            "nannyml_available": self.nannyml_available,
            "nannyml_summary": self.nannyml_summary,
            "negative_trip_distance_count": self.negative_trip_distance_count,
            "negative_fare_amount_count": self.negative_fare_amount_count,
            "negative_tip_amount_count": self.negative_tip_amount_count,
            "invalid_time_count": self.invalid_time_count,
            "reference_rows_after_cleaning": self.reference_rows_after_cleaning,
            "batch_rows_after_cleaning": self.batch_rows_after_cleaning,
            "reference_rows_dropped": self.reference_rows_dropped,
            "batch_rows_dropped": self.batch_rows_dropped,
            "rmse_baseline": self.rmse_baseline,
            "r2_baseline": self.r2_baseline,
            "rmse_increase_pct": self.rmse_increase_pct,
            "max_rmse_increase_pct": float(self.max_rmse_increase_pct),
            "max_champion_rmse": float(self.max_champion_rmse),
            "min_champion_r2": float(self.min_champion_r2),
            "retrain_needed": self.retrain_needed,
            "retrain_reason": self.retrain_reason,
            "retrain_recommended": self.retrain_recommended,
            "promotion_recommended": self.promotion_recommended,
            "candidate_status": self.candidate_status,
            "candidate_version": self.candidate_version,
            "rmse_candidate": self.rmse_candidate,
            "r2_candidate": self.r2_candidate,
            "candidate_improvement_pct": self.candidate_improvement_pct,
            "min_candidate_improvement": float(self.min_candidate_improvement),
            "promotion_status": self.promotion_status,
            "promotion_reason": self.promotion_reason,
            "promoted_version": self.promoted_version,
            "previous_champion_version": self.previous_champion_version,
            "candidate_training_path_used": candidate_training_path_used,
            "candidate_training_rows": candidate_training_rows,
            "candidate_extra_training_rows": candidate_extra_training_rows,
            "inference_status": self.inference_status,
            "prediction_rows": self.prediction_rows,
            "predictions_artifact_path": self.predictions_artifact_path,
        }

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("flow_stage", "monitoring_retrain_promotion_decision")
            mlflow.set_tag("batch_path", str(self.batch_path))
            mlflow.set_tag("reference_path", str(self.reference_path))
            mlflow.set_tag("integrity_status", self.integrity_status)
            mlflow.set_tag("batch_rejected", str(self.batch_rejected).lower())
            mlflow.set_tag(
                "feature_engineering_status",
                self.feature_engineering_status,
            )
            mlflow.set_tag("action", self.action)
            mlflow.set_tag("champion_status", self.champion_status)
            mlflow.set_tag("champion_version", self.champion_version)
            mlflow.set_tag("model_name", self.model_name)
            mlflow.set_tag("champion_evaluation_status", self.champion_evaluation_status)
            mlflow.set_tag("integrity_warn", str(self.integrity_warn).lower())
            mlflow.set_tag("integrity_warning_count", str(self.integrity_warning_count))
            mlflow.set_tag("soft_monitoring_status", self.nannyml_status)
            mlflow.set_tag("soft_monitoring_warn", str(self.nannyml_warn).lower())
            mlflow.set_tag("soft_monitoring_warning_count", str(self.nannyml_warning_count))
            mlflow.set_tag("nannyml_available", str(self.nannyml_available).lower())
            mlflow.set_tag("inference_status", self.inference_status)
            mlflow.set_tag("retrain_recommended", self.retrain_recommended)
            mlflow.set_tag("promotion_recommended", self.promotion_recommended)
            mlflow.set_tag("retrain_reason", self.retrain_reason)
            mlflow.set_tag("candidate_status", self.candidate_status)
            mlflow.set_tag("candidate_version", self.candidate_version)
            mlflow.set_tag("promotion_status", self.promotion_status)
            mlflow.set_tag("promotion_reason", self.promotion_reason)
            mlflow.set_tag("promoted_version", self.promoted_version)
            mlflow.set_tag("previous_champion_version", self.previous_champion_version)

            mlflow.log_metric("reference_rows", self.reference_rows)
            mlflow.log_metric("batch_rows", self.batch_rows)
            mlflow.log_metric("hard_failure_count", self.hard_failure_count)
            mlflow.log_metric("reference_feature_rows", self.reference_feature_rows)
            mlflow.log_metric("batch_feature_rows", self.batch_feature_rows)
            mlflow.log_metric("n_features", self.n_features)
            mlflow.log_metric("bootstrap_rmse", self.bootstrap_rmse)
            mlflow.log_metric("bootstrap_r2", self.bootstrap_r2)
            mlflow.log_metric("rmse_champion", self.rmse_champion)
            mlflow.log_metric("r2_champion", self.r2_champion)
            mlflow.log_metric("negative_trip_distance_count", self.negative_trip_distance_count)
            mlflow.log_metric("negative_fare_amount_count", self.negative_fare_amount_count)
            mlflow.log_metric("negative_tip_amount_count", self.negative_tip_amount_count)
            mlflow.log_metric("invalid_time_count", self.invalid_time_count)
            mlflow.log_metric("soft_monitoring_warning_count", self.nannyml_warning_count)
            mlflow.log_metric("prediction_rows", self.prediction_rows)
            mlflow.log_metric("reference_rows_after_cleaning", self.reference_rows_after_cleaning)
            mlflow.log_metric("batch_rows_after_cleaning", self.batch_rows_after_cleaning)
            mlflow.log_metric("reference_rows_dropped", self.reference_rows_dropped)
            mlflow.log_metric("batch_rows_dropped", self.batch_rows_dropped)
            mlflow.log_metric("rmse_baseline", self.rmse_baseline)
            mlflow.log_metric("r2_baseline", self.r2_baseline)
            mlflow.log_metric("rmse_increase_pct", self.rmse_increase_pct)
            mlflow.log_metric("max_rmse_increase_pct", float(self.max_rmse_increase_pct))
            mlflow.log_metric("max_champion_rmse", float(self.max_champion_rmse))
            mlflow.log_metric("min_champion_r2", float(self.min_champion_r2))
            mlflow.log_metric("min_candidate_improvement", float(self.min_candidate_improvement))
            mlflow.log_metric("rmse_candidate", self.rmse_candidate)
            mlflow.log_metric("r2_candidate", self.r2_candidate)
            mlflow.log_metric("candidate_improvement_pct", self.candidate_improvement_pct)
            mlflow.log_metric("candidate_training_rows", candidate_training_rows)
            mlflow.log_metric("candidate_extra_training_rows", candidate_extra_training_rows)

            artifact_dir = Path("artifacts")
            artifact_dir.mkdir(exist_ok=True)

            decision_path = artifact_dir / "decision.json"
            with decision_path.open("w", encoding="utf-8") as f:
                json.dump(self.decision, f, indent=2)

            mlflow.log_artifact(str(decision_path))

            if self.feature_spec:
                feature_spec_path = artifact_dir / "feature_spec.json"
                with feature_spec_path.open("w", encoding="utf-8") as f:
                    json.dump(self.feature_spec, f, indent=2)

                mlflow.log_artifact(str(feature_spec_path))
                print(f"Feature spec artifact: {feature_spec_path}")

            nannyml_summary_path = artifact_dir / "nannyml_summary.json"
            with nannyml_summary_path.open("w", encoding="utf-8") as f:
                json.dump(self.nannyml_summary, f, indent=2)

            mlflow.log_artifact(str(nannyml_summary_path))
            print(f"NannyML summary artifact: {nannyml_summary_path}")

            if self.predictions_artifact_path:
                predictions_path = Path(self.predictions_artifact_path)
                if predictions_path.exists():
                    mlflow.log_artifact(str(predictions_path))
                    print(f"Predictions artifact: {predictions_path}")

            print("Logged decision to MLflow")
            print(f"Decision artifact: {decision_path}")

        self.next(self.end)

    @step
    def end(self):
        print("Flow finished")
        print(f"Batch rejected: {self.batch_rejected}")
        print(f"Integrity status: {self.integrity_status}")
        print(f"Feature engineering status: {self.feature_engineering_status}")
        print(f"Action: {self.action}")

        if self.hard_failures:
            print("Hard failure reasons:")
            for reason in self.hard_failures:
                print(f"- {reason}")


if __name__ == "__main__":
    GreenTaxiCapstoneFlow()