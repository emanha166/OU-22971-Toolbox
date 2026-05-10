# MLOps Unit 8 Capstone Project: Green Taxi Tip Prediction

## Project Overview

This project implements a complete MLOps workflow for NYC Green Taxi TLC trip data. The goal is to predict the tip amount for each trip and maintain model quality through continuous monitoring, evaluation, retraining, and promotion. Rather than simply training a static model, this workflow demonstrates a production-ready system that:

- Monitors incoming batches for data quality and drift
- Evaluates the current champion model against new data
- Detects performance degradation and decides when to retrain
- Trains candidate models on optional updated training windows
- Promotes candidates only if they demonstrate sufficient improvement
- Performs offline batch inference with the current champion
- Logs comprehensive evidence to MLflow for auditability and governance

The workflow is orchestrated using **Metaflow**, experiment tracking is managed with **MLflow**, and data quality monitoring is inspired by **NannyML** principles.

---

## Data Used

The project uses publicly available NYC Green Taxi TLC trip data in Parquet format:

- **green_tripdata_2020-01.parquet**: Reference dataset (January 2020)
  - Used as the initial training data to bootstrap the champion model
  - Represents the baseline distribution of taxi features

- **green_tripdata_2020-04.parquet**: Intermediate batch and optional candidate training data (April 2020)
  - Used as the monitoring batch to detect drift and performance issues
  - Can optionally be used as additional training data for the candidate model
  - Serves as a test case for negative improvement (rejection scenario)

- **green_tripdata_2020-08.parquet**: Later evaluation batch (August 2020)
  - Used in the promotion run to demonstrate successful candidate retraining and promotion
  - Shows improved model performance after incorporating additional training data

All datasets contain raw taxi trip features including `pickup_datetime`, `dropoff_datetime`, `trip_distance`, `fare_amount`, `tip_amount`, and location information.

---

## Workflow Steps

The Metaflow pipeline implements a complete MLOps lifecycle:

```
start 
  → load_data 
  → integrity_gate 
  → nannyml_soft_gate 
  → feature_engineering 
  → load_or_bootstrap_champion 
  → evaluate_champion 
  → decide_retrain 
  → retrain_candidate 
  → promotion_gate 
  → batch_inference 
  → log_mlflow 
  → end
```

### Step Details

**start**: Initializes the workflow, prints configuration parameters, and sets the MLflow experiment.

**load_data**: Loads reference and batch datasets from Parquet files. Validates file existence and reports the number of rows loaded in each dataset.

**integrity_gate**: Performs hard data quality checks on the batch dataset (hard gate). Rejects the batch if required columns are missing. Allows small fractions of invalid rows (negative fare, invalid timestamps) within a configurable tolerance. Logs integrity warnings for rows within tolerance and hard failures for rows exceeding tolerance.

**nannyml_soft_gate**: Detects distributional drift between reference and batch data by computing drift scores for key numeric features (passenger_count, trip_distance, fare_amount, tip_amount). Does not reject the batch; instead, logs warnings and saves a NannyML summary artifact for monitoring and alerting.

**feature_engineering**: Builds model-ready feature tables from raw data by applying the same cleaning rules (removing negative amounts and invalid timestamps) and engineering stable features (log-transformed distances, hour-of-day, day-of-week). Validates that reference and batch feature schemas match.

**load_or_bootstrap_champion**: Retrieves the current champion model from the MLflow Model Registry using the `@champion` alias. If no champion exists, bootstraps an initial RandomForest model trained on the reference dataset and registers it.

**evaluate_champion**: Loads the current champion and evaluates it on the cleaned batch data. Computes baseline RMSE/R² on reference and batch RMSE/R². Calculates the percentage increase in RMSE from baseline to batch.

**decide_retrain**: Determines whether retraining is needed by comparing champion metrics against configurable thresholds (max_champion_rmse, min_champion_r2, max_rmse_increase_pct). Sets `retrain_needed=true` if any threshold is exceeded.

**retrain_candidate**: Trains a new candidate model if `retrain_needed=true`. By default, trains on reference data only. If `candidate_training_path` is provided, loads and cleans that dataset, verifies feature compatibility, and concatenates it with reference data before training. Logs candidate metrics and computes improvement percentage versus champion.

**promotion_gate**: Evaluates whether the candidate model meets promotion criteria. Promotes the candidate to `@champion` alias only if `candidate_improvement_pct >= min_candidate_improvement` and no hard integrity failures exist. Otherwise, rejects the candidate and keeps the current champion. Tags model versions with metadata and decision reasons.

**batch_inference**: Performs offline inference using the current champion on the cleaned batch data. Saves predictions (row_id, y_true, y_pred, errors, model_name, champion_version, batch_path) to `artifacts/predictions.parquet` for later analysis.

**log_mlflow**: Creates an MLflow run logging all pipeline decisions, metrics, parameters, and artifacts. Saves decision.json, feature_spec.json, nannyml_summary.json, and predictions.parquet to enable audit trails and reproducibility.

**end**: Finalizes the workflow and prints summary information.

---

## Integrity Gate

The integrity gate performs hard quality checks to determine if a batch is usable for modeling:

- **Hard failures** reject the batch outright: missing required columns, zero rows, or schema mismatches
- **Soft warnings** are logged when small numbers of invalid rows are detected but remain within the `invalid_row_tolerance` (default 1%)
- Invalid rows include those with negative fare amounts, negative tip amounts, negative trip distances, or invalid pickup/dropoff datetime pairs
- All invalid rows are removed before feature engineering and modeling, ensuring the feature and training data are clean

Example: If a batch has 5,000 rows and 20 have negative trip_distance (0.4% ratio), this is within the 1% tolerance and logged as a warning. If 60 rows are affected (1.2%), this exceeds tolerance and becomes a hard failure, rejecting the batch.

---

## NannyML Soft Gate

The soft monitoring gate detects data drift without blocking the workflow:

- Computes drift scores for key numeric features using a simple standardized mean difference metric
- Drift score = |batch_mean - reference_mean| / reference_std
- Warnings are emitted for strong drift (score ≥ 1.0) and moderate drift (score ≥ 0.5)
- Unlike the hard integrity gate, drift warnings do not stop batch processing; the workflow continues
- Saves a detailed summary to `nannyml_summary.json` including drift scores and warnings for observability and alerting

Example: If trip_distance has significantly shorter trips in the April batch compared to January, this is detected and logged. This informs stakeholders that the model may be operating in a shifted domain, even if data quality is acceptable.

---

## Champion Evaluation and Retrain Decision

The system maintains a single champion model in the MLflow Model Registry and continuously evaluates its performance:

- **Champion loading**: Retrieves the model using the `models:/green_taxi_tip_model@champion` URI
- **Baseline metrics**: Computes RMSE and R² on the reference dataset (training domain)
- **Batch metrics**: Computes RMSE and R² on the cleaned batch data (production domain)
- **Degradation detection**: Calculates `rmse_increase_pct = (batch_rmse - baseline_rmse) / baseline_rmse * 100`
- **Retrain decision**: Sets `retrain_needed=true` if any threshold is exceeded:
  - RMSE increases by more than `max_rmse_increase_pct` (default 10%)
  - Absolute RMSE exceeds `max_champion_rmse` (default 2.0)
  - R² falls below `min_champion_r2` (default 0.0)

This approach separates concerns: hard quality checks happen first (integrity gate), then the champion is only retrained if performance warrants it.

---

## Candidate Training and Promotion Gate

When retraining is needed, a candidate model is trained and evaluated for promotion:

- **Candidate training**: Trained on reference data by default using RandomForestRegressor (n_estimators=120, min_samples_leaf=3)
- **Optional extended training**: If `candidate_training_path` is provided, loads and cleans that dataset, verifies feature compatibility, and concatenates with reference data before training
  - This allows the candidate to learn from more recent or diverse data, improving generalization
  - Same cleaning rules (negative amounts, invalid timestamps) are applied
  - Training data size is logged to MLflow for reproducibility
- **Evaluation**: Candidate is evaluated on the batch data (same as champion)
- **Improvement metric**: `candidate_improvement_pct = (champion_rmse - candidate_rmse) / champion_rmse * 100`
- **Promotion criteria**: Candidate is promoted to `@champion` only if:
  - `candidate_improvement_pct >= min_candidate_improvement` (default 1.0%)
  - No hard integrity failures in the batch
  - Batch was not rejected by the integrity gate
- **Metadata tagging**: Approved candidates are tagged as `role=champion`, rejected candidates are tagged with decision_reason explaining why

This gatekeeping ensures that only models demonstrating clear improvement over the incumbent are put into production.

---

## Offline Inference

After all evaluation and promotion decisions are made, the current champion model produces predictions on the cleaned batch:

- **Model selection**: Always uses the current champion (updated by promotion_gate if applicable)
- **Input data**: Cleaned batch features (invalid rows removed, features engineered)
- **Predictions**: Scores for tip_amount on each batch record
- **Output schema**: DataFrame with row_id, y_true, y_pred, prediction_error, absolute_error, model_name, champion_version, batch_path
- **Storage**: Saved to `artifacts/predictions.parquet` and logged to MLflow for downstream analysis, dashboarding, and auditing

This artifact enables stakeholders to inspect model behavior on production data without accessing model internals.

---

## MLflow Evidence

All decisions, metrics, parameters, and data artifacts are logged to MLflow for full auditability:

### Artifacts
- **decision.json**: Comprehensive JSON object capturing the entire pipeline state (batch_rejected, integrity_status, retrain_needed, promotion_status, etc.)
- **feature_spec.json**: Schema and statistics of engineered features for reproducibility
- **nannyml_summary.json**: Drift detection results and warnings from soft monitoring
- **predictions.parquet**: Offline batch predictions for analysis and auditing

### Key Metrics
- `rmse_baseline`: Champion RMSE on reference data
- `rmse_champion`: Champion RMSE on batch data
- `rmse_increase_pct`: Percentage degradation in RMSE
- `rmse_candidate`: Candidate RMSE on batch data (if trained)
- `candidate_improvement_pct`: Percentage improvement of candidate over champion (if trained)
- `prediction_rows`: Number of predictions generated

### Important Tags
- `flow_stage`: Pipeline stage name
- `integrity_status`: "passed" or "failed"
- `nannyml_warn`: "true" or "false" soft monitoring warning flag
- `retrain_recommended`: "true" or "false"
- `promotion_recommended`: "true" or "false"
- `promotion_status`: "promoted", "rejected", "not_applicable", or "skipped"

---

## Example Commands

### Start MLflow server
```bash
~/venvs/project1/bin/mlflow server \
  --workers 1 \
  --port 5000 \
  --backend-store-uri sqlite:////home/eman_hasan/mlflow_tlc_capstone/mlflow.db \
  --default-artifact-root /home/eman_hasan/mlflow_tlc_capstone/mlruns
```

### Run April monitoring / candidate rejection scenario
```bash
~/venvs/project1/bin/python capstone_flow.py run \
  --reference-path ../6_monitoring_data_drift/TLC_data/green_tripdata_2020-01.parquet \
  --batch-path ../6_monitoring_data_drift/TLC_data/green_tripdata_2020-04.parquet
```

This run:
- Loads January as reference and April as the monitoring batch
- Bootstraps the champion on January data if none exists
- Detects data drift in April (e.g., shorter trips, lower fares)
- Detects champion degradation (rmse_increase_pct > threshold)
- Trains a candidate on January data only, which shows negative improvement
- Rejects the candidate because it does not improve over champion
- Champion remains unchanged

### Run August promotion scenario (with extended training)
```bash
~/venvs/project1/bin/python capstone_flow.py run \
  --reference-path ../6_monitoring_data_drift/TLC_data/green_tripdata_2020-01.parquet \
  --candidate-training-path ../6_monitoring_data_drift/TLC_data/green_tripdata_2020-04.parquet \
  --batch-path ../6_monitoring_data_drift/TLC_data/green_tripdata_2020-08.parquet
```

This run:
- Loads January as reference and August as the monitoring batch
- Detects August batch conditions (new domain shift)
- Trains a candidate on January + April (extended training window)
- Candidate learns from both baseline and intermediate data
- Candidate shows strong improvement on August batch (candidate_improvement_pct > threshold)
- Candidate is promoted to champion
- Model Registry alias `@champion` is updated to the new version

---

## Demonstration Results

### April Monitoring Run (Rejection)
- Integrity gate: PASSED (no hard failures)
- NannyML soft gate: WARN (drift detected, especially in trip_distance)
- Champion evaluation: RMSE increased by about 31.46% on the April batch
- Retrain decision: YES (degradation exceeds threshold)
- Candidate training: Trained on January data only
- Candidate performance: candidate_improvement_pct = -0.52% (negative)
- Promotion gate: REJECTED (candidate did not improve)
- Outcome: Champion remains unchanged, no promotion

### August Promotion Run (Promotion)
- Integrity gate: PASSED
- NannyML soft gate: WARN (drift detected, but monitored)
- Champion evaluation: RMSE increased by about 38.55% on the August batch
- Retrain decision: YES
- Candidate training: Trained on January + April data, with 481,291 total training rows after cleaning
- Candidate performance: candidate_improvement_pct = +17.27% (strong improvement)
- Promotion gate: APPROVED (candidate_improvement_pct > min_candidate_improvement)
- Outcome: Version 7 promoted to @champion, Model Registry updated

This demonstrates the full MLOps cycle: detection, decision, experimentation, and safe deployment.

---

## Repository Notes

- **Local runtime**: `.metaflow/` directory contains Metaflow metadata (excluded from Git)
- **MLflow artifacts**: `mlruns/` and `artifacts/` directories contain experiment outputs and predictions (excluded from Git)
- **Data files**: `*.parquet` files in the TLC_data directory are excluded from Git (large files)
- **Configuration**: All parameters are exposed via Metaflow `@Parameter` decorators and can be customized per run
- **Reproducibility**: Run IDs, model versions, and decision artifacts are logged to enable audit trails and replication

This project is suitable for submission as a capstone demonstration of MLOps principles in production workflows.
