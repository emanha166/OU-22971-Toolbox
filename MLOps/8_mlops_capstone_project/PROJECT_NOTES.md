# MLOps Unit 8 Capstone Project: Green Taxi Tip Prediction

## Project Overview
This project implements a manually run MLOps workflow for predicting tip amounts for Green Taxi rides in New York City. The target variable is `tip_amount`, and the workflow focuses on data integrity, feature engineering, and experiment tracking.

## Workflow Pipeline
The current Metaflow pipeline consists of the following steps:
1. **start** - Initialize the workflow
2. **load_data** - Load and prepare the dataset
3. **integrity_gate** - Perform data integrity checks
4. **feature_engineering** - Create and transform features
5. **log_mlflow** - Log metrics, tags, and artifacts to MLflow
6. **end** - Complete the workflow

## Technologies Used
- **Metaflow**: Orchestrates the workflow steps and manages the pipeline execution
- **MLflow**: Handles experiment tracking, logging of metrics, tags, and artifacts
- **NannyML**: Planned for the monitoring/drift detection stage that will be added next, as part of the capstone workflow

## Data Integrity Checks
The following integrity checks are currently implemented:
- Required columns must exist in the dataset
- `trip_distance` values must not be negative
- `fare_amount` values must not be negative
- `dropoff_datetime` must not occur before `pickup_datetime`

## Feature Engineering
The current feature set includes:
- `pickup_hour`: Hour of the day when the trip started
- `pickup_dayofweek`: Day of the week when the trip started
- `pickup_month`: Month when the trip started
- `passenger_count`: Number of passengers
- `trip_distance_log`: Log-transformed trip distance
- `fare_amount_log`: Log-transformed fare amount
- `PULocationID`: Pickup location ID
- `DOLocationID`: Dropoff location ID

## MLflow Artifacts
The following artifacts are currently logged to MLflow:
- `decision.json`: Contains decision-making information from the workflow
- `feature_spec.json`: Generated during successful feature engineering, contains feature specifications

## Future Enhancements
- Complete the required NannyML-based monitoring and drift detection stage.
- Automation of the manual workflow execution
- Additional feature engineering techniques
- Model deployment and serving capabilities

## Project Structure
- `capstone_flow.py`: Main Metaflow pipeline implementation
- `PROJECT_NOTES.md`: This documentation file
- Additional supporting files as needed

## Usage Instructions
1. Ensure all dependencies are installed
2. Run the Metaflow pipeline manually
3. Check MLflow UI for logged experiments and artifacts
4. Review integrity check results before proceeding with modeling

## Current Status

The project currently implements the first part of the MLOps workflow:

load data
↓
integrity gate
↓
feature engineering
↓
MLflow logging

The workflow already demonstrates two important cases:
1. A valid batch that passes the integrity gate and completes feature engineering.
2. A rejected batch that fails hard integrity checks and skips feature engineering.

## Next Implementation Steps

The next development stages are:

1. Champion bootstrap:
   - train an initial model on the reference dataset
   - register the model as `green_taxi_tip_model`
   - assign the `champion` alias

2. Champion evaluation:
   - load the current champion model
   - evaluate it on the new batch
   - log `rmse_champion`

3. Retraining decision:
   - decide whether retraining is needed based on performance degradation and monitoring signals

4. Candidate training and promotion:
   - train a candidate model if retraining is needed
   - compare candidate performance against champion
   - promote the candidate only if it passes the quality gate

5. NannyML monitoring:
   - add soft monitoring checks for drift or data quality warnings
   - log monitoring results to MLflow

## Demonstration Guide
- Project goal: Predict tip amounts for Green Taxi rides using a robust MLOps workflow
- Reference dataset and batch dataset: Explain the datasets used for training and evaluation
- Metaflow pipeline structure: Describe the current pipeline steps (start, load_data, integrity_gate, feature_engineering, log_mlflow, end)
- Valid batch run: Demonstrate a successful run where data passes integrity checks and features are engineered
- Rejected batch run: Show a run that fails integrity checks and skips feature engineering
- MLflow experiment evidence: Highlight how experiments are tracked and logged
- Metrics, tags, decision.json, and feature_spec.json: Detail the logged artifacts and their contents
- Auditability and reproducibility: Emphasize how the workflow ensures traceable and repeatable results