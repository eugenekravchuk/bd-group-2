# Bordeaux holdout evaluation

## Evaluation protocol

Bordeaux is the unseen-city evaluation dataset for a model developed on Paris and Lyon. The A/B model and training-median baseline are saved before Bordeaux evaluation. The same ETL accepts `city=bordeaux`; the model notebook uses `mode=evaluate_c` with the original run ID and does not refit preprocessing, regression or baseline statistics.

| Item | Value |
|---|---|
| Snapshot date | 2026-09-17 |
| Gold feature rows | 12,063 |
| Labeled evaluation rows | 7,722 |
| Run ID | `france_v1` |
| Currency | EUR |

The frozen model and manifest are stored under `/Volumes/airbnb_lakehouse/gold/model_artifacts/france_v1/`. This is an application run ID, not an MLflow run ID.

## Results

| Metric | MAE (EUR) |
|---|---:|
| A/B combined held-out model | 144.13 |
| Bordeaux training-median baseline | 118.51 |
| Bordeaux unchanged model | 223.16 |

Model MAE increases by EUR 79.03 (54.8%) relative to the A/B test set. The baseline outperforms the model by EUR 104.65. Entire homes and private rooms exhibit systematic overprediction. Full room-type diagnostics and interpretation are in the [model evaluation report](model_evaluation_report.md).

## Changes and limitations

No execution errors were recorded in the completed workflow. No pipeline fixes or model changes were made. No after-change model result is included. Detailed parse-failure and missing-column statistics require inspection of `silver.quality_report`; successful scoring alone does not establish that ETL had no data-quality issues.

All 7,722 valid-priced Bordeaux listings contribute to MAE. The remaining listings cannot be evaluated without actual prices. A separate Gold-table completeness query confirmed 12,063 rows, 12,063 unique listing IDs and 12,063 non-null predictions for this run and evaluation stage. Prediction coverage is therefore complete. This successful execution does not remove the observed accuracy limitation: the model performs worse than the baseline on Bordeaux.

## Further investigation

Compare feature missingness, accommodation mix and numerical distributions with A/B training data. Inspect high-prediction listings before attributing transfer failure to one mechanism. Candidate model experiments include log-price fitting, robust loss, balanced training and transferable geographic features. Select parameters using an inner A/B training split. Any further Bordeaux results are post-diagnosis evaluations and must retain the unchanged-model result as the original reference.
