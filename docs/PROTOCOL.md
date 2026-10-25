# Fraud-monitoring experiment protocol

## Source and identity limits

The public ULB/Worldline benchmark contains anonymised PCA fields V1–V28, relative event time, amount and a fraud label. There are no usable account, card, merchant or customer IDs. This project therefore cannot construct velocity by customer, account networks, AML typologies or interpretable customer reasons. No such fields are fabricated. Amount units are reported as source units rather than assuming a currency.

Remove exact event-profile repeats using Time, all PCA fields and Amount, independent of Class. Conflicting labels for an identical event profile stop the run. The audited file removes 1,081 repeats. No resampling, SMOTE or test-set balancing is applied; rare-event prevalence remains visible.

## Chronological windows

Seconds are relative to the first transaction:

| Partition | Interval | Purpose |
| --- | --- | --- |
| Train | t < 82,800 | Fit scaler and candidate estimators |
| Calibration | 86,400 ≤ t < 104,400 | Fit probability calibrators |
| Validation | 108,000 ≤ t < 126,000 | Select candidate and probability threshold |
| Test | t ≥ 129,600 | Final held-out period |

Each adjacent window has a one-hour gap. Other rows in the gaps do not enter the experiment. This prevents adjacent windows touching; it does **not** simulate delayed fraud confirmation. With roughly two days of data, realistic multi-day chargeback delays and seasonal behaviour cannot be measured. Upstream PCA fitting is outside this pipeline's control, so full-source preprocessing leakage cannot be ruled out.

## Models and calibration

Features are V1–V28 plus log(1 + Amount); Time is used for ordering and queue buckets, not fitting. A logistic pipeline fits its scaler on training rows only. Histogram gradient boosting is the nonlinear challenger. Both use balanced class weights for fitting, then a sigmoid calibrator fits original-prevalence labels on the independent calibration window. No estimator is refitted after validation.

Select first by validation recall at ten reviews per closed hour, then by validation average precision to break ties. This selection has only 17 validation frauds in the audited run, so it can be unstable. The 99.5th percentile of validation scores independently defines a fixed probability-threshold policy.

## Review capacity and evaluation

Sort each **closed hour** by score descending and original row ID ascending for stable ties. Take at most K rows, with K=5/10/20/40. This is delayed batch triage, not instantaneous transaction blocking. Selection sees all scores in a completed hour; up to one hour of review batching is implied. Largest Amount is an equal-capacity baseline.

Report average precision, ROC AUC, Brier score, prevalence, hourly queue precision/recall, flagged fraud amount share and fixed-threshold review share. Flagged amount is not prevented loss: an alert may fail to stop or recover funds. A fixed probability threshold can exceed analyst capacity; the strict hourly queue and threshold policy are not interchangeable.

The average-precision uncertainty interval resamples observed test hours. With only 12 test hours it is a sensitivity diagnostic, not evidence across independent cards or future days. PSI uses training-quantile score bins with smoothing, without an asserted universal cutoff. Validation permutation importance ranks anonymised components and cannot support customer-level business explanations.
