# Tabular Classification Dataset Specification

## Required logical schema

A training table contains feature columns plus one classification target column.

- target name: configured by `target_column` (default `target`)
- minimum classes: 2
- minimum rows per class for a random stratified holdout: 2
- duplicate column names: rejected
- feature columns: numeric, boolean, string, or categorical values supported
- empty feature set: rejected

## Preprocessing rules

1. Fit all learned preprocessing on the training partition only.
2. Numeric values are converted to floating point; missing/invalid numeric values become `NaN` and are handled by TabDPT's training-fitted imputer.
3. Categorical values are mapped using training-only vocabularies. Missing and unseen values have distinct codes.
4. Target labels are mapped to integer IDs for TabDPT and decoded back to string labels for user-facing predictions.
5. Inference tables must contain exactly the fitted feature names. Order is normalized to training order.

## Split rules

Random stratification is acceptable only for approximately IID rows. Preserve user-provided partitions for temporal, grouped, panel, repeated-entity, patient, household, geographic, or leakage-sensitive data.

## Operational envelope

The initial DIMER manifest defaults to at most 10,000 training rows and a prediction context of 2,048 rows. These are service controls, not intrinsic TabDPT model limits. Increase only after resource testing.

## Evaluation

Report accuracy and log loss. Report ROC-AUC when its assumptions are satisfied. Keep an independent test set whenever model or operational settings are selected using a validation partition.