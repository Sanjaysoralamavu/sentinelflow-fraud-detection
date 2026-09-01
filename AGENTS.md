# SentinelFlow Agent Guidance

## Code Review Rules

### Time-aware fraud evaluation
- Preserve chronological transaction order. Do not use random train/test splits for final evaluation.
- Fit scaling, sampling, threshold selection, and feature transformations only on training data.

### Drift experiments
- Compare static and adaptive models on the same stream batches and with the same evaluation windows.
- Record every simulated or detected drift event, its configuration, and the model adaptation taken.

### Security and reproducibility
- Never commit datasets, credentials, tokens, or personally identifiable information.
- Use fixed random seeds and document all model parameters and dataset versions.
- Treat PR-AUC, precision, recall, false-positive rate, and detection delay as primary metrics; do not use accuracy as the main result.
