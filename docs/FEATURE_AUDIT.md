# High-signal feature audit and ablation (issue #9)

Owner: Aravindan Chidambaram. Data: verified CreditTransAct file
(SHA-256 `4a336232…0e823`), strict CNP rows only (3,843,653 rows, 14.68% fraud).

## Bottom line

1. **No evidence of target or future-information leakage.** Every flagged feature
   describes the session, the device, or the account's past, so each could be
   known when the transaction is scored. `chargeback_history_count` carries no
   signal at all (single-column AUC 0.500).
2. **The near-perfect values are generator signatures.** Fraud was synthesised
   from four "compound signal clusters" (dataset description), and several
   features take values that the generator gives *only* to fraud rows: no
   legitimate row is faster than 5.2 s, travels above 449.5 km/h, or is more
   than 385.5 km from home. A real system would see legitimate transactions in
   those ranges, so these separations will not transfer to real data.
3. **Removing the flagged features barely changes performance.** The reduced
   and strict feature sets keep ROC-AUC ≈ 0.999 and PR-AUC ≈ 0.994–0.998
   (diagnostic below). The separation is spread across about 20 engineered
   fraud-signal features; ROC-AUC only falls below 0.95 after 23 of 31 features
   are removed. The final report should present near-0.99 scores as a property
   of this synthetic dataset, not as evidence of a strong detector.

## Method and limits

The dataset authors have not published the generator code. The page lists
33 features, four fraud signal clusters (Card-Not-Present, Bot/Card Testing,
Account Takeover, Geo-Velocity) and "controlled label noise in borderline
cases". There is also no event timestamp, only the `time_of_transaction`
bucket. Provenance is therefore judged from feature semantics plus four
empirical tests on the file, all reproducible with the commands at the end:

- value-level fraud rates for every categorical and integer column
- legitimate-class range of every numeric column, and the share of fraud rows outside it
- whether each bin column is a function of a numeric column (disjoint ranges)
- full-versus-reduced model comparison on identical rows

## Evidence

Values with at least 99% (or at most 1%) fraud and ≥1,000 rows
(`results/metrics/feature_audit/value_screen.csv`):

| Feature = value | Rows | Fraud rate | Share of rows |
| --- | ---: | ---: | ---: |
| `session_duration_bin` = bot_speed | 295,741 | 100.000% | 7.7% |
| `geo_velocity_bin` = impossible | 212,941 | 100.000% | 5.5% |
| `geo_distance_bin` = long | 212,944 | 100.000% | 5.5% |
| `merchant_category_vs_history` = never_before_category | 288,883 | 99.989% | 7.5% |
| `cards_on_device_30d` = 5 … 17 (each value) | 1,371–37,629 each | 99.98–100% | 11.6% total |
| `failed_attempts_before_success` = 4, 6, 10 | 66,920 / 44,695 / 22,315 | 99.99% | 3.5% total |
| `failed_attempts_before_success` = 5 | 32,626 | **0.46%** | 0.8% |
| `device_fingerprint_match` = known_device | 2,451,528 | 0.97% | 63.8% |

Fraud-only numeric ranges (`numeric_support.csv`):

| Numeric feature | Legitimate range | Fraud range | Fraud rows outside legitimate range |
| --- | --- | --- | ---: |
| `session_duration_sec` | 5.2 – 1,800 s | 0.3 – 1,800 s | 52.4% |
| `geo_distance_km` | 0 – 385.5 km | 0 – 8,000 km | 38.6% |
| `geo_velocity_kmh` | 0 – 449.5 km/h | 0 – 2,500 km/h | 38.4% |
| `cards_on_device_30d` | 1 – 15 | 1 – 17 | 1.2% |

Bins that are exact functions of a numeric column (non-overlapping ranges):

| Bin | Source | Flagged value |
| --- | --- | --- |
| `session_duration_bin` | `session_duration_sec` | bot_speed = 0.3–5.0 s |
| `geo_velocity_bin` | `geo_velocity_kmh` | impossible = 500.1–2,500 km/h |
| `geo_distance_bin` | `geo_distance_km` | long = 600–8,000 km |
| `transaction_amount_bin` | `transaction_amount` | none |

Other observations:

- `impossible` velocity and `long` distance mark the same rows (212,941 of 212,944).
  That is one Geo-Velocity cluster encoded twice.
- `never_before_category` has 31 legitimate rows out of 288,883. This matches the
  documented label noise, not real customer behaviour. Guest checkouts (segment D)
  never have this value; they are `no_history`.
- `failed_attempts_before_success` takes only 0–6 and 10. Values 7–9 never
  occur, and 5 is almost entirely legitimate while 4 and 6 are almost entirely
  fraud. Class-specific value sets like this are a sampling artifact.
- `chargeback_history_count`: fraud rate is 14.9% at 0 and about 12.5% at 1–3, so
  prior chargebacks *lower* the fraud rate slightly. The feature does not encode
  this transaction's outcome.

## Provenance assessment

| Feature | Available at scoring time? | Future or post-transaction information? | Derived from the label? | Domain signal or target proxy? | Production-style feature set? |
| --- | --- | --- | --- | --- | --- |
| `session_duration_sec`, `session_duration_bin` | Yes: session length up to checkout | No | Not from the label itself, but ≤5.2 s exists only in fraud rows (Bot/Card Testing cluster) | Valid concept; the hard cut-off is a synthetic proxy | Keep in full; drop in ablation |
| `geo_velocity_kmh`, `geo_velocity_bin` | Yes: needs the previous transaction's place and time, which are in the past | No | Same: >449.5 km/h exists only in fraud rows | Valid concept; real data has VPN and geolocation noise | Keep in full; drop in ablation |
| `geo_distance_km`, `geo_distance_bin` | Yes: distance from home or last location | No | Same: >385.5 km exists only in fraud rows; duplicates geo velocity | Valid concept; real travellers exist | Keep in full; drop in ablation |
| `merchant_category_vs_history` | Yes: computed from past purchases | No | 99.99% fraud for never_before_category; the 31 exceptions match label noise | Valid concept; real customers try new categories | Keep in full; drop in ablation |
| `chargeback_history_count` | Yes, if counted only from disputes resolved before scoring. Cannot be verified without timestamps | Not detectable: no relationship with this transaction's label | No (AUC 0.500) | Domain signal, currently uninformative | Keep; dropped in ablation as requested, expected to have no effect |
| `cards_on_device_30d` | Yes: trailing 30 days | No | 5–17 cards ⇒ 99.98–100% fraud | Valid concept; deterministic above 4 | Keep in full; drop in strict ablation |
| `failed_attempts_before_success` | Yes: attempts before this one | No | 4/6/10 ⇒ fraud, 5 ⇒ legitimate, 7–9 absent | Class-specific value sets; synthetic artifact | Keep in full; drop in strict ablation |

None of these features should be removed from the full configuration as leakage.
They are realistic fraud signals with unrealistically clean synthetic
distributions. The ablation configurations measure how much the headline
numbers depend on that cleanliness.

## Ablation configurations

`configs/full.yaml` is unchanged. Two new configurations are identical to it
except for `data.preprocessing.exclude_columns` (a column-to-reason map, so
every reason is written into the run manifest and the feature manifest) and
`paths.experiment_output` (so runs do not overwrite each other). A test fails
if any other setting drifts from `full.yaml`.

| Configuration | Removed columns | Model input columns |
| --- | --- | ---: |
| `full.yaml` | none (only target and identifiers) | 31 |
| `ablation_reduced.yaml` | the 4 flagged bins, their 3 numeric sources, `chargeback_history_count` | 23 |
| `ablation_strict.yaml` | reduced set + `cards_on_device_30d`, `failed_attempts_before_success` | 21 |

The numeric sources are removed with their bins because each bin is a function
of its numeric column. Dropping only `bot_speed`'s bin would leave
`session_duration_sec ≤ 5.0`, which encodes the same rule. The diagnostic
"bins only" row below shows this: it changes nothing.

Excluded columns are still loaded and still pass through drift injection; the
preprocessor drops them afterwards. So the stream, batch order, label delay
and drift are byte-identical across configurations (tested), and only the
model inputs differ. The target and both identifiers are removed in every
configuration. Listing them in `exclude_columns` is rejected, and so is a
misspelled column name.

## Diagnostic: how much do these features matter?

Diagnostic only, not the official runner. scikit-learn LogisticRegression and
HistGradientBoosting were trained on about 480,000 CNP rows from 60 randomly
chosen row groups (all segments) and tested on about 493,000 rows from 60
different row groups (14.5% fraud), threshold 0.5:

| Feature set | Features | Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| full | 31 | logistic | 0.9995 | 0.9983 | 0.9992 | 0.9929 | 0.9961 |
| full | 31 | gradient boosting | 0.9996 | 0.9985 | 0.9989 | 0.9924 | 0.9956 |
| bins only removed | 26 | logistic | 0.9994 | 0.9981 | 0.9993 | 0.9924 | 0.9959 |
| reduced | 23 | logistic | 0.9994 | 0.9980 | 0.9991 | 0.9923 | 0.9957 |
| reduced | 23 | gradient boosting | 0.9990 | 0.9980 | 0.9986 | 0.9921 | 0.9953 |
| strict | 21 | logistic | 0.9993 | 0.9979 | 0.9976 | 0.9913 | 0.9944 |
| strict | 21 | gradient boosting | 0.9994 | 0.9944 | 0.9960 | 0.9923 | 0.9942 |

In the test rows, 85.1% of fraud carries one of the four flagged values, and
96.5% once the two integer features are added (0.004% of legitimate rows do).
Removing them still leaves performance almost unchanged, because the same
fraud rows also carry other cluster signals: device, IP type, velocity,
CVV, 3-D Secure, credential changes and country.

Starting from the strict set, removing the remaining features in order of
single-column strength (logistic, about 240k train / 245k test rows):

| Features left | ROC-AUC | PR-AUC | F1 |
| ---: | ---: | ---: | ---: |
| 21 (strict) | 0.999 | 0.998 | 0.994 |
| 16 | 0.997 | 0.992 | 0.966 |
| 13 | 0.991 | 0.966 | 0.909 |
| 10 | 0.975 | 0.906 | 0.824 |
| 8 | 0.919 | 0.751 | 0.654 |
| 6 | 0.784 | 0.497 | 0.394 |

## Implications for the experiments

- Run the experiment matrix with both `full` and `ablation_strict`. Expect
  similar absolute scores; the useful comparison is how each set behaves under
  drift. The injected drift shifts numeric fraud features, and the strict set
  removes three of the most shifted ones.
- The camouflage drift scenario (issue #8) is where feature choice matters
  most: a scenario that moves fraud values into the legitimate ranges above
  will defeat the full feature set's hard cut-offs.
- In the report, describe performance as an upper bound on clean synthetic
  data and cite this audit.

## Reproduce

From the repository root, with the dataset in `data/raw/`:

```bash
# Value-level screen (about 1 minute): value_screen.csv, numeric_support.csv, signal_audit.json
python -m sentinelflow.data.signal_audit --config configs/full.yaml

# Feature manifests for all configurations, plus a static-baseline comparison
python -m sentinelflow.data.ablation_probe \
    --configs configs/full.yaml configs/ablation_reduced.yaml configs/ablation_strict.yaml

# Official end-to-end runs; artifacts go to results/metrics/experiment/<config>/
sentinelflow run --config configs/full.yaml
sentinelflow run --config configs/ablation_reduced.yaml
sentinelflow run --config configs/ablation_strict.yaml
```

Sources: dataset description at
https://data.mendeley.com/datasets/y64bbnm2s3/1 (CC BY 4.0).
