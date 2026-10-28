# Drift scenarios

SentinelFlow injects controlled drift into the evaluation stream before
preprocessing, so every model scores and learns from identical values. The
scenario is selected with `drift.scenario` in `configs/full.yaml`:

| `drift.scenario` | Role | What changes |
| --- | --- | --- |
| `none` | No-drift control | Nothing. All batches stay in phase `pre_drift`. |
| `separability` | Original scenario (default) | Fraud rows are shifted along a fixed random direction per numeric column. |
| `camouflage` | Degrading scenario | Fraud rows move toward legitimate behavior, so the two classes become harder to separate. |

`configs/full.yaml` keeps `enabled: true`, which means `separability`. Configs
without `drift.scenario` behave exactly as before (`enabled: false` means
`none`). When `drift.scenario` is present it wins over `drift.enabled`, so a
scenario can be applied as an override on top of any base config.

The drift schedule is shared by all scenarios: `start_fraction`,
`end_fraction` and `max_intensity` give the `pre_drift`, `ramp_drift` and
`drifted` phases, with intensity ramping linearly from 0 to `max_intensity`.
Labels and row counts are never changed, and the label is never added to the
feature frame.

## Interpretation

**`none`** is the stability and false-alarm baseline. Any degradation or
detector alarm here is not caused by injected drift.

**`separability`** is a covariate shift confined to the fraud class. Because
each column's direction is random (seeded), it can widen or narrow the gap to
legitimate traffic, so read its effect from the diagnostics. On the real data
with seed 543 it widens the gap (see below). A frozen model is not hurt by it;
it tests whether stale decision boundaries still catch fraud that moved.

**`camouflage`** models attackers who adapt to look normal. For the selected
columns, a fraud value moves toward the legitimate distribution as intensity
grows:

- numeric columns, with `a = intensity * strength`:
  `x' = (1 - a) * x + a * (legit_mean + (x - fraud_mean) * legit_std / fraud_std)`.
  `a = 0` is the identity and `a = 1` gives fraud the legitimate mean and
  spread.
- categorical columns: a share `a` of fraud rows takes its category from a
  random legitimate row of the same batch.

Legitimate rows are untouched, so the false-positive rate stays flat while
recall falls. Columns are the `top_k` numeric columns by `|d|` and the
`categorical_top_k` categorical columns by total variation distance in the
first batch containing both classes, frozen for the rest of the stream, or an
explicit `drift.camouflage.columns` list.

```yaml
drift:
  scenario: camouflage        # none | separability | camouflage
  camouflage:
    strength: 1.0             # 0..1, scales the maximum pull
    top_k: 4                  # numeric columns
    categorical_top_k: 8      # 0 disables categorical camouflage
    # columns: [avg_amount_deviation_sigma, cvv_match_status]
```

Determinism: numeric camouflage uses no random numbers. Category draws use a
generator keyed by `drift.random_seed` and the batch's stream progress, so a
batch is reproduced exactly regardless of which other batches ran.

Limitations:

- This is a stress scenario, not a claim about real attackers, and it is not
  tuned so that adaptive models win.
- Only the selected columns change. Categorical bin columns derived from
  numerics (for example `geo_velocity_bin`) are only camouflaged if they are
  selected, and are resampled independently of their numeric source.
- Legitimate statistics come from the same batch, so camouflage tracks
  legitimate behavior even if it moves.
- Why categorical columns matter: on CreditTransAct most signal is
  categorical (fraud-versus-legitimate total variation distance 0.6 to 0.77
  for `cvv_match_status`, `device_fingerprint_match`, `ip_address_type` and
  others). Numeric-only camouflage cut `|d|` from 1.07 to 0.58 but left
  frozen-baseline recall near 0.98, so categorical camouflage is on by
  default.

## Diagnostics

`sentinelflow.streaming.diagnostics` measures separation without any model:
mean absolute standardized mean difference `|d|` over numeric columns, and mean
total variation distance (TVD) over categorical columns.

`sentinelflow.streaming.scenario_report` compares the three scenarios. It fits
the preprocessor and frozen static baselines once on the warm-up window,
replays the same stream under each scenario, and reports separation, ROC-AUC,
recall and false-positive rate per drift phase, with deltas against the
no-drift control. All scenarios are grouped by the configured drift schedule,
so the same rows are compared.

```bash
python -m sentinelflow.streaming.scenario_report --config configs/full.yaml \
    --max-evaluation-rows 600000
```

The JSON report is written to
`results/metrics/drift_scenarios/scenario_report.json`.
`--max-evaluation-rows` shrinks the stream but keeps the schedule, because
phases are defined by progress through the evaluated rows.

### Result on CreditTransAct (seed 543, 600,000 evaluation rows)

Phase rows: 250,000 `pre_drift`, 150,000 `ramp_drift`, 200,000 `drifted`.
Frozen models are trained once on the 500,000 warm-up rows. Threshold 0.5.

| Scenario | Phase | `\|d\|` | TVD | Logistic AUC | Logistic recall | RF AUC | RF recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | drifted | 1.058 | 0.555 | 1.000 | 0.992 | 0.999 | 0.992 |
| separability | drifted | 1.760 | 0.555 | 1.000 | 0.994 | 1.000 | 1.000 |
| camouflage | pre_drift | 1.073 | 0.560 | 1.000 | 0.994 | 0.999 | 0.993 |
| camouflage | ramp_drift | 0.837 | 0.425 | 0.996 | 0.856 | 0.997 | 0.903 |
| camouflage | drifted | 0.583 | 0.315 | 0.978 | 0.373 | 0.991 | 0.616 |

In the `drifted` phase, camouflage lowers frozen-logistic recall by 0.62 and
ROC-AUC by 0.022 against the no-drift control, and lowers frozen-RF recall by
0.38. The false-positive rate does not change (about 0.001 or below). The
`pre_drift` rows are identical in every scenario, which confirms that the
no-drift configuration is unchanged.

Reading the numbers: ROC-AUC barely moves because the legitimate rows keep
their scores and the camouflaged fraud rows still tend to rank above them.
The damage shows at the fixed 0.5 threshold, where recall drops. Report both.
