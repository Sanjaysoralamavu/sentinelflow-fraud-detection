# Detector study: ADWIN and DDM under drift (issue #10)

This document reports the multi-seed detector study on the full audited
evaluation stream, reviews the detector and recovery parameters, and explains
the timing limits they impose.

## Summary

- **No false alarms from ADWIN** in any no-drift run. DDM raised false alarms
  in one of three seeds.
- **Camouflage drift is detected by both detectors in every seed**, typically
  7–9 source batches after drift starts (14–16 observed batches, including the
  7-batch label delay). ADWIN did this with no false alarms.
- **Detection plus recovery pays off.** In the fully drifted camouflage phase,
  adaptive_adwin keeps recall at 0.921 and F1 at 0.953, against 0.616 and
  0.760 for the same learner without a detector (adaptive_none) and 0.616 and
  0.762 for the frozen random forest.
- **Separability drift is not detected, and does not need to be.** It makes
  fraud easier to separate, so every model's error falls and there is nothing
  to recover from.
- **Recovery is correct in every run:** zero recovery violations across all
  nine runs.

## How to run

```bash
# Default scenarios: no_drift, separability and camouflage
python scripts/run_detector_study.py --seeds 543 544 545

# Also write per-run evaluation CSVs, figures and dashboards
python scripts/run_detector_study.py --seeds 543 544 545 --evaluation-outputs

# Choose scenarios explicitly (keys override the drift config section)
python scripts/run_detector_study.py --seeds 543 544 545 \
    --scenario no_drift:enabled=false,scenario=none \
    --scenario camouflage:enabled=true,scenario=camouflage

# The 30-batch experiment (first 1.5M evaluation rows)
python scripts/run_detector_study.py --seeds 543 --max-evaluation-rows 1500000
```

Artifacts go to `results/metrics/experiment/detector_study/` (git-ignored):

| File | Contents |
| --- | --- |
| `detector_study.json` | All parameter settings and every run |
| `detector_summary.csv` | One row per run and detector: false alarms, missed detection, source- and observed-batch delays, alarm counts by state, recovery violations |
| `model_phase_summary.csv` | One row per run, model and phase (plus `overall`): precision, recall, F1, FPR, ROC-AUC, average precision, precision@100 for every static and adaptive model |
| `<scenario>/seed_<seed>/drift_events.json` | Every logged alarm, confirmed or suppressed |
| `<scenario>/seed_<seed>/study_run.json` | Everything above for one run |
| `<scenario>/seed_<seed>/run_manifest.json` | Full configuration and run summary |
| `<scenario>/seed_<seed>/evaluation/` | Evaluation CSVs, figures and dashboards (with `--evaluation-outputs`) |

Only `project.random_seed` changes between seeds. `drift.random_seed` is kept
fixed, so every seed sees the same drift and the seeds measure model,
shuffling and detector-sampling variability.

Detector quality is scored by `evaluation.detectors` against
`drift_start_batch`, the first source batch outside `pre_drift` (`None`
without drift). With no drift every confirmed alarm is a false alarm. A
detection counted there may have `observed_batch` after the last scored batch;
`confirmed_after_stream_end` in `detector_summary.csv` shows those, since they
cannot change any prediction.

The script exits with a non-zero status if any confirmed alarm breaks the
recovery contract: retraining on 1 to `recent_window_rows` rows, only from
batches whose labels were already released, with the reset count increasing by
one. Suppressed alarms must never retrain.

## Parameter review

| Parameter | Value | Role | Review |
| --- | --- | --- | --- |
| `label_delay_batches` | 7 | Labels for batch *b* are learned at observed batch *b* + 7 | Sets the minimum observed-batch detection delay to 7 |
| `detector_grace_batches` | 12 | Alarms in the first 12 learned batches are logged but suppressed | Learned batch *k* is observed batch *k* + 6, so grace covers observed batches 7–18. It ends before any drifted labels arrive, so it only suppresses early, pre-drift alarms |
| `detector_cooldown_batches` | 20 | Minimum observed batches between confirmed resets; a signal inside it is held | In the 30-batch run this allows at most one confirmed reset after drift |
| `adwin_delta` | 0.002 | ADWIN confidence; smaller is more conservative | River default. ADWIN reacts to a change in mean error in either direction |
| `ddm_warm_start` | 30 | Errors DDM needs before it may signal | Small against 2,000 errors per batch; reached within the first released batch |
| `detector_sample_size` | 2000 | Errors sampled per released batch | 4% of a 50,000-row batch; keeps detector input equal across batches |
| `recent_window_rows` | 200,000 | Rows used to retrain after a confirmed alarm | Four 50,000-row batches |
| `threshold` | 0.50 | Turns scores into the 0/1 predictions behind each error | Fixed; not tuned for detection |

## Timing limits

With 50,000-row batches, `start_fraction` 0.45 and `end_fraction` 0.70:

| Stream | Batches | Drift starts (source) | Fully drifted (source) | First drifted labels observed | Drifted-label batches observed while scoring |
| --- | --- | --- | --- | --- | --- |
| 1.5M rows (30-batch experiment) | 0–29 | 13 | 20 | 20 | 20–29 (10 batches) |
| Full stream, 3,343,653 rows | 0–66 | 30 | 46 | 37 | 37–66 (30 batches) |

Consequences:

1. **Detection delay floor.** No alarm about drifted batch *b* can come before
   observed batch *b* + 7.
2. **Short post-drift window in the 30-batch run.** Only source batches 13–22
   have labels released while the stream is still being scored. Labels for
   source batches 23–29 are released after the last scored batch, so alarms on
   them (`confirmed_after_stream_end`) cannot change any prediction.
3. **One reset at most.** With a 20-batch cooldown and 10 observed post-drift
   batches, a second confirmed reset cannot happen in the 30-batch run.
4. **Recovery data is partly pre-drift.** A reset at observed batch *t* retrains
   on source batches *t* − 10 to *t* − 7. An early detection at observed batch 20
   retrains on source batches 10–13, which are mostly pre-drift.

## Results

Full audited evaluation stream (3,343,653 rows, 67 batches of 50,000, labels
released after 7 batches), drift starting at source batch 30, seeds 543, 544
and 545, settings as in the parameter table. Raw numbers are in
`detector_summary.csv` and `model_phase_summary.csv`.

### Detector quality

Confirmed alarms only. Delays are counted from drift start (source batch 30);
"seed 543 / 544 / 545" lists one value per seed.

| Scenario | Detector | False alarms | Missed detection | Source delay | Observed delay |
| --- | --- | --- | --- | --- | --- |
| no_drift | ADWIN | 0 / 0 / 0 | – | – | – |
| no_drift | DDM | 0 / 3 / 0 | – | – | – |
| separability | ADWIN | 0 / 0 / 0 | yes / no* / yes | – / 36 / – | – / 43 / – |
| separability | DDM | 0 / 1 / 0 | yes / no† / yes | – / 2 / – | – / 9 / – |
| camouflage | ADWIN | 0 / 0 / 0 | no / no / no | 8 / 7 / 9 | 15 / 14 / 16 |
| camouflage | DDM | 0 / 1 / 0 | no / no / no | 8 / 2† / 8 | 15 / 9† / 15 |

\* The ADWIN alarm in separability seed 544 was released at observed batch 73,
after the last scored batch (66), so it could not change any prediction. It is
effectively a miss.

† DDM in seed 544 is noisy in every scenario: it raises three confirmed alarms
with no drift at all, so its early "detections" in that seed are not evidence
of drift.

Every run had **zero recovery violations**: each confirmed alarm retrained a
fresh learner on at most 200,000 rows of already-released labels, and no
suppressed alarm retrained.

### Model performance

Mean over the three seeds (± standard deviation where it is not negligible).
FPR is in percent.

**No drift (whole stream).** All models are near ceiling and indistinguishable:
recall 0.993–0.994, F1 0.995–0.997, FPR 0.006–0.055%.

**Separability, drifted phase.** Every model improves: recall 0.995 (frozen
logistic) to 1.000 (frozen random forest), F1 ≥ 0.996. There is nothing for the
detectors to act on.

**Camouflage, drifted phase:**

| Model | Recall | Precision | F1 | FPR % | ROC-AUC | Avg. precision |
| --- | --- | --- | --- | --- | --- | --- |
| static_logistic | 0.478 ± 0.088 | 0.994 | 0.642 ± 0.083 | 0.054 | 0.982 | 0.939 |
| static_random_forest | 0.616 ± 0.037 | 0.999 | 0.762 ± 0.029 | 0.007 | 0.993 | 0.977 |
| adaptive_none | 0.616 ± 0.023 | 0.995 | 0.760 ± 0.018 | 0.051 | 0.989 | 0.965 |
| **adaptive_adwin** | **0.921 ± 0.008** | 0.989 | **0.953 ± 0.004** | 0.190 | 0.994 | 0.987 |
| adaptive_ddm | 0.906 ± 0.054 | 0.987 | 0.944 ± 0.031 | 0.218 | 0.994 | 0.985 |
| online_ensemble | 0.607 ± 0.035 | 0.996 | 0.754 ± 0.027 | 0.045 | 0.989 | 0.964 |

Over the whole camouflage stream, recall is 0.949 for adaptive_adwin and 0.944
for adaptive_ddm, against 0.849 for adaptive_none and 0.854 for the frozen
random forest.

### Interpretation

1. **The detectors are what makes adaptation work under camouflage.** Plain
   incremental learning (adaptive_none) and online bagging do no better than
   the frozen random forest, because one partial-fit step per batch cannot
   undo a model fitted on the warm-up and the long pre-drift stream. Resetting and retraining
   on the recent window raises drifted-phase recall by about 0.3.
2. **The cost is a small rise in false positives.** FPR goes from about 0.05%
   to about 0.2% for the detector models in the drifted phase. Precision stays
   above 0.98.
3. **Detection comes about halfway through the ramp.** Source batches 37–39
   are at roughly 45–60% of the ramp, so the detectors react once camouflage
   has made a measurable dent in accuracy, not to the first small shift.
4. **Delay is dominated by the label delay.** Seven of the 14–16 observed
   batches are the label delay itself.
5. **ADWIN is the more reliable detector here.** It matches DDM on camouflage,
   is tighter across seeds (recall ± 0.008 against ± 0.054), and never
   false-alarmed. DDM's sensitivity to the error stream makes one seed noisy.
6. **Recovery data is post-drift on the full stream.** The first reset comes at
   observed batch 44–46, so the 200,000-row window covers source batches
   34–39, all inside the drift ramp. The partly pre-drift window noted under
   timing limits applies only to the shorter 30-batch run.
7. **Ranking metrics hide the damage.** ROC-AUC barely moves (≥ 0.978 for every
   model), and precision@100 stays at 0.87–1.00 because fraud is about 15% of
   each batch. The loss shows at the 0.5 decision threshold, in recall.

## Interpretation notes

- **DDM only detects rising error.** Separability drift increases
  fraud-versus-legitimate separation, which lowers model error, so DDM is not
  expected to signal on it; the results confirm this. ADWIN can signal on
  falling error too, but did not do so before the end of the scored stream.
- **No tuning to force alarms.** Parameters are fixed for every scenario. A
  missed detection under these settings is reported as a result.
