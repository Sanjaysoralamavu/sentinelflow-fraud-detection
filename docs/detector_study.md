# Detector study: ADWIN and DDM under drift (issue #10)

This document reviews the detector and recovery parameters, explains the
timing limits they impose, and describes how to run the multi-seed study.
Results will be added after the full-stream runs.

## How to run

```bash
# Default scenarios: no_drift, separability, and camouflage (once #14 is on main)
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

## Interpretation notes

- **DDM only detects rising error.** The current controlled drift increases
  fraud-versus-legitimate separation (issue #8), which is expected to lower
  model error, so DDM is not expected to signal on it. ADWIN can signal on falling error too, and
  a reset after an improvement is reported as such, not as a detection.
- **No tuning to force alarms.** Parameters are fixed for every scenario. A
  missed detection under these settings is reported as a result.
