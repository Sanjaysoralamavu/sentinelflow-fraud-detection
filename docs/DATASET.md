# Dataset setup, audit and data pipeline

Owner: Aravindan Chidambaram (work package 2)

## 1. Get the file

1. Open https://data.mendeley.com/datasets/y64bbnm2s3/1 (CreditTransAct, version 1).
2. Download `credit_card_fraud.parquet` (about 487 MB). If you use "Download All",
   unzip the archive and take only that Parquet file.
3. Put it in the repository, unchanged (do not rename columns or re-save it):

```bash
mkdir -p data/raw data/processed
mv /path/to/download/credit_card_fraud.parquet data/raw/
ls -lh data/raw/credit_card_fraud.parquet
shasum -a 256 data/raw/credit_card_fraud.parquet
```

The verified SHA-256 is:

```
4a33623216213beb121cdc85006cf36f336601af7645e73feabe1abdf94e0823
```

If your hash differs, Mendeley may have replaced the file. Stop, record the new
version and hash in an issue, and do not run experiments on it silently.

`data/raw/` and every `*.parquet` file are git-ignored. Never commit the dataset.

## 2. Run the audit

```bash
python -m sentinelflow.data.audit --config configs/full.yaml
```

It reads the file one row group at a time (memory stays bounded) and writes to
`results/metrics/audit/`:

| File | Contents |
| --- | --- |
| `dataset_audit.json` | Checksum, schema, row counts, nulls, channel and fraud counts, pass/fail checks, warnings |
| `channel_summary.csv` | Rows, fraud rows and fraud rate per `card_present_cnp` value |
| `column_profile.csv` | Type, nulls, min/max/mean or distinct count and examples for every column |
| `leakage_screen.csv` | How well each column alone predicts `is_fraud` on a 200,000-row CNP sample |

Acceptance checks (the command exits with status 1 if any fails):

| Check | Expected |
| --- | --- |
| `sha256_matches` | verified hash above |
| `source_rows` | 15,000,000 |
| `cnp_rows` | 3,843,653 |
| `cnp_fraud_rows` | 564,272 |
| `target_is_binary` | only 0/1, no nulls |

Useful flags: `--skip-checksum` (faster reruns), `--no-expected` (other files),
`--screen-rows N`, `--output DIR`.

### Reading the warnings

- **CNP look-alike values**: a channel value containing `cnp` that the strict
  filter excludes. Confirm with the team whether it belongs in `data.cnp_values`.
- **Near-perfect single columns** (`flag = True` in `leakage_screen.csv`, AUC
  strength at or above 0.99): a column that separates fraud almost on its own may
  be derived from the label (for example a generator's fraud-pattern or
  risk-score field). Review it. If it is label-derived, add it to
  `data.preprocessing.exclude_columns` and document why.
- **Suspicious names**: names containing words such as fraud, label, signal,
  cluster, pattern, chargeback or dispute. This is a prompt to look, not proof.

## 3. Pipeline behaviour

```
Parquet row group ─► strict CNP filter ─► split off is_fraud ─► drop IDs ─► CNPChunk
                                                                             │
          first warmup_rows CNP rows ◄── split_warmup ◄──────────────────────┘
                 │                               │
     FraudPreprocessor.fit (once)        evaluation chunks ─► stream (WP3) ─► drift ─► transform
```

- `sentinelflow.data.loader.iter_cnp_chunks` yields `CNPChunk` objects: raw
  feature `DataFrame`, `int8` labels, global CNP row positions, source row group.
  The target and `customer_id`/`transaction_id` never appear in `features`.
- `split_warmup` / `load_warmup_and_stream` return the first `data.warmup_rows`
  CNP rows plus an iterator over every remaining row, with no gap or overlap.
- Row-group order follows `stream.row_group_shuffle` (on in `configs/full.yaml`):
  a permutation seeded by `project.random_seed`. See section 4 for why this is
  required on this file.
- `frame_hash(features, labels)` is the reproducibility hash for a batch.

### Preprocessing

`FraudPreprocessor` is fitted once on the warm-up frame and then frozen; a second
`fit` raises. Column roles are inferred from dtypes:

| Role | Rule | Transform |
| --- | --- | --- |
| Numeric | int, float, bool | median imputation, standard scaling |
| Categorical | string/category with 2 to `max_categories` values | `__missing__` imputation, one-hot; unseen categories become all zeros |
| Dropped | more than `max_categories` values, constant, datetime, or excluded | recorded with a reason in the manifest |

Output is a dense `float32` matrix by default. Passing the target or an
identifier column to `fit` or `transform` raises `LeakageError`.

Optional configuration (defaults shown), under `data:` in the YAML:

```yaml
preprocessing:
  max_categories: 100
  sparse_output: false
  exclude_columns: []
```

`preprocessor.manifest.write_json(path)` writes the feature manifest: input,
included, numeric, categorical and dropped columns, output feature names and the
number of warm-up rows used to fit.

## 4. The file is sorted by segment (audit finding, 2026-10-17)

The verified file (480 row groups of 32,768 rows) is stored in segment order, and
every row group holds a single segment. Strict-CNP rows in file order:

| CNP rows | Row groups | Segment | CNP fraud rate |
| --- | --- | --- | --- |
| 0 to 1,461,787 | 0-191 | A_established | about 8% |
| 1,461,787 to 2,594,440 | 192-335 | B_regular | about 12.5% |
| 2,594,440 to 3,399,898 | 336-431 | C_new | about 20% |
| 3,399,898 to 3,843,653 | 432-479 | D_guest | about 31.6% |

Read in file order, the 500,000-row warm-up is 100% segment A (8.1% fraud), and
the evaluation stream contains three natural shifts (near 29%, 63% and 87% of
the stream) that overlap the injected drift window. That would confound
controlled drift with segment change, and no-drift-control alarms would not be
false alarms.

Seeded row-group shuffling fixes the warm-up but leaves batch-level noise,
because a 50,000-row batch contains only about six single-segment row groups:

| Order | Warm-up fraud | Warm-up segment A share | Batch fraud rate range (std) |
| --- | --- | --- | --- |
| File order | 8.1% | 100% | 8.0%-32.2% (0.075) |
| Shuffle, seed 543 | 14.2% | 36% | 8.9%-22.6% (0.031) |
| Shuffle, seed 1543 | 15.1% | 41% | 8.9%-23.6% (0.035) |
| Shuffle, seed 2543 | 14.4% | 38% | 9.0%-21.9% (0.028) |

The full strict-CNP population has a 14.7% fraud rate. Open team decision: accept
this batch-level noise (and report it), or interleave rows across a window of
row groups so each batch approaches the population mix.

Other audit results on the verified file: every acceptance check passes; no
column reaches the 0.99 single-column leakage threshold (the strongest,
`geo_distance_km`, scores 0.945); `chargeback_history_count` is flagged by name
only and carries no signal (AUC 0.500). High single-column scores across many
engineered fraud-signal columns explain why combined models score near 0.99.

## 5. Tests

```bash
make lint
make test
```

`tests/test_loader.py`, `tests/test_preprocessing.py` and `tests/test_audit.py` use
a small synthetic Parquet file with several row groups (`tests/conftest.py`), so
they run in CI without the real dataset.
