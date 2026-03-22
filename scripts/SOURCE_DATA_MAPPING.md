# Raw Data to Source Data Mapping

Mapping established by value-matching "Network Burst Duration - Avg (sec)"
values from each `data/raw/` file against all neuralMetrics CSVs under
`data/source_data/`. Only folders with significant match rates (>=10%) are
listed as sources. Low match rates (<10%) across many unrelated folders
indicate coincidental numeric overlap, not a true source relationship.

## Summary Table

| Raw File | Source Folder(s) | Match Rate | Coverage |
|----------|-----------------|------------|----------|
| `MEA_r1_dataset_0915_per_well_genoko_corrected.csv` | `Genetic KD data/r1/{p1,p2,p3}_excel only` | 70% p2, 18% p3, 16% p1 | 100% |
| `MEA_r2_dataset_0821_per_well_genoko.csv` | `Genetic KD data/r3/{WT,PTEN} envelope` | 58% WT, 45% PTEN | 100% |
| `MEA_r3_dataset_0821_per_well_genoko.csv` | `Genetic KD data/r2/{WT Envelope, Envelop network PTEN}` | 60% PTEN, 41% WT | 100% |
| `R2_GENO_KD_ISIthreshold.csv` | **Not found** | max 4.8% (coincidental) | 36% |
| `R3_GENO_KD_ISIthreshold.csv` | **Not found** | max 3.3% (coincidental) | 34% |
| `MEA_r2_dataset_0804_per_well.csv` | `20241030MEA/round 2/spikes50/{plate2, plateD}` | 54% plate2, 48% plateD | 100% |
| `MEA_r3_dataset_0804_per_well.csv` | `20241030MEA/round 3/spikes50/{plate_PTEN_correct, plate_WT}` | 64% PTEN, 39% WT | 100% |
| `MEA_r4_dataset_0804_per_well.csv` | `20241030MEA/round4/{PTEN KO, WT}` | 70% PTEN KO, 31% WT | 100% |

## Important: Round Labels Are Swapped for Genetic KO Envelope Files

Value-matching reveals that the round numbers in `MEA_r2_dataset_0821_per_well_genoko.csv` and `MEA_r3_dataset_0821_per_well_genoko.csv` are **reversed** relative to their actual source data:

- `MEA_r2_..._genoko.csv` (labeled "R2") actually contains data from **`Genetic KD data/r3/`** (round 3 source folders)
- `MEA_r3_..._genoko.csv` (labeled "R3") actually contains data from **`Genetic KD data/r2/`** (round 2 source folders)

This swap is confirmed purely by value-matching (58%/45% and 60%/41% match rates respectively) and is not an artifact of naming assumptions.

## Detailed Evidence

All results below were produced by matching every non-zero NBD value from
each raw file against every neuralMetrics CSV in `data/source_data/`,
using `numpy.isclose(rtol=1e-4)`.

### `MEA_r1_dataset_0915_per_well_genoko_corrected.csv`

417 non-zero NBD values. **417/417 (100%) matched to at least one source folder.**

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 291/417 | 69.8% | `Genetic KD data/r1/p2_excel only` |
| 77/417 | 18.5% | `Genetic KD data/r1/p3_excel only` |
| 68/417 | 16.3% | `Genetic KD data/r1/p1_excel only` |

The combined coverage from these three folders accounts for 100% of values.
Match percentages overlap because a single raw file value can match values
in multiple source folders.

### `MEA_r2_dataset_0821_per_well_genoko.csv`

602 non-zero NBD values. **602/602 (100%) matched.**

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 349/602 | 58.0% | `Genetic KD data/r3/WT envelope` |
| 271/602 | 45.0% | `Genetic KD data/r3/PTEN envelope` |

Note: despite the raw filename containing "r2", the values match
`Genetic KD data/r3/`, not `r2/`. This may reflect a naming convention
difference between the raw file and the source folder.

### `MEA_r3_dataset_0821_per_well_genoko.csv`

483 non-zero NBD values. **483/483 (100%) matched.**

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 290/483 | 60.0% | `Genetic KD data/r2/Envelop network PTEN` |
| 197/483 | 40.8% | `Genetic KD data/r2/WT Envelope` |

Note: despite the raw filename containing "r3", the values match
`Genetic KD data/r2/`, not `r3/`. This mirrors the cross-labeling seen
with R2 above.

### `R2_GENO_KD_ISIthreshold.csv`

393 non-zero NBD values. **No source folder found (max 4.8% coincidental match).**

The highest-matching folder is `20241030MEA/round 2/spikes50/plateD` at
only 19/393 (4.8%). All other folders are below 5%. The total coverage
across all source folders combined is only 143/393 (36.4%), with no single
folder exceeding 5% -- indicating scattered coincidental numeric overlaps,
not a true source relationship.

This file's values do not exist in `data/source_data/`.

### `R3_GENO_KD_ISIthreshold.csv`

552 non-zero NBD values. **No source folder found (max 3.3% coincidental match).**

The highest-matching folder is `20241030MEA/round 3/spikes50/plate_PTEN_correct`
at only 18/552 (3.3%). Total coverage is 190/552 (34.4%) spread thinly across
38 folders.

This file's values do not exist in `data/source_data/`.

### `MEA_r2_dataset_0804_per_well.csv`

337 non-zero NBD values. **337/337 (100%) matched.**

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 182/337 | 54.0% | `20241030MEA/round 2/spikes50/plate2` |
| 161/337 | 47.8% | `20241030MEA/round 2/spikes50/plateD` |

Secondary matches exist at lower rates in other spike thresholds and
duplicate folders, because multiple spike-threshold folders share a subset
of source recordings for overlapping date ranges:

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 64/337 | 19.0% | `20240904/plate2` |
| 64/337 | 19.0% | `20240806/plate2` |
| 53/337 | 15.7% | `20241030MEA/round 2/{spikes150,spikes75,spikes20,spikes100}/plateD` (each) |
| 47/337 | 13.9% | `20241030MEA/round 2/{spikes150,spikes75,spikes20,spikes100}/plate2` (each) |
| 40/337 | 11.9% | `20240904/plateD` |
| 40/337 | 11.9% | `20240806/plateD` |

The `spikes50` folders have the highest match rate because they contain
more dates (20 and 19 files) than other spike thresholds (12-13 files),
making them the primary source.

### `MEA_r3_dataset_0804_per_well.csv`

611 non-zero NBD values. **611/611 (100%) matched.**

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 388/611 | 63.5% | `20241030MEA/round 3/spikes50/plate_PTEN_correct` |
| 248/611 | 40.6% | `20241030MEA/round 3/network500/plate_PTEN_correct` |
| 248/611 | 40.6% | `20241030MEA/round 3/network500_spike50/plate_PTEN_correct` |
| 236/611 | 38.6% | `20241030MEA/round 3/spikes50/plate_WT` |

Secondary matches at lower rates across other spike thresholds and
subsets of the same data:

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 166/611 | 27.2% | `20241030MEA/round 3/{spikes150,spikes75,spikes20,spikes100}/plate_PTEN_correct` (each) |
| 135/611 | 22.1% | `20241030MEA/round 3/{network500,network500_spike50}/plate_WT` (each) |
| 95/611 | 15.5% | `20241003_indu/plate_PTEN_correct` |
| 82/611 | 13.4% | `20241030MEA/round 3/late days network #spikes500/platePTEN_latedays_#spikes500` |
| 74/611 | 12.1% | `20241030MEA/round 3/late days network #spikes500/plateWT_latedays_#spikes500` |
| 63/611 | 10.3% | `20241030MEA/round 3/{spikes150,spikes75,spikes20,spikes100}/plate_WT` (each) |

The `spikes50` folders have the highest match rate (64% and 39%), making
them the primary source.

### `MEA_r4_dataset_0804_per_well.csv`

259 non-zero NBD values. **259/259 (100%) matched.**

| Match Count | Match % | Source Folder |
|-------------|---------|---------------|
| 180/259 | 69.5% | `20241030MEA/round4/PTEN KO` |
| 81/259 | 31.3% | `20241030MEA/round4/WT` |

No other folders exceeded 10%.

## Additional Note

The `data/source_data/Genetic KD data/` folder also contains copies of
some raw CSV files at its root level:

- `MEA_r1_dataset_0915_per_well_plate_corrected.csv`
- `MEA_r2_dataset_0821_per_well_genoko.csv`
- `MEA_r3_dataset_0821_per_well_genoko.csv`
