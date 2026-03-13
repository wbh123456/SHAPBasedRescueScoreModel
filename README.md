# SHAP-Based Rescue Score Model

A machine learning pipeline that quantifies pharmacological rescue of PTEN-related neuronal phenotypes using SHAP (SHapley Additive exPlanations) values from Multi-Electrode Array (MEA) recordings.

## Overview

This project trains a neural network to classify Wild-Type (WT) vs. PTEN-knockout neurons based on electrophysiological features, then uses SHAP DeepExplainer to measure how much a treatment shifts PTEN phenotypes back toward WT. The resulting **Rescue Score** ranks features by the degree of phenotypic rescue, supported by non-parametric statistical tests.

### Key Concept

```
Rescue Score = (μ_Treated − μ_PTEN) / (μ_WT − μ_PTEN)
```

A score of 1.0 means the treatment fully restores the feature to WT-level SHAP importance; 0.0 means no rescue.

## Pipeline

```
Raw MEA CSVs (long format, one row per well × parameter × recording date)
  │
  ▼
DataLoader
  • Concatenates CSVs across biological rounds
  • Computes recording day relative to each round's first date
  • Pivots long → wide (one row per well × day, one column per parameter)
  • Labels each sample: WTFalse | PTENFalse | PTENTrue
  • Drops WT-treated samples (not used in the classification task)
  • Filters to a window around the day of interest
    (DAY_OF_INTEREST ± extra_days, default DAY_OF_INTEREST = 21)
  │
  ▼
DataPreprocessor
  • Drops highly correlated features (Pearson r > 0.95)
  • Drops high-VIF features (multicollinearity)
  • Drops Std-suffix features (only Avg-level features retained)
  • Drops zero-variance columns
  • Batch correction via neuroHarmonize (treats recording round as site)
  • Z-score normalization via StandardScaler
  │
  ▼
ModelPipeline  ─── repeated R times ──────────────────────────────────────┐
  │                                                                        │
  ▼                                                                        │
  Stratified 5-Fold Cross-Validation  (classifier trained on WT vs PTEN)  │
  │                                                                        │
  ├─ For each fold:                                                        │
  │    1. Train feedforward neural network (WT vs PTEN only)              │
  │    2. Evaluate on held-out fold (accuracy, AUC-ROC)                   │
  │    3. Initialize SHAP DeepExplainer with 100-sample background        │
  │       drawn from the training set                                      │
  │    4. Compute mean SHAP values (WT class) for each group:             │
  │         • μ_WT      — Wild-Type wells                                 │
  │         • μ_PTEN    — PTEN-KO untreated wells                        │
  │         • μ_Treated — PTEN-KO treated wells                          │
  │    5. Store (μ_WT, μ_PTEN, μ_Treated) per feature per fold           │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
  │
  ▼
Result
  • Aggregates fold results across all repeats
  • Per feature, computes:
      – shift  = μ_Treated − μ_PTEN   (treatment effect in SHAP space)
      – denom  = μ_WT − μ_PTEN        (genotype gap in SHAP space)
      – Rescue Score = shift / denom
        (runs with |denom| < 10% of median |denom| are excluded)
  • Statistical tests (per feature, across all valid runs):
      – Wilcoxon signed-rank test on Rescue Scores (median ≠ 0)
      – Kruskal-Wallis H-test across WT, PTEN, Treated SHAP distributions
      – Dunn's post-hoc test: WT–PTEN and PTEN–Treated pairs
  • Only features with all p < 0.05 and MeanRescue > 0 are visualized
  • Outputs: Rescue Score bar chart, SHAP shift strip plot
```

## Raw Data Format

Each CSV file contains per-well electrophysiology recordings in **long format** — one row per unique combination of well, parameter, and recording date. Multiple CSV files (biological rounds) are specified in `data_config.py` and concatenated automatically.

### Required Columns

| Column | Type | Description |
|---|---|---|
| `filename` | string | Source recording filename (used as a sample identifier) |
| `well` | string | Well position (e.g. `A1`, `B3`) |
| `parameter` | string | Electrophysiology metric name (e.g. `Weighted Mean Firing Rate (Hz)`) |
| `bioDup` | integer | Biological replicate index |
| `techDup` | string | Technical replicate label (e.g. `A`, `B`) |
| `geno` | string | Genotype — must be `WT` or `PTEN` |
| `treatment` | boolean string | Whether the well received treatment — `true` or `false` |
| `date` | integer | Recording date in `YYYYMMDD` format |
| `value` | float | Measured value for this parameter |

> **Note:** The `round` column is added programmatically by `DataLoader` from `data_config.py` and must not appear in the raw CSV.

### Example Rows

```csv
filename,well,parameter,bioDup,techDup,plate,geno,treatment,date,value
20241116_PTEN(000)_neuralMetrics.csv,A1,Weighted Mean Firing Rate (Hz),1,A,PTEN,PTEN,true,20241116,0.42
20241116_PTEN(000)_neuralMetrics.csv,A1,Number of Bursts,1,A,PTEN,PTEN,true,20241116,14
20241116_PTEN(000)_neuralMetrics.csv,A2,Weighted Mean Firing Rate (Hz),2,A,WT,WT,false,20241116,1.87
20241116_PTEN(000)_neuralMetrics.csv,A2,Number of Bursts,2,A,WT,WT,false,20241116,31
```

Each unique `(filename, well, bioDup, techDup, geno, treatment, date)` combination corresponds to one well recording. All parameters measured for that well on that date appear as separate rows and are pivoted into columns during loading.

### Sample Labels

The pipeline derives a label from `geno` + `treatment` for each row after pivoting:

| `geno` | `treatment` | Label | Role |
|---|---|---|---|
| `WT` | `false` | `WTFalse` | Classifier training target (healthy baseline) |
| `PTEN` | `false` | `PTENFalse` | Classifier training target (disease phenotype) |
| `PTEN` | `true` | `PTENTrue` | Held out — SHAP evaluated but not used in training |
| `WT` | `true` | `WTTrue` | Excluded from all analyses |

### Adding a New Dataset

Register new CSV paths in `data_config.py`:

```python
data_config = {
    "my_experiment": [
        {"round": 1, "path": "data/raw/my_experiment_r1.csv"},
        {"round": 2, "path": "data/raw/my_experiment_r2.csv"},
    ]
}
```

Then run with `--dataset my_experiment`.

## Project Structure

```
├── main.py                 # Entry point — orchestrates load, preprocess, train, analyze
├── data_loader.py          # Loads raw CSVs, pivots, labels samples, filters by day
├── data_preprocessor.py    # Feature selection, neuroHarmonize batch correction, scaling
├── pipeline.py             # Repeated stratified K-fold CV with SHAP analysis per fold
├── network.py              # Keras feedforward classifier
├── shap_analyzer.py        # SHAP DeepExplainer wrapper for mean SHAP computation
├── result.py               # Rescue Score calculation, statistical tests, plotting
├── visualization.py        # Raw feature distribution plots by group
├── utils.py                # Helper functions (filtering, one-hot encoding, trimmed mean)
├── constants.py            # Labels, column lists, feature drop lists, day-of-interest
├── data_config.py          # Dataset name → round/path mapping
├── requirements.txt        # Python dependencies
├── data/raw/               # Raw MEA CSV files (git-ignored)
├── results/                # Saved .pkl results and analysis PDFs (git-ignored)
├── convergence/            # Convergence plots (git-ignored)
└── visualization/          # Feature distribution PDFs (git-ignored)
```

## Setup

### Prerequisites

- Python 3.9+

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data

Place raw MEA CSV files under `data/raw/` following the paths defined in `data_config.py`.

## Usage

```bash
# Train and analyze (80 repeated 5-fold CV runs), save results and plots
python main.py --dataset dtx --repeats 80 --epochs 60 --save results/dtx.pkl

# Load saved results and re-analyze without re-training
python main.py --dataset dtx --load results/dtx.pkl

# Plot convergence of rescue scores and SHAP values vs. number of repeats
python main.py --load results/dtx.pkl --convergence

# Visualize raw feature distributions for selected features
python main.py --dataset dtx --visualize_features

# Same, with a side-by-side panel showing zeros removed
python main.py --dataset dtx --visualize_features --show_zeros_removed
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `dtx` | Dataset key defined in `data_config.py` |
| `--load` | `None` | Path to a `.pkl` file to load previous results from |
| `--save` | `None` | Path to save results after training (e.g. `results/run.pkl`) |
| `--epochs` | `60` | Number of training epochs per fold |
| `--repeats` | `0` | Number of repeated 5-fold CV runs (0 skips training) |
| `--visualize_features` | `False` | Plot raw feature distributions by group (requires `--dataset`) |
| `--convergence` | `False` | Plot rescue-score and SHAP convergence vs. repeats (requires `--load`) |
| `--show_zeros_removed` | `False` | Add a side-by-side subplot with zeros removed to feature distribution plots |

### Output Directories

When `--save` is provided, plots are written automatically:

| Directory | Contents |
|---|---|
| `results/<name>/` | `MeanRescueScore.pdf`, `MeanSHAPShift.pdf` |
| `convergence/<name>/` | `rescue_score_convergence.pdf`, `shap_convergence.pdf` |
| `visualization/<dataset>/` | One PDF per feature (created by `--visualize_features`) |

## Reproducing Results

> **Note:** The raw MEA data used in this study is not publicly available. To reproduce the analysis you must supply your own MEA recordings in the format described in [Raw Data Format](#raw-data-format).

### Step 1 — Prepare your data

1. Export per-well MEA recordings as CSV files in the long format shown above.
2. Place them under `data/raw/`.
3. Register the files in `data_config.py` (see [Adding a New Dataset](#adding-a-new-dataset)), or reuse an existing key if your files match the expected paths.

### Step 2 — Train and save results

Each run performs repeated 5-fold cross-validation. Training 80 repeats (~400 models) can take several hours per dataset.

```bash
python main.py --dataset <your_dataset> --repeats 80 --epochs 60 --save results/<your_dataset>.pkl
```

This automatically produces:
- `results/<your_dataset>.pkl` — serialized fold-level SHAP results
- `results/<your_dataset>/MeanRescueScore.pdf` — rescue score bar chart
- `results/<your_dataset>/MeanSHAPShift.pdf` — SHAP shift strip plot
- `convergence/<your_dataset>/` — convergence plots

### Step 3 — Re-analyze or inspect saved results

```bash
# Re-run analysis from a saved pickle (no training)
python main.py --dataset <your_dataset> --load results/<your_dataset>.pkl

# Generate only convergence plots from a saved pickle
python main.py --load results/<your_dataset>.pkl --convergence
```

### Step 4 — Visualize raw feature distributions

```bash
python main.py --dataset <your_dataset> --visualize_features
python main.py --dataset <your_dataset> --visualize_features --show_zeros_removed
```

PDFs are saved to `visualization/<your_dataset>/`.

## Statistical Tests

For each neural feature, the pipeline reports:

- **Wilcoxon signed-rank test** on Rescue Scores (testing median ≠ 0)
- **Kruskal-Wallis H-test** across WT, PTEN, and Treated SHAP distributions
- **Dunn's post-hoc test** for pairwise comparisons (WT–PTEN, PTEN–Treated)

Only features passing all tests (p < 0.05) with positive Rescue Scores are included in the final visualization.

## Dependencies

- numpy, pandas, scipy
- scikit-learn
- tensorflow (Keras)
- shap
- neuroHarmonize, nibabel, statsmodels
- matplotlib, seaborn
- scikit-posthocs
