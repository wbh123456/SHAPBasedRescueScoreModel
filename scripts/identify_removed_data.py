"""
Identify which raw data file each removed (NA) value in
'network burst duration all rounds.csv' originates from.

The modified CSV is a wide-format extraction of Network Burst Duration
values from the genetic_ko_new dataset (3 rounds). Some values were
manually turned to NA to remove suspected corrupted data. This script:

1. Loads the 3 raw files that compose genetic_ko_new
2. Computes day numbers identically to DataLoader
3. Maps each modified-CSV column to a (raw_file, well) via value matching
4. Identifies every NA cell that had a real value in the raw data
5. Reports results with numerical evidence

Usage:
    python scripts/identify_removed_data.py
"""

import os

import numpy as np
import pandas as pd

MODIFIED_CSV = "data/raw/network burst duration all rounds.csv"
OUTPUT_CSV = "scripts/removed_data_report.csv"

START_DAY = 9
FEATURE = "Network Burst Duration - Avg (sec)"
RTOL = 1e-4

GENETIC_KO_NEW = [
    {"round": 1, "path": "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected.csv"},
    {"round": 2, "path": "data/raw/R2_GENO_KD_ISIthreshold.csv"},
    {"round": 3, "path": "data/raw/MEA_r2_dataset_0821_per_well_genoko.csv"},
]


def load_raw_nbd():
    """Load NBD time-series from the 3 genetic_ko_new raw files.

    Computes day numbers identically to DataLoader: min_date is taken
    from ALL rows in the file (not just NBD), then day = delta + START_DAY.

    Returns:
        well_series: dict (basename, geno, treat_bool, well) -> {day: value}
    """
    well_series = {}

    for entry in GENETIC_KO_NEW:
        path = entry["path"]
        fname = os.path.basename(path)
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        min_date = df["date"].min()

        nbd = df[df["parameter"] == FEATURE].copy()
        nbd["day"] = (nbd["date"] - min_date).dt.days + START_DAY

        treat_col = nbd["treatment"]
        if treat_col.dtype == object:
            nbd["treat_bool"] = treat_col.str.lower() == "true"
        else:
            nbd["treat_bool"] = treat_col.astype(bool)

        for _, row in nbd.iterrows():
            val = float(row["value"]) if pd.notna(row["value"]) else np.nan
            key = (fname, row["geno"], row["treat_bool"], row["well"])
            well_series.setdefault(key, {})[row["day"]] = val

    return well_series


def parse_modified_csv():
    """Parse the wide-format modified CSV.

    Returns:
        group_labels: list[str] of length n_cols (header labels for data columns)
        days:         list[int] (e.g. [9, 10, ..., 24])
        data:         list[list[float|None]], shape (n_days, n_cols)
    """
    with open(MODIFIED_CSV, encoding="utf-8-sig") as f:
        header_line = f.readline().strip()
    group_labels = header_line.split(",")[1:]

    df = pd.read_csv(MODIFIED_CSV, header=None, skiprows=1, encoding="utf-8-sig")
    days = [int(tp.strip().replace("d", "")) for tp in df[0]]

    data = []
    for _, row in df.iterrows():
        day_vals = []
        for col_idx in range(1, len(row)):
            v = row[col_idx]
            if pd.isna(v) or str(v).strip() == "":
                day_vals.append(None)
            else:
                day_vals.append(float(v))
        data.append(day_vals)

    return group_labels, days, data


def find_structural_empties(data, n_cols):
    """Columns that are NA on every single day (round 1 has fewer wells)."""
    return {
        col for col in range(n_cols)
        if all(row[col] is None for row in data)
    }


def match_column(nonzero_vals, well_series, max_misses=0):
    """Find the well whose time-series best matches the given values.

    nonzero_vals: {day: float} — values to match (typically non-zero only)
    max_misses:   tolerate up to this many value disagreements (to handle
                  minor edits in the modified CSV beyond pure NA removal)

    Returns ((fname, geno, treat, well), n_hits, n_misses) or None.
    """
    best_key = None
    best_hits = 0
    best_misses = 0

    for key, series in well_series.items():
        hits = 0
        misses = 0
        for day, val in nonzero_vals.items():
            raw_val = series.get(day)
            if raw_val is None or np.isnan(raw_val):
                continue
            if np.isclose(val, raw_val, rtol=RTOL):
                hits += 1
            else:
                misses += 1
        if misses <= max_misses and hits > best_hits:
            best_key = key
            best_hits = hits
            best_misses = misses

    return (best_key, best_hits, best_misses) if best_key else None


def main():
    well_series = load_raw_nbd()
    print(f"Loaded {len(well_series)} well time-series from "
          f"{len(GENETIC_KO_NEW)} raw files\n")

    group_labels, days, data = parse_modified_csv()
    n_cols = len(group_labels)
    n_days = len(days)

    structural = find_structural_empties(data, n_cols)
    print(f"Modified CSV: {n_cols} data columns, {n_days} days (d{days[0]}-d{days[-1]})")
    print(f"Structural empty columns (always NA): {len(structural)}")

    # ---- Step 3: map each column to (raw_file, well) ----
    col_mapping = {}
    unmatched_cols = []

    for col in range(n_cols):
        if col in structural:
            continue

        nonzero = {
            days[i]: data[i][col]
            for i in range(n_days)
            if data[i][col] is not None and data[i][col] != 0.0
        }

        # Try exact match first, then allow up to 2 misses (handles
        # minor value edits beyond pure NA removal in the modified CSV).
        result = match_column(nonzero, well_series, max_misses=0)
        if result is None:
            result = match_column(nonzero, well_series, max_misses=2)

        if result is None:
            all_vals = {
                days[i]: data[i][col]
                for i in range(n_days)
                if data[i][col] is not None
            }
            result = match_column(all_vals, well_series, max_misses=2)

        if result:
            (fname, geno, treat, well), n_hits, n_misses = result
            col_mapping[col] = {
                "raw_file": fname,
                "geno": geno,
                "treatment": treat,
                "well": well,
                "group": group_labels[col],
                "n_matched": n_hits,
                "n_misses": n_misses,
            }
        else:
            unmatched_cols.append(col)

    mapped_count = len(col_mapping)
    active_cols = n_cols - len(structural)
    print(f"Mapped {mapped_count}/{active_cols} active columns")
    if unmatched_cols:
        print(f"WARNING: {len(unmatched_cols)} unmatched columns: {unmatched_cols}")
    fuzzy = [c for c, info in col_mapping.items() if info["n_misses"] > 0]
    if fuzzy:
        print(f"  ({len(fuzzy)} columns matched with 1-2 value mismatches — "
              f"likely d24 edits in modified CSV)")

    # ---- Step 4: find removed data ----
    removed = []

    for col, info in col_mapping.items():
        key = (info["raw_file"], info["geno"], info["treatment"], info["well"])
        raw_series = well_series.get(key, {})

        for i in range(n_days):
            day = days[i]
            if data[i][col] is None:
                raw_val = raw_series.get(day)
                if raw_val is not None and not np.isnan(raw_val):
                    removed.append({
                        "day": f"d{day}",
                        "group": info["group"],
                        "col_index_1based": col + 1,
                        "raw_file": info["raw_file"],
                        "geno": info["geno"],
                        "treatment": info["treatment"],
                        "well": info["well"],
                        "original_value": raw_val,
                    })

    removed_df = pd.DataFrame(removed)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    removed_df.to_csv(OUTPUT_CSV, index=False)

    # ---- Step 5: report ----
    print(f"\n{'=' * 72}")
    print("REMOVED DATA REPORT")
    print(f"{'=' * 72}")
    print(f"Total removed cells (excluding structural empties): {len(removed)}")

    if not removed:
        print("No removed data found.")
        return

    print(f"\n--- Breakdown by raw file ---")
    for fname in sorted(removed_df["raw_file"].unique()):
        n = len(removed_df[removed_df["raw_file"] == fname])
        print(f"  {fname}: {n} values removed")

    print(f"\n--- Breakdown by day ---")
    for day_label in removed_df["day"].unique():
        n = len(removed_df[removed_df["day"] == day_label])
        print(f"  {day_label}: {n} values removed")

    print(f"\n--- Breakdown by group ---")
    for grp in sorted(removed_df["group"].unique()):
        n = len(removed_df[removed_df["group"] == grp])
        print(f"  {grp}: {n} values removed")

    print(f"\n--- Column-to-file mapping summary ---")
    file_cols = {}
    for col, info in sorted(col_mapping.items()):
        file_cols.setdefault(info["raw_file"], []).append(col)
    for fname in sorted(file_cols):
        print(f"  {fname}: {len(file_cols[fname])} columns mapped")

    print(f"\n--- Evidence: every removed cell with its original raw value ---")
    header = (f"{'Day':<5} {'Group':<22} {'Col':>4} {'Well':<5} "
              f"{'Original Value':>15}  Raw File")
    print(header)
    print("-" * len(header) + "-" * 40)
    for _, r in removed_df.iterrows():
        print(f"{r['day']:<5} {r['group']:<22} {r['col_index_1based']:>4} "
              f"{r['well']:<5} {r['original_value']:>15.6f}  {r['raw_file']}")

    print(f"\nDetailed CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
