"""
Analyze which raw data file(s) and source_data folder(s) each value
in 'network burst duration all rounds.csv' originates from.

All mappings are determined by value-matching Network Burst Duration
values -- no assumptions from naming conventions.

Usage:
    python scripts/analyze_data_source.py

Results are saved to scripts/:
  - per_value_matches.csv : every non-zero value with matched raw
                            file(s) and source_data folder(s)
  - summary.csv           : per-group, per-raw-file match counts
  - unmatched.csv         : values with no match in any raw file
  - raw_to_source.csv     : raw file -> source_data folder mapping
  - report.txt            : full human-readable report
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict

# --------------- Configuration ---------------

MYSTERY_CSV = "data/raw/network burst duration all rounds.csv"
SOURCE_DATA_DIR = "data/source_data"
OUTPUT_DIR = "scripts"

RAW_FILES = [
    "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected.csv",
    "data/raw/MEA_r2_dataset_0821_per_well_genoko.csv",
    "data/raw/MEA_r3_dataset_0821_per_well_genoko.csv",
    "data/raw/R2_GENO_KD_ISIthreshold.csv",
    "data/raw/R3_GENO_KD_ISIthreshold.csv",
    "data/raw/MEA_r2_dataset_0804_per_well.csv",
    "data/raw/MEA_r3_dataset_0804_per_well.csv",
    "data/raw/MEA_r4_dataset_0804_per_well.csv",
]

FEATURE = "Network Burst Duration - Avg (sec)"
RTOL = 1e-4
MIN_SOURCE_PCT = 10.0

# --------------- Data loading ---------------


def load_mystery_csv(path):
    """Load the wide-format mystery CSV into long form.

    The first row contains group labels (e.g. 'WT + sc', 'PTEN + sc').
    Subsequent rows are timepoint x group numeric values.
    Returns a DataFrame with columns: timepoint, group, value, col_idx.
    """
    with open(path) as f:
        header_line = f.readline().strip()
    group_labels = header_line.split(",")
    group_labels[0] = "timepoint"

    df = pd.read_csv(path, header=None, skiprows=1)
    df.rename(columns={0: "timepoint"}, inplace=True)

    records = []
    for _, row in df.iterrows():
        tp = row["timepoint"]
        for col_idx in range(1, len(row)):
            val = row[col_idx]
            group = group_labels[col_idx]
            if pd.isna(val) or str(val).strip() == "":
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            if val == 0.0:
                continue
            records.append({
                "timepoint": tp,
                "group": group,
                "value": val,
                "col_idx": col_idx,
            })
    return pd.DataFrame(records)


def load_raw_nbd_values(path):
    """Load non-zero NBD values from a long-format raw CSV."""
    df = pd.read_csv(path)
    mask = df["parameter"] == FEATURE
    values = pd.to_numeric(
        df.loc[mask, "value"], errors="coerce"
    ).dropna()
    return values[values != 0.0].values


def extract_nbd_from_source_csv(path):
    """Extract non-zero NBD values from a wide-format source neuralMetrics CSV.

    Scans line-by-line for the 'Network Burst Duration - Avg' row and
    parses all numeric cells from that row.
    """
    values = []
    with open(path) as f:
        for line in f:
            if not line.strip().startswith("Network Burst Duration - Avg"):
                continue
            for cell in line.split(",")[1:]:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    v = float(cell)
                    if v != 0.0:
                        values.append(v)
                except ValueError:
                    pass
            break
    return np.array(values) if values else np.array([])


# --------------- Indexing & matching ---------------


def build_source_index(source_dir):
    """Walk source_data/ and index all NBD values by folder.

    Returns dict: {relative_folder_path: np.array of NBD values}.
    """
    index = {}
    for root, _dirs, files in os.walk(source_dir):
        for f in files:
            if not f.endswith("_neuralMetrics.csv"):
                continue
            path = os.path.join(root, f)
            vals = extract_nbd_from_source_csv(path)
            if len(vals) == 0:
                continue
            rel = os.path.relpath(root, source_dir)
            if rel not in index:
                index[rel] = []
            index[rel].append(vals)
    for k in index:
        index[k] = np.concatenate(index[k])
    return index


def match_raw_to_source(raw_vals, source_index):
    """Value-match a raw file's NBD values against the source index.

    Returns list of (folder, hit_count, hit_pct) sorted by hit count desc.
    """
    n = len(raw_vals)
    if n == 0:
        return [], 0
    results = []
    for folder, src_vals in source_index.items():
        hits = sum(
            1 for v in raw_vals
            if np.any(np.isclose(src_vals, v, rtol=RTOL))
        )
        if hits > 0:
            results.append((folder, hits, round(hits / n * 100, 1)))
    results.sort(key=lambda x: -x[1])

    total_any = sum(
        1 for v in raw_vals if any(
            np.any(np.isclose(sv, v, rtol=RTOL))
            for sv in source_index.values()
        )
    )
    coverage = round(total_any / n * 100, 1)
    return results, coverage


# --------------- Analysis steps ---------------


def analyze_raw_to_source(source_index, log):
    """Part 1: Map each raw file to its source_data folder(s).

    Returns:
        raw_to_source: dict {raw_basename: [(folder, hits, pct), ...]}
        raw_source_rows: list of dicts for CSV output
    """
    raw_to_source = {}
    raw_source_rows = []

    log("=" * 70)
    log("RAW FILE -> SOURCE DATA MAPPING")
    log(f"(by value-matching NBD values, >={MIN_SOURCE_PCT}% threshold)")
    log("=" * 70)

    for raw_path in RAW_FILES:
        fname = os.path.basename(raw_path)
        raw_vals = load_raw_nbd_values(raw_path)
        n = len(raw_vals)
        matches, coverage = match_raw_to_source(raw_vals, source_index)

        log(f"\n  {fname} ({n} values, {coverage}% total coverage)")

        primary = [(f, h, p) for f, h, p in matches if p >= MIN_SOURCE_PCT]
        raw_to_source[fname] = primary

        if primary:
            for folder, hits, pct in primary:
                log(f"    {hits}/{n} ({pct}%) -> {folder}")
                raw_source_rows.append({
                    "raw_file": fname,
                    "source_folder": folder,
                    "matched_count": hits,
                    "total_values": n,
                    "match_pct": pct,
                    "coverage_pct": coverage,
                })
        else:
            max_pct = matches[0][2] if matches else 0
            log(f"    No source folder found (max {max_pct}% coincidental)")
            raw_source_rows.append({
                "raw_file": fname,
                "source_folder": "NOT FOUND",
                "matched_count": 0,
                "total_values": n,
                "match_pct": 0.0,
                "coverage_pct": coverage,
            })

    return raw_to_source, raw_source_rows


def analyze_mystery_to_raw(raw_to_source, log):
    """Part 2: Match each mystery CSV value to raw files and source folders.

    Returns:
        per_value_records, match_counts, group_file_counts, unmatched_records
    """
    mystery = load_mystery_csv(MYSTERY_CSV)
    log("\n" + "=" * 70)
    log("MYSTERY CSV -> RAW FILE MATCHING")
    log("=" * 70)
    log(f"\nExtracted {len(mystery)} non-zero values")
    log(f"Groups: {mystery['group'].unique().tolist()}")
    log(f"Timepoints: {mystery['timepoint'].unique().tolist()}")

    raw_values = {}
    for raw_path in RAW_FILES:
        fname = os.path.basename(raw_path)
        raw_values[fname] = load_raw_nbd_values(raw_path)
        log(f"  {fname}: {len(raw_values[fname])} non-zero NBD")

    per_value_records = []
    match_counts = defaultdict(int)
    group_file_counts = defaultdict(lambda: defaultdict(int))
    unmatched_records = []

    for _, row in mystery.iterrows():
        matched_raw = [
            fname for fname, vals in raw_values.items()
            if np.any(np.isclose(vals, row["value"], rtol=RTOL))
        ]

        src_folders = set()
        for fname in matched_raw:
            for folder, _h, _p in raw_to_source.get(fname, []):
                src_folders.add(folder)

        per_value_records.append({
            "timepoint": row["timepoint"],
            "group": row["group"],
            "col_idx": row["col_idx"],
            "value": row["value"],
            "matched_raw_files": "; ".join(matched_raw) if matched_raw else "NONE",
            "source_data_folders": "; ".join(sorted(src_folders)) if src_folders else "UNKNOWN",
            "num_raw_matches": len(matched_raw),
        })

        if matched_raw:
            for m in matched_raw:
                match_counts[m] += 1
                group_file_counts[row["group"]][m] += 1
        else:
            unmatched_records.append({
                "timepoint": row["timepoint"],
                "group": row["group"],
                "col_idx": row["col_idx"],
                "value": row["value"],
            })

    return mystery, per_value_records, match_counts, group_file_counts, unmatched_records


def save_results(mystery, per_value_records, match_counts,
                 group_file_counts, unmatched_records,
                 raw_to_source, raw_source_rows, log):
    """Save all CSVs and build the report body."""

    # raw_to_source.csv
    rs_df = pd.DataFrame(raw_source_rows)
    rs_path = os.path.join(OUTPUT_DIR, "raw_to_source.csv")
    rs_df.to_csv(rs_path, index=False)
    log(f"\nSaved raw->source map   -> {rs_path}")

    # per_value_matches.csv
    pv_df = pd.DataFrame(per_value_records)
    pv_path = os.path.join(OUTPUT_DIR, "per_value_matches.csv")
    pv_df.to_csv(pv_path, index=False)
    log(f"Saved per-value matches -> {pv_path}")

    # unmatched.csv
    um_df = pd.DataFrame(unmatched_records)
    um_path = os.path.join(OUTPUT_DIR, "unmatched.csv")
    um_df.to_csv(um_path, index=False)
    log(f"Saved unmatched values  -> {um_path}")

    # summary.csv
    raw_basenames = [os.path.basename(p) for p in RAW_FILES]
    summary_rows = []
    for group in mystery["group"].unique():
        total = len(mystery[mystery["group"] == group])
        for fname in raw_basenames:
            cnt = group_file_counts[group].get(fname, 0)
            src_list = [f for f, _h, _p in raw_to_source.get(fname, [])]
            summary_rows.append({
                "group": group,
                "raw_file": fname,
                "source_data_folders": "; ".join(src_list) if src_list else "NOT FOUND",
                "matched_count": cnt,
                "total_nonzero": total,
                "match_pct": round(cnt / total * 100, 1) if total else 0,
            })
    sm_df = pd.DataFrame(summary_rows)
    sm_path = os.path.join(OUTPUT_DIR, "summary.csv")
    sm_df.to_csv(sm_path, index=False)
    log(f"Saved summary           -> {sm_path}")

    return raw_basenames, um_df


def print_report(mystery, match_counts, group_file_counts,
                 unmatched_records, um_df, raw_to_source,
                 raw_basenames, log):
    """Print the human-readable report section."""

    log("\n--- Overall match counts per raw file ---")
    for fname in raw_basenames:
        cnt = match_counts[fname]
        src_list = [
            f"{f} ({p}%)" for f, _h, p in raw_to_source.get(fname, [])
        ]
        log(f"\n  {fname}")
        log(f"    matched: {cnt} mystery CSV values")
        if src_list:
            log(f"    source:  {'; '.join(src_list)}")
        else:
            log("    source:  NOT FOUND in source_data")

    n_um = len(unmatched_records)
    log(f"\n--- Unmatched values: {n_um} ---")
    if unmatched_records:
        for group in um_df["group"].unique():
            subset = um_df[um_df["group"] == group]
            log(f"  {group}: {len(subset)} unmatched")
            for _, r in subset.head(5).iterrows():
                log(f"    {r['timepoint']} col={r['col_idx']} val={r['value']}")
            if len(subset) > 5:
                log(f"    ... and {len(subset) - 5} more")

    log("\n--- Per-group breakdown ---")
    for group in mystery["group"].unique():
        total = len(mystery[mystery["group"] == group])
        log(f"\n  [{group}] ({total} non-zero values)")
        for fname in raw_basenames:
            cnt = group_file_counts[group].get(fname, 0)
            if cnt > 0:
                pct = cnt / total * 100
                log(f"    {fname}: {cnt} ({pct:.1f}%)")


# --------------- Main ---------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("DATASET SOURCE ANALYSIS")
    log(f"Mystery file: {MYSTERY_CSV}")
    log("=" * 70)

    log("\nBuilding source_data NBD value index...")
    source_index = build_source_index(SOURCE_DATA_DIR)
    log(f"Indexed {len(source_index)} source folders")

    raw_to_source, raw_source_rows = analyze_raw_to_source(source_index, log)

    (mystery, per_value_records, match_counts,
     group_file_counts, unmatched_records) = analyze_mystery_to_raw(
        raw_to_source, log
    )

    raw_basenames, um_df = save_results(
        mystery, per_value_records, match_counts,
        group_file_counts, unmatched_records,
        raw_to_source, raw_source_rows, log,
    )

    print_report(
        mystery, match_counts, group_file_counts,
        unmatched_records, um_df, raw_to_source,
        raw_basenames, log,
    )

    rpt_path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(rpt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved full report       -> {rpt_path}")


if __name__ == "__main__":
    main()
