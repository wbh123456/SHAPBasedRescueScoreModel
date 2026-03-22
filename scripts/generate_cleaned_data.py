"""
Generate cleaned versions of the 3 genetic_ko_new raw data files by
applying the NA removals identified in scripts/removed_data_report.csv.

For each raw file, corrupted Network Burst Duration values are set to
NaN. After generation, a 6-point validation suite confirms that the
cleaned files differ from the originals in exactly the expected cells
and nowhere else.

Usage:
    python scripts/generate_cleaned_data.py
"""

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

REPORT_CSV = "scripts/removed_data_report.csv"
FEATURE = "Network Burst Duration - Avg (sec)"
START_DAY = 9
RTOL = 1e-4

GENETIC_KO_NEW = [
    {
        "round": 1,
        "path": "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected.csv",
    },
    {"round": 2, "path": "data/raw/R2_GENO_KD_ISIthreshold.csv"},
    {
        "round": 3,
        "path": "data/raw/MEA_r2_dataset_0821_per_well_genoko.csv",
    },
]


def cleaned_path(original_path):
    base, ext = os.path.splitext(original_path)
    return f"{base}_cleaned{ext}"


def day_to_date(day_num, min_date):
    """Convert a day number (e.g. 22) back to a datetime date."""
    return min_date + timedelta(days=day_num - START_DAY)


def generate_cleaned_files():
    """Apply NA removals from the report CSV.

    Returns dict {basename: (orig_path, cleaned_path, expected_count)}.
    """
    report = pd.read_csv(REPORT_CSV)
    report["day_num"] = report["day"].str.replace("d", "").astype(int)

    file_map = {}

    for entry in GENETIC_KO_NEW:
        orig_path = entry["path"]
        fname = os.path.basename(orig_path)
        out_path = cleaned_path(orig_path)
        removals = report[report["raw_file"] == fname]
        expected_count = len(removals)

        df = pd.read_csv(orig_path)
        df["date"] = pd.to_numeric(df["date"])
        min_date = pd.to_datetime(str(df["date"].min()), format="%Y%m%d")

        applied = 0
        for _, rem in removals.iterrows():
            target_date = day_to_date(rem["day_num"], min_date)
            target_date_int = int(target_date.strftime("%Y%m%d"))

            treat_val = rem["treatment"]
            mask = (
                (df["parameter"] == FEATURE)
                & (df["well"] == rem["well"])
                & (df["geno"] == rem["geno"])
                & (df["treatment"] == treat_val)
                & (df["date"] == target_date_int)
            )
            matched_rows = df.index[mask]

            if len(matched_rows) != 1:
                print(f"  WARNING: expected 1 row for {fname} "
                      f"well={rem['well']} geno={rem['geno']} "
                      f"treat={treat_val} date={target_date_int}, "
                      f"got {len(matched_rows)}")
                continue

            idx = matched_rows[0]
            orig_val = df.at[idx, "value"]
            if not np.isclose(orig_val, rem["original_value"], rtol=RTOL):
                print(f"  WARNING: value mismatch at {fname} "
                      f"well={rem['well']} d{rem['day_num']}: "
                      f"expected {rem['original_value']}, got {orig_val}")

            df.at[idx, "value"] = np.nan
            applied += 1

        df.to_csv(out_path, index=False)
        print(f"  {fname}: {applied}/{expected_count} "
              f"removals applied -> {out_path}")
        file_map[fname] = (orig_path, out_path, expected_count)

    return file_map


def validate(file_map):
    """Run 6-point validation on each file pair. Returns True if all pass."""
    all_passed = True

    for fname, (orig_path, clean_path, expected_count) in (
        file_map.items()
    ):
        print(f"\n  Validating {fname} "
              f"({expected_count} expected removals)")
        orig = pd.read_csv(orig_path)
        clean = pd.read_csv(clean_path)
        passed = True

        # 1. Shape check
        if orig.shape == clean.shape:
            print(f"    [PASS] Shape: {orig.shape}")
        else:
            print(f"    [FAIL] Shape: orig={orig.shape} "
                  f"vs clean={clean.shape}")
            passed = False

        # 2. Non-NBD rows identical
        non_nbd_orig = orig[orig["parameter"] != FEATURE]
        non_nbd_orig = non_nbd_orig.reset_index(drop=True)
        non_nbd_clean = clean[clean["parameter"] != FEATURE]
        non_nbd_clean = non_nbd_clean.reset_index(drop=True)
        if non_nbd_orig.equals(non_nbd_clean):
            n = len(non_nbd_orig)
            print(f"    [PASS] Non-NBD rows: all {n} identical")
        else:
            diffs = (non_nbd_orig != non_nbd_clean).sum().sum()
            print(f"    [FAIL] Non-NBD rows: "
                  f"{diffs} cell differences found")
            passed = False

        # Identify which NBD rows were removed
        nbd_orig = orig[orig["parameter"] == FEATURE].copy()
        nbd_clean = clean[clean["parameter"] == FEATURE].copy()

        orig_nan_mask = nbd_orig["value"].isna()
        clean_nan_mask = nbd_clean["value"].isna()
        new_nan_mask = ~orig_nan_mask & clean_nan_mask

        # 3. NBD unchanged check
        unchanged_mask = ~new_nan_mask
        nbd_unch_orig = nbd_orig.loc[unchanged_mask]
        nbd_unch_orig = nbd_unch_orig.reset_index(drop=True)
        nbd_unch_clean = nbd_clean.loc[unchanged_mask]
        nbd_unch_clean = nbd_unch_clean.reset_index(drop=True)
        if nbd_unch_orig.equals(nbd_unch_clean):
            n = len(nbd_unch_orig)
            print(f"    [PASS] NBD unchanged: all {n} identical")
        else:
            d = (nbd_unch_orig != nbd_unch_clean).sum().sum()
            print(f"    [FAIL] NBD unchanged: "
                  f"{d} unexpected differences")
            passed = False

        # 4. NBD removed check
        new_nan_count = new_nan_mask.sum()
        if new_nan_count == expected_count:
            print(f"    [PASS] NBD removed: "
                  f"{new_nan_count} cells set to NaN")
        else:
            print(f"    [FAIL] NBD removed: "
                  f"{new_nan_count} NaNs vs "
                  f"{expected_count} expected")
            passed = False

        # 5. Count check (total NaN diff)
        orig_total_nan = orig["value"].isna().sum()
        clean_total_nan = clean["value"].isna().sum()
        nan_diff = clean_total_nan - orig_total_nan
        if nan_diff == expected_count:
            print(f"    [PASS] NaN count: "
                  f"orig={orig_total_nan}, "
                  f"clean={clean_total_nan}, "
                  f"diff={nan_diff}")
        else:
            print(f"    [FAIL] NaN count: "
                  f"diff={nan_diff}, "
                  f"expected {expected_count}")
            passed = False

        # 6. No extra NaN check (only NBD rows should have new NaNs)
        for col in orig.columns:
            if col == "value":
                continue
            orig_nans = orig[col].isna().sum()
            clean_nans = clean[col].isna().sum()
            if orig_nans != clean_nans:
                print(f"    [FAIL] Extra NaN in column '{col}': "
                      f"orig={orig_nans}, clean={clean_nans}")
                passed = False
                break
        else:
            print("    [PASS] No extra NaN: "
                  "non-value columns untouched")

        if passed:
            print("    --- ALL CHECKS PASSED ---")
        else:
            print("    --- SOME CHECKS FAILED ---")
            all_passed = False

    return all_passed


def main():
    print("=" * 60)
    print("GENERATING CLEANED DATA FILES")
    print("=" * 60)

    file_map = generate_cleaned_files()

    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    all_passed = validate(file_map)

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL FILES VALIDATED SUCCESSFULLY")
    else:
        print("VALIDATION FAILED — see details above")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
