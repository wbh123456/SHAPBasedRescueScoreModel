data_config = {
    "dtx": [
        {"round": 4, "path": "data/raw/MEA_r4_dataset_0804_per_well.csv"},
        {"round": 3, "path": "data/raw/MEA_r3_dataset_0804_per_well.csv"},
        {"round": 2, "path": "data/raw/MEA_r2_dataset_0804_per_well.csv"},
    ],
    "genetic_ko": [
        {"round": 2, "path": "data/raw/R2_GENO_KD_ISIthreshold.csv"},
        {"round": 3, "path": "data/raw/R3_GENO_KD_ISIthreshold.csv"},
    ],
    "genetic_ko_3_rounds": [
        {
            "round": 1,
            "path": "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected.csv",
        },
        {"round": 2, "path": "data/raw/R2_GENO_KD_ISIthreshold.csv"},
        {"round": 3, "path": "data/raw/R3_GENO_KD_ISIthreshold.csv"},
    ],
    "genetic_ko_env_2_rounds": [
        {"round": 2, "path": "data/raw/MEA_r2_dataset_0821_per_well_genoko.csv"},
        {"round": 3, "path": "data/raw/MEA_r3_dataset_0821_per_well_genoko.csv"},
    ],
    "genetic_ko_env_3_rounds": [
        {
            "round": 1,
            "path": "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected.csv",
        },
        {"round": 2, "path": "data/raw/MEA_r2_dataset_0821_per_well_genoko.csv"},
        {"round": 3, "path": "data/raw/MEA_r3_dataset_0821_per_well_genoko.csv"},
    ],
    # Final data config used for genetic ko analysis, which matches the data config used for the paper
    # Note that the data name used in round 3 is  incorrectly labeled as round 2. It is actually round 3.
    "genetic_ko_new": [
        {
            "round": 1,
            "path": "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected.csv",
        },
        {"round": 2, "path": "data/raw/R2_GENO_KD_ISIthreshold.csv"},
        {"round": 3, "path": "data/raw/MEA_r2_dataset_0821_per_well_genoko.csv"},
    ],
    "genetic_ko_new_cleaned": [
        {
            "round": 1,
            "path": "data/raw/MEA_r1_dataset_0915_per_well_genoko_corrected_cleaned.csv",
        },
        {
            "round": 2,
            "path": "data/raw/R2_GENO_KD_ISIthreshold_cleaned.csv",
        },
        {
            "round": 3,
            "path": "data/raw/MEA_r2_dataset_0821_per_well_genoko_cleaned.csv",
        },
    ],
}
