from enum import Enum


# Labels
PTENTrue = "PTENTrue"
PTENFalse = "PTENFalse"
WTTrue = "WTTrue"
WTFalse = "WTFalse"

PT_WT_LABELS = [WTFalse, PTENFalse]


KEY_INDEXES = [
    "round",
    "filename",
    "well",
    "geno",
    "treatment",
    "day",
    "bioDup",
    "techDup",
]

START_DAY = 9
DAY_OF_INTEREST = 21

# Data Correlation
"""
Highly correlated feature pairs (corr > 0.95):
[Mean Firing Rate (Hz)]  <-->  [Number of Spikes] : correlation = 1.000
[Mean ISI within Burst - Avg (sec)]  <-->  [Median ISI within Burst - Avg (sec)] : correlation = 0.996
[Mean ISI within Burst - Std (sec)]  <-->  [Median ISI within Burst - Std (sec)] : correlation = 0.990
[Mean ISI within Network Burst - Avg (sec)]  <-->  [Median ISI within Network Burst - Avg (sec)] : correlation = 0.993
[Mean ISI within Network Burst - Std (sec)]  <-->  [Median ISI within Network Burst - Std (sec)] : correlation = 0.990
[Network Burst Frequency]  <-->  [Number of Network Bursts] : correlation = 1.000
[Number of Active Electrodes]  <-->  [Number of Bursting Electrodes] : correlation = 0.955
[Number of Spikes per Network Burst - Avg]  <-->  [Number of Spikes per Network Burst per Channel - Avg] : correlation = 0.991
[Number of Spikes per Network Burst - Std]  <-->  [Number of Spikes per Network Burst per Channel - Std] : correlation = 0.990
"""
to_drop_095 = [
    "Number of Spikes",
    "Median ISI within Burst - Avg (sec)",
    "Median ISI within Burst - Std (sec)",
    "Median ISI within Network Burst - Avg (sec)",
    "Median ISI within Network Burst - Std (sec)",
    "Number of Network Bursts",
    "Number of Active Electrodes",
    "Number of Spikes per Network Burst per Channel - Avg",
    "Number of Spikes per Network Burst per Channel - Std",
]

high_vif_feat = [
    "Burst Percentage - Avg",
    "Number of Spikes per Burst - Avg",
    "Number of Elecs Participating in Burst - Avg",
    "Number of Bursting Electrodes",
    "Mean Firing Rate (Hz)",
    "Area Under Cross-Correlation",
    "Median/Mean ISI within Burst - Avg",
    "Median/Mean ISI within Network Burst - Avg",
    "Full Width at Half Height of Cross-Correlation",

    # "Network Burst Percentage",
    # "Burst Duration - Avg (sec)"
]
