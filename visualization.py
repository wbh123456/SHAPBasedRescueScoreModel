from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import WTFalse, PTENFalse, PTENTrue


GROUP_PALETTE = {"WT": "#4C72B0", "PTEN": "#DD8452", "Treated": "#55A868"}
GROUP_ORDER = ["WT", "PTEN", "Treated"]
LABEL_TO_GROUP = {WTFalse: "WT", PTENFalse: "PTEN", PTENTrue: "Treated"}


def _extract_raw_long(features, raw_X, raw_y, label_to_group):
    rows = []
    for label, group in label_to_group.items():
        mask = raw_y == label
        subset = raw_X.loc[mask]
        for feat in features:
            if feat not in subset.columns:
                continue
            for val in subset[feat]:
                rows.append({"feature": feat, "group": group, "value": val})
    return pd.DataFrame(rows)


def visualize_feature_distributions(features, raw_X, raw_y):
    raw_long = _extract_raw_long(features, raw_X, raw_y, LABEL_TO_GROUP)

    for feature in features:
        feat_df = raw_long[raw_long["feature"] == feature]
        if feat_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        for grp in GROUP_ORDER:
            grp_data = feat_df[feat_df["group"] == grp]["value"]
            grp_data = grp_data.dropna()
            if len(grp_data) < 2:
                continue
            sns.kdeplot(
                grp_data,
                ax=ax,
                fill=True,
                alpha=0.25,
                color=GROUP_PALETTE[grp],
                label=grp,
                linewidth=1.5,
            )
            grp_mean = grp_data.mean()
            ax.axvline(
                grp_mean,
                color=GROUP_PALETTE[grp],
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
            )

        stat_lines = [
            f"{'':>8} {'N':>5} {'Mean':>9} {'Med':>9} {'Std':>9} {'Skew':>7}"
        ]
        for grp in GROUP_ORDER:
            grp_data = feat_df[feat_df["group"] == grp]["value"].dropna()
            n = len(grp_data)
            if n == 0:
                continue
            m = grp_data.mean()
            med = grp_data.median()
            sd = grp_data.std()
            sk = stats.skew(grp_data) if n >= 3 else np.nan
            stat_lines.append(
                f"{grp:>8} {n:5d} {m:9.3f} {med:9.3f}" f" {sd:9.3f} {sk:7.2f}"
            )

        stat_text = "\n".join(stat_lines)
        ax.text(
            0.98,
            0.97,
            stat_text,
            transform=ax.transAxes,
            fontsize=8,
            fontfamily="monospace",
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                alpha=0.85,
                edgecolor="grey",
            ),
        )

        ax.set_title(feature, fontsize=13, fontweight="bold")
        ax.set_xlabel("Raw Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
