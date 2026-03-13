from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

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


def _draw_kde_panel(ax, feat_df, title, drop_zeros=False):
    """Draw a KDE distribution panel onto ax. Optionally removes zero values."""
    stat_lines = [
        f"{'':>8} {'N':>5} {'Zeros':>6} {'Mean':>9} {'Med':>9} {'Std':>9} {'Skew':>7}"
    ]

    for grp in GROUP_ORDER:
        grp_data = feat_df[feat_df["group"] == grp]["value"].dropna()
        n_total = len(grp_data)
        zeros = int((grp_data == 0).sum())

        if drop_zeros:
            grp_data = grp_data[grp_data != 0]

        n = len(grp_data)
        if n < 2:
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
        ax.axvline(
            grp_data.mean(),
            color=GROUP_PALETTE[grp],
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

        m = grp_data.mean()
        med = grp_data.median()
        sd = grp_data.std()
        sk = stats.skew(grp_data) if n >= 3 else np.nan
        stat_lines.append(
            f"{grp:>8} {n_total:5d} {zeros:6d} {m:9.3f} {med:9.3f}"
            f" {sd:9.3f} {sk:7.2f}"
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

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Raw Value")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    ax.grid(axis="both", linestyle="--", alpha=0.3)


def visualize_feature_distributions(
    features, raw_X, raw_y, output_dir=None, show_zeros_removed=False
):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    raw_long = _extract_raw_long(features, raw_X, raw_y, LABEL_TO_GROUP)

    for feature in features:
        feat_df = raw_long[raw_long["feature"] == feature]
        if feat_df.empty:
            continue

        if show_zeros_removed:
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 5))
            fig.suptitle(feature, fontsize=13, fontweight="bold")
            _draw_kde_panel(ax_left, feat_df, title="All Data")
            _draw_kde_panel(ax_right, feat_df, title="Zeros Removed", drop_zeros=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_kde_panel(ax, feat_df, title=feature)

        plt.tight_layout()

        if output_dir is not None:
            safe_name = re.sub(r'[^\w\-]', '_', feature)
            fig.savefig(os.path.join(output_dir, f"{safe_name}.pdf"))
            plt.close(fig)
        else:
            plt.show()
