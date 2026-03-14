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

# ── Plot configuration ──────────────────────────────────────────
VIS_SUPTITLE_FONTSIZE = 15
VIS_TITLE_FONTSIZE = 18
VIS_AXIS_LABEL_FONTSIZE = 15
VIS_TICK_FONTSIZE = 13
VIS_LEGEND_FONTSIZE = 14
VIS_STAT_TEXT_FONTSIZE = 11
VIS_BRACKET_FONTSIZE = 12


def _p_to_stars(p):
    if p <= 0.0001:
        return "****"
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "NS"


def _draw_significance_bracket(ax, x1, x2, y_frac, text, color="black",
                               narrow_thresh=100):
    """Draw a bracket anchored on two x-positions (data coords) at y_frac (axes coords).

    If the bracket spans more than *narrow_thresh* of the axis width, the
    text is centered on the bracket; otherwise it is placed to the right.
    """
    left, right = min(x1, x2), max(x1, x2)
    xlim = ax.get_xlim()
    span_frac = (right - left) / (xlim[1] - xlim[0])

    trans = ax.get_xaxis_transform()
    bracket_drop = 0.02
    ax.plot(
        [left, left, right, right],
        [y_frac - bracket_drop, y_frac, y_frac, y_frac - bracket_drop],
        transform=trans,
        color=color,
        linewidth=1.2,
    )

    if span_frac >= narrow_thresh:
        tx, ha, label = (left + right) / 2, "center", text
    else:
        tx, ha, label = right, "left", f" {text}"

    ax.text(
        tx, y_frac + 0.01, label,
        transform=trans,
        ha=ha,
        va="bottom",
        fontsize=VIS_BRACKET_FONTSIZE,
        fontweight="bold",
        color=color,
    )


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


def _draw_kde_panel(ax, feat_df, title, drop_zeros=False, dunn_stats=None):
    """Draw a KDE distribution panel onto ax. Optionally removes zero values."""
    stat_lines = [
        f"{'':>8} {'Mean':>9} {'Med':>9} {'Std':>9} {'Skew':>7}"
    ]

    group_means = {}

    for grp in GROUP_ORDER:
        grp_data = feat_df[feat_df["group"] == grp]["value"].dropna()

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
        group_means[grp] = m
        med = grp_data.median()
        sd = grp_data.std()
        sk = stats.skew(grp_data) if n >= 3 else np.nan
        stat_lines.append(
            f"{grp:>8} {m:9.3f} {med:9.3f} {sd:9.3f} {sk:7.2f}"
        )

    stat_text = "\n".join(stat_lines)
    ax.text(
        0.98,
        0.97,
        stat_text,
        transform=ax.transAxes,
        fontsize=VIS_STAT_TEXT_FONTSIZE,
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

    if dunn_stats is not None:
        if "WT" in group_means and "PTEN" in group_means:
            _draw_significance_bracket(
                ax, group_means["WT"], group_means["PTEN"],
                0.6, _p_to_stars(dunn_stats["WT-PT"]),
                color=GROUP_PALETTE["WT"],
            )
        if "PTEN" in group_means and "Treated" in group_means:
            _draw_significance_bracket(
                ax, group_means["PTEN"], group_means["Treated"],
                0.4, _p_to_stars(dunn_stats["PT-PTx"]),
                color=GROUP_PALETTE["Treated"],
            )

    ax.set_title(title, fontsize=VIS_TITLE_FONTSIZE, fontweight="bold")
    ax.set_xlabel("Raw Value", fontsize=VIS_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Density", fontsize=VIS_AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=VIS_TICK_FONTSIZE)
    ax.legend(loc="upper left", fontsize=VIS_LEGEND_FONTSIZE)
    ax.grid(axis="both", linestyle="--", alpha=0.3)


def visualize_feature_distributions(
    features, raw_X, raw_y, output_dir=None, show_zeros_removed=False,
    dunn_stats=None,
):
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    raw_long = _extract_raw_long(features, raw_X, raw_y, LABEL_TO_GROUP)

    for feature in features:
        feat_df = raw_long[raw_long["feature"] == feature]
        if feat_df.empty:
            continue

        feat_dunn = dunn_stats.get(feature) if dunn_stats else None

        if show_zeros_removed:
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 5))
            fig.suptitle(feature, fontsize=VIS_SUPTITLE_FONTSIZE, fontweight="bold")
            _draw_kde_panel(ax_left, feat_df, title="All Data", dunn_stats=feat_dunn)
            _draw_kde_panel(ax_right, feat_df, title="Zeros Removed", drop_zeros=True, dunn_stats=feat_dunn)
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
            _draw_kde_panel(ax, feat_df, title=feature, dunn_stats=feat_dunn)

        plt.tight_layout()

        if output_dir is not None:
            safe_name = re.sub(r'[^\w\-]', '_', feature)
            fig.savefig(os.path.join(output_dir, f"{safe_name}.pdf"))
            plt.close(fig)
        else:
            plt.show()
