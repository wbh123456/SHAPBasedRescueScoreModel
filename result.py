from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy import stats
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import os
import pickle
import io
import pandas as pd

from utils import tmean


PARAMETER_CATEGORY = pd.read_csv(
    io.StringIO(
        """
Weighted Mean Firing Rate (Hz),1
Full Width at Half Height of Normalized Cross-Correlation,2
Number of Spikes per Network Burst - Avg,3
Time to Burst Peak (ms),3
Network Burst Duration - Avg (sec),3
Burst Duration - Avg (sec),4
Number of Bursts,4
Area Under Normalized Cross-Correlation,2
Burst Frequency - Avg (Hz),4
Mean ISI within Network Burst - Avg (sec),3
Start Electrode,3
Network Burst Frequency,3
ISI Coefficient of Variation - Avg,1
Network Burst Percentage,3
Percent Bursts with Start Electrode,3
IBI Coefficient of Variation - Avg,4
Burst Peak (Max Spikes per sec),3
Inter-Burst Interval - Avg (sec),4
Mean ISI within Burst - Avg (sec),4
Network IBI Coefficient of Variation,3
Network Normalized Duration IQR,3
"""
    ),
    header=None,
)
PARAMETER_CATEGORY.columns = ["parameter", "category"]
category_dict = {
    1: "general parameter",
    2: "synchrony parameter",
    3: "network burst parameter",
    4: "single burst parameter",
}

PARAMETER_CATEGORY["category"] = PARAMETER_CATEGORY["category"].map(category_dict)


class Result:
    def __init__(self, results=None):
        self.results = results

    def load(self, results):
        self.results = results

    def merge_result(self, results):
        if self.results and results:
            self.results = self.results + results
        elif results:
            self.results = results

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.results, f)

    def load_from_file(self, file_path):
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
            self.load(loaded_data)

    def analyze_results(self, output_dir=None, normtest=False):
        print("####")
        print("#### Analyzing Result")
        print("Result size =", len(self.results))
        rescue_scores_summary, rescue_full_list, shift_denom = self._get_rescue_scores(
            self.results, normtest=normtest
        )
        self._visualize_results(rescue_scores_summary, rescue_full_list, shift_denom, output_dir)
        return rescue_scores_summary, shift_denom

    def analyze_convergence(self, output_dir=None):
        print("####")
        print("#### Convergence Analysis")
        print("Result size =", len(self.results))

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        # --- Step A: flatten all folds into per-fold records ---
        records = []
        for repeat_idx, folds in enumerate(self.results):
            for fold_idx, fold in enumerate(folds):
                df = fold["mean_shap_df"]
                for param in df.index:
                    records.append(
                        {
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "parameter": param,
                            "mu_WT": df.loc[param, "mu_WT"],
                            "mu_PT": df.loc[param, "mu_PT"],
                            "mu_Tx": df.loc[param, "mu_Tx"],
                        }
                    )
        flat = pd.DataFrame(records)

        # Pre-compute rescue score per row, applying the per-parameter min_delta guard
        flat["denom"] = flat["mu_WT"] - flat["mu_PT"]
        flat["shift"] = flat["mu_Tx"] - flat["mu_PT"]
        flat["rescue"] = np.nan
        for param in flat["parameter"].unique():
            mask = flat["parameter"] == param
            d = flat.loc[mask, "denom"]
            s = flat.loc[mask, "shift"]
            min_delta = np.nanmedian(np.abs(d)) * 0.1
            valid = mask & (flat["denom"].abs() >= min_delta)
            flat.loc[valid, "rescue"] = (flat.loc[valid, "shift"] / flat.loc[valid, "denom"])

        R = len(self.results)
        params = flat["parameter"].unique()

        # --- Step B: cumulative statistics per repeat prefix ---
        rows = []
        for k in range(1, R + 1):
            subset = flat[flat["repeat"] < k]
            for param, g in subset.groupby("parameter"):
                rescue_vals = g["rescue"].dropna().to_numpy()
                n = len(rescue_vals)
                mean_rescue = tmean(rescue_vals) if n > 0 else np.nan
                se_rescue = (np.nanstd(rescue_vals) / np.sqrt(n)) if n > 1 else np.nan
                rows.append(
                    {
                        "k": k,
                        "parameter": param,
                        "mean_rescue": mean_rescue,
                        "se_rescue": se_rescue,
                        "mean_WT": g["mu_WT"].mean(),
                        "se_WT": g["mu_WT"].sem(),
                        "mean_PT": g["mu_PT"].mean(),
                        "se_PT": g["mu_PT"].sem(),
                        "mean_Tx": g["mu_Tx"].mean(),
                        "se_Tx": g["mu_Tx"].sem(),
                    }
                )
        conv = pd.DataFrame(rows)

        # --- Step C: Plot convergence figures ---
        self._plot_rescue_convergence(conv, params, R, output_dir)
        self._plot_shap_convergence(conv, params, R, output_dir)

    def _plot_rescue_convergence(self, conv, params, R, output_dir):
        n_params = len(params)
        ncols = min(3, n_params)
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        xs = np.arange(1, R + 1)
        for ax_idx, param in enumerate(sorted(params)):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            g = conv[conv["parameter"] == param].sort_values("k")
            y = g["mean_rescue"].to_numpy()
            se = g["se_rescue"].to_numpy()
            final_val = y[-1]

            ax.plot(xs, y, color="#2271B5", linewidth=1.5)
            ax.fill_between(xs, y - se, y + se, alpha=0.2, color="#2271B5")
            ax.axhline(final_val, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(param, fontsize=8, fontweight="bold")
            ax.set_xlabel("Repeats", fontsize=7)
            ax.set_ylabel("Mean Rescue Score", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(linestyle="--", alpha=0.3)

        for ax_idx in range(n_params, nrows * ncols):
            axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

        fig.suptitle("Rescue Score Convergence vs. Number of Repeats", fontsize=11, fontweight="bold")
        plt.tight_layout()

        if output_dir is not None:
            fig.savefig(os.path.join(output_dir, "rescue_score_convergence.pdf"))
            plt.close(fig)
        else:
            plt.show()

    def _plot_shap_convergence(self, conv, params, R, output_dir):
        shap_palette = {"WT": "#4C72B0", "PTEN": "#DD8452", "Treated": "#55A868"}
        shap_cols = {"WT": ("mean_WT", "se_WT"), "PTEN": ("mean_PT", "se_PT"), "Treated": ("mean_Tx", "se_Tx")}

        n_params = len(params)
        ncols = min(3, n_params)
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        xs = np.arange(1, R + 1)
        for ax_idx, param in enumerate(sorted(params)):
            ax = axes[ax_idx // ncols][ax_idx % ncols]
            g = conv[conv["parameter"] == param].sort_values("k")

            for grp, (mean_col, se_col) in shap_cols.items():
                y = g[mean_col].to_numpy()
                se = g[se_col].to_numpy()
                color = shap_palette[grp]
                ax.plot(xs, y, color=color, linewidth=1.5, label=grp)
                ax.fill_between(xs, y - se, y + se, alpha=0.2, color=color)

            ax.set_title(param, fontsize=8, fontweight="bold")
            ax.set_xlabel("Repeats", fontsize=7)
            ax.set_ylabel("Mean SHAP Value", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(linestyle="--", alpha=0.3)
            ax.legend(fontsize=6, loc="best")

        for ax_idx in range(n_params, nrows * ncols):
            axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

        fig.suptitle("Mean SHAP Value Convergence vs. Number of Repeats", fontsize=11, fontweight="bold")
        plt.tight_layout()

        if output_dir is not None:
            fig.savefig(os.path.join(output_dir, "shap_convergence.pdf"))
            plt.close(fig)
        else:
            plt.show()

    def _get_rescue_scores(self, all_results, normtest=False):
        shap_df_list = []
        accuracy_list = []
        auc_list = []

        for runs in all_results:
            for fold in runs:
                shap_df_list.append(fold["mean_shap_df"])

                accuracy_list.append(fold["Test Accuracy"])
                auc_list.append(fold["Test AUC-ROC"])

        stacked = pd.concat(
            shap_df_list, keys=range(len(shap_df_list)), names=["run", "Parameter"]
        )

        results = []

        rescue_list = []
        param_list = []
        shift_denom_list = {
            "shift": [],
            "denom": [],
            "wt": [],
            "pten": [],
            "trt": [],
            "parameter": [],
        }

        def show_stat_result(value):
            return value
            # return ((f"{value:.5f}"), value<alpha)

        for param, g in stacked.groupby(level="Parameter"):
            param_list.append(param)

            wt = g["mu_WT"].to_numpy()
            pten = g["mu_PT"].to_numpy()
            trt = g["mu_Tx"].to_numpy()

            denom = wt - pten
            shift = trt - pten
            # Compute rescue per run; guard tiny denominators
            rescue = np.full_like(denom, np.nan, dtype=float)
            min_delta = np.nanmedian(np.abs(denom)) * 0.1
            valid = np.abs(denom) >= min_delta
            rescue[valid] = shift[valid] / denom[valid]

            rescue_list.append(rescue[valid])
            shift_denom_list["shift"] += shift[valid].tolist()
            shift_denom_list["denom"] += denom[valid].tolist()
            shift_denom_list["wt"] += wt[valid].tolist()
            shift_denom_list["pten"] += pten[valid].tolist()
            shift_denom_list["trt"] += trt[valid].tolist()
            shift_denom_list["parameter"] += [param] * len(shift[valid])

            # Summary stats (ignore NaNs)
            mean_rescue = tmean(rescue)
            std_rescue = np.nanstd(rescue)
            n_valid = np.sum(~np.isnan(rescue))

            # ===Stats Test===
            stat, p_val_wilcoxon = wilcoxon(
                rescue[~np.isnan(rescue)], alternative="two-sided"
            )

            # Htest
            kruskal_stat, kruskal_p = kruskal(
                wt[~np.isnan(wt)], pten[~np.isnan(pten)], trt[~np.isnan(trt)]
            )

            # Dunn Test
            dunn_result = sp.posthoc_dunn(
                [wt[~np.isnan(wt)], pten[~np.isnan(pten)], trt[~np.isnan(trt)]]
            )
            p_val_wt_pt_dunn = dunn_result.iloc[0, 1]
            p_val_pt_ptx_dunn = dunn_result.iloc[1, 2]

            if normtest:
                data = pten
                # Create density plot
                plt.figure(figsize=(10, 6))

                # Histogram with density
                plt.hist(
                    data,
                    density=True,
                    alpha=0.6,
                    color="lightblue",
                    edgecolor="black",
                    label="Histogram",
                )

                # KDE curve
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                plt.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")

                # Add mean and median lines
                plt.axvline(
                    np.mean(data),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {np.mean(data):.2f}",
                )
                plt.axvline(
                    np.median(data),
                    color="green",
                    linestyle="--",
                    label=f"Median: {np.median(data):.2f}",
                )

                plt.title(param)
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()

                # Normality tests
                print(f"=== Normality Tests for {param} ===")

                # Shapiro-Wilk test
                shapiro_stat, shapiro_p = stats.shapiro(data)
                print(f"Shapiro-Wilk Test:")
                print(f"  Statistic: {shapiro_stat:.4f}")
                print(f"  P-value: {shapiro_p:.4f}")
                print(
                    f"  Result: {'Data appears normal' if shapiro_p > 0.05 else 'Data does NOT appear normal'} (α=0.05)"
                )

            # Helpful context metrics
            mean_wt = np.nanmean(wt[valid])
            mean_pten = np.nanmean(pten[valid])
            mean_trt = np.nanmean(trt[valid])
            mean_denom = np.nanmean(denom[valid])
            mean_shift = np.nanmean(shift[valid])  # raw shift from PTEN

            results.append(
                {
                    "Parameter": param,
                    "N": int(n_valid),
                    "MeanRescue": mean_rescue,
                    "kruskal_p": show_stat_result(kruskal_p),
                    "WT-PT Dunn": show_stat_result(p_val_wt_pt_dunn),
                    "PT-PTx Dunn": show_stat_result(p_val_pt_ptx_dunn),
                    "p_value_wilcoxon": show_stat_result(p_val_wilcoxon),
                    "Mean_WT": mean_wt,
                    "Mean_PTEN": mean_pten,
                    "Mean_Treated": mean_trt,
                    "std_WT": np.nanstd(wt[valid]),
                    "std_PTEN": np.nanstd(pten[valid]),
                    "std_Treated": np.nanstd(trt[valid]),
                    "Mean_Denom": mean_denom,
                    "Mean_Shift": mean_shift,
                }
            )

        rescue_scores_summary = (
            pd.DataFrame(results)
            .set_index("Parameter")
            .sort_values(["MeanRescue"], ascending=False)
        )
        rescue_scores = pd.DataFrame(rescue_list, index=param_list)
        shift_denom = pd.DataFrame(shift_denom_list)

        mean_accuracy = np.mean(accuracy_list)
        mean_auc = np.mean(auc_list)

        print("Mean test accuracy:", mean_accuracy)
        print("Mean AUC-ROC:", mean_auc)

        return rescue_scores_summary, rescue_scores, shift_denom

    def _rescue_to_long(self, rescue_full_list, trim=0.1):
        """Convert wide rescue scores to long-form DataFrame.

        Applies the same count-based trim as tmean so that
        downstream np.mean on this data equals tmean on the
        raw data.
        """
        long_list = []
        for param in rescue_full_list.index:
            vals = np.sort(
                rescue_full_list.loc[param].dropna().values
            )
            n = len(vals)
            cut = int(n * trim)
            if cut > 0:
                vals = vals[cut:-cut]
            long_list.append(
                pd.DataFrame({
                    "parameter": param,
                    "rescue_score": vals,
                })
            )
        return pd.concat(long_list, ignore_index=True)

    def _visualize_results(self, resuce_scores, rescue_full_list, shift_denom, output_dir=None):
        rescue_long = self._rescue_to_long(rescue_full_list)

        df = resuce_scores.copy()
        p_thresh = 0.05

        plot_df = df.copy()
        plot_df["param"] = plot_df.index.astype(str)

        # Sort and filter
        sorted_df = plot_df.sort_values(
            "MeanRescue", ascending=False
        ).copy()
        sorted_df = sorted_df[
            (sorted_df["kruskal_p"] < p_thresh)
            & (sorted_df["p_value_wilcoxon"] < p_thresh)
            & (sorted_df["WT-PT Dunn"] < p_thresh)
            & (sorted_df["PT-PTx Dunn"] < p_thresh)
        ]
        sorted_df = sorted_df[(sorted_df["MeanRescue"] > 0)]

        ### ===Rescue Plot===
        plt.figure(figsize=(12, 4))

        order = sorted_df.index
        order = order.drop("Start Electrode", errors="ignore")

        rescue_long = rescue_long[
            rescue_long["parameter"].isin(order)
        ]
        rescue_long = rescue_long.merge(
            PARAMETER_CATEGORY, on="parameter"
        )
        rescue_long = rescue_long[
            rescue_long["parameter"] != "Start Electrode"
        ]

        # Data is already count-trimmed, so np.mean = tmean
        param_means = (
            rescue_long.groupby("parameter")["rescue_score"]
            .mean()
            .sort_values(ascending=False)
        )
        order = param_means.index

        self._visualize_mean_rescue_score(
            rescue_long, order, output_dir
        )
        self._visualize_shap_shift(shift_denom, output_dir)
        # self._visualize_mean_shap_values(shift_denom)

    def _visualize_mean_rescue_score(self, rescue_long, order, output_dir=None):
        palette = sns.color_palette("Spectral")
        palette_mapping = {
            "general parameter": palette[4],
            "synchrony parameter": palette[5],
            "network burst parameter": palette[1],
            "single burst parameter": palette[2],
        }

        sns.barplot(
            data=rescue_long,
            x="rescue_score",
            y="parameter",
            order=order,
            palette=palette_mapping,
            hue="category",
            legend=True,
            estimator=np.mean,
            errorbar="se",
            err_kws={"color": "grey", "linewidth": 2.5},
        )

        plt.title("Mean Rescue Score")
        plt.xlabel("Rescue Score")
        plt.ylabel("")
        plt.xlim(0, 1)
        plt.grid(axis="x", which="major", linestyle="-", alpha=0.8, visible=True)
        plt.grid(axis="x", which="minor", linestyle="--", alpha=0.1, visible=True)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "MeanRescueScore.pdf"))
        plt.show()

    def _visualize_shap_shift(self, shift_denom, output_dir=None):
        shap_df = shift_denom.reset_index().melt(
            id_vars=["parameter"],
            value_vars=["shift", "denom"],
        )
        plt.figure(figsize=(12, 6))
        sns.stripplot(
            data=shap_df,
            x="value",
            y="parameter",
            alpha=0.3,
            jitter=0.2,
            palette="Set2",
            zorder=0,
            dodge=True,
            size=3,
            hue="variable",
            legend=True,
        )
        sns.pointplot(
            data=shap_df,
            x="value",
            y="parameter",
            hue="variable",
            linestyle="none",
            errorbar=None,
            marker="|",
            markersize=10,
            markeredgewidth=2,
            zorder=1,
            palette="Set2",
            dodge=0.4,
            alpha=1,
            legend=False,
            estimator=tmean,
        )

        plt.title("Mean SHAP Shifts")
        plt.xlabel("SHAP Difference")
        plt.ylabel("")
        plt.grid(axis="x", which="major", linestyle="-", alpha=0.8, visible=True)
        plt.grid(axis="x", which="minor", linestyle="--", alpha=0.1, visible=True)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xlim(-0.1, 0.2)
        plt.legend(loc="upper right", labels=["Treatment Shift", "Genotype Shift"])
        plt.tight_layout()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "MeanSHAPShift.pdf"))
        plt.show()

    def _extract_shap_long(self, features):
        rows = []
        for runs in self.results:
            for fold in runs:
                df = fold["mean_shap_df"]
                for feat in features:
                    if feat not in df.index:
                        continue
                    rows.append(
                        {
                            "feature": feat,
                            "group": "WT",
                            "shap_value": df.loc[feat, "mu_WT"],
                        }
                    )
                    rows.append(
                        {
                            "feature": feat,
                            "group": "PTEN",
                            "shap_value": df.loc[feat, "mu_PT"],
                        }
                    )
                    rows.append(
                        {
                            "feature": feat,
                            "group": "Treated",
                            "shap_value": df.loc[feat, "mu_Tx"],
                        }
                    )
        return pd.DataFrame(rows)

    def _visualize_mean_shap_values(self, shift_denom):
        shap_df = shift_denom.reset_index().melt(
            id_vars=["parameter"],
            value_vars=["pten", "wt", "trt"],
        )

        plt.figure(figsize=(12, 9))
        sns.stripplot(
            data=shap_df,
            x="value",
            y="parameter",
            alpha=0.3,
            jitter=0.2,
            palette="Set2",
            zorder=0,
            dodge=True,
            size=3,
            hue="variable",
            legend=True,
        )
        sns.pointplot(
            data=shap_df,
            x="value",
            y="parameter",
            hue="variable",
            linestyle="none",
            errorbar=None,
            marker="|",
            markersize=10,
            markeredgewidth=2,
            zorder=1,
            palette="Set2",
            dodge=0.4,
            alpha=1,
            legend=False,
            estimator=tmean,
        )

        plt.title("Mean SHAP Values")
        plt.xlabel("SHAP Values")
        plt.ylabel("")
        plt.grid(axis="x", which="major", linestyle="-", alpha=0.8, visible=True)
        plt.grid(axis="x", which="minor", linestyle="--", alpha=0.1, visible=True)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.xlim(-0.1, 0.1)
        # plt.legend(loc='upper right', labels=['Treatment Shift', 'Genotype Shift'])
        plt.tight_layout()
        # plt.savefig("MeanSHAPShift_DTX.pdf")
        plt.show()
