from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy import stats
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
import numpy as np
import scikit_posthocs as sp
import matplotlib.pyplot as plt
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
        if results:
            self.results = results

    def save_to_file(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.results, f)

    def load_from_file(self, file_path):
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
        self.load(loaded_data)

    def analyze_results(self, normtest=False):
        print("####")
        print("#### Analyzing Result")
        print("Result size =", len(self.results))
        rescue_scores_summary, rescue_full_list, shift_denom = self._get_rescue_scores(
            self.results, normtest=normtest
        )
        self._visualize_results(rescue_scores_summary, rescue_full_list, shift_denom)
        return rescue_scores_summary, shift_denom

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

        alpha = (0.05,)
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

    def _trim_scores(self, rescue_full_list):
        # Function to trim 10% at both ends
        def trim_series(s, proportion=0.1):
            q_low = s.quantile(proportion)
            q_high = s.quantile(1 - proportion)
            return s[(s >= q_low) & (s <= q_high)]

        # Apply trimming to each parameter (row), collect in long-form list
        trimmed_list = []
        for param in rescue_full_list.index:
            trimmed = trim_series(rescue_full_list.loc[param])
            trimmed_list.append(
                pd.DataFrame({"parameter": param, "rescue_score": trimmed.values})
            )

        # Concatenate all trimmed results into a long-form DataFrame
        df_trimmed = pd.concat(trimmed_list, ignore_index=True)
        return df_trimmed

    def _visualize_results(self, resuce_scores, rescue_full_list, shift_denom):
        rescue_trimmed = self._trim_scores(rescue_full_list)

        df = resuce_scores.copy()
        p_thresh = 0.05

        plot_df = df.copy()
        plot_df["param"] = plot_df.index.astype(str)

        # Sort and filter
        sorted_df = plot_df.sort_values("MeanRescue", ascending=False).copy()
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

        rescue_trimmed = rescue_trimmed[rescue_trimmed["parameter"].isin(order)]
        rescue_trimmed = rescue_trimmed.merge(PARAMETER_CATEGORY, on="parameter")

        #!!! Remove Start Electrode in the graph
        rescue_trimmed = rescue_trimmed[
            rescue_trimmed["parameter"] != "Start Electrode"
        ]

        palette = sns.color_palette("Spectral")
        palette_mapping = {
            "general parameter": palette[4],
            "synchrony parameter": palette[5],
            "network burst parameter": palette[1],
            "single burst parameter": palette[2],
        }

        sns.barplot(
            data=rescue_trimmed,
            x="rescue_score",
            y="parameter",
            order=order,
            palette=palette_mapping,
            hue="category",
            legend=True,
            estimator=tmean,
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
        # plt.savefig("MeanRescueScore_kv1_1_kd.pdf")
        plt.show()

        ### ===Mean SHAP Shifts===
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
        plt.xlim(-0.5, 0.5)
        plt.legend(loc="upper right", labels=["Treatment Shift", "Genotype Shift"])
        plt.tight_layout()
        # plt.savefig("MeanSHAPShift_DTX.pdf")
        plt.show()

        ### ===Mean SHAP Values===
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
