import pandas as pd

from constants import high_vif_feat, to_drop_095
from utils import get_unique_items, is_substr
from neuroHarmonize import harmonizationLearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    def __init__(self, x: pd.DataFrame) -> None:
        self.x = x

    # ====== Filters
    def _remove_high_colinear_features(self, x):
        x = x[[c for c in x.columns if not is_substr(c, to_drop_095)]]
        # X = X[[c for c in X.columns if not is_substr(c, to_drop_manual)]]
        x = x[[c for c in x.columns if c not in high_vif_feat]]
        return x.copy()

    def _remove_std_features(self, x):
        x = x[[c for c in x.columns if not "Std" in c]]
        return x.copy()

    def _remove_zero_features(self, x):
        x = x.loc[:, (x != 0).any(axis=0)]
        return x

    # ====== Normalizations
    def _batch_normalize_neuro_harmonize(self, x):
        x = x.copy()
        covars = x[["day", "round"]]
        covars["SITE"] = covars["round"]
        covars.drop(columns=["round"], inplace=True)

        x = self._remove_zero_features(x)

        feature_cols = [
            col for col in x.columns if col not in ["round", "day"]
        ]  # numerical features

        # neuroHarmonize cannot handle NaN; impute with per-round median
        for col in feature_cols:
            if x[col].isna().any():
                x[col] = x.groupby("round")[col].transform(
                    lambda s: s.fillna(s.median())
                )
                # Fall back to global median if an entire round is NaN
                x[col] = x[col].fillna(x[col].median())

        X_Feat = x[feature_cols]

        # Learn model and apply to the same data
        my_model, X_harmonized = harmonizationLearn(
            X_Feat.values,
            covars,
        )

        x[feature_cols] = X_harmonized
        return x

    def _normalize_data(self, df):
        exclude_cols = ["round"]
        df = df.copy()
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        return df

    # ====== Visualizations
    def _plot_pca_by_round(self, X, title="PCA"):
        X = X.copy()
        round = X.pop("round")
        X.pop("day")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_plot["round"] = round.reset_index(drop=True)

        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df_plot,
            x="PC1",
            y="PC2",
            hue="round",
            palette="tab10",
            alpha=0.8,
            s=20,
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def _compare_feature_by_round_kde(self, X_raw, X_scaled, features):
        for feature in features:
            plt.figure(figsize=(10, 4))

            # Raw
            plt.subplot(1, 2, 1)
            sns.kdeplot(
                data=X_raw, x=feature, hue="round", fill=True, common_norm=False
            )
            plt.title(f"Before Normalization: {feature}")

            # Normalized
            plt.subplot(1, 2, 2)
            sns.kdeplot(
                data=X_scaled, x=feature, hue="round", fill=True, common_norm=False
            )
            plt.title(f"After Normalization: {feature}")
            plt.xlabel("Normalized Value")

            plt.tight_layout()
            plt.show()

    # 1. Remove high colinear & zero features
    # 2. Normalization
    def preprocess(self, is_normalize=True, show_plot=False):
        print("=== Preprocessing Started")
        x = self.x.copy()

        x = self._remove_high_colinear_features(x)
        x = self._remove_std_features(x)
        x = self._remove_zero_features(x)

        X_before_scaling = x.copy()

        # Scale the data (Normalize)
        if is_normalize:
            x = self._batch_normalize_neuro_harmonize(x)
            X_after_batch_norm = x.copy()
            x = self._normalize_data(x)

            if show_plot:
                self._plot_pca_by_round(
                    X_before_scaling, title="PCA - Before Normolization"
                )
                self._plot_pca_by_round(
                    X_after_batch_norm, title="PCA - After Batch Normolization"
                )
                self._plot_pca_by_round(x, title="PCA - After Overall Normolization")

                self._compare_feature_by_round_kde(
                    X_before_scaling,
                    x,
                    [
                        "Burst Duration - Avg (sec)",
                        "Number of Bursts",
                        "Weighted Mean Firing Rate (Hz)",
                        #  'Full Width at Half Height of Normalized Cross-Correlation'
                    ],
                )

        # X = onehot_encode_round(X)

        x.pop("day")
        x.pop("round")

        print("Features:", x.columns)

        return x
