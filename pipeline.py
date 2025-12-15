import random
import time

from sklearn.model_selection import StratifiedKFold
from constants import PT_WT_LABELS, PTENFalse, PTENTrue, WTFalse
from network import Network
from shap_analyzer import ShapDeepExplainer
from utils import filter_labels, get_Y
import pandas as pd


class ModelPipeline:
    def __init__(self, X, y, batch_size=8, epochs=80):
        self.batch_size = batch_size
        self.epochs = epochs

        X, y = X.copy(), y.copy()
        self.X_to_train, self.y_to_train = filter_labels(X, y, PT_WT_LABELS)
        self.X_PT_treated, self.y_PT_treated = filter_labels(X, y, [PTENTrue])
        self.X_PT_untreated, self.y_PT_untreated = filter_labels(X, y, [PTENFalse])
        self.X_WT, self.y_WT = filter_labels(X, y, [WTFalse])
        return

    def run_repeated_5_fold(self, repeat, is_run_1_fold=False):
        all_results = []
        for i in range(repeat):
            results = self._run_5_fold(i, is_run_1_fold)
            all_results.append(results)
        return all_results

    def _run_5_fold(self, run_idx=0, is_run_1_fold=False):

        folds = self._get_5fold_splits()

        results_all_folds = []

        for i, fold in enumerate(folds):
            print("\n###############")
            print("###############")
            print("##### Running --- Run #", run_idx, "- Fold #", i)
            X_train, X_test, Y_train, Y_test = fold

            print("\n##### Training")
            model = Network()
            model.train_model(
                X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs
            )
            test_scrores = model.evaluate_model(X_test, Y_test)

            print("\n##### SHAP Analysis")
            shap_interested_label_index = PT_WT_LABELS.index(WTFalse)
            explainer = ShapDeepExplainer(
                model.get_model(), X_train, shap_interested_label_index
            )

            mean_shap_PT_untreated = explainer.run_and_get_mean_shap(
                self.X_PT_untreated
            )
            mean_shap_PT_treated = explainer.run_and_get_mean_shap(self.X_PT_treated)
            mean_shap_WT = explainer.run_and_get_mean_shap(self.X_WT)

            mean_shap_PT_treated = mean_shap_PT_treated.reindex_like(
                mean_shap_PT_untreated
            )
            mean_shap_WT = mean_shap_WT.reindex_like(mean_shap_PT_untreated)

            mean_shap_df = pd.DataFrame(
                {
                    "mu_WT": mean_shap_WT,
                    "mu_PT": mean_shap_PT_untreated,
                    "mu_Tx": mean_shap_PT_treated,
                }
            )

            results_all_folds.append(
                {
                    "Test Accuracy": test_scrores["accuracy"],
                    "Test AUC-ROC": test_scrores["auc-roc"],
                    "mean_shap_df": mean_shap_df,
                }
            )

            if is_run_1_fold:
                break
        return results_all_folds

    def _get_5fold_splits(self, n_splits=5):
        seed = int(time.time_ns() & 0xFFFFFFFF)
        random_state = random.seed(seed)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        folds = []
        X, y = self.X_to_train, self.y_to_train

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            Y_train = get_Y(y_train, PT_WT_LABELS)
            Y_test = get_Y(y_test, PT_WT_LABELS)

            folds.append((X_train, X_test, Y_train, Y_test))

        return folds
