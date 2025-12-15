import numpy as np
import shap
import pandas as pd

class ShapDeepExplainer:
    def __init__(self, model, X_train, shap_interested_label_index, background_size=100):
        X_train_np = X_train.to_numpy()
        # If X_train is a NumPy array:
        background = X_train_np[np.random.choice(X_train_np.shape[0], background_size, replace=False)]
        # If X_train is a pandas DataFrame
        # background = X_train.sample(background_size, random_state=0)
        self.deepExplainer = shap.DeepExplainer(model, background)
        self.shap_interested_label_index = shap_interested_label_index

    def explain_SHAP(self, X_explain):
        X_explain_np = X_explain.to_numpy()        
        return self.deepExplainer.shap_values(X_explain_np)

    def get_mean_shap(self, shap_values, columns):
        shap_df = pd.DataFrame(shap_values[..., self.shap_interested_label_index], columns=columns)
        mean_shap = shap_df.mean().sort_values(ascending=False)
        return mean_shap

    def run_and_get_mean_shap(self, X_explain):
        shap_values = self.explain_SHAP(X_explain)
        return self.get_mean_shap(shap_values, X_explain.columns)