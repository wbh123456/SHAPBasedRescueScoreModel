import pandas as pd
from pandas.core.api import CategoricalDtype
from scipy.stats import trim_mean
import numpy as np


def is_substr(s, l):
    for i in l:
        if s in i:
            return True
    return False


def get_unique_items(y: pd.Series) -> list:
    return y.unique().tolist()


def filter_labels(X, y, labels_to_keep):
    mask = y.isin(labels_to_keep)
    return X[mask], y[mask]


def get_Y(y, labels):
    label_type = CategoricalDtype(categories=labels, ordered=False)
    y = y.astype(label_type)
    return pd.get_dummies(y)

def tmean(x, p=0.1):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return trim_mean(x, proportiontocut=p)
