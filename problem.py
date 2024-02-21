import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import rampwf as rw

problem_title = "Sneakers brand classification"

_prediction_label_names = ['adidas','Nike','Jordan']

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),

]

def _read_data(path,split='train'):
    """
    Idée : avoir un dossier test et un dossier train avec dedans labels.csv qui
    contient une colonne filename et une colonne label,
    inspiré de https://github.com/ramp-kits/follicles_detection
    """
    base_data_path = os.path.abspath(os.path.join(path, "data", split))
    labels_path = os.path.join(base_data_path, "labels.csv")
    labels = pd.read_csv(labels_path)
    filepaths = []
    brands = []
    for filename, group in labels.groupby("filename"):
        filepath = os.path.join(base_data_path, filename)
        filepaths.append(filepath)
        brands.append(group)
    X = np.array(filepaths, dtype=object)
    y = np.array(brands, dtype=object)
    assert len(X) == len(y)
    
    return X, y



def get_train_data(path='.'):
    return _read_data(path, split='train')


def get_test_data(path='.'):
    return _read_data(path, split='test')

groups = None

def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=33)
    return cv.split(X, y, groups)