import os
import sys
import git
import pandas as pd

root_path = git.Repo('.', search_parent_directories=True).working_tree_dir

sys.path.append(root_path)

# import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from cotatenis_sneakers.sneaker_dataset import SneakerDataset
from cotatenis_sneakers.sneaker_transforms import get_transform
from torch.utils.data import DataLoader

import rampwf as rw

problem_title = "Sneakers brand classification"

_prediction_label_names = ["adidas", "Nike", "Jordan"]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.ImageClassifier(
    test_batch_size=32,
    chunk_size=256,
    n_jobs=8,
    n_classes=len(_prediction_label_names),
)

transform = get_transform()

score_types = [
    rw.score_types.BalancedAccuracy(name="bal_acc", precision=3, adjusted=False),
    rw.score_types.Accuracy(name="acc", precision=3),
]


# def _read_data(path, split="train"):
#     """
#     Idée : avoir un dossier test et un dossier train avec dedans labels.csv qui
#     contient une colonne filename et une colonne label,
#     inspiré de https://github.com/ramp-kits/follicles_detection
#     """
#     base_data_path = os.path.abspath(os.path.join(path, "data", split))
#     labels_path = os.path.join(base_data_path, "labels.csv")
#     labels = pd.read_csv(labels_path)
#     filepaths = []
#     brands = []
#     for filename, group in labels.groupby("filename"):
#         filepath = os.path.join(base_data_path, filename)
#         filepaths.append(filepath)
#         brands.append(group)
#     X = np.array(filepaths, dtype=object)
#     y = np.array(brands, dtype=object)
#     assert len(X) == len(y)

#     return X, y


def _read_data(folder, split):
    if split not in ["train", "test"]:
        raise ValueError("split must be either 'train' or 'test'")
    path = os.path.join(folder, split, split + ".csv")
    path_img = os.path.join(folder, split)
    data = pd.read_csv(path)
    X = data.iloc[:, 0]
    y = data.iloc[:, 1]
    return (path_img, X), y


def get_train_data(path='.'):
    return _read_data(os.path.join(path, "data", "private"), "train")


def get_test_data(path='.'):
    return _read_data(os.path.join(path, "data", "private"), "test")


def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=33)
    return cv.split(X, y)
