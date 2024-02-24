import sys
import git
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


root_path = git.Repo('.', search_parent_directories=True).working_tree_dir

sys.path.append(root_path)

from cotatenis_sneakers.sneaker_dataset import SneakerDataset
from cotatenis_sneakers.sneaker_transforms import get_transform, UnNormalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = "data/private/"

class BatchClassifier():
    def __init__(self):
        self.correct = 0
        self.total_correct = 0
        self.total = 0
        self.print_every = 10
        self.build_model()

    def build_model(self):
        model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        model.fc = torch.nn.Linear(2048, 3)
        self.model_ = model.to(device)

    def fit(self, X=None, y=None):
        pass # pretrained model

    def predict_proba(self, images):
        predictions = []
        with torch.no_grad():
            for _, img in enumerate(images):
                img = torch.Tensor(img.reshape((-1, 3, 400, 400))).to(device)
                outputs = self.model_(img)
                predictions.append(outputs.squeeze().detach().cpu().numpy())
        return predictions
    