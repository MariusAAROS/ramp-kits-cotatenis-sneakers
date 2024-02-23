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
folder = "data/"

class Classifier():
    def __init__(self):
        transform = get_transform()
        train_data = pd.read_csv(f"{folder}/train/train.csv")
        self.train_dataset = SneakerDataset(
            train_data, folder=f"{folder}/train", device=device, transform=transform
        )
        test_data = pd.read_csv(f"{folder}/test/test.csv")
        self.test_dataset = SneakerDataset(
            test_data, folder=f"{folder}/test", device=device, transform=transform
        )

    def fit(self, X=None, y=None):
        model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        model.fc = torch.nn.Linear(2048, 3)
        self.model_ = model.to(device)

        self.model_.eval()
        self.correct = 0
        self.total_correct = 0
        self.total = 0
        self.print_every = 10

        if device == "cuda":
            self.test_loader = DataLoader(
                self.test_dataset, batch_size=32, shuffle=False, pin_memory=True
            )
        else:
            self.test_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
        
    def predict(self, images):
        predictions = []
        with torch.no_grad():
            for i, (images, _ ) in enumerate(self.test_loader):
                images= images.to(device)
                outputs = self.model_(images)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted)

    def predict_proba(self, images):
        predictions = []
        with torch.no_grad():
            for i, (images, _ ) in enumerate(self.test_loader):
                images= images.to(device)
                outputs = self.model_(images)
                predictions.append(outputs)
    