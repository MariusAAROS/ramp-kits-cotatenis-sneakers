import sys
import git
import torch
import pandas as pd
from torch.utils.data import DataLoader


root_path = git.Repo(".", search_parent_directories=True).working_tree_dir
sys.path.append(root_path)
from cotatenis_sneakers.sneaker_dataset import SneakerDataset
from cotatenis_sneakers.sneaker_transforms import get_transform


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder = "data/private/"


class BatchClassifier:
    def __init__(self):
        self.correct = 0
        self.total_correct = 0
        self.total = 0
        self.print_every = 10
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.transform = get_transform(in_notebook=True)
        self.loss = None

    def build_model(self):
        model = torch.hub.load("pytorch/vision", "mobilenet_v2", pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = torch.nn.Linear(model.last_channel, 3)

        for param in model.classifier.parameters():
            param.requires_grad = True

        model = model.to(device)
        return model

    def fit(self, gen_builder):
        # J'ai été obligée de reconstruire data à partir des attributs de gen_builder
        # puis de le passer à SneakerDataset car j'avais une key error en parcourant
        # gen_builder que je n'ai pas réussi à corriger

        data = pd.concat([gen_builder.X_array, gen_builder.y_array], axis=1)

        sneaker_dataset = SneakerDataset(
            data=data,
            folder=gen_builder.folder,
            device=device,
            transform=self.transform,
        )
        sneaker_loader = DataLoader(sneaker_dataset, batch_size=32, shuffle=True)

        print_every = 50
        running_loss = 0
        n_epochs = 2

        self.model.train()

        for epoch in range(n_epochs):
            for i, data in enumerate(sneaker_loader):

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                self.loss = self.criterion(outputs, labels)
                running_loss += self.loss.item()
                self.loss.backward()
                self.optimizer.step()

                if i % print_every == 0 and i != 0:
                    avg_loss = running_loss / print_every
                    print(f"Epoch {epoch}, Iteration {i}, Loss: {avg_loss}")
                    running_loss = 0

    def predict_proba(self, images):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for _, img in enumerate(images):
                img = torch.Tensor(img.reshape((-1, 3, 400, 400))).to(device)
                outputs = self.model(img)
                predictions.append(outputs.squeeze().detach().cpu().numpy())
        return predictions
