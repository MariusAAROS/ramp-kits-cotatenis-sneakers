from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms


class SneakerDataset(Dataset):
    def __init__(self, data, folder, transform=None):
        """
        Args:
            data (DataFrame): Pandas DataFrame containing the path to the image
            as index and its label.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pandas DataFrame")

        self.data = data
        self.folder = folder
        self.transform = transform
        self.labels = self.data.iloc[:, 1]
        self.unique_labels = self.labels.unique()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        path = self.data.iloc[idx, 0]
        image = Image.open(f"{self.folder}/{path}").convert("RGB")
        label = self.data.iloc[idx, 1]
        if not self.transform:
            self.transform = transforms.ToTensor()  # minimum transformation

        image = self.transform(image)
        return image, label

    def get_untransformed_tuple(self, idx):
        path = self.data.iloc[idx, 0]
        image = Image.open(f"{self.folder}/{path}").convert("RGB")
        label = self.data.iloc[idx, 1]
        return image, label
