import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image

class Orchid(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        # self.img_lebels = pd.read_csv("data/label.csv")
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = image.convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        self.transform(image)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
