import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import pandas as pd

class FireDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, weather_csv=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.weather_data = pd.read_csv(weather_csv) if weather_csv else None

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.to_tensor = transforms.ToTensor()  # <--- important

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Convert to tensor before returning
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        if self.transform:
            image = self.transform(image)

        return image, mask
