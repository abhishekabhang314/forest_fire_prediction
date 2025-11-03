# forest_fire_detection/src/dataset_loader.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd


class FireDataset(Dataset):
    def __init__(self, img_dir=None, mask_dir=None, transform=None, weather_csv=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        self.transform = transform
        self.weather_data = pd.read_csv(weather_csv) if weather_csv else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".tif", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.weather_data is not None:
            date_str = img_name.split("_")[-1].split(".")[0]
            row = self.weather_data[self.weather_data["date"] == date_str]
            if not row.empty:
                w = row.iloc[0]
                temp = torch.full((1, image.shape[1], image.shape[2]), w["temp_max"] / 50.0)
                hum = torch.full((1, image.shape[1], image.shape[2]), w["humidity"] / 100.0)
                wind = torch.full((1, image.shape[1], image.shape[2]), w["wind_speed"] / 50.0)
                image = torch.cat([image, temp, hum, wind], dim=0)

        return image, mask
