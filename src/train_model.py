# forest_fire_detection/src/train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import os

from src.model_unet import get_unet
from src.dataset_loader import FireDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(region_name="alberta"):
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    region_dir = os.path.join(BASE_DIR, "data", region_name, "processed")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = FireDataset(
        os.path.join(region_dir, "images"),
        os.path.join(region_dir, "masks"),
        weather_csv=os.path.join(region_dir, "weather.csv"),
        transform=transform
    )

    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    # Adjust input channels (3 weather vars)
    model = get_unet(in_channels=3, classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):  # quick demo training
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    out_model = os.path.join(BASE_DIR, "models", f"unet_{region_name}.pth")
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    torch.save(model.state_dict(), out_model)
    print(f"✅ Model saved at {out_model}")


if __name__ == "__main__":
    train_model("alberta")
