import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset_loader import FireDataset
from src.config import MODEL_CONFIG
import segmentation_models_pytorch as smp


def evaluate_model(region_name="alberta", model_path="models/unet_alberta.pth", num_samples=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ Load test dataset
    data_dir = os.path.join("data", region_name, "processed")
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    test_dataset = FireDataset(
        image_dir,  # first positional argument
        mask_dir,   # second positional argument
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2️⃣ Load trained model
    print("📦 Loading model...")
    model = smp.Unet(
        encoder_name=MODEL_CONFIG["encoder"],
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def iou_score(pred, target, threshold=0.5, eps=1e-6):
        pred = (pred > threshold).float()
        target = (target > threshold).float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + eps) / (union + eps)
        return iou.mean().item()

    def dice_score(pred, target, threshold=0.5, eps=1e-6):
        pred = (pred > threshold).float()
        target = (target > threshold).float()
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + eps) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
        return dice.mean().item()

    total_iou, total_dice, count = 0.0, 0.0, 0

    # 4️⃣ Evaluate
    print("🚀 Evaluating model...")
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds = torch.sigmoid(preds)

            total_iou += iou_score(preds, masks)
            total_dice += dice_score(preds, masks)
            count += 1

    avg_iou = total_iou / count
    avg_dice = total_dice / count

    print(f"\n✅ Evaluation Results:")
    print(f"  Mean IoU:   {avg_iou:.4f}")
    print(f"  Mean Dice:  {avg_dice:.4f}")

    # 5️⃣ Visualization (optional)
    os.makedirs("outputs", exist_ok=True)
    for i in range(num_samples):
        img, mask = test_dataset[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        axs[0].imshow(np.transpose(img.numpy(), (1, 2, 0)))
        axs[0].set_title("Input Image")
        axs[1].imshow(mask.squeeze().numpy(), cmap="gray")
        axs[1].set_title("Ground Truth")
        axs[2].imshow(pred > 0.5, cmap="Reds")
        axs[2].set_title("Predicted Fire Mask")

        for ax in axs: ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"outputs/pred_{i+1}.png")
        plt.close()

    print(f"📸 Saved {num_samples} prediction samples to 'outputs/'")


if __name__ == "__main__":
    evaluate_model(region_name="alberta", model_path="models/unet_alberta.pth")
