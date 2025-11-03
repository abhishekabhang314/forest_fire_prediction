import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

from src.dataset_loader import FireDataset
import segmentation_models_pytorch as smp

from src.config import get_region_paths, OUTPUT_DIR

region_name = "alberta" 
paths = get_region_paths(region_name)

image_dir = paths["images_dir"]
mask_dir = paths["masks_dir"]


# --- Metrics ---
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

def evaluate_model(region_name="alberta", model_path="models/unet_alberta.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Loading model...")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("🚀 Evaluating model...")

    region_name = "alberta"
    paths = get_region_paths(region_name)

    image_dir = paths["images_dir"]
    mask_dir = paths["masks_dir"]

    test_dataset = FireDataset(image_dir=image_dir, mask_dir=mask_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

    total_iou, total_dice, total_acc, count = 0.0, 0.0, 0.0, 0
    os.makedirs(os.path.join(OUTPUT_DIR, "predictions"), exist_ok=True)

    for imgs, masks in tqdm(test_loader):
        imgs, masks = imgs.to(device), masks.to(device)

        with torch.no_grad():
            preds = model(imgs)
            preds = torch.sigmoid(preds)

        total_iou += iou_score(preds, masks)
        total_dice += dice_score(preds, masks)
        acc = ((preds > 0.5) == (masks > 0.5)).float().mean().item()
        total_acc += acc
        count += 1

        # Save sample predictions
        if count <= 5:
            for i in range(len(preds)):
                save_image(preds[i], os.path.join(OUTPUT_DIR, "predictions", f"pred_{count}_{i}.png"))

    avg_iou = total_iou / count
    avg_dice = total_dice / count
    avg_acc = total_acc / count

    print("\n✅ Evaluation Results:")
    print(f"  Mean IoU:   {avg_iou:.4f}")
    print(f"  Mean Dice:  {avg_dice:.4f}")
    print(f"  Accuracy:   {avg_acc:.4f}")

    # --- Plot metrics ---
    metrics = {"IoU": avg_iou, "Dice": avg_dice, "Accuracy": avg_acc}
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values(), color=["orange", "blue", "green"])
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, "metrics_chart.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"📊 Saved chart to {chart_path}")
    print(f"📸 Saved predictions to {os.path.join(OUTPUT_DIR, 'predictions')}")


if __name__ == "__main__":
    evaluate_model(region_name="alberta", model_path="models/unet_alberta.pth")
