import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

MODEL_PATH = "models/unet_alberta.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Prediction pipeline
def predict_fire(image_path, output_path="outputs/custom_prediction.png"):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        mask = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Threshold mask
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Resize mask to original
    mask_resized = cv2.resize(mask, img.size)

    # Overlay mask
    img_np = np.array(img)
    overlay = img_np.copy()
    overlay[mask_resized > 0] = [255, 0, 0]  # red overlay
    blended = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    plt.imshow(blended)
    plt.title("Fire Prediction Overlay")
    plt.axis("off")
    plt.show()

    print(f"✅ Saved overlay to: {output_path}")

# Example
if __name__ == "__main__":
    test_image = "data/alberta/processed/images/sample.png"  # change this path
    predict_fire(test_image)
