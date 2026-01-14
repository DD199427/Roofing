import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from utils.ridge_dataset import RidgeSegmentationDataset
from utils.transforms import get_train_transforms, get_val_transforms
from models.unet import UNet

# ----------------------------
# CONFIGURATION
# ----------------------------
IMAGE_HEIGHT = 600
IMAGE_WIDTH = 596
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMAGE_DIR = "/content/drive/MyDrive/RidgeData/data/train/image"
TRAIN_MASK_DIR  = "/content/drive/MyDrive/RidgeData/data/train/masks"

VAL_IMAGE_DIR   = "/content/drive/MyDrive/RidgeData/data/valid/image"
VAL_MASK_DIR    = "/content/drive/MyDrive/RidgeData/data/valid/masks"


# ----------------------------
# DATA LOADERS
# ----------------------------
train_dataset = RidgeSegmentationDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    transform=get_train_transforms(),
)

val_dataset = RidgeSegmentationDataset(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    transform=get_val_transforms(),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ----------------------------
# MODEL SETUP
# ----------------------------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# TRAINING LOOP
# ----------------------------
# def train_one_epoch(loader, model, optimizer, loss_fn):
#     model.train()
#     epoch_loss = 0

#     for images, masks in tqdm(loader, desc="Training"):
#         images = images.to(DEVICE)
#         masks = masks.to(DEVICE).unsqueeze(1).float()  # (B, 1, H, W)

#         preds = model(images)
#         loss = loss_fn(preds, masks)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()

#     return epoch_loss / len(loader)
def train_one_epoch(loader, model, optimizer, loss_fn):
    model.train()
    epoch_loss = 0.0
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True).float()
        # 🔒 normalize shapes to [B,1,H,W]
        if masks.ndim == 5 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [B,1,H,W]
        assert masks.ndim == 4 and masks.shape[1] == 1, f"Mask shape bad: {masks.shape}"

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / max(1, len(loader))

# ----------------------------
# VALIDATION LOOP
# ----------------------------
def evaluate(loader, model):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE, non_blocking=True).float()
            if masks.ndim == 5 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            preds = model(images)
            loss = loss_fn(preds, masks)
            loss_total += loss.item()

    return loss_total / len(loader)

# ----------------------------
#  MAIN TRAINING ROUTINE
# ----------------------------
mlflow.set_experiment("RidgeSegmentation")

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params({
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model": "UNet"
    })
    
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
    val_loss = evaluate(val_loader, model)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("Saved best model")

print("\n🎉 Training Complete.")
