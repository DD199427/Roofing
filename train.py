# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from utils.ridge_dataset import RidgeSegmentationDataset
from utils.transforms import get_train_transforms, get_val_transforms
from models.unet import UNet

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE      = 512
BATCH_SIZE    = 4           # reduce if OOM
EPOCHS        = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
BANDS         = (1, 2, 3)   # rasterio band order to read
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (Drive layout)
TRAIN_IMAGE_DIR = "/content/drive/MyDrive/RidgeData/data/train/image"
TRAIN_MASK_DIR  = "/content/drive/MyDrive/RidgeData/data/train/masks"

VAL_IMAGE_DIR   = "/content/drive/MyDrive/RidgeData/data/valid/image"
VAL_MASK_DIR    = "/content/drive/MyDrive/RidgeData/data/valid/masks"
CHECKPOINT_DIR  = "/content/drive/MyDrive/RidgeData/checkpoints"
MLFLOW_URI      = "file:/content/drive/MyDrive/RidgeData/mlruns"
EXPERIMENT_NAME = "RidgeSegmentation"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# LOSSES
# ----------------------------
bce = nn.BCEWithLogitsLoss()

def dice_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3)) + eps
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return 1.0 - (num / den).mean()

def combo_loss(logits, targets):
    return 0.5 * bce(logits, targets) + 0.5 * dice_loss(logits, targets)

# ----------------------------
# BATCH NORMALIZER (robust)
# ----------------------------
def _normalize_batch(images, masks, device):
    # images -> [B,3,H,W] float32 [0,1]
    images = images.to(device, non_blocking=True)
    if images.ndim == 5 and images.shape[1] == 1:
        images = images.squeeze(1)
    assert images.ndim == 4, f"Images must be 4D, got {images.shape}"
    assert images.shape[1] == 3, f"Images must have 3 channels, got {images.shape}"
    images = images.float().clamp(0.0, 1.0)

    # masks -> [B,1,H,W] float32 {0,1}
    masks = masks.to(device, non_blocking=True)
    if masks.ndim == 5 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    masks = masks.float()
    if masks.max() > 1.0:
        masks = masks / 255.0
    masks = masks.clamp(0.0, 1.0)
    return images, masks

# ----------------------------
# LOOPS
# ----------------------------
def train_one_epoch(loader, model, optimizer, scaler):
    model.train()
    running = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images, masks = _normalize_batch(images, masks, DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(images)
            loss   = combo_loss(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def validate(loader, model):
    model.eval()
    running = 0.0
    for images, masks in tqdm(loader, desc="Valid", leave=False):
        images, masks = _normalize_batch(images, masks, DEVICE)
        logits = model(images)
        loss   = combo_loss(logits, masks)
        running += loss.item()
    return running / max(1, len(loader))

# ----------------------------
# MAIN
# ----------------------------
def main():
    # Datasets & loaders
    train_ds = RidgeSegmentationDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir =TRAIN_MASK_DIR,
        transform=get_train_transforms(IMG_SIZE),
        bands=BANDS,
    )
    val_ds = RidgeSegmentationDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir =VAL_MASK_DIR,
        transform=get_val_transforms(IMG_SIZE),
        bands=BANDS,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # Sanity check one batch
    xb, yb = next(iter(train_loader))
    print("Batch images:", xb.shape, xb.dtype)
    print("Batch masks :", yb.shape, yb.dtype)

    # Model, optim, scaler
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_val = float("inf")
    with mlflow.start_run():
        mlflow.log_params({
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "bands": str(BANDS),
            "model": "UNet",
        })

        for epoch in range(1, EPOCHS + 1):
            tl = train_one_epoch(train_loader, model, optimizer, scaler)
            vl = validate(val_loader, model)

            print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tl:.4f} | val_loss={vl:.4f}")
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

            if vl < best_val:
                best_val = vl
                ckpt_path = os.path.join(CHECKPOINT_DIR, "unet_best.pth")
                torch.save({"state_dict": model.state_dict(), "val_loss": vl, "epoch": epoch}, ckpt_path)
                print(f" Saved best: {ckpt_path} (val_loss={vl:.4f})")
                mlflow.pytorch.log_model(model, artifact_path="best_model")

    print("\n Training Complete.")

if __name__ == "__main__":
    main()
