from ridge_dataset import RidgeSegmentationDataset
from transforms import get_train_transforms  # assuming you have albumentations transforms

dataset = RidgeSegmentationDataset(
    image_dir="/content/drive/MyDrive/ridge_data/train/images",
    mask_dir="/content/drive/MyDrive/ridge_data/train/masks",
    transform=get_train_transforms(512)
)
