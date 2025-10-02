import random
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms as T
from torchvision import datasets
from PIL import Image, ImageOps
import cv2
import numpy as np
import albumentations as A # type: ignore
from albumentations.pytorch import ToTensorV2 # type: ignore


def get_transforms():
    train_transforms = A.Compose([
        # A.Resize(224, 224),
        # Tăng khả năng zoom-in
        A.OneOf([
            A.RandomResizedCrop(size=(224, 224), scale=(0.3, 1.0), ratio=(0.75, 1.33), p=0.7), # zoom mạnh
            A.Resize(224, 224)
        ], p=1.0),
        # tăng tương phản / làm nét
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
            A.Sharpen(alpha=(0.2,0.5), lightness=(0.5,1.0), p=0.3),
        ], p=0.8),
        
        # Giả lập ảnh bé / mất nét
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ImageCompression(p=0.4),
        ], p=0.5),
        
        # occlusion + noise (bắt model học với mất mát thông tin)
        A.CoarseDropout(p=0.3),
        A.GaussNoise(p=0.3),
        
        A.RandomShadow(shadow_dimension=5, p=0.5),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_test_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transforms, val_test_transforms

class TailLightDataset(Dataset):
    def __init__(self, root_dir, transform=None, flip_transform=False):
        self.root_dir = root_dir
        self.transform = transform
        self.flip_transform = flip_transform
        self.dataset = datasets.ImageFolder(root=root_dir)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        flip = False
        if self.flip_transform:
            flip = random.choice([True, False])
            if flip:
                img = ImageOps.mirror(img)
                if label == 0:
                    label = 2
                elif label == 2:
                    label = 0
        if self.transform is not None:
            img = np.array(img)
            img = self.transform(image=img)['image']
        return img, label
    
def get_dataloaders(train_dir, val_dir, test_dir, batch_size=16):
    train_transforms, val_test_transforms = get_transforms()
    train_dataset = TailLightDataset(train_dir, transform=train_transforms, flip_transform=True)
    val_dataset = TailLightDataset(val_dir, transform=val_test_transforms)
    test_dataset = TailLightDataset(test_dir, transform=val_test_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_dataloader, val_dataloader, test_dataloader