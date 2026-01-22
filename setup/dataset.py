import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as Aug


class GoProDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        """
        GoPro Deblurring Dataset
        Args:
            root_dir: GOPRO_Large root directory
            mode: 'train' or 'test'
            transform: Albumentations transform
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.blur_paths = []
        self.sharp_paths = []
        
        mode_dir = os.path.join(root_dir, mode)
        sequence_dirs = sorted(os.listdir(mode_dir))
        
        for seq in sequence_dirs:
            blur_dir = os.path.join(mode_dir, seq, 'blur_gamma')
            sharp_dir = os.path.join(mode_dir, seq, 'sharp')
            if not os.path.isdir(blur_dir):
                continue
            
            blur_imgs = sorted(os.listdir(blur_dir))
            for img_name in blur_imgs:
                blur_path = os.path.join(blur_dir, img_name)
                sharp_path = os.path.join(sharp_dir, img_name)
                if os.path.exists(blur_path) and os.path.exists(sharp_path):
                    self.blur_paths.append(blur_path)
                    self.sharp_paths.append(sharp_path)
    
    def __len__(self):
        return len(self.blur_paths)
    
    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_paths[idx]).convert("RGB")
        sharp_img = Image.open(self.sharp_paths[idx]).convert("RGB")
        
        blur_np = np.array(blur_img)
        sharp_np = np.array(sharp_img)
        
        if self.transform is not None:
            augmented = self.transform(image=blur_np, target=sharp_np)
            blur_np = augmented["image"]
            sharp_np = augmented["target"]
        
        # Convert to tensor and normalize to [0, 1]
        blur_tensor = torch.from_numpy(blur_np).permute(2, 0, 1).float() / 255.0
        sharp_tensor = torch.from_numpy(sharp_np).permute(2, 0, 1).float() / 255.0
        
        return blur_tensor, sharp_tensor


def get_dataloaders(root_dir, batch_size=4, num_workers=4):
    """
    Create train and validation dataloaders
    """
    train_transform = Aug.Compose([
        Aug.RandomCrop(height=256, width=256, p=1.0),
        Aug.HorizontalFlip(p=0.5),
    ], additional_targets={"target": "image"}, p=1.0)
    
    val_transform = Aug.Compose([
        Aug.RandomCrop(height=256, width=256, p=1.0),
    ], additional_targets={"target": "image"}, p=1.0)
    
    train_dataset = GoProDataset(root_dir=root_dir, mode="train", transform=train_transform)
    val_dataset = GoProDataset(root_dir=root_dir, mode="test", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader