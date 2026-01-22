"""
Simple GoPro Deblurring Training Script
A example for training deblurring models on GoPro dataset
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from setup.dataset import get_dataloaders
from setup.trainer import train_model

from model.rt_focuser_model import RT_Focuser_Standard

def main():
    # Configuration
    DATA_ROOT = "/path/to/your/gopro/dataset"  # Change this to your dataset path
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    NUM_EPOCHS = 3000
    LEARNING_RATE = 1e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        root_dir=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    model = RT_Focuser_Standard()
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Train model
    best_psnr = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        save_dir='./Pretrained_Weights'
    )
    
    print(f"\nFinal best PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()