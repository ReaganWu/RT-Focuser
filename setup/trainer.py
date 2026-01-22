import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import skimage.metrics


def train_model(model, train_loader, val_loader, optimizer, lr_scheduler, 
                num_epochs=3000, device='cuda', save_dir='./Pretrained_Weights'):
    """
    Simple training loop for deblurring model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)
    
    model.to(device)
    best_psnr = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {save_dir}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for blur, sharp in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            blur = blur.to(device)
            sharp = sharp.to(device)
            
            optimizer.zero_grad()
            output = model(blur)
            output = torch.clamp(output, 0.0, 1.0)
            
            loss = F.mse_loss(output, sharp)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation phase
        val_loss, val_psnr, val_ssim = validate(model, val_loader, device)
        
        # Print metrics
        avg_train_loss = np.mean(train_losses)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"PSNR: {val_psnr:.2f} - "
              f"SSIM: {val_ssim:.4f} - "
              f"LR: {lr:.6f}")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Best model saved! PSNR: {val_psnr:.2f}")
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
        
        lr_scheduler.step()
    
    print(f"\nTraining completed! Best PSNR: {best_psnr:.2f}")
    return best_psnr


def validate(model, val_loader, device):
    """
    Validation function
    """
    model.eval()
    val_loss = 0
    psnr_total = 0
    ssim_total = 0
    total_images = 0
    
    with torch.no_grad():
        for blur, sharp in val_loader:
            blur = blur.to(device)
            sharp = sharp.to(device)
            
            output = model(blur)
            output = torch.clamp(output, 0.0, 1.0)
            
            loss = F.mse_loss(output, sharp)
            val_loss += loss.item()
            
            batch_size = output.shape[0]
            for i in range(batch_size):
                output_np = output[i].cpu().numpy().transpose(1, 2, 0)
                sharp_np = sharp[i].cpu().numpy().transpose(1, 2, 0)
                
                psnr = skimage.metrics.peak_signal_noise_ratio(sharp_np, output_np, data_range=1.0)
                ssim = skimage.metrics.structural_similarity(sharp_np, output_np, channel_axis=2, data_range=1.0)
                
                psnr_total += psnr
                ssim_total += ssim
            
            total_images += batch_size
    
    avg_loss = val_loss / len(val_loader)
    avg_psnr = psnr_total / total_images
    avg_ssim = ssim_total / total_images
    
    return avg_loss, avg_psnr, avg_ssim