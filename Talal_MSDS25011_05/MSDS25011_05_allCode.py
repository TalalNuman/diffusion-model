import os
import math
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid

# ---------------------------------------------------------
# 1. Dataset & DataLoader Definition
# ---------------------------------------------------------
class AnimalSubsetDataset(Dataset):
    """
    Custom Dataset to load exactly 20 images from 5 selected animal classes
    from the unzipped animal_data folder.
    """
    def __init__(self, data_path, classes, num_images_per_class=20, transform=None):
        self.data_path = data_path
        self.classes = classes
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(data_path, cls)
            if not os.path.isdir(cls_dir):
                print(f"[Warning] Directory {cls_dir} does not exist.")
                continue
            
            # Match standard image file formats
            valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            files = []
            for ext in valid_extensions:
                files.extend(glob.glob(os.path.join(cls_dir, ext)))
            
            # Sort to keep loading deterministic
            files = sorted(files)
            
            # Select exactly 20 images per class
            cls_files = files[:num_images_per_class]
            print(f"Loaded {len(cls_files)} images from class '{cls}'")
            
            for file_path in cls_files:
                self.image_paths.append(file_path)
                self.labels.append(idx)
                
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


# ---------------------------------------------------------
# 2. Diffusion Noise Schedule Setup
# ---------------------------------------------------------
T = 1000
# Linear variance schedule as per original DDPM paper
beta = torch.linspace(1e-4, 0.02, T)

# Useful intermediate coefficients
alpha = 1. - beta
alpha_cumprod = torch.cumprod(alpha, dim=0)
alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]], dim=0)

# Constants for the forward process
sqrt_alphas_cumprod = torch.sqrt(alpha_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alpha_cumprod)

# Constants for the backward process
posterior_variance = beta * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

def extract(a, t, x_shape):
    """
    Extract schedule values for a batch of indices.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ---------------------------------------------------------
# 3. Forward Diffusion Process
# ---------------------------------------------------------
def q_sample(x_0, t, noise=None):
    """
    Computes x_t directly from x_0 at timestep t using the closed-form formulation.
    DO NOT APPLY NOISE DIRECTLY OR ITERATIVELY TO PREVENT SLOW TRAINING AND GRADS ISSUES.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
        
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # x_t = sqrt(\bar{alpha}_t) * x_0 + sqrt(1 - \bar{alpha}_t) * epsilon
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise


# ---------------------------------------------------------
# 4. Model Architecture (U-Net with Timestep Embedding)
# ---------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embedding for representing timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """
    Double convolution layer with Group Normalization and timestep injection.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.relu1 = nn.SiLU()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.relu2 = nn.SiLU()

    def forward(self, x, t):
        h = self.relu1(self.norm1(self.conv1(x)))
        # Map time embedding to correct channels and add
        time_emb = self.time_mlp(t)
        time_emb = time_emb[..., None, None] # Broadcast over spatial dims
        h = h + time_emb
        h = self.relu2(self.norm2(self.conv2(h)))
        return h


class SimpleUNet(nn.Module):
    """
    A lightweight U-Net optimized for quick training and convergence on small image sizes.
    """
    def __init__(self, time_emb_dim=32):
        super().__init__()
        # Time MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Encoder (Downsample)
        self.down1 = ConvBlock(3, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = ConvBlock(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.mid = ConvBlock(128, 128, time_emb_dim)
        
        # Decoder (Upsample)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = ConvBlock(128 + 128, 64, time_emb_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv2 = ConvBlock(64 + 64, 64, time_emb_dim)
        
        self.out_conv = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Downsample
        x1 = self.down1(x, t_emb)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1, t_emb)
        p2 = self.pool2(x2)
        
        # Mid
        m = self.mid(p2, t_emb)
        
        # Upsample & Skip Connections
        u1 = self.up1(m)
        c1 = torch.cat((u1, x2), dim=1)
        h1 = self.up_conv1(c1, t_emb)
        
        u2 = self.up2(h1)
        c2 = torch.cat((u2, x1), dim=1)
        h2 = self.up_conv2(c2, t_emb)
        
        return self.out_conv(h2)


# ---------------------------------------------------------
# 5. Custom Loss Function
# ---------------------------------------------------------
class CustomLoss(nn.Module):
    """
    Custom L1 or L2 loss as required by instructions.
    Uses custom formulation instead of calling standard torch loss.
    """
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, pred_noise, true_noise):
        if self.loss_type == 'l2':
            # Mean Squared Error (MSE)
            return torch.mean((pred_noise - true_noise) ** 2)
        elif self.loss_type == 'l1':
            # Mean Absolute Error (MAE)
            return torch.mean(torch.abs(pred_noise - true_noise))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ---------------------------------------------------------
# 6. Sampling (Test Function)
# ---------------------------------------------------------
@torch.no_grad()
def p_sample(model, x, t, t_idx):
    """
    Samples x_{t-1} from x_t using the reverse process formula.
    """
    # Predict noise from current x_t and t
    predicted_noise = model(x, t)
    
    # Extract constants
    beta_t = extract(beta, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_alpha_t = torch.sqrt(extract(alpha, t, x.shape))
    
    # Equation 11 from the DDPM paper:
    # x_{t-1} = 1 / sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - \bar{alpha}_t)) * epsilon_theta)
    mean = (1.0 / sqrt_alpha_t) * (x - (beta_t / sqrt_one_minus_alphas_cumprod_t) * predicted_noise)
    
    if t_idx == 0:
        return mean
    else:
        # Add random noise for stochastic sampling steps
        noise = torch.randn_like(x)
        var_t = extract(posterior_variance, t, x.shape)
        return mean + torch.sqrt(var_t) * noise


@torch.no_grad()
def sample_images(model, n_samples, img_size, device):
    """
    Generates new images starting from pure Gaussian noise.
    """
    model.eval()
    x = torch.randn(n_samples, 3, img_size, img_size, device=device)
    
    # Reverse loop from T-1 down to 0
    for i in reversed(range(0, T)):
        t = torch.full((n_samples,), i, dtype=torch.long, device=device)
        x = p_sample(model, x, t, i)
        
    # Scale from [-1, 1] back to [0, 1] for saving/visualization
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
    return x


# ---------------------------------------------------------
# 7. Main Training Execution Loop
# ---------------------------------------------------------
def train(args):
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Ensure save directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Map to [-1, 1]
    ])
    
    # Setup dataset & loader
    selected_classes = ['Cat', 'Dog', 'Elephant', 'Lion', 'Panda']
    dataset = AnimalSubsetDataset(
        data_path=args.data_path,
        classes=selected_classes,
        num_images_per_class=20,
        transform=train_transform
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Total dataset size: {len(dataset)} | Total batches: {len(dataloader)}")
    
    # Setup model, loss, optimizer
    model = SimpleUNet(time_emb_dim=64).to(device)
    loss_fn = CustomLoss(loss_type=args.loss_type)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    losses = []
    
    print("\n--- Starting Diffusion Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x_0, _) in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # Sample random timesteps t uniformly
            t = torch.randint(0, T, (batch_size,), device=device).long()
            
            # Sample Gaussian noise
            noise = torch.randn_like(x_0)
            
            # Forward process: get x_t
            x_t = q_sample(x_0, t, noise)
            
            # Predict noise
            predicted_noise = model(x_t, t)
            
            # Custom loss
            loss = loss_fn(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Logs
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch}/{args.epochs}] | Custom Loss: {avg_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        # Periodic Generation / Visualization
        if epoch % 50 == 0 or epoch == args.epochs:
            model.eval()
            print(f"Saving samples and checkpoint at epoch {epoch}...")
            # Generate 10 sample images (2 for each class average style)
            samples = sample_images(model, n_samples=10, img_size=args.image_size, device=device)
            grid = make_grid(samples, nrow=5)
            save_image(grid, os.path.join(args.output_dir, f"samples_epoch_{epoch}.png"))
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, "diffusion_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, checkpoint_path)
            
    # Save Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Custom MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
    plt.close()
    
    print("\n--- Training Complete! ---")
    print(f"Final model checkpoint saved at: {os.path.join(args.output_dir, 'diffusion_model.pth')}")
    print(f"Loss curve saved at: {os.path.join(args.output_dir, 'loss_curve.png')}")


# ---------------------------------------------------------
# 8. Command line arguments
# ---------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train custom DDPM diffusion model on animal subset.")
    parser.add_argument('--data_path', type=str, default='animal_data', help='Path to unzipped dataset root')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training (fits 100 images evenly)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=64, help='Image resolution (width/height)')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2'], help='Custom loss type')
    parser.add_argument('--output_dir', type=str, default='saved_models', help='Directory to save outputs')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='Compute hardware device')
    
    args = parser.parse_args()
    
    # Check data path
    if not os.path.exists(args.data_path):
        print(f"Error: dataset path '{args.data_path}' not found. Please provide path using --data_path")
    else:
        train(args)
