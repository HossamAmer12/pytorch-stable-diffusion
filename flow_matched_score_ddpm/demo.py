# ---------------------------
# Install dependencies if needed
# ---------------------------
# !pip install torch torchvision diffusers matplotlib

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

import os
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. Load MNIST
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # scale to [-1,1]
])

mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# ---------------------------
# 2. Define tiny UNet
# ---------------------------
def create_unet():
    return UNet2DModel(
        sample_size=28,        # The input image size (MNIST = 28x28)
        in_channels=1,         # Number of input channels (MNIST is grayscale → 1 channel)
        out_channels=1,        # Number of output channels (we predict noise in the same shape → 1 channel)
        layers_per_block=1,    # How many ResNet layers per block (keep small for MNIST)
        block_out_channels=(32, 64),  # Feature map sizes at each block (like CNN filters, grows deeper)
        down_block_types=("DownBlock2D", "DownBlock2D"),  # Downsampling path (no attention for MNIST)
        up_block_types=("UpBlock2D", "UpBlock2D")         # Upsampling path to reconstruct
    ).to(device)


# ---------------------------
# 3. DDPM Setup
# ---------------------------
ddpm_model = create_unet()
ddpm_scheduler = DDPMScheduler(num_train_timesteps=50, beta_start=0.0001, beta_end=0.02)
ddpm_optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=1e-3)

# ---------------------------
# 4. Score-Based Setup
# ---------------------------
score_model = create_unet()
score_optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-3)
T = 1.0
sigma_max = 1.0

# ---------------------------
# 5. Flow Matching Setup
# ---------------------------
flow_model = create_unet()
flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)

# ---------------------------
# 6. Training (toy, 1 batch each)
# ---------------------------
for images, _ in train_loader:
    images = images.to(device)
    batch_size = images.size(0)

    # -------- DDPM Training --------
    timesteps = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(images)
    noisy_images = ddpm_scheduler.add_noise(images, noise, timesteps)
    noise_pred = ddpm_model(noisy_images, timesteps).sample
    loss_ddpm = nn.MSELoss()(noise_pred, noise)
    ddpm_optimizer.zero_grad(); loss_ddpm.backward(); ddpm_optimizer.step()
    
    # -------- Score-Based Training --------
    t = torch.rand(batch_size, device=device) * T
    sigma = t.view(-1,1,1,1) * sigma_max
    noise_s = torch.randn_like(images) * sigma
    x_t = images + noise_s
    target_score = -noise_s / (sigma ** 2 + 1e-5)
    pred_score = score_model(x_t, t).sample
    loss_score = nn.MSELoss()(pred_score, target_score)
    score_optimizer.zero_grad(); loss_score.backward(); score_optimizer.step()
    
    # -------- Flow Matching Training --------
    t_f = torch.rand(batch_size, device=device)
    x_t_f = (1-t_f).view(-1,1,1,1)*images + t_f.view(-1,1,1,1)*torch.randn_like(images)
    v_true = (torch.randn_like(images) - images)
    v_pred = flow_model(x_t_f, t_f).sample
    loss_flow = nn.MSELoss()(v_pred, v_true)
    flow_optimizer.zero_grad(); loss_flow.backward(); flow_optimizer.step()
    break  # Only one batch for demo

# ---------------------------
# 7. Sampling
# ---------------------------

# --- DDPM Sampling ---
# Correct DDPM sampling for small toy model
ddpm_pipeline = DDPMPipeline(unet=ddpm_model, scheduler=ddpm_scheduler).to(device)
with torch.no_grad():
    ddpm_samples = ddpm_pipeline(
        batch_size=8,
        generator=torch.manual_seed(0),
        num_inference_steps=50  # <= ddpm_scheduler.config.num_train_timesteps
    ).images


# --- Score-Based Sampling ---
steps = 50
x_score = torch.randn(8, 1, 28, 28, device=device)
dt = T / steps
for i in range(steps):
    t_i = T - i*dt
    sigma = t_i * sigma_max
    t_tensor = torch.full((x_score.size(0),), t_i, device=device)
    s = score_model(x_score, t_tensor).sample
    # x_score = x_score + dt * (s * sigma**2) + torch.sqrt(dt) * torch.randn_like(x_score) * sigma
    x_score = x_score + dt * (s * sigma**2) + (dt ** 0.5) * torch.randn_like(x_score) * sigma

score_samples = x_score.cpu()

# --- Flow Matching Sampling ---
x_flow = torch.randn(8, 1, 28, 28, device=device)
for i in range(steps):
    t_f_i = torch.full((x_flow.size(0),), i*dt, device=device)
    v = flow_model(x_flow, t_f_i).sample
    x_flow = x_flow + dt * v
flow_samples = x_flow.cpu()

# ---------------------------
# 8. Visualization
# ---------------------------
def save_samples(samples, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(samples):
        # Convert tensor to PIL image if needed
        if isinstance(img, torch.Tensor):
            # Detach, move to CPU, scale [-1,1] -> [0,255]
            img_np = ((img.squeeze().detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        else:
            pil_img = img  # already PIL
        pil_img.save(os.path.join(folder, f"{prefix}_{i}.png"))


save_samples(ddpm_samples, folder="samples_ddpm", prefix="ddpm")
save_samples(score_samples, folder="samples_score", prefix="score")
save_samples(flow_samples, folder="samples_flow", prefix="flow")

# show_samples(ddpm_samples, "DDPM Samples")
# show_samples(score_samples, "Score-Based Samples")
# show_samples(flow_samples, "Flow Matching Samples")

