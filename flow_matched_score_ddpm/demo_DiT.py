import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
import os
from PIL import Image
import numpy as np
from tiny_vit import TinyVit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1. MNIST dataset
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # [-1,1]
])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# ---------------------------
# 2. Model builders
# ---------------------------
def create_dit():
    return TinyVit(patch_size=2, emb_dim=64, depth=2, num_heads=2).to(device)

# ---------------------------
# 3. Models & optimizers
# ---------------------------
ddpm_model = create_dit()
ddpm_scheduler = DDPMScheduler(num_train_timesteps=50, beta_start=0.0001, beta_end=0.02)
ddpm_optimizer = torch.optim.Adam(ddpm_model.parameters(), lr=1e-3)

score_model = create_dit()
score_optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-3)

flow_model = create_dit()
flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-3)

# ---------------------------
# 4. Training (1 batch demo)
# ---------------------------
T = 1.0
sigma_max = 1.0
for images, _ in train_loader:
    images = images.to(device)
    batch_size = images.size(0)

    # ---- DDPM ----
    timesteps = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
    noise = torch.randn_like(images)
    noisy_images = ddpm_scheduler.add_noise(images, noise, timesteps)
    noise_pred = ddpm_model(noisy_images, timesteps)
    loss_ddpm = nn.MSELoss()(noise_pred, noise)
    ddpm_optimizer.zero_grad(); loss_ddpm.backward(); ddpm_optimizer.step()

    # ---- Score-based ----
    t = torch.rand(batch_size, device=device) * T
    sigma = t.view(-1,1,1,1) * sigma_max
    noise_s = torch.randn_like(images) * sigma
    x_t = images + noise_s
    target_score = -noise_s / (sigma ** 2 + 1e-5)
    pred_score = score_model(x_t, t)
    loss_score = nn.MSELoss()(pred_score, target_score)
    score_optimizer.zero_grad(); loss_score.backward(); score_optimizer.step()

    # ---- Flow Matching ----
    t_f = torch.rand(batch_size, device=device)
    x_t_f = (1-t_f).view(-1,1,1,1)*images + t_f.view(-1,1,1,1)*torch.randn_like(images)
    v_true = (torch.randn_like(images) - images)
    v_pred = flow_model(x_t_f, t_f)
    loss_flow = nn.MSELoss()(v_pred, v_true)
    flow_optimizer.zero_grad(); loss_flow.backward(); flow_optimizer.step()
    break  # 1 batch for demo

# ---------------------------
# 5. Sampling
# ---------------------------
steps = 50

# --- Score-based ---
x_score = torch.randn(8,1,28,28,device=device)
dt = T/steps
for i in range(steps):
    t_i = T - i*dt
    sigma = t_i*sigma_max
    t_tensor = torch.full((x_score.size(0),), t_i, device=device)
    s = score_model(x_score, t_tensor)
    x_score = x_score + dt * (s*sigma**2) + (dt**0.5)*torch.randn_like(x_score)*sigma
score_samples = x_score.cpu()

# --- Flow ---
x_flow = torch.randn(8,1,28,28,device=device)
for i in range(steps):
    t_f_i = torch.full((x_flow.size(0),), i*dt, device=device)
    v = flow_model(x_flow, t_f_i)
    x_flow = x_flow + dt*v
flow_samples = x_flow.cpu()

# ---------------------------
# 6. Save samples
# ---------------------------
def save_samples(samples, folder, prefix):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(samples):
        if isinstance(img, torch.Tensor):
            img_np = ((img.squeeze().detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        else:
            pil_img = img
        pil_img.save(os.path.join(folder,f"{prefix}_{i}.png"))

save_samples(score_samples, folder="samples_score_DiT", prefix="score")
save_samples(flow_samples, folder="samples_flow_DiT", prefix="flow")

# --- DDPM Sampling (manual) ---
batch_size = 8
ddpm_samples = torch.randn(batch_size, 1, 28, 28, device=device)  # start from noise
num_steps = ddpm_scheduler.config.num_train_timesteps

for t in reversed(range(num_steps)):
    for i in range(batch_size):
        t_scalar = torch.tensor(t, device=device, dtype=torch.long)
        noise_pred = ddpm_model(ddpm_samples[i:i+1], t_scalar)
        prev_sample, *_ = ddpm_scheduler.step(noise_pred, t_scalar, ddpm_samples[i:i+1], return_dict=False)
        ddpm_samples[i:i+1] = prev_sample
        
ddpm_samples = ddpm_samples.cpu()

save_samples(ddpm_samples, folder="samples_ddpm_DiT", prefix="ddpm")


print("✅ Samples saved!")
