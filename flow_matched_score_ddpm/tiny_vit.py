import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVit(nn.Module):
    def __init__(self, patch_size=2, emb_dim=64, depth=2, num_heads=2, img_size=28):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding (grayscale → emb_dim)
        self.patch_embed = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, emb_dim))

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Map back to pixels
        self.to_out = nn.Linear(emb_dim, patch_size * patch_size)

        # ---- diffusers compatibility ----
        class Config:
            sample_size = img_size
            in_channels = 1
            out_channels = 1
        self.config = Config()

    def forward(self, x, t=None):
        b = x.size(0)
        device = next(self.parameters()).device

        # Patchify
        patches = self.patch_embed(x)              # (B, emb_dim, H', W')
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, emb_dim)

        # Add position embedding
        patches = patches + self.pos_embed

        # Add timestep embedding if given
        if t is not None:
            t = t.view(-1, 1).float().to(device)
            t_emb = self.time_mlp(t)               # (B, emb_dim)
            patches = patches + t_emb.unsqueeze(1)

        # Transformer
        patches = self.transformer(patches)        # (B, num_patches, emb_dim)

        # Reassemble image
        out = self.to_out(patches)                 # (B, num_patches, p^2)
        h = w = int(x.size(-1) / self.patch_size)
        out = out.transpose(1, 2).reshape(b, 1, h*self.patch_size, w*self.patch_size)
        return out

    # diffusers-like attributes
    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device


# ----------------
# ✅ Usage Example
# ----------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TinyVit().to(device)

# x = torch.randn(8, 1, 28, 28, device=device)  # batch of MNIST-like images
# out = model(x)  # (8, 1, 28, 28)

# print(out.shape)
