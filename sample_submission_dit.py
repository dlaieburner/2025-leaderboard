#!pip install -Uqq torch torchvision gdown wandb tqdm matplotlib torch_ema safetensors

#| export  <-- export flags are an artifact of SolveIt; ignore them
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.utils.checkpoint')
warnings.filterwarnings('ignore', message='.*repr.*attribute.*Field.*')
warnings.filterwarnings('ignore', message='.*frozen.*attribute.*Field.*')

import os
import torch
from torch import nn, optim, utils
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomAffine, RandomErasing
import torchvision
from tqdm.auto import tqdm
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import wandb
import gdown
from safetensors.torch import load_file

# Adjust to whichever platform we're running on
platform = 'local'
dir_prefix = './'
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print("device =",device)

"""## Setup the VAE

### Define VAE Model
"""

#| export

class VAEResidualBlock(nn.Module):
    def __init__(self, channels, use_skip=True, use_bn=True, act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.use_skip, self.act = use_skip, act
    def forward(self, x):
        if self.use_skip: x0 = x
        out = self.act()(self.bn1(self.conv1(x)))
        #out = F.dropout(out, 0.4, training=self.training)
        out = self.bn2(self.conv2(out))
        if self.use_skip: out = out + x0
        return self.act()(out)

class ResNetVAEEncoderSpatial(nn.Module):
    "this shrinks down to a wee image for its latents, e.g. for MNIST: 1x28x28 -> 1x7x7 for two downsampling operations"
    def __init__(self, in_channels, latent_channels=1, base_channels=32, blocks_per_level=4, use_skips=True, use_bn=True,  act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(base_channels) if use_bn else nn.Identity()
        channels = [base_channels, base_channels*2, base_channels*4]# , base_channels*8]
        self.levels = nn.ModuleList([nn.ModuleList([VAEResidualBlock(ch, use_skips, use_bn, act=act) for _ in range(blocks_per_level)]) for ch in channels])
        self.transitions = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], 1, bias=not use_bn) for i in range(len(channels)-1) ])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_proj = nn.Conv2d(in_channels=channels[-1],  out_channels=2*latent_channels, kernel_size=1 ) # 1x1 conv
        self.act = act
    def forward(self, x, pad=False):
        if pad: x = F.pad(x, (2, 2, 2, 2)) # 28x28->32x32 to make sizes cleaner
        x = self.act()(self.bn1(self.conv1(x)))
        for i in range(len(self.levels)):
            if i > 0:  # shrink down
                x = F.avg_pool2d(x, 2)
                x = self.transitions[i-1](x)
            for block in self.levels[i]:
                x = block(x)
        x = self.channel_proj(x)
        mean, logvar = x.chunk(2, dim=1)  # mean and log variance
        return mean, logvar


class ResNetVAEDecoderSpatial(nn.Module):
    """this is just the mirror image of ResNetVAEEnoderSpatial"""
    def __init__(self, out_channels, latent_channels=1, base_channels=32, blocks_per_level=4, use_skips=True, use_bn=True,  act=nn.GELU):
        super().__init__()
        channels = [base_channels, base_channels*2, base_channels*4]# , base_channels*8][::-1]
        channels = channels[::-1] # reversed from encoder
        self.channels = channels
        self.channel_proj = nn.Conv2d(in_channels=latent_channels,  out_channels=channels[0], kernel_size=1 ) # 1x1 conv
        self.levels = nn.ModuleList([nn.ModuleList([VAEResidualBlock(ch, use_skips, use_bn, act=act) for _ in range(blocks_per_level)]) for ch in channels])
        self.transitions = nn.ModuleList([  nn.Conv2d(channels[i], channels[i+1], 1, bias=not use_bn) for i in range(len(channels)-1)])
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.act = act
    def forward(self, z):
        x = self.channel_proj(z)
        for i in range(len(self.levels)):
            for block in self.levels[i]:
                x = block(x)
            if i < len(self.levels) - 1:  # not last level
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x = self.transitions[i](x)
        x = self.final_conv(x)
        if x.shape[2:] != (28, 28):  x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        return x


class ResNetVAESpatial(nn.Module):
    """Main VAE class"""
    def __init__(self,
                 data_channels=1, # 1 channel for MNIST, 3 for CFAR10, etc.
                 act = nn.GELU,
                 ):
        super().__init__()
        self.encoder = ResNetVAEEncoderSpatial(data_channels, latent_channels=1, act=act)
        self.decoder = ResNetVAEDecoderSpatial(data_channels, latent_channels=1, act=act)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = torch.cat([mu, log_var], dim=1)      # this is unnecessary/redundant but our other Lesson code expects z
        z_hat = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        x_hat = self.decoder(z_hat)
        return z, x_hat, mu, log_var, z_hat

"""### Load VAE Weights"""

#| export
# @title load_vae code
#         import gdown
#         weights_url = 'https://drive.google.com/file/d/1d8dG3ArhpLvlrJ0jlngG2vMOwm_RC3i-/view?usp=drive_link'
#         gdown.download(weights_url, vae_checkpoint, quiet=False, fuzzy=True)
#     ckpt_dict = torch.load(vae_checkpoint, weights_only=True, map_location=device)
#     vae.load_state_dict(ckpt_dict['model'])
#     return vae

#vae = load_vae().to(device)

"""## Encode MNIST to Latents"""

def encode_dataset(model, dataset, batch_size=512):
    """Encode entire dataset into VAE latents (z, mu)"""

    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_latents = []
    all_labels = []
    must_flatten = None
    with torch.no_grad():
        for data, labels in tqdm(loader):
            x = data.to(device)
            # next bit is so it should work with linear layers or conv
            if must_flatten is None or must_flatten==False:
                try:
                    z = model.encoder(x)
                except RuntimeError:
                    z = model.encoder(x.view(x.size(0), -1))
                    must_flatten = True
            else: z = model.encoder(x.view(x.size(0), -1))
            mu, logvar = z
            all_latents.append(mu.cpu())
            all_labels.append(labels)

    return torch.cat(all_latents), torch.cat(all_labels)

#| export
def encode_mnist(vae, filename=None, batch_size=512):
    print("Acquiring train & test MNIST image datasets...")
    train_ds = MNIST(root='./data', train=True,  download=True, transform=ToTensor())
    test_ds  = MNIST(root='./data', train=False, download=True, transform=ToTensor())

    print(f"Encoding dataset to latents...")
    train_latents, train_labels = encode_dataset(vae, train_ds, batch_size=batch_size)
    test_latents, test_labels = encode_dataset(vae, test_ds, batch_size=batch_size)

    if filename is not None:
        print(f"Saving to {dir_prefix+filename} ...")
        torch.save({ 'train_z': train_latents,     'test_z': test_latents,
                     'train_labels': train_labels, 'test_labels': test_labels # don't need labels though
                     }, dir_prefix+filename)
    return train_latents, train_labels

latent_data_filename = 'latent_encodings_spatial.pt'

#if not os.path.exists(latent_data_filename):
    #if 'solveit' == platform: # there's a timeout we need to circumvent
        #import threading
        #thread = threading.Thread(target=encode_mnist, args=(vae, latent_data_filename))
        #thread.start()
    #else:
        #train_latents, train_labels = encode_mnist(vae, filename=latent_data_filename)
#
#print(f"{thread.is_alive() if 'thread' in locals() else 'No thread'}")
#
#"""## Load Encoded Data (for easy restarting)"""
#
## @title load_encoded_data code
#def load_encoded_data(filename):
    #if 'MyDrive' in filename:
        #from google.colab import drive
        #drive.mount('/content/drive')
    #data_dict = torch.load(filename, weights_only=True)
    #return data_dict
#
#data_dict = load_encoded_data('latent_encodings_spatial.pt')
#train_z, test_z = data_dict['train_z'], data_dict['test_z']
#train_z.shape, test_z.shape

# "Sample Smart": Fast Track to accelerating training convergence via a good initial guess:
#     measure the means & std's of each "pixel" in latent space:
#SAMPLE_SMART = False
#if SAMPLE_SMART:
    #latent_mean = train_z.mean(dim=0).to(device)  # shape: e.g.  [1,4,4] or 1,7,7
    #latent_std = 1.1*train_z.std(dim=0).to(device)
#else:
    #latent_mean = 0* train_z[0].to(device)
    #latent_std = 1.0 + 0* train_z[0].to(device)

"""## Define the Flow Model"""

class Config(dict):
    # endows dicts with "." attribute syntax for brevity/readability
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value

flow_config = Config({
    "batch_size": 1024, # 256, #power of 2 is fastest; 2048 causes OOM
    "epochs": 1000000,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "steps": 50,  # integration steps for inference
})

"""#### Oh: Define DataLoaders (using config)"""

#from torch.utils.data import TensorDataset, DataLoader
#
## Create datasets from the latent tensors
#train_latent_ds = TensorDataset(train_z, data_dict['train_labels'][:train_z.shape[0]])
#test_latent_ds = TensorDataset(test_z, data_dict['test_labels'])
#
## Create dataloaders with large batch size since latents are tiny
#train_latent_dl = DataLoader(train_latent_ds, batch_size=flow_config.batch_size, shuffle=True)
#test_latent_dl = DataLoader(test_latent_ds, batch_size=flow_config.batch_size, shuffle=False)

#print(f"Train batches: {len(train_latent_dl)}, Test batches: {len(test_latent_dl)}")
# print single latent size
#print(f"Latent size: {train_latent_ds[0][0].shape}")

"""## DiT Model spec"""

# code by Scott Hawley (heavily revised by hand after Claude.ai's initial gen)
def timestep_embedding(t, dim):
    import math
    half_dim = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim).to(t.device)
    args = t[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

def get_2d_rope(embed_dim: int, grid_size: int | tuple) -> tuple:
    """Generate 2D rotary position embeddings (RoPE)
    grid_size: int for square grids, or tuple (ngrid_h, ngrid_w) for rectangular
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D RoPE"
    grid_size = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size

    # Since it's 2D, we'll have both height-wise (_h) and width-wise (_w) quantities
    dim_h = dim_w = embed_dim // 2
    freq_h = 1.0 / (10000 ** (torch.arange(0, dim_h, 2).float() / dim_h))
    freq_w = 1.0 / (10000 ** (torch.arange(0, dim_w, 2).float() / dim_w))

    grid_h, grid_w = [torch.arange(g, dtype=torch.float32) for g in grid_size]
    pos_h = grid_h[:, None] * freq_h[None, :] # height-wise positional embeddings
    pos_w = grid_w[:, None] * freq_w[None, :] # width-wise positional embeddings
    pos_h_all = pos_h[None, :, None, :].expand(-1, -1, grid_size[1], -1)
    pos_w_all = pos_w[None, None, :, :].expand(-1, grid_size[0], -1, -1)
    pos_h_flat = pos_h_all.reshape(grid_size[0] * grid_size[1], dim_h // 2)
    pos_w_flat = pos_w_all.reshape(grid_size[0] * grid_size[1], dim_w // 2)

    cos = torch.cat([torch.cos(pos_h_flat), torch.cos(pos_w_flat)], dim=1)
    sin = torch.cat([torch.sin(pos_h_flat), torch.sin(pos_w_flat)], dim=1)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embedding by rotating pairs of dimensions"""
    x1, x2 = x[..., ::2], x[..., 1::2]  # split embedding into pairs: evens & odds (by index numbers)
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)  # interleave rotated pairs back to original shape


class PatchEmbed(nn.Module):
    """Convert image to sequence of patch embeddings via convolution
    For MNIST, using patch_size=2:
    Input: (B, 1, 28, 28)
    We don't actually ever make the "patches" explicitly.  Rather, we use the conv kernel which directly maps each (disjoint)
    2x2 region/patch of the image to a 256-dim embedding. There are 14x14 2x2 patches, yielding 14x14 256-dim embeddings.
    i.e., After conv with kernel=2, stride=2, new dimensions are (B, 256, 14, 14)
    After flatten(2): (B, 256, 196) - flattens the spatial dimensions: 14x14=196
    After transpose: (B, 196, 256) - 196 patches, each with 256-dim embedding
    """
    def __init__(self, patch_size=2, in_channels=1, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, n_patches, embed_dim)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)  # will be modulated
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        # AdaLN modulation - scale and shift for both norms
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size)  # scale1, shift1, scale2, shift2
        )

    def forward(self, x, c):  # c is time conditioning
        shift1, scale1, shift2, scale2 = self.adaLN(c).chunk(4, dim=1)

        # Modulated attention
        norm_x = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.attn(norm_x, norm_x, norm_x)[0]

        # Modulated MLP
        norm_x = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.mlp(norm_x)
        return x


class DiT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, patch_size=2,
                 # hidden_size=128, depth=4, num_heads=4, time_dim=64):
                 hidden_size=256, depth=8, num_heads=8, time_dim=128,
                 grad_checkpoint=False):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.patch_size, self.hidden_size, self.time_dim = patch_size, hidden_size, time_dim
        self.grad_checkpoint = grad_checkpoint

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)


    def unpatchify(self, x, img_h, img_w):
        B, nph, npw = x.shape[0], img_h // self.patch_size, img_w // self.patch_size
        x = x.reshape(B, nph, npw, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, self.out_channels, nph * self.patch_size, npw * self.patch_size)
        return x

    def forward(self, x, t):
        B, C, H, W = x.shape
        orig_h, orig_w = H, W  # Save original size
        # Pad to 8×8 (nearest multiple of patch_size=2)
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, H, W = x.shape

        # Time conditioning
        if t.dim() == 0:
            t = torch.full((x.shape[0],), t.item()).to(x.device)
        t = torch.full((x.shape[0],), t.item() if t.dim() == 0 else t[0]).to(x.device)
        t_embed = self.time_mlp(timestep_embedding(t, self.time_dim))  # reuse your function

        # RoPE (your caching logic)
        grid_size = (H // self.patch_size, W // self.patch_size)
        if (not hasattr(self, '_rope_cache') or self._rope_cache is None or
            self._rope_cache[0] != grid_size or self._rope_cache[1].device != x.device):
            cos, sin = get_2d_rope(self.hidden_size, grid_size)
            self._rope_cache = (grid_size, cos.to(x.device), sin.to(x.device))
        _, cos, sin = self._rope_cache

        x = self.patch_embed(x)
        x = apply_rope(x, cos, sin)

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t_embed, use_reentrant=False)
            else:
                x = block(x, t_embed)  # pass time conditioning

        x = self.norm(x)
        x = self.output_proj(x)
        x = self.unpatchify(x, H, W)
        x = x[:, :, :orig_h, :orig_w]
        return x  # NO sigmoid!


"""## Integration / Inference / Sampling"""

# @title Integration code, same as blog post
def warp_time(t, dt=None, s=.5):
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s<0 or s>1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t
    if dt:                           # warped time-step requested; use derivative
        return tw,  dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s)
    return tw

@torch.no_grad()
def fwd_euler_step(model, current_points, current_t, dt):
    velocity = model(current_points, current_t)
    return current_points + velocity * dt

def rk4_step(f, # function that takes (y,t) and returns dy/dt, i.e. velocity
             y, # current location
             t, # current t value
             dt, # requested time step size
             ):
    k1 =  f(y, t)
    k2 =  f(y + dt*k1/2, t + dt/2)
    k3 =  f(y + dt*k2/2, t + dt/2)
    k4 =  f(y + dt*k3, t + dt)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


@torch.no_grad()
def integrate_path(model, initial_points, step_fn=rk4_step, n_steps=100,
                   save_trajectories=False, warp_fn=warp_time, pbar=False):
    """this 'sampling' routine is primarily used for visualization."""
    device = next(model.parameters()).device
    current_points = initial_points.clone()
    ts =  torch.linspace(0,1,n_steps).to(device)
    if warp_fn: ts = warp_fn(ts)
    if save_trajectories: trajectories = [current_points]
    iterator = tqdm(range(len(ts)-1)) if pbar else range(len(ts)-1)
    for i in iterator:
        current_points = step_fn(model, current_points, ts[i], ts[i+1]-ts[i])
        if save_trajectories: trajectories.append(current_points)
    if save_trajectories: return current_points, torch.stack(trajectories).cpu()
    return current_points

generate_samples = integrate_path # alias for the probability




"""# Submission Class"""

class SubmissionInterface(nn.Module):
    """All teams must implement this for automated evaluation.
    When you subclass/implement these methods, replace the NotImplementedError."""

    def __init__(self):
        super().__init__() 
        import gdown 
        #--- REQUIRED INFO:
        self.info = {
            'team': 'sample-dit',  # REPLACE with your team name. This will be public
            'names': 'Dr. Hawley', # or single name. This will be kept private
        }
        #----

        # keep support for full auto-initialization:
        self.load_vae()
        self.load_flow_model()
        self.device = 'cpu' # we can change this later via .to()

    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.vae = ResNetVAESpatial()
        vae_weights_file = 'downloaded_vae.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1kPK3ZPadOUEfH8ZycrG3k27pl9-lGUeL/view?usp=sharing"
        gdown.download(safetensors_link, vae_weights_file, quiet=False, fuzzy=True)
        self.vae.load_state_dict(load_file(vae_weights_file))

    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.flow_model = DiT()
        flow_weights_file = 'downloaded_flow.safetensors'
        safetensors_link = "https://drive.google.com/file/d/1q9Iguf--2_MqUjsosS7iAGGzYtVfJ3UF/view?usp=sharing"
        gdown.download(safetensors_link, flow_weights_file, quiet=False, fuzzy=True)
        self.flow_model.load_state_dict(load_file(flow_weights_file))

    def generate_samples(self, n_samples:int) -> torch.Tensor:
        z0 = torch.randn([n_samples, 1, 7, 7]).to(self.device)
        z1 = integrate_path(self.flow_model, z0)
        gen_xhat = F.sigmoid(self.decode(z1).view(-1, 28, 28))
        return gen_xhat

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # if your vae has linear layers, flatten first
        # if your vae has conv layers, comment out next line
        #images = images.view(images.size(0), -1)
        with torch.no_grad():
            z, logvar = self.vae.encoder(images.to(self.device))
            mu = z  # return only first half (mu)
            return mu

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(latents)

    def to(self, device):
        self.device = device
        self.vae.to(self.device)
        self.flow_model.to(self.device)
        return self


# Sample usage:
# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# mysub = SubmissionInterface().to(device) # loads vae and flow models
# xhat_gen = mysub.generate_samples(n_samples=10, n_steps=100)
