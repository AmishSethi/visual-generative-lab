"""
UNet-based Diffusion Models with Continuous Radius Conditioning
Updated with improved block structure matching DDPM paper implementation
"""

import torch
import torch.nn as nn
import numpy as np
import math


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class RadiusEmbedder(nn.Module):
    def __init__(self, hidden_size, embedding_type='sinusoidal'):
        super().__init__()
        self.embedding_type = embedding_type
        self.hidden_size = hidden_size
        
        if embedding_type == 'sinusoidal':
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        elif embedding_type == 'linear':
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def forward(self, radius):
        if self.embedding_type == 'sinusoidal':
            half_dim = self.hidden_size // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=radius.device) * -emb)
            emb = radius[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            if self.hidden_size % 2 == 1:
                emb = torch.nn.functional.pad(emb, (0, 1), mode='constant')
        elif self.embedding_type == 'linear':
            emb = radius.unsqueeze(-1)
        
        return self.mlp(emb)


class TimeEmbedding(nn.Module):
    """Maps the Time/Radius Embedding to the Required output Dimension."""
    def __init__(self, n_out, t_emb_dim=128):
        super().__init__()
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, n_out)
        )
    
    def forward(self, x):
        return self.te_block(x)


class NormActConv(nn.Module):
    """Perform GroupNorm, Activation, and Convolution operations."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_groups=8, 
                 kernel_size=3, 
                 norm=True,
                 act=True):
        super().__init__()
        
        self.g_norm = nn.GroupNorm(
            num_groups,
            in_channels
        ) if norm else nn.Identity()
        
        self.act = nn.SiLU() if act else nn.Identity()
        
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size, 
            padding=(kernel_size - 1)//2
        )
    
    def forward(self, x):
        x = self.g_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x


class SelfAttentionBlock(nn.Module):
    """Perform GroupNorm and Multiheaded Self Attention operation."""
    def __init__(self, 
                 num_channels,
                 num_groups=8, 
                 num_heads=4,
                 norm=True):
        super().__init__()
        
        self.g_norm = nn.GroupNorm(
            num_groups,
            num_channels
        ) if norm else nn.Identity()
        
        self.attn = nn.MultiheadAttention(
            num_channels,
            num_heads, 
            batch_first=True
        )
    
    def forward(self, x):
        batch_size, channels, h, w = x.shape
        x = x.reshape(batch_size, channels, h*w)
        x = self.g_norm(x)
        x = x.transpose(1, 2)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).reshape(batch_size, channels, h, w)
        return x


class Downsample(nn.Module):
    """Perform Downsampling by the factor of k across Height and Width."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 k=2,
                 use_conv=True,
                 use_mpool=True):
        super().__init__()
        
        self.use_conv = use_conv
        self.use_mpool = use_mpool
        
        # Downsampling using Convolution
        self.cv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_mpool else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            )
        ) if use_conv else nn.Identity()
        
        # Downsampling using Maxpool
        self.mpool = nn.Sequential(
            nn.MaxPool2d(k, k), 
            nn.Conv2d(
                in_channels, 
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_mpool else nn.Identity()
    
    def forward(self, x):
        if not self.use_conv:
            return self.mpool(x)
        
        if not self.use_mpool:
            return self.cv(x)
            
        return torch.cat([self.cv(x), self.mpool(x)], dim=1)


class Upsample(nn.Module):
    """Perform Upsampling by the factor of k across Height and Width."""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 k=2,
                 use_conv=True,
                 use_upsample=True):
        super().__init__()
        
        self.use_conv = use_conv
        self.use_upsample = use_upsample
        
        # Upsampling using conv
        self.cv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels//2 if use_upsample else out_channels, 
                kernel_size=4, 
                stride=k, 
                padding=1
            ),
            nn.Conv2d(
                out_channels//2 if use_upsample else out_channels, 
                out_channels//2 if use_upsample else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_conv else nn.Identity()
        
        # Upsampling using nn.Upsample
        self.up = nn.Sequential(
            nn.Upsample(
                scale_factor=k, 
                mode='bilinear', 
                align_corners=False
            ),
            nn.Conv2d(
                in_channels,
                out_channels//2 if use_conv else out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        ) if use_upsample else nn.Identity()
    
    def forward(self, x):
        if not self.use_conv:
            return self.up(x)
        
        if not self.use_upsample:
            return self.cv(x)
        
        return torch.cat([self.cv(x), self.up(x)], dim=1)


class DownC(nn.Module):
    """
    Down-convolution block with radius conditioning support.
    1. Conv + TimeEmbedding + RadiusEmbedding
    2. Conv
    3. Skip-connection from input x
    4. Self-Attention
    5. Skip-Connection from 3
    6. Downsampling
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_emb_dim=128,
                 r_emb_dim=128,
                 num_layers=2,
                 down_sample=True,
                 num_heads=4,
                 conditioning_method='adaln'):
        super().__init__()
        
        self.num_layers = num_layers
        self.conditioning_method = conditioning_method
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                       out_channels) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                       out_channels) for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        # Radius embedding blocks
        if conditioning_method == 'adaln':
            self.re_block = nn.ModuleList([
                TimeEmbedding(out_channels, r_emb_dim) for _ in range(num_layers)
            ])
        elif conditioning_method == 'concat':
            self.radius_proj = nn.Conv2d(r_emb_dim, out_channels, kernel_size=1)
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        self.down_block = Downsample(out_channels, out_channels) if down_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
    
    def forward(self, x, t_emb, r_emb=None):
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            
            # Add radius conditioning if provided
            if r_emb is not None:
                if self.conditioning_method == 'adaln':
                    out = out + self.re_block[i](r_emb)[:, :, None, None]
                elif self.conditioning_method == 'concat':
                    B, C, H, W = out.shape
                    r_spatial = r_emb[:, :, None, None].expand(B, r_emb.shape[1], H, W)
                    r_proj = self.radius_proj(r_spatial)
                    out = out + r_proj
            
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)
            
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
        
        # Downsampling
        out = self.down_block(out)
        
        return out


class MidC(nn.Module):
    """
    Middle convolution block that refines features.
    1. Resnet Block with Time/Radius Embedding
    2. A Series of Self-Attention + Resnet Block with Time/Radius Embedding
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_emb_dim=128,
                 r_emb_dim=128,
                 num_layers=2,
                 num_heads=4,
                 conditioning_method='adaln'):
        super().__init__()
        
        self.num_layers = num_layers
        self.conditioning_method = conditioning_method
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                       out_channels) for i in range(num_layers + 1)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                       out_channels) for _ in range(num_layers + 1)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers + 1)
        ])
        
        # Radius embedding blocks
        if conditioning_method == 'adaln':
            self.re_block = nn.ModuleList([
                TimeEmbedding(out_channels, r_emb_dim) for _ in range(num_layers + 1)
            ])
        elif conditioning_method == 'concat':
            self.radius_proj = nn.Conv2d(r_emb_dim, out_channels, kernel_size=1)
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers + 1)
        ])
    
    def forward(self, x, t_emb, r_emb=None):
        out = x
        
        # First Resnet Block
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        
        if r_emb is not None:
            if self.conditioning_method == 'adaln':
                out = out + self.re_block[0](r_emb)[:, :, None, None]
            elif self.conditioning_method == 'concat':
                B, C, H, W = out.shape
                r_spatial = r_emb[:, :, None, None].expand(B, r_emb.shape[1], H, W)
                r_proj = self.radius_proj(r_spatial)
                out = out + r_proj
        
        out = self.conv2[0](out)
        out = out + self.res_block[0](resnet_input)
        
        # Sequence of Self-Attention + Resnet Blocks
        for i in range(self.num_layers):
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.conv1[i+1](out)
            out = out + self.te_block[i+1](t_emb)[:, :, None, None]
            
            if r_emb is not None:
                if self.conditioning_method == 'adaln':
                    out = out + self.re_block[i+1](r_emb)[:, :, None, None]
                elif self.conditioning_method == 'concat' and i == 0:  # Only add once
                    B, C, H, W = out.shape
                    r_spatial = r_emb[:, :, None, None].expand(B, r_emb.shape[1], H, W)
                    r_proj = self.radius_proj(r_spatial)
                    out = out + r_proj
            
            out = self.conv2[i+1](out)
            out = out + self.res_block[i+1](resnet_input)
        
        return out


class UpC(nn.Module):
    """
    Up-convolution block with radius conditioning support.
    1. Upsampling
    2. Conv + TimeEmbedding + RadiusEmbedding
    3. Conv
    4. Skip-connection from 1
    5. Self-Attention
    6. Skip-Connection from 3
    """
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_emb_dim=128,
                 r_emb_dim=128,
                 num_layers=2,
                 up_sample=True,
                 num_heads=4,
                 conditioning_method='adaln'):
        super().__init__()
        
        self.num_layers = num_layers
        self.conditioning_method = conditioning_method
        
        self.conv1 = nn.ModuleList([
            NormActConv(in_channels if i==0 else out_channels, 
                       out_channels) for i in range(num_layers)
        ])
        
        self.conv2 = nn.ModuleList([
            NormActConv(out_channels, 
                       out_channels) for _ in range(num_layers)
        ])
        
        self.te_block = nn.ModuleList([
            TimeEmbedding(out_channels, t_emb_dim) for _ in range(num_layers)
        ])
        
        # Radius embedding blocks
        if conditioning_method == 'adaln':
            self.re_block = nn.ModuleList([
                TimeEmbedding(out_channels, r_emb_dim) for _ in range(num_layers)
            ])
        elif conditioning_method == 'concat':
            self.radius_proj = nn.Conv2d(r_emb_dim, out_channels, kernel_size=1)
        
        self.attn_block = nn.ModuleList([
            SelfAttentionBlock(out_channels, num_heads=num_heads) for _ in range(num_layers)
        ])
        
        self.up_block = Upsample(in_channels, in_channels//2) if up_sample else nn.Identity()
        
        self.res_block = nn.ModuleList([
            nn.Conv2d(
                in_channels if i==0 else out_channels, 
                out_channels, 
                kernel_size=1
            ) for i in range(num_layers)
        ])
    
    def forward(self, x, down_out, t_emb, r_emb=None):
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            
            # Add radius conditioning if provided
            if r_emb is not None:
                if self.conditioning_method == 'adaln':
                    out = out + self.re_block[i](r_emb)[:, :, None, None]
                elif self.conditioning_method == 'concat':
                    B, C, H, W = out.shape
                    r_spatial = r_emb[:, :, None, None].expand(B, r_emb.shape[1], H, W)
                    r_proj = self.radius_proj(r_spatial)
                    out = out + r_proj
            
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)
            
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
        
        return out


class UNet_Continuous(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        out_channels=None,
        down_channels=[64, 128, 256, 512],
        mid_channels=[512, 512, 256],
        up_channels=[512, 256, 128, 64],
        down_sample=[True, True, False],
        num_downc_layers=2,
        num_midc_layers=2,
        num_upc_layers=2,
        t_emb_dim=256,
        radius_dim=256,
        radius_embedding_type='sinusoidal',
        conditioning_method='adaln',
        learn_sigma=True,
        radius_dropout_prob=0.1,
        null_radius=0.0,
        null_embedding_type='learnable',
        num_heads=8,
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels * 2 if learn_sigma else in_channels
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_sigma = learn_sigma
        self.radius_dropout_prob = radius_dropout_prob
        self.conditioning_method = conditioning_method
        
        # Time and Radius embedders
        self.time_embed = TimestepEmbedder(t_emb_dim, frequency_embedding_size=t_emb_dim)
        self.radius_embedder = RadiusEmbedder(radius_dim, embedding_type=radius_embedding_type)
        
        # Time projection
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, padding=1)
        
        self.up_sample = list(reversed(down_sample))
        
        # DownC Blocks
        self.downs = nn.ModuleList([
            DownC(
                down_channels[i],
                down_channels[i+1],
                t_emb_dim,
                radius_dim,
                num_downc_layers,
                down_sample[i],
                num_heads,
                conditioning_method
            ) for i in range(len(down_channels) - 1)
        ])
        
        # MidC Block
        self.mids = nn.ModuleList([
            MidC(
                mid_channels[i],
                mid_channels[i+1],
                t_emb_dim,
                radius_dim,
                num_midc_layers,
                num_heads,
                conditioning_method
            ) for i in range(len(mid_channels) - 1)
        ])
        
        # UpC Blocks
        self.ups = nn.ModuleList([
            UpC(
                up_channels[i],
                up_channels[i+1],
                t_emb_dim,
                radius_dim,
                num_upc_layers,
                self.up_sample[i],
                num_heads,
                conditioning_method
            ) for i in range(len(up_channels) - 1)
        ])
        
        # Final convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, up_channels[-1]),
            nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, padding=1)
        )
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)
        
        # Zero-out output layers
        nn.init.constant_(self.conv_out[-1].weight, 0)
        nn.init.constant_(self.conv_out[-1].bias, 0)

    def forward(self, x, t, r=None, **kwargs):
        """
        Forward pass of UNet.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of continuous radius values
        """
        # Handle both positional and keyword argument for radius
        if r is None and 'r' in kwargs:
            r = kwargs['r']
        # Also check for 'radius' key for backward compatibility
        if r is None and 'radius' in kwargs:
            r = kwargs['radius']
        assert r is not None, "Radius values 'r' must be provided"
        
        # Use 'radius' internally for consistency with the rest of the model
        radius = r
        
        cfg_mask = kwargs.get('cfg_mask', None)
        # Handle CFG dropout - exactly like in DiT
        if self.training and np.random.random() < self.radius_dropout_prob:
            # Create random CFG mask if not provided
            if cfg_mask is None:
                cfg_mask = torch.rand(x.shape[0], device=x.device) < self.radius_dropout_prob
        
        # Get embeddings
        t_emb = self.time_embed(t)
        t_emb = self.t_proj(t_emb)
        
        # Process radius embedding - no normalization, exactly like DiT
        r_emb = self.radius_embedder(radius)
        
        # Apply CFG masking if needed
        if cfg_mask is not None:
            # Zero out embeddings for CFG samples
            r_emb = torch.where(cfg_mask[:, None], torch.zeros_like(r_emb), r_emb)
        
        # Initial convolution
        out = self.conv_in(x)
        
        # DownC outputs for skip connections
        down_outs = []
        
        # Downsampling
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, r_emb)
        
        # Middle processing
        for mid in self.mids:
            out = mid(out, t_emb, r_emb)
        
        # Upsampling with skip connections
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, r_emb)
        
        # Final convolution
        out = self.conv_out(out)
        
        return out

    def forward_with_cfg(self, x, t, r, cfg_scale):
        """Forward pass with classifier-free guidance."""
        # For conditional generation
        cond_out = self.forward(x, t, r, cfg_mask=None)
        
        # For unconditional generation  
        cfg_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        uncond_out = self.forward(x, t, r, cfg_mask=cfg_mask)
        
        # Apply CFG
        return uncond_out + cfg_scale * (cond_out - uncond_out)


def UNet_XL(**kwargs):
    return UNet_Continuous(
        down_channels=[128, 256, 512, 1024],
        mid_channels=[1024, 1024, 512], 
        up_channels=[1024, 512, 256, 128],
        num_heads=16,
        t_emb_dim=512,
        radius_dim=512,
        **kwargs
    )

def UNet_L(**kwargs):
    return UNet_Continuous(
        down_channels=[128, 256, 384, 512],
        mid_channels=[512, 512, 384],
        up_channels=[512, 384, 256, 128],
        num_heads=12,
        t_emb_dim=384,
        radius_dim=384,
        **kwargs
    )

def UNet_B(**kwargs):
    return UNet_Continuous(
        down_channels=[64, 128, 256, 512],
        mid_channels=[512, 512, 256],
        up_channels=[512, 256, 128, 64],
        num_heads=8,
        t_emb_dim=256,
        radius_dim=256,
        **kwargs
    )

def UNet_M(**kwargs):
    return UNet_Continuous(
        down_channels=[48, 96, 192, 384],
        mid_channels=[384, 384, 192],
        up_channels=[384, 192, 96, 48],
        num_heads=4,
        t_emb_dim=192,
        radius_dim=192,
        **kwargs
    )

def UNet_S(**kwargs):
    return UNet_Continuous(
        down_channels=[32, 64, 128, 256],
        mid_channels=[256, 256, 128],
        up_channels=[256, 128, 64, 32],
        num_heads=4,
        t_emb_dim=128,
        radius_dim=128,
        **kwargs
    )

def UNet_DiT_S2_matched(**kwargs):
    """UNet sized to match DiT-S/2 parameter count (~22.2M)."""
    return UNet_Continuous(
        down_channels=[48, 96, 192, 352],
        mid_channels=[352, 352, 192],
        up_channels=[384, 192, 96, 48],
        num_heads=4,
        t_emb_dim=192,
        radius_dim=192,
        **kwargs
    )


UNet_models = {
    'UNet-XL': UNet_XL,
    'UNet-L': UNet_L,
    'UNet-B': UNet_B,
    'UNet-M': UNet_M,
    'UNet-S': UNet_S,
    'UNet-DiT-S2-matched': UNet_DiT_S2_matched,
}
