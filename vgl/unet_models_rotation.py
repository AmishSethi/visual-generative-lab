"""
UNet-based Diffusion Models with Rotation Conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


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


class RotationEmbedder(nn.Module):
    def __init__(self, hidden_size, embedding_type='circular'):
        super().__init__()
        self.embedding_type = embedding_type
        self.hidden_size = hidden_size
        
        if embedding_type == 'circular':
            # Use sine and cosine to encode circular nature of rotation
            self.mlp = nn.Sequential(
                nn.Linear(2, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        elif embedding_type == 'sinusoidal':
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
    
    def forward(self, rotation):
        """
        rotation: angle in radians
        """
        if self.embedding_type == 'circular':
            # Encode rotation using sine and cosine to preserve circular nature
            emb = torch.stack([torch.sin(rotation), torch.cos(rotation)], dim=-1)
        elif self.embedding_type == 'sinusoidal':
            # Normalize rotation to [0, 1] for sinusoidal embedding
            rotation_normalized = rotation / (2 * math.pi)
            half_dim = self.hidden_size // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=rotation.device) * -emb)
            emb = rotation_normalized[:, None] * emb[None, :]
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            if self.hidden_size % 2 == 1:
                emb = torch.nn.functional.pad(emb, (0, 1), mode='constant')
        elif self.embedding_type == 'linear':
            # Normalize rotation to [0, 1]
            emb = (rotation / (2 * math.pi)).unsqueeze(-1)
        
        return self.mlp(emb)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, rotation_channels=None, 
                 conditioning_method='adaln', dropout=0.0):
        super().__init__()
        self.conditioning_method = conditioning_method
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels * 2)
        )
        
        if rotation_channels is not None:
            if conditioning_method == 'adaln':
                self.rotation_mlp = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(rotation_channels, out_channels * 2)
                )
            elif conditioning_method == 'concat':
                self.rotation_proj = nn.Conv2d(rotation_channels, out_channels, kernel_size=1)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, t, rot=None):
        h = x
        
        # First convolution
        h = self.conv1(h)
        h = self.norm1(h)
        
        # Time conditioning (always AdaLN-style)
        t_emb = self.time_mlp(t)
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        h = self.act(h)
        h = self.dropout(h)
        
        # Second convolution
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Rotation conditioning if provided
        if rot is not None:
            if self.conditioning_method == 'adaln':
                rot_emb = self.rotation_mlp(rot)
                scale, shift = rot_emb.chunk(2, dim=1)
                h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            elif self.conditioning_method == 'concat':
                # Add rotation as spatial feature
                B, C, H, W = h.shape
                rot_spatial = rot[:, :, None, None].expand(B, rot.shape[1], H, W)
                rot_proj = self.rotation_proj(rot_spatial)
                h = h + rot_proj
        
        h = self.act(h)
        h = self.dropout(h)
        
        # Residual connection
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, heads, HW, dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)
        
        return x + out


class UNet_Rotation(nn.Module):
    def __init__(
        self,
        input_size=32,
        in_channels=4,
        model_channels=128,
        out_channels=None,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(16, 8),
        dropout=0.0,
        rotation_dim=256,
        rotation_embedding_type='circular',
        conditioning_method='adaln',
        learn_sigma=True,
        class_dropout_prob=0.1,
        num_heads=8,
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels * 2 if learn_sigma else in_channels
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.learn_sigma = learn_sigma
        self.class_dropout_prob = class_dropout_prob
        self.conditioning_method = conditioning_method
        
        time_channels = model_channels * 4
        self.time_embed = TimestepEmbedder(time_channels, frequency_embedding_size=model_channels)
        
        self.rotation_embedder = RotationEmbedder(rotation_dim, embedding_type=rotation_embedding_type)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        resolution = input_size
        
        for level, mult in enumerate(channel_mult):
            ch_out = model_channels * mult
            
            for _ in range(num_res_blocks):
                block = ConvBlock(
                    ch, ch_out, time_channels, rotation_dim,
                    conditioning_method=conditioning_method,
                    dropout=dropout
                )
                self.down_blocks.append(block)
                
                if resolution in attention_resolutions:
                    self.down_blocks.append(AttentionBlock(ch_out, num_heads))
                
                ch = ch_out
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
                resolution //= 2
        
        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ConvBlock(ch, ch, time_channels, rotation_dim, 
                     conditioning_method=conditioning_method, dropout=dropout),
            AttentionBlock(ch, num_heads),
            ConvBlock(ch, ch, time_channels, rotation_dim,
                     conditioning_method=conditioning_method, dropout=dropout),
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            ch_out = model_channels * mult
            
            for _ in range(num_res_blocks):
                skip_ch = model_channels * mult
                
                block = ConvBlock(
                    ch + skip_ch, ch_out, time_channels, rotation_dim,
                    conditioning_method=conditioning_method,
                    dropout=dropout
                )
                self.up_blocks.append(block)
                
                if resolution in attention_resolutions:
                    self.up_blocks.append(AttentionBlock(ch_out, num_heads))
                
                ch = ch_out
            
            if level != 0:
                self.up_blocks.append(nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2))
                resolution *= 2
        
        # Output projection
        self.norm_out = nn.GroupNorm(8, ch)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        
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
        nn.init.constant_(self.conv_out.weight, 0)
        nn.init.constant_(self.conv_out.bias, 0)

    def forward(self, x, t, rotation=None, **kwargs):
        """
        Forward pass of UNet.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        rotation: (N,) tensor of rotation angles in radians
        """
        # Handle both positional and keyword argument for rotation
        if rotation is None and 'rotation' in kwargs:
            rotation = kwargs['rotation']
        assert rotation is not None, "Rotation angle 'rotation' must be provided"
        
        cfg_mask = kwargs.get('cfg_mask', None)
        # Handle CFG dropout
        if cfg_mask is not None and self.training:
            # For CFG dropout samples, set rotation to a special value
            rotation = torch.where(cfg_mask, torch.ones_like(rotation) * -999, rotation)
        
        # Get embeddings
        t_emb = self.time_embed(t)
        
        # Handle CFG dropout indicator
        rot_processed = torch.where(rotation == -999, torch.zeros_like(rotation), rotation)
        rot_emb = self.rotation_embedder(rot_processed)
        
        # For CFG dropout, zero out rotation embeddings
        if cfg_mask is not None:
            rot_emb = torch.where((rotation == -999)[:, None], torch.zeros_like(rot_emb), rot_emb)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling
        skips = []
        for block in self.down_blocks:
            if isinstance(block, ConvBlock):
                h = block(h, t_emb, rot_emb)
                skips.append(h)
            elif isinstance(block, AttentionBlock):
                h = block(h)
                skips[-1] = h  # Update the last skip connection
            else:  # Downsampling conv
                h = block(h)
        
        # Middle
        for block in self.middle_blocks:
            if isinstance(block, ConvBlock):
                h = block(h, t_emb, rot_emb)
            else:  # AttentionBlock
                h = block(h)
        
        # Upsampling
        for block in self.up_blocks:
            if isinstance(block, ConvBlock):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb, rot_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else:  # Upsampling conv
                h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        
        return h

    def forward_with_cfg(self, x, t, rotation, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half_batch = x.shape[0]
        combined = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        combined_rotation = torch.cat([rotation, torch.ones_like(rotation) * -999], dim=0)
        
        # Forward pass
        model_out = self.forward(combined, combined_t, combined_rotation)
        
        # Split conditional and unconditional outputs
        cond_out, uncond_out = model_out.chunk(2, dim=0)
        
        # Apply CFG
        return uncond_out + cfg_scale * (cond_out - uncond_out)


def UNet_XL_Rotation(**kwargs):
    return UNet_Rotation(model_channels=192, channel_mult=(1, 2, 3, 4), num_heads=8, **kwargs)

def UNet_L_Rotation(**kwargs):
    return UNet_Rotation(model_channels=160, channel_mult=(1, 2, 3, 4), num_heads=8, **kwargs)

def UNet_B_Rotation(**kwargs):
    return UNet_Rotation(model_channels=128, channel_mult=(1, 2, 4, 4), num_heads=8, **kwargs)

def UNet_M_Rotation(**kwargs):
    return UNet_Rotation(model_channels=72, channel_mult=(1, 2, 3, 4), num_heads=4, **kwargs)

def UNet_S_Rotation(**kwargs):
    return UNet_Rotation(model_channels=96, channel_mult=(1, 2, 3, 4), num_heads=4, **kwargs)

def UNet_DiT_S2_matched_Rotation(**kwargs):
    """UNet sized to match DiT-S/2 parameter count (~22.2M)."""
    return UNet_Rotation(model_channels=64, channel_mult=(1, 2, 4, 4), num_heads=4, **kwargs)


UNet_models_rotation = {
    'UNet-XL': UNet_XL_Rotation,
    'UNet-L': UNet_L_Rotation,
    'UNet-B': UNet_B_Rotation,
    'UNet-M': UNet_M_Rotation,
    'UNet-S': UNet_S_Rotation,
    'UNet-DiT-S2-matched': UNet_DiT_S2_matched_Rotation,
}
