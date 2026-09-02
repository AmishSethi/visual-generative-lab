"""
UNet-based Diffusion Models with Position Conditioning
Updated with improved block structure matching DDPM paper implementation
"""

import torch
import torch.nn as nn
import numpy as np
import math


# Import common blocks from base unet_models
from vgl.unet_models import (
    TimestepEmbedder, TimeEmbedding, NormActConv, 
    SelfAttentionBlock, Downsample, Upsample
)


class PositionEmbedder(nn.Module):
    def __init__(self, hidden_size, embedding_type='sinusoidal'):
        super().__init__()
        self.embedding_type = embedding_type
        self.hidden_size = hidden_size
        
        if embedding_type == 'sinusoidal':
            self.x_embedder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.y_embedder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        elif embedding_type == 'linear':
            self.x_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.y_embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def forward(self, x, y):
        # No normalization - keep raw position values like DiT
        # x and y are used directly without division by max_position_value
        
        if self.embedding_type == 'sinusoidal':
            # Create sinusoidal embeddings
            half_dim = self.hidden_size // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
            
            x_emb = x[:, None] * emb[None, :]
            x_emb = torch.cat((x_emb.sin(), x_emb.cos()), dim=-1)
            if self.hidden_size % 2 == 1:
                x_emb = torch.nn.functional.pad(x_emb, (0, 1), mode='constant')
            
            y_emb = y[:, None] * emb[None, :]
            y_emb = torch.cat((y_emb.sin(), y_emb.cos()), dim=-1)
            if self.hidden_size % 2 == 1:
                y_emb = torch.nn.functional.pad(y_emb, (0, 1), mode='constant')
        
        elif self.embedding_type == 'linear':
            x_emb = x.unsqueeze(-1)
            y_emb = y.unsqueeze(-1)
        
        x_emb = self.x_embedder(x_emb)
        y_emb = self.y_embedder(y_emb)
        
        return x_emb + y_emb  # Combine x and y embeddings


class DownC_Position(nn.Module):
    """Down-convolution block with position conditioning support."""
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_emb_dim=128,
                 pos_emb_dim=128,
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
        
        # Position embedding blocks
        if conditioning_method == 'adaln':
            self.pos_block = nn.ModuleList([
                TimeEmbedding(out_channels, pos_emb_dim) for _ in range(num_layers)
            ])
        elif conditioning_method == 'concat':
            self.position_proj = nn.Conv2d(pos_emb_dim, out_channels, kernel_size=1)
        
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
    
    def forward(self, x, t_emb, pos_emb=None):
        out = x
        
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            
            # Add position conditioning if provided
            if pos_emb is not None:
                if self.conditioning_method == 'adaln':
                    out = out + self.pos_block[i](pos_emb)[:, :, None, None]
                elif self.conditioning_method == 'concat':
                    B, C, H, W = out.shape
                    pos_spatial = pos_emb[:, :, None, None].expand(B, pos_emb.shape[1], H, W)
                    pos_proj = self.position_proj(pos_spatial)
                    out = out + pos_proj
            
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)
            
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
        
        # Downsampling
        out = self.down_block(out)
        
        return out


class MidC_Position(nn.Module):
    """Middle convolution block with position conditioning."""
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_emb_dim=128,
                 pos_emb_dim=128,
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
        
        # Position embedding blocks
        if conditioning_method == 'adaln':
            self.pos_block = nn.ModuleList([
                TimeEmbedding(out_channels, pos_emb_dim) for _ in range(num_layers + 1)
            ])
        elif conditioning_method == 'concat':
            self.position_proj = nn.Conv2d(pos_emb_dim, out_channels, kernel_size=1)
        
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
    
    def forward(self, x, t_emb, pos_emb=None):
        out = x
        
        # First Resnet Block
        resnet_input = out
        out = self.conv1[0](out)
        out = out + self.te_block[0](t_emb)[:, :, None, None]
        
        if pos_emb is not None:
            if self.conditioning_method == 'adaln':
                out = out + self.pos_block[0](pos_emb)[:, :, None, None]
            elif self.conditioning_method == 'concat':
                B, C, H, W = out.shape
                pos_spatial = pos_emb[:, :, None, None].expand(B, pos_emb.shape[1], H, W)
                pos_proj = self.position_proj(pos_spatial)
                out = out + pos_proj
        
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
            
            if pos_emb is not None:
                if self.conditioning_method == 'adaln':
                    out = out + self.pos_block[i+1](pos_emb)[:, :, None, None]
                elif self.conditioning_method == 'concat' and i == 0:
                    B, C, H, W = out.shape
                    pos_spatial = pos_emb[:, :, None, None].expand(B, pos_emb.shape[1], H, W)
                    pos_proj = self.position_proj(pos_spatial)
                    out = out + pos_proj
            
            out = self.conv2[i+1](out)
            out = out + self.res_block[i+1](resnet_input)
        
        return out


class UpC_Position(nn.Module):
    """Up-convolution block with position conditioning support."""
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 t_emb_dim=128,
                 pos_emb_dim=128,
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
        
        # Position embedding blocks
        if conditioning_method == 'adaln':
            self.pos_block = nn.ModuleList([
                TimeEmbedding(out_channels, pos_emb_dim) for _ in range(num_layers)
            ])
        elif conditioning_method == 'concat':
            self.position_proj = nn.Conv2d(pos_emb_dim, out_channels, kernel_size=1)
        
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
    
    def forward(self, x, down_out, t_emb, pos_emb=None):
        # Upsampling
        x = self.up_block(x)
        x = torch.cat([x, down_out], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            
            # Resnet Block
            out = self.conv1[i](out)
            out = out + self.te_block[i](t_emb)[:, :, None, None]
            
            # Add position conditioning if provided
            if pos_emb is not None:
                if self.conditioning_method == 'adaln':
                    out = out + self.pos_block[i](pos_emb)[:, :, None, None]
                elif self.conditioning_method == 'concat':
                    B, C, H, W = out.shape
                    pos_spatial = pos_emb[:, :, None, None].expand(B, pos_emb.shape[1], H, W)
                    pos_proj = self.position_proj(pos_spatial)
                    out = out + pos_proj
            
            out = self.conv2[i](out)
            out = out + self.res_block[i](resnet_input)
            
            # Self Attention
            out_attn = self.attn_block[i](out)
            out = out + out_attn
        
        return out


class UNet_Position(nn.Module):
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
        position_dim=256,
        position_embedding_type='sinusoidal',
        conditioning_method='adaln',
        learn_sigma=True,
        position_dropout_prob=None,
        class_dropout_prob=0.1,
        max_position_value=None,
        num_heads=8,
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels * 2 if learn_sigma else in_channels
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_sigma = learn_sigma
        if position_dropout_prob is None:
            position_dropout_prob = class_dropout_prob
        self.position_dropout_prob = position_dropout_prob
        self.class_dropout_prob = class_dropout_prob
        self.conditioning_method = conditioning_method
        self.max_position_value = max_position_value
        
        # Time and Position embedders
        self.time_embed = TimestepEmbedder(t_emb_dim, frequency_embedding_size=t_emb_dim)
        self.position_embedder = PositionEmbedder(
            position_dim, 
            embedding_type=position_embedding_type
        )
        
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
            DownC_Position(
                down_channels[i],
                down_channels[i+1],
                t_emb_dim,
                position_dim,
                num_downc_layers,
                down_sample[i],
                num_heads,
                conditioning_method
            ) for i in range(len(down_channels) - 1)
        ])
        
        # MidC Block
        self.mids = nn.ModuleList([
            MidC_Position(
                mid_channels[i],
                mid_channels[i+1],
                t_emb_dim,
                position_dim,
                num_midc_layers,
                num_heads,
                conditioning_method
            ) for i in range(len(mid_channels) - 1)
        ])
        
        # UpC Blocks
        self.ups = nn.ModuleList([
            UpC_Position(
                up_channels[i],
                up_channels[i+1],
                t_emb_dim,
                position_dim,
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

    def forward(self, x, t, pos=None, **kwargs):
        """
        Forward pass of UNet.
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        pos: (N, 2) tensor of (x,y) position coordinates
        """
        # Handle both positional and keyword argument for pos
        if pos is None and 'pos' in kwargs:
            pos = kwargs['pos']
        assert pos is not None, "Position coordinates 'pos' must be provided"
        
        # Extract x and y coordinates from pos tensor
        position_x = pos[:, 0]
        position_y = pos[:, 1]
        
        cfg_mask = kwargs.get('cfg_mask', None)
        if cfg_mask is None and self.training and self.position_dropout_prob > 0:
            cfg_mask = torch.rand(x.shape[0], device=x.device) < self.position_dropout_prob
        # Handle CFG dropout
        if cfg_mask is not None and self.training:
            position_x = torch.where(cfg_mask, torch.ones_like(position_x) * -1, position_x)
            position_y = torch.where(cfg_mask, torch.ones_like(position_y) * -1, position_y)
        
        # Get embeddings
        t_emb = self.time_embed(t)
        t_emb = self.t_proj(t_emb)
        
        # Handle CFG dropout indicator
        pos_x_processed = torch.where(position_x == -1, torch.zeros_like(position_x), position_x)
        pos_y_processed = torch.where(position_y == -1, torch.zeros_like(position_y), position_y)
        pos_emb = self.position_embedder(pos_x_processed, pos_y_processed)
        
        # For CFG dropout, zero out position embeddings
        if cfg_mask is not None:
            pos_emb = torch.where(cfg_mask[:, None], torch.zeros_like(pos_emb), pos_emb)
        
        # Initial convolution
        out = self.conv_in(x)
        
        # DownC outputs for skip connections
        down_outs = []
        
        # Downsampling
        for down in self.downs:
            down_outs.append(out)
            out = down(out, t_emb, pos_emb)
        
        # Middle processing
        for mid in self.mids:
            out = mid(out, t_emb, pos_emb)
        
        # Upsampling with skip connections
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb, pos_emb)
        
        # Final convolution
        out = self.conv_out(out)
        
        return out

    def forward_with_cfg(self, x, t, pos, cfg_scale):
        """Forward pass with classifier-free guidance."""
        half_batch = x.shape[0]
        combined = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        # Create unconditional positions
        uncond_pos = torch.ones_like(pos) * -1
        combined_pos = torch.cat([pos, uncond_pos], dim=0)
        
        model_out = self.forward(combined, combined_t, combined_pos)
        cond_out, uncond_out = model_out.chunk(2, dim=0)
        
        return uncond_out + cfg_scale * (cond_out - uncond_out)


def UNet_XL_Position(**kwargs):
    return UNet_Position(
        down_channels=[128, 256, 512, 1024],
        mid_channels=[1024, 1024, 512], 
        up_channels=[1024, 512, 256, 128],
        num_heads=16,
        t_emb_dim=512,
        position_dim=512,
        **kwargs
    )

def UNet_L_Position(**kwargs):
    return UNet_Position(
        down_channels=[128, 256, 384, 512],
        mid_channels=[512, 512, 384],
        up_channels=[512, 384, 256, 128],
        num_heads=12,
        t_emb_dim=384,
        position_dim=384,
        **kwargs
    )

def UNet_B_Position(**kwargs):
    return UNet_Position(
        down_channels=[64, 128, 256, 512],
        mid_channels=[512, 512, 256],
        up_channels=[512, 256, 128, 64],
        num_heads=8,
        t_emb_dim=256,
        position_dim=256,
        **kwargs
    )

def UNet_M_Position(**kwargs):
    return UNet_Position(
        down_channels=[48, 96, 192, 384],
        mid_channels=[384, 384, 192],
        up_channels=[384, 192, 96, 48],
        num_heads=4,
        t_emb_dim=192,
        position_dim=192,
        **kwargs
    )

def UNet_S_Position(**kwargs):
    return UNet_Position(
        down_channels=[32, 64, 128, 256],
        mid_channels=[256, 256, 128],
        up_channels=[256, 128, 64, 32],
        num_heads=4,
        t_emb_dim=128,
        position_dim=128,
        **kwargs
    )

def UNet_DiT_S2_matched_Position(**kwargs):
    """UNet sized to match DiT-S/2 parameter count (~22.2M)."""
    return UNet_Position(
        down_channels=[48, 96, 192, 352],
        mid_channels=[352, 352, 192],
        up_channels=[384, 192, 96, 48],
        num_heads=4,
        t_emb_dim=192,
        position_dim=192,
        **kwargs
    )


UNet_models_position = {
    'UNet-XL': UNet_XL_Position,
    'UNet-L': UNet_L_Position,
    'UNet-B': UNet_B_Position,
    'UNet-M': UNet_M_Position,
    'UNet-S': UNet_S_Position,
    'UNet-DiT-S2-matched': UNet_DiT_S2_matched_Position,
}
