"""
Modified SongUNet implementation with compositional conditioning support.
Based on the original SongUNet but extended to handle multiple properties simultaneously.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
import math

# Import the base classes from the original SongUNet implementation
from vgl.unet_models_song import (
    weight_init, Linear, Conv2d, GroupNorm, UNetBlock, 
    PositionalEmbedding, RadiusEmbedder, PositionEmbedder
)


class CompositionalSongUNet(nn.Module):
    """
    SongUNet with support for compositional conditioning on multiple properties.
    """
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        
        # Compositional conditioning parameters
        active_properties=['radius'],       # List of active properties: 'radius', 'position', 'shape', 'color', etc.
        conditioning_method='adaln',        # 'adaln' or 'concat'
        radius_embedding_type='sinusoidal',
        position_embedding_type='sinusoidal',
        property_dropout_prob=0.1,
        
        # Shape and color parameters (for categorical properties)
        num_shapes=4,
        num_colors=8,
        
        # Original SongUNet parameters
        model_channels=128,                 # Base multiplier for the number of channels.
        channel_mult=[1,2,2,2],            # Per-resolution multipliers for the number of channels.
        channel_mult_emb=4,                 # Multiplier for the dimensionality of the embedding vector.
        num_blocks=4,                       # Number of residual blocks per resolution.
        attn_resolutions=[16],              # List of resolutions with self-attention.
        dropout=0.10,                       # Dropout probability of intermediate activations.
        
        embedding_type='positional',        # Timestep embedding type: 'positional' for DDPM++
        channel_mult_noise=1,               # Timestep embedding size: 1 for DDPM++
        encoder_type='standard',            # Encoder architecture: 'standard' for DDPM++
        decoder_type='standard',            # Decoder architecture: 'standard' for DDPM++
        resample_filter=[1,1],              # Resampling filter: [1,1] for DDPM++
        learn_sigma=False,                  # Whether to learn sigma
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.active_properties = active_properties
        # Force AdaLN for color+shape (discrete) to improve conditioning; keep others as requested
        if set(active_properties) == set(['shape', 'color']) and conditioning_method != 'adaln':
            print('CompositionalSongUNet: Forcing conditioning_method="adaln" for color+shape experiment')
            self.conditioning_method = 'adaln'
        else:
            self.conditioning_method = conditioning_method
        self.learn_sigma = learn_sigma
        self.num_shapes = num_shapes
        self.num_colors = num_colors
        
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
            conditioning_method=self.conditioning_method,
        )

        # Mapping for timestep
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        
        # Create embedders for each active property
        self.property_embedders = nn.ModuleDict()
        
        if 'radius' in active_properties:
            self.property_embedders['radius'] = RadiusEmbedder(
                noise_channels, radius_embedding_type, property_dropout_prob
            )
        
        if 'position' in active_properties:
            self.property_embedders['position'] = PositionEmbedder(
                noise_channels, position_embedding_type, property_dropout_prob
            )
        
        if 'shape' in active_properties:
            # Categorical embedding for shapes
            self.property_embedders['shape'] = nn.Embedding(
                num_shapes + (1 if property_dropout_prob > 0 else 0), noise_channels
            )
        
        if 'color' in active_properties:
            # Categorical embedding for colors
            self.property_embedders['color'] = nn.Embedding(
                num_colors + (1 if property_dropout_prob > 0 else 0), noise_channels
            )
        
        # MLP layers for embedding processing
        self.map_layer0 = Linear(noise_channels, emb_channels, **init)
        self.map_layer1 = Linear(emb_channels, emb_channels, **init)

        # Calculate input channels for concat conditioning
        encoder_in_channels = in_channels
        self.concat_proj = nn.ModuleDict()
        # Only support raw-channel concat for backward-compatible experiments (e.g., radius+position)
        if self.conditioning_method == 'concat':
            extra_channels = 0
            if 'radius' in active_properties:
                extra_channels += 1
            if 'position' in active_properties:
                extra_channels += 2
            if 'shape' in active_properties:
                extra_channels += 1
            if 'color' in active_properties:
                extra_channels += 1
            encoder_in_channels = in_channels + extra_channels

        # Build encoder
        self.enc = nn.ModuleDict()
        cout = encoder_in_channels
        caux = encoder_in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(cout, cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(caux, caux, kernel=0, down=True, **init)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(caux, cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(caux, cout, kernel=3, down=True, **init_zero)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(cin, cout, attention=attn, **block_kwargs)

        # Build decoder
        self.dec = nn.ModuleDict()
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(cout, cout, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(cout, cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(cout, cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(cin, cout, attention=attn, **block_kwargs)
                cin = cout

        self.out_conv = Conv2d(cout, out_channels, kernel=3, **init_zero)

    def forward(self, x, t, radius=None, position=None, shape=None, color=None, **kwargs):
        """
        Forward pass with compositional conditioning.
        
        Args:
            x: (N, C, H, W) input images
            t: (N,) timesteps
            radius: (N,) radius values (if 'radius' in active_properties)
            position: (N, 2) position coordinates (if 'position' in active_properties)
            shape: (N,) shape indices (if 'shape' in active_properties)
            color: (N,) color indices (if 'color' in active_properties)
        """
        # Handle kwargs for backward compatibility
        if radius is None and 'radius' in kwargs:
            radius = kwargs['radius']
        if position is None and 'pos' in kwargs:
            position = kwargs['pos']
        if shape is None and 'shape' in kwargs:
            shape = kwargs['shape']
        if color is None and 'color' in kwargs:
            color = kwargs['color']
        
        # Timestep embedding
        t_emb = self.map_noise(t)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape(*t_emb.shape)  # swap sin/cos
        
        # Combine all property embeddings
        emb = t_emb
        
        if 'radius' in self.active_properties and radius is not None:
            radius_emb = self.property_embedders['radius'](radius)
            emb = emb + radius_emb
        
        if 'position' in self.active_properties and position is not None:
            pos_emb = self.property_embedders['position'](position)
            emb = emb + pos_emb
        
        if 'shape' in self.active_properties and shape is not None:
            shape_emb = self.property_embedders['shape'](shape)
            emb = emb + shape_emb
        
        if 'color' in self.active_properties and color is not None:
            color_emb = self.property_embedders['color'](color)
            emb = emb + color_emb
        
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Handle concat conditioning
        if self.conditioning_method == 'concat':
            H, W = x.shape[2], x.shape[3]
            concat_channels = []
            # Backward-compatible raw-channel concat
            if 'radius' in self.active_properties and radius is not None:
                radius_channel = radius.view(-1, 1, 1, 1).expand(-1, 1, H, W)
                concat_channels.append(radius_channel)
            if 'position' in self.active_properties and position is not None:
                x_channel = position[:, 0].view(-1, 1, 1, 1).expand(-1, 1, H, W)
                y_channel = position[:, 1].view(-1, 1, 1, 1).expand(-1, 1, H, W)
                concat_channels.extend([x_channel, y_channel])
            if 'shape' in self.active_properties and shape is not None:
                shape_channel = shape.float().view(-1, 1, 1, 1).expand(-1, 1, H, W)
                concat_channels.append(shape_channel)
            if 'color' in self.active_properties and color is not None:
                color_channel = color.float().view(-1, 1, 1, 1).expand(-1, 1, H, W)
                concat_channels.append(color_channel)
            if concat_channels:
                x = torch.cat([x] + concat_channels, dim=1)

        # Encoder
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = x + block(aux)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder
        for name, block in self.dec.items():
            if 'block' in name:
                x = block(torch.cat([x, skips.pop()], dim=1), emb)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)

        return self.out_conv(x)


# Model configurations
def CompositionalSongUNet_XL(**kwargs):
    return CompositionalSongUNet(
        model_channels=256,
        channel_mult=[1,2,2,2],
        num_blocks=4,
        **kwargs
    )

def CompositionalSongUNet_L(**kwargs):
    return CompositionalSongUNet(
        model_channels=192,
        channel_mult=[1,2,2,2],
        num_blocks=3,
        **kwargs
    )

def CompositionalSongUNet_B(**kwargs):
    return CompositionalSongUNet(
        model_channels=160,
        channel_mult=[1,2,2],
        num_blocks=3,
        **kwargs
    )

def CompositionalSongUNet_S(**kwargs):
    return CompositionalSongUNet(
        model_channels=128,
        channel_mult=[1,2,2],
        num_blocks=3,
        **kwargs
    )

# Model name mappings
CompositionalSongUNet_models = {
    'CompSongUNet-XL': CompositionalSongUNet_XL,
    'CompSongUNet-L': CompositionalSongUNet_L,
    'CompSongUNet-B': CompositionalSongUNet_B,
    'CompSongUNet-S': CompositionalSongUNet_S,
}
