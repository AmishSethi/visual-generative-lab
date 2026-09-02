"""
UNet implementation based on the SongUNet architecture from the EDM paper,
adapted for continuous conditioning with radius/position/rotation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import silu
import math


# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


# ----------------------------------------------------------------------------
# Fully-connected layer.

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

class Conv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = F.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = F.conv2d(x, w, padding=w_pad+f_pad)
            x = F.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = F.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = F.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = F.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


# ----------------------------------------------------------------------------
# Group normalization.

class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = F.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


# ----------------------------------------------------------------------------
# Attention weight computation

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.

class UNetBlock(nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
        conditioning_method='adaln',  # Added for our conditioning
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.conditioning_method = conditioning_method

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        
        if conditioning_method == 'adaln':
            self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        if self.conditioning_method == 'adaln':
            params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = silu(self.norm1(x.add_(params)))
        else:
            # For concat conditioning, we just apply normalization without modulation
            x = silu(self.norm1(x))

        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ architecture.

class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


# ----------------------------------------------------------------------------
# Custom embedding for radius/position/rotation conditioning

class RadiusEmbedder(nn.Module):
    """Embeds continuous radius values into vector representations."""
    def __init__(self, hidden_size, embedding_type='sinusoidal', dropout_prob=0.0):
        super().__init__()
        self.embedding_type = embedding_type
        self.dropout_prob = dropout_prob
        
        if embedding_type == 'sinusoidal':
            self.embedder = PositionalEmbedding(hidden_size)
        elif embedding_type == 'linear':
            self.embedder = nn.Sequential(
                nn.Linear(1, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def forward(self, radius):
        """
        Args:
            radius: (N,) tensor of radius values
        Returns:
            (N, hidden_size) tensor of embeddings
        """
        # Handle CFG dropout
        if self.training and self.dropout_prob > 0:
            mask = torch.rand(radius.shape[0], device=radius.device) < self.dropout_prob
            radius = torch.where(mask, torch.ones_like(radius) * -1, radius)
        
        if self.embedding_type == 'sinusoidal':
            # Sinusoidal embeddings
            emb = self.embedder(radius)
        else:
            # Linear embeddings
            emb = self.embedder(radius.unsqueeze(-1))
        
        return emb


class PositionEmbedder(nn.Module):
    """Embeds (x,y) position coordinates into vector representations."""
    def __init__(self, hidden_size, embedding_type='sinusoidal', dropout_prob=0.0):
        super().__init__()
        self.embedding_type = embedding_type
        self.dropout_prob = dropout_prob
        self.hidden_size = hidden_size
        
        if embedding_type == 'sinusoidal':
            self.x_embedder = PositionalEmbedding(hidden_size // 2)
            self.y_embedder = PositionalEmbedding(hidden_size // 2)
        elif embedding_type == 'linear':
            self.x_embedder = nn.Sequential(
                nn.Linear(1, hidden_size // 2, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, hidden_size // 2, bias=True),
            )
            self.y_embedder = nn.Sequential(
                nn.Linear(1, hidden_size // 2, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size // 2, hidden_size // 2, bias=True),
            )
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def forward(self, pos):
        """
        Args:
            pos: (N, 2) tensor of (x, y) positions
        Returns:
            (N, hidden_size) tensor of embeddings
        """
        x = pos[:, 0]
        y = pos[:, 1]
        
        # Handle CFG dropout
        if self.training and self.dropout_prob > 0:
            mask = torch.rand(x.shape[0], device=x.device) < self.dropout_prob
            x = torch.where(mask, torch.ones_like(x) * -1, x)
            y = torch.where(mask, torch.ones_like(y) * -1, y)
        
        if self.embedding_type == 'sinusoidal':
            x_emb = self.x_embedder(x)
            y_emb = self.y_embedder(y)
        else:
            x_emb = self.x_embedder(x.unsqueeze(-1))
            y_emb = self.y_embedder(y.unsqueeze(-1))
        
        return torch.cat([x_emb, y_emb], dim=-1)


# ----------------------------------------------------------------------------
# Main SongUNet model adapted for continuous conditioning

class SongUNet(nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        
        # Conditioning parameters (our additions)
        conditioning_type='radius',         # 'radius', 'position', 'rotation', etc.
        conditioning_method='adaln',        # 'adaln' or 'concat'
        radius_embedding_type='sinusoidal', # 'sinusoidal' or 'linear'
        position_embedding_type='sinusoidal',
        rotation_embedding_type='sinusoidal',
        radius_dropout_prob=0.1,
        position_dropout_prob=0.1,
        rotation_dropout_prob=0.1,
        
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
        learn_sigma=False,                  # Whether to learn sigma (False matches original SongUNet defaults)
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.conditioning_type = conditioning_type
        self.conditioning_method = conditioning_method
        self.learn_sigma = learn_sigma
        
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
            conditioning_method=conditioning_method,
        )

        # Mapping for timestep
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        
        # Mapping for conditioning (radius/position/rotation)
        if conditioning_type == 'radius':
            self.cond_embedder = RadiusEmbedder(noise_channels, radius_embedding_type, radius_dropout_prob)
        elif conditioning_type == 'position':
            self.cond_embedder = PositionEmbedder(noise_channels, position_embedding_type, position_dropout_prob)
        elif conditioning_type == 'rotation':
            self.cond_embedder = RadiusEmbedder(noise_channels, rotation_embedding_type, rotation_dropout_prob)
        else:
            self.cond_embedder = None
        
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Adjust input channels for concat conditioning
        encoder_in_channels = in_channels
        if conditioning_method == 'concat':
            if conditioning_type == 'radius' or conditioning_type == 'rotation':
                encoder_in_channels = in_channels + 1  # Add 1 channel for radius/rotation
            elif conditioning_type == 'position':
                encoder_in_channels = in_channels + 2  # Add 2 channels for (x,y)

        # Encoder
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
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder
        self.dec = nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels * (2 if learn_sigma else 1), kernel=3, **init_zero)

    def forward(self, x, t, r=None, pos=None, rotation=None, **kwargs):
        """
        Forward pass with conditioning.
        Args:
            x: (N, C, H, W) tensor of spatial inputs
            t: (N,) tensor of diffusion timesteps
            r: (N,) tensor of radius values (optional)
            pos: (N, 2) tensor of position values (optional)
            rotation: (N,) tensor of rotation values (optional)
        """
        # Get the conditioning value based on type
        cond_value = None
        if self.conditioning_type == 'radius':
            cond_value = r if r is not None else kwargs.get('radius')
            assert cond_value is not None, "Radius conditioning required but not provided"
        elif self.conditioning_type == 'position':
            cond_value = pos if pos is not None else kwargs.get('pos')
            assert cond_value is not None, "Position conditioning required but not provided"
        elif self.conditioning_type == 'rotation':
            cond_value = rotation if rotation is not None else kwargs.get('rotation')
            assert cond_value is not None, "Rotation conditioning required but not provided"
        
        # Timestep embedding
        t_emb = self.map_noise(t)
        t_emb = t_emb.reshape(t_emb.shape[0], 2, -1).flip(1).reshape(*t_emb.shape)  # swap sin/cos
        
        # Add conditioning embedding
        if self.cond_embedder is not None:
            cond_emb = self.cond_embedder(cond_value)
            emb = t_emb + cond_emb
        else:
            emb = t_emb
        
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Handle concat conditioning
        if self.conditioning_method == 'concat':
            if self.conditioning_type == 'radius' or self.conditioning_type == 'rotation':
                # Add radius/rotation as an extra channel
                cond_channel = cond_value.view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
                x = torch.cat([x, cond_channel], dim=1)
            elif self.conditioning_type == 'position':
                # Add x and y as two extra channels
                x_channel = cond_value[:, 0].view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
                y_channel = cond_value[:, 1].view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
                x = torch.cat([x, x_channel, y_channel], dim=1)

        # Encoder
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        
        return aux


# ----------------------------------------------------------------------------
# Model constructors for different sizes

def SongUNet_XL(**kwargs):
    return SongUNet(
        model_channels=192,
        channel_mult=[1,2,3,4],
        num_blocks=3,
        **kwargs
    )

def SongUNet_L(**kwargs):
    return SongUNet(
        model_channels=192,
        channel_mult=[1,2,2,2],
        num_blocks=3,
        **kwargs
    )

def SongUNet_B(**kwargs):
    return SongUNet(
        model_channels=128,
        channel_mult=[1,2,2,2],
        num_blocks=4,
        **kwargs
    )

def SongUNet_S(**kwargs):
    return SongUNet(
        model_channels=128,
        channel_mult=[1,2,2],
        num_blocks=3,
        **kwargs
    )


# Model name mappings
SongUNet_models = {
    'SongUNet-XL': SongUNet_XL,
    'SongUNet-L': SongUNet_L,
    'SongUNet-B': SongUNet_B,
    'SongUNet-S': SongUNet_S,
}