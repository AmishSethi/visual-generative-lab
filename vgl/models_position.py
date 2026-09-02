# models_position.py
"""
Modified DiT model with continuous position (x,y) conditioning.
Fixed to handle models trained without dropout (no CFG).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
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
        """
        Create sinusoidal timestep embeddings.
        """
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
    """
    Embeds radius values into vector representations. Also handles radius dropout for classifier-free guidance.
    This is the original embedder from models.py, used for individual coordinate embedding.
    Fixed to handle models without dropout properly.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="sinusoidal", 
                 max_period=10000, dropout_prob=0.0, null_radius=0.0, 
                 null_embedding_type="learnable"):
        super().__init__()
        self.max_period = max_period
        self.embedding_type = embedding_type
        self.dropout_prob = dropout_prob
        self.null_radius = null_radius
        self.null_embedding_type = null_embedding_type
        
        # Only create learnable null embedding if dropout is enabled
        if dropout_prob > 0 and null_embedding_type == "learnable":
            self.null_embedding = nn.Parameter(torch.randn(frequency_embedding_size))
        else:
            self.null_embedding = None
        
        if embedding_type == "sinusoidal":
            self.rope_base = None
        elif embedding_type == "rotary":
            self.rope_base = nn.Parameter(torch.randn(frequency_embedding_size))
            half = frequency_embedding_size // 2
            inv_freq = torch.exp(
                -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        elif embedding_type == "linear":
            self.scalar_proj = nn.Linear(1, frequency_embedding_size)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def token_drop(self, radius, force_drop_ids=None):
        """
        Drops radius values to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(radius.shape[0], device=radius.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        radius = torch.where(drop_ids, self.null_radius, radius)
        return radius

    @staticmethod
    def radius_embedding(r, dim, max_period=10000):
        """
        Create sinusoidal radius embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=r.device)
        args = r[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, r, train=False, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            r = self.token_drop(r, force_drop_ids)
        
        r = torch.clamp(r, min=-100.0, max=100.0)  # Allow negative values for coordinates
        
        if torch.isnan(r).any() or torch.isinf(r).any():
            print(f"Warning: NaN/Inf in coordinate input: {r}")
            r = torch.nan_to_num(r, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Only check for null coordinates if dropout is enabled
        if use_dropout:
            # Check for null coordinates
            is_null = (r == self.null_radius)
        else:
            # No null coordinates when dropout is disabled
            is_null = torch.zeros_like(r, dtype=torch.bool)
        
        # Get embeddings based on type
        if self.embedding_type == "sinusoidal":
            r_freq = self.radius_embedding(r, self.frequency_embedding_size)
        elif self.embedding_type == "rotary":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                r_freq = self.rope_embedding(r.float())
            if torch.isnan(r_freq).any() or torch.isinf(r_freq).any():
                print("Warning: NaN/Inf in RoPE output")
                r_freq = torch.nan_to_num(r_freq, nan=0.0, posinf=1.0, neginf=-1.0)
        elif self.embedding_type == "linear":
            r_expanded = r.unsqueeze(-1)
            r_freq = self.scalar_proj(r_expanded).squeeze(1)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        # Replace null embeddings with zero or learnable vector (only if dropout is enabled and nulls exist)
        if use_dropout and is_null.any():
            if self.null_embedding_type == "zero":
                r_freq[is_null] = 0.0
            elif self.null_embedding_type == "learnable" and self.null_embedding is not None:
                # Expand learnable embedding to batch dimension
                null_emb = self.null_embedding.unsqueeze(0).expand(is_null.sum(), -1)
                r_freq[is_null] = null_emb
        
        # Pass through MLP
        r_emb = self.mlp(r_freq)
        
        return r_emb
    
    def rope_embedding(self, radius: torch.Tensor) -> torch.Tensor:
        """RoPE-style encoding"""
        B = radius.size(0)
        D = self.frequency_embedding_size
        half = D // 2

        if isinstance(self.rope_base, nn.Parameter):
            base = F.normalize(self.rope_base, dim=0, eps=1e-8)
        else:                               
            base = self.rope_base
        base = base.to(radius).expand(B, -1)

        freqs = self.inv_freq.to(radius)
        theta = radius[:, None] * freqs

        cos, sin = torch.cos(theta), torch.sin(theta)

        x_even = base[:, 0::2]
        x_odd  = base[:, 1::2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd  = x_even * sin + x_odd * cos

        out = torch.stack((rot_even, rot_odd), dim=-1).flatten(1)

        if D % 2:
            out = torch.cat((out, base[:, -1:]), dim=-1)

        return out


class PositionEmbedder(nn.Module):
    """
    Embeds (x,y) position coordinates into separate vector representations for x and y tokens.
    Also handles position dropout for classifier-free guidance.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="sinusoidal", 
                 max_period=10000, dropout_prob=0.0, null_position=(0.0, 0.0), 
                 null_embedding_type="learnable"):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.null_position = null_position
        
        # Create separate embedders for x and y coordinates - each produces full hidden_size
        self.x_embedder = RadiusEmbedder(
            hidden_size=hidden_size,  # Full hidden size for each coordinate
            frequency_embedding_size=frequency_embedding_size,
            embedding_type=embedding_type,
            max_period=max_period,
            dropout_prob=dropout_prob,
            null_radius=null_position[0],
            null_embedding_type=null_embedding_type
        )
        
        self.y_embedder = RadiusEmbedder(
            hidden_size=hidden_size,  # Full hidden size for each coordinate
            frequency_embedding_size=frequency_embedding_size,
            embedding_type=embedding_type,
            max_period=max_period,
            dropout_prob=dropout_prob,
            null_radius=null_position[1],
            null_embedding_type=null_embedding_type
        )

    def forward(self, positions, train=False, force_drop_ids=None):
        """
        Forward pass for position embedding.
        positions: (N, 2) tensor of (x,y) coordinates
        Returns: tuple of (x_emb, y_emb) each of shape (N, hidden_size)
        """
        # Extract x and y coordinates
        x_coords = positions[:, 0]  # (N,)
        y_coords = positions[:, 1]  # (N,)
        
        # Embed x and y separately
        x_emb = self.x_embedder(x_coords, train=train, force_drop_ids=force_drop_ids)  # (N, hidden_size)
        y_emb = self.y_embedder(y_coords, train=train, force_drop_ids=force_drop_ids)  # (N, hidden_size)
        
        return x_emb, y_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class StandardDiTBlock(nn.Module):
    """
    Standard transformer block without conditioning (for concat method).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class StandardFinalLayer(nn.Module):
    """
    Standard final layer without conditioning (for concat method).
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class DiT_Position(nn.Module):
    """
    Diffusion model with a Transformer backbone and continuous position (x,y) conditioning.
    Uses separate continuous embedders for x and y coordinates.
    Supports both AdaLN and in-context (concatenation) conditioning.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        position_embedding_type="sinusoidal",
        conditioning_method="concat",  # Default to concat for position experiment
        position_dropout_prob=0.0,  # CFG dropout probability
        null_position=(0.0, 0.0),  # Null position for unconditional generation
        null_embedding_type="learnable"
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.conditioning_method = conditioning_method

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.pos_embedder = PositionEmbedder(
            hidden_size, 
            embedding_type=position_embedding_type, 
            dropout_prob=position_dropout_prob, 
            null_position=null_position,
            null_embedding_type=null_embedding_type
        )
        num_patches = self.x_embedder.num_patches
        
        # Position embeddings for patches
        if conditioning_method == "concat":
            # Add 3 extra positions for time, x, and y conditioning tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 3, hidden_size), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Transformer blocks
        if conditioning_method == "adaln":
            self.blocks = nn.ModuleList([
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
            self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        else:  # concat
            self.blocks = nn.ModuleList([
                StandardDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            ])
            self.final_layer = StandardFinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.conditioning_method == "concat":
            # Initialize image patch positions
            pos_embed = get_2d_sincos_pos_embed(self.hidden_size, int(self.x_embedder.num_patches ** 0.5))
            # Add three extra learned positions for conditioning tokens (t, x, y)
            extra_tokens = torch.zeros(3, self.hidden_size)
            full_pos_embed = np.concatenate([extra_tokens, pos_embed], axis=0)
            self.pos_embed.data.copy_(torch.from_numpy(full_pos_embed).float().unsqueeze(0))
        else:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Initialize position embedding MLPs:
        nn.init.normal_(self.pos_embedder.x_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.pos_embedder.x_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.pos_embedder.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.pos_embedder.y_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks (only for adaln):
        if self.conditioning_method == "adaln":
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        
        # Initialize final linear layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, pos=None, **kwargs):
        """
        Forward pass of DiT with continuous position conditioning.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        pos: (N, 2) tensor of (x,y) position coordinates
        """
        # Handle both positional and keyword argument for pos
        if pos is None and 'pos' in kwargs:
            pos = kwargs['pos']
        assert pos is not None, "Position coordinates 'pos' must be provided"
        
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Embed inputs
        x = self.x_embedder(x)  # (N, num_patches, D)
        t_emb = self.t_embedder(t)  # (N, D)
        x_emb, y_emb = self.pos_embedder(pos, self.training)  # (N, D), (N, D)
        
        if self.conditioning_method == "adaln":
            # AdaLN approach - combine x and y embeddings for conditioning
            x = x + self.pos_embed  # (N, T, D)
            c = t_emb + x_emb + y_emb  # (N, D)
            
            for block in self.blocks:
                x = block(x, c)  # (N, T, D)
            
            x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
            
        else:  # concat
            # In-context conditioning approach with separate x and y tokens
            batch_size = x.shape[0]
            
            # Create three separate conditioning tokens
            t_token = t_emb.unsqueeze(1)  # (N, 1, D)
            x_token = x_emb.unsqueeze(1)  # (N, 1, D)
            y_token = y_emb.unsqueeze(1)  # (N, 1, D)
            
            # Concatenate conditioning tokens with image tokens
            x = torch.cat([t_token, x_token, y_token, x], dim=1)  # (N, 3 + num_patches, D)
            
            # Add position embeddings
            x = x + self.pos_embed  # (N, 3 + num_patches, D)
            
            # Pass through transformer blocks
            for block in self.blocks:
                x = block(x)  # (N, 3 + num_patches, D)
            
            # Remove conditioning tokens
            x = x[:, 3:, :]  # (N, num_patches, D)
            
            # Final layer
            x = self.final_layer(x)  # (N, num_patches, patch_size ** 2 * out_channels)
        
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, pos, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # The input x should be the original batch size (not doubled)
        # pos should be doubled: [conditional_positions, unconditional_positions]
        batch_size = x.size(0)
        
        # Double the input for conditional and unconditional
        x_doubled = torch.cat([x, x], dim=0)
        
        # Handle timesteps - duplicate for conditional and unconditional
        if t.size(0) == batch_size:
            t_doubled = torch.cat([t, t], dim=0)
        else:
            t_doubled = t
        
        # pos should already be doubled with [conditional, unconditional] positions
        model_out = self.forward(x_doubled, t_doubled, pos)
        
        # Apply CFG
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, batch_size, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        # Handle the rest channels
        if rest.size(0) > batch_size:
            rest_cond, rest_uncond = torch.split(rest, batch_size, dim=0)
            return torch.cat([half_eps, rest_cond], dim=1)
        else:
            return torch.cat([half_eps, rest], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Model configurations with position conditioning
def DiT_XL_2_position(**kwargs):
    return DiT_Position(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_2_position(**kwargs):
    return DiT_Position(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_B_2_position(**kwargs):
    return DiT_Position(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_S_2_position(**kwargs):
    return DiT_Position(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4_position(**kwargs):
    return DiT_Position(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_1_position(**kwargs):
    # 8x8 VAE latent at patch 1 gives 64 tokens.
    return DiT_Position(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_S_16_position(**kwargs):
    # 64x64 pixels at patch 16 gives 16 tokens, matching the VAE-latent baseline
    # (8x8 latent at patch 2) so the two can be compared at equal sequence length.
    return DiT_Position(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)


DiT_models_position = {
    'DiT-XL/2': DiT_XL_2_position,
    'DiT-L/2':  DiT_L_2_position,
    'DiT-B/2':  DiT_B_2_position,
    'DiT-S/2':  DiT_S_2_position,
    'DiT-S/4':  DiT_S_4_position,
    'DiT-S/1':  DiT_S_1_position,
    'DiT-S/16': DiT_S_16_position,
}