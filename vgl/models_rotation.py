# models_rotation.py
"""
Modified DiT model with continuous rotation angle conditioning.
Supports rotation angles in degrees [0, 360].
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


class RotationEmbedder(nn.Module):
    """
    Embeds rotation angle values into vector representations.
    Handles rotation angles in degrees with special circular encoding.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="circular_sinusoidal",
                 dropout_prob=0.0, null_angle=180.0, null_embedding_type="learnable",
                 text_table_path=None):
        super().__init__()
        self.embedding_type = embedding_type
        self.dropout_prob = dropout_prob
        self.null_angle = null_angle
        self.null_embedding_type = null_embedding_type
        
        # Learnable null embedding if needed
        if dropout_prob > 0 and null_embedding_type == "learnable":
            self.null_embedding = nn.Parameter(torch.randn(frequency_embedding_size))
        else:
            self.null_embedding = None
        
        if embedding_type == "circular_sinusoidal":
            # Use circular encoding for angles (sin and cos of angle)
            # This ensures 0° and 360° have the same representation
            pass
        elif embedding_type == "sinusoidal":
            # Standard sinusoidal encoding (treats angle as linear value)
            pass
        elif embedding_type == "linear":
            self.scalar_proj = nn.Linear(2, frequency_embedding_size)  # 2 for sin and cos
        elif embedding_type == "rotary":
            # RoPE-style encoding for rotation angles
            self.rope_base = nn.Parameter(torch.randn(frequency_embedding_size))
            half = frequency_embedding_size // 2
            inv_freq = torch.exp(
                -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        elif embedding_type == "text":
            # Frozen text-encoder conditioning: the angle selects a precomputed CLIP
            # embedding of a sentence describing it, so the model is conditioned the
            # way a text-to-image system is rather than on the number itself.  Mirrors
            # the radius implementation in models.py.  The table is fixed; only the
            # projection is learned.
            if text_table_path is None:
                raise ValueError("embedding_type='text' requires text_table_path")
            table = torch.load(text_table_path, map_location="cpu", weights_only=False)
            self.register_buffer("text_values", table["values"].float(), persistent=True)
            self.register_buffer("text_embeddings", table["embeddings"].float(), persistent=True)
            self.text_proj = nn.Linear(table["embeddings"].shape[1], frequency_embedding_size)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def text_lookup(self, values):
        """Nearest entry in the precomputed text-embedding table, then project."""
        idx = torch.argmin((values.reshape(-1, 1) - self.text_values.reshape(1, -1)).abs(), dim=1)
        return self.text_proj(self.text_embeddings[idx])

    def token_drop(self, angles, force_drop_ids=None):
        """
        Drops angle values to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(angles.shape[0], device=angles.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        angles = torch.where(drop_ids, self.null_angle, angles)
        return angles

    @staticmethod
    def circular_angle_embedding(angles_deg, dim):
        """
        Create circular embeddings for rotation angles.
        Uses sin and cos to ensure continuity at 0/360 degrees.
        """
        # Convert to radians
        angles_rad = angles_deg * (math.pi / 180.0)
        
        # Create embeddings using multiple frequencies
        embeddings = []
        for freq in range(1, dim // 2 + 1):
            embeddings.append(torch.sin(freq * angles_rad))
            embeddings.append(torch.cos(freq * angles_rad))
        
        # Stack and transpose to get (batch, dim)
        embedding = torch.stack(embeddings, dim=1)
        
        # Truncate if dim is odd
        if dim % 2:
            embedding = embedding[:, :dim]
        
        return embedding

    @staticmethod
    def angle_embedding(angles_deg, dim, max_period=10000):
        """
        Create standard sinusoidal embeddings for angles (treats as linear value).
        """
        # Normalize to [0, 1]
        normalized = angles_deg / 360.0
        
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=angles_deg.device)
        args = normalized[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def rope_embedding(self, angles: torch.Tensor) -> torch.Tensor:
        """RoPE-style encoding for rotation angles"""
        # Convert degrees to radians and normalize to [0, 2π)
        angles_rad = (angles % 360.0) * (math.pi / 180.0)
        
        B = angles_rad.size(0)
        D = self.frequency_embedding_size
        half = D // 2

        # Normalize base vector
        base = torch.nn.functional.normalize(self.rope_base, dim=0, eps=1e-8)
        base = base.to(angles_rad).expand(B, -1)

        freqs = self.inv_freq.to(angles_rad)
        theta = angles_rad[:, None] * freqs

        cos, sin = torch.cos(theta), torch.sin(theta)

        x_even = base[:, 0::2]
        x_odd  = base[:, 1::2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd  = x_even * sin + x_odd * cos

        out = torch.stack((rot_even, rot_odd), dim=-1).flatten(1)

        if D % 2:
            out = torch.cat((out, base[:, -1:]), dim=-1)

        return out

    def forward(self, angles, train=False, force_drop_ids=None):
        """
        angles: (B,) tensor of rotation angles in degrees
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            angles = self.token_drop(angles, force_drop_ids)
        
        # Ensure angles are in valid range [0, 360)
        angles = angles % 360.0
        
        # Check for null angles
        if use_dropout:
            is_null = (angles == self.null_angle)
        else:
            is_null = torch.zeros_like(angles, dtype=torch.bool)
        
        # Get embeddings based on type
        if self.embedding_type == "circular_sinusoidal":
            angle_freq = self.circular_angle_embedding(angles, self.frequency_embedding_size)
        elif self.embedding_type == "sinusoidal":
            angle_freq = self.angle_embedding(angles, self.frequency_embedding_size)
        elif self.embedding_type == "linear":
            # Use sin and cos as features
            angles_rad = angles * (math.pi / 180.0)
            sin_cos = torch.stack([torch.sin(angles_rad), torch.cos(angles_rad)], dim=1)
            angle_freq = self.scalar_proj(sin_cos)
        elif self.embedding_type == "text":
            angle_freq = self.text_lookup(angles)
        elif self.embedding_type == "rotary":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                angle_freq = self.rope_embedding(angles.float())
            if torch.isnan(angle_freq).any() or torch.isinf(angle_freq).any():
                print("Warning: NaN/Inf in RoPE output")
                angle_freq = torch.nan_to_num(angle_freq, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
        
        # Replace null embeddings if needed
        if use_dropout and is_null.any():
            if self.null_embedding_type == "zero":
                angle_freq[is_null] = 0.0
            elif self.null_embedding_type == "learnable" and self.null_embedding is not None:
                null_emb = self.null_embedding.unsqueeze(0).expand(is_null.sum(), -1)
                angle_freq[is_null] = null_emb
        
        # Pass through MLP
        angle_emb = self.mlp(angle_freq)
        
        return angle_emb


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


class DiT_Rotation(nn.Module):
    """
    Diffusion model with a Transformer backbone and continuous rotation angle conditioning.
    Uses circular encoding to ensure continuity at 0/360 degrees.
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
        rotation_embedding_type="circular_sinusoidal",
        rotation_text_table=None,  # path to a precomputed text-embedding table
        conditioning_method="concat",
        rotation_dropout_prob=0.0,
        null_angle=180.0,
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
        self.rotation_embedder = RotationEmbedder(
            hidden_size,
            embedding_type=rotation_embedding_type,
            text_table_path=rotation_text_table,
            dropout_prob=rotation_dropout_prob,
            null_angle=null_angle,
            null_embedding_type=null_embedding_type
        )
        num_patches = self.x_embedder.num_patches
        
        # Position embeddings for patches
        if conditioning_method == "concat":
            # Add 2 extra positions for time and rotation conditioning tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, hidden_size), requires_grad=False)
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
            # Add two extra learned positions for conditioning tokens (t, rotation)
            extra_tokens = torch.zeros(2, self.hidden_size)
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
        
        # Initialize rotation embedding MLP:
        nn.init.normal_(self.rotation_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.rotation_embedder.mlp[2].weight, std=0.02)

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

    def forward(self, x, t, rotation=None, **kwargs):
        """
        Forward pass of DiT with continuous rotation conditioning.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        rotation: (N,) tensor of rotation angles in degrees
        """
        # Handle both positional and keyword argument for rotation
        if rotation is None and 'rotation' in kwargs:
            rotation = kwargs['rotation']
        assert rotation is not None, "Rotation angle 'rotation' must be provided"
        
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Embed inputs
        x = self.x_embedder(x)  # (N, num_patches, D)
        t_emb = self.t_embedder(t)  # (N, D)
        rotation_emb = self.rotation_embedder(rotation, self.training)  # (N, D)
        
        if self.conditioning_method == "adaln":
            # AdaLN approach - combine time and rotation embeddings for conditioning
            x = x + self.pos_embed  # (N, T, D)
            c = t_emb + rotation_emb  # (N, D)
            
            for block in self.blocks:
                x = block(x, c)  # (N, T, D)
            
            x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
            
        else:  # concat
            # In-context conditioning approach with separate time and rotation tokens
            batch_size = x.shape[0]
            
            # Create two separate conditioning tokens
            t_token = t_emb.unsqueeze(1)  # (N, 1, D)
            rotation_token = rotation_emb.unsqueeze(1)  # (N, 1, D)
            
            # Concatenate conditioning tokens with image tokens
            x = torch.cat([t_token, rotation_token, x], dim=1)  # (N, 2 + num_patches, D)
            
            # Add position embeddings
            x = x + self.pos_embed  # (N, 2 + num_patches, D)
            
            # Pass through transformer blocks
            for block in self.blocks:
                x = block(x)  # (N, 2 + num_patches, D)
            
            # Remove conditioning tokens
            x = x[:, 2:, :]  # (N, num_patches, D)
            
            # Final layer
            x = self.final_layer(x)  # (N, num_patches, patch_size ** 2 * out_channels)
        
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, rotation, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # The input x should be the original batch size (not doubled)
        # rotation should be doubled: [conditional_angles, unconditional_angles]
        batch_size = x.size(0)
        
        # Double the input for conditional and unconditional
        x_doubled = torch.cat([x, x], dim=0)
        
        # Handle timesteps - duplicate for conditional and unconditional
        if t.size(0) == batch_size:
            t_doubled = torch.cat([t, t], dim=0)
        else:
            t_doubled = t
        
        # rotation should already be doubled with [conditional, unconditional] angles
        model_out = self.forward(x_doubled, t_doubled, rotation)
        
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


# Model configurations with rotation conditioning
def DiT_XL_2_rotation(**kwargs):
    return DiT_Rotation(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_2_rotation(**kwargs):
    return DiT_Rotation(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_B_2_rotation(**kwargs):
    return DiT_Rotation(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_S_2_rotation(**kwargs):
    return DiT_Rotation(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4_rotation(**kwargs):
    return DiT_Rotation(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_1_rotation(**kwargs):
    # 8x8 VAE latent at patch 1 gives 64 tokens, for the token-matched
    # pixel-versus-latent comparison.
    return DiT_Rotation(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)

def DiT_S_16_rotation(**kwargs):
    # 64x64 pixels at patch 16 gives 16 tokens, matching the token count of the
    # VAE-latent baseline (8x8 latent at patch 2).
    return DiT_Rotation(depth=12, hidden_size=384, patch_size=16, num_heads=6, **kwargs)


DiT_models_rotation = {
    'DiT-XL/2': DiT_XL_2_rotation,
    'DiT-L/2':  DiT_L_2_rotation,
    'DiT-B/2':  DiT_B_2_rotation,
    'DiT-S/2':  DiT_S_2_rotation,
    'DiT-S/4':  DiT_S_4_rotation,
    'DiT-S/1':  DiT_S_1_rotation,
    'DiT-S/16': DiT_S_16_rotation,
}
