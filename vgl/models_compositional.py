# models_compositional.py
"""
DiT model with compositional conditioning on radius, position, shape, and color.
Supports both AdaLN and in-context (concatenation) conditioning.
Only initializes embedders for active properties.
Updated: Count now uses linear embedder for extrapolation.
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
    """Embeds scalar timesteps into vector representations."""
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
        """Create sinusoidal timestep embeddings."""
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


class PropertyEmbedder(nn.Module):
    """Base class for property embedders with dropout for classifier-free guidance."""
    def __init__(self, hidden_size, frequency_embedding_size=256, dropout_prob=0.0, 
                 null_embedding_type="learnable"):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.null_embedding_type = null_embedding_type
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size
        
        # Learnable null embedding (before MLP)
        if null_embedding_type == "learnable" and dropout_prob > 0:
            self.null_embedding = nn.Parameter(torch.randn(frequency_embedding_size))
        else:
            self.null_embedding = None

    def token_drop(self, batch_size, device, force_drop_ids=None):
        """Determine which samples to drop for classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(batch_size, device=device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return drop_ids


class RadiusEmbedder(PropertyEmbedder):
    """Embeds radius values into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="sinusoidal", 
                 max_period=10000, dropout_prob=0.0, null_radius=0.0, 
                 null_embedding_type="learnable"):
        super().__init__(hidden_size, frequency_embedding_size, dropout_prob, null_embedding_type)
        self.max_period = max_period
        self.embedding_type = embedding_type
        self.null_radius = null_radius
        
        if embedding_type == "sinusoidal":
            pass  # No additional parameters
        elif embedding_type == "linear":
            self.scalar_proj = nn.Linear(1, frequency_embedding_size)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def radius_embedding(r, dim, max_period=10000):
        """Create sinusoidal radius embeddings."""
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
        # Handle dropout
        drop_ids = self.token_drop(r.shape[0], r.device, force_drop_ids) if (train and self.dropout_prob > 0) or (force_drop_ids is not None) else torch.zeros(r.shape[0], device=r.device, dtype=torch.bool)
        
        # Replace dropped values with null
        r = torch.where(drop_ids, self.null_radius, r)
        r = torch.clamp(r, min=0.0, max=100.0)
        
        # Get embeddings
        if self.embedding_type == "sinusoidal":
            r_freq = self.radius_embedding(r, self.frequency_embedding_size)
        elif self.embedding_type == "linear":
            r_expanded = r.unsqueeze(-1)
            r_freq = self.scalar_proj(r_expanded).squeeze(-1)
        
        # Replace null embeddings
        if drop_ids.any() and self.null_embedding is not None:
            null_emb = self.null_embedding.unsqueeze(0).expand(drop_ids.sum(), -1)
            r_freq[drop_ids] = null_emb
        elif drop_ids.any() and self.null_embedding_type == "zero":
            r_freq[drop_ids] = 0.0
        
        return self.mlp(r_freq)


class PositionEmbedder(PropertyEmbedder):
    """Embeds (x,y) position coordinates into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="sinusoidal", 
                 max_period=10000, dropout_prob=0.0, null_position=(0.0, 0.0), 
                 null_embedding_type="learnable"):
        super().__init__(hidden_size, frequency_embedding_size, dropout_prob, null_embedding_type)
        self.embedding_type = embedding_type
        self.null_position = null_position
        
        if embedding_type == "sinusoidal":
            pass  # No additional parameters
        elif embedding_type == "linear":
            self.coord_proj = nn.Linear(2, frequency_embedding_size)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def position_embedding(self, pos, dim, max_period=10000):
        """Create sinusoidal position embeddings for (x,y) coordinates."""
        # Split embedding dimension between x and y
        half_dim = dim // 2
        
        # Embed x coordinate
        x_embedding = RadiusEmbedder.radius_embedding(pos[:, 0], half_dim, max_period)
        # Embed y coordinate  
        y_embedding = RadiusEmbedder.radius_embedding(pos[:, 1], half_dim, max_period)
        
        # Concatenate x and y embeddings
        embedding = torch.cat([x_embedding, y_embedding], dim=-1)
        
        # Handle odd dimensions
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding

    def forward(self, pos, train=False, force_drop_ids=None):
        # Handle dropout
        drop_ids = self.token_drop(pos.shape[0], pos.device, force_drop_ids) if (train and self.dropout_prob > 0) or (force_drop_ids is not None) else torch.zeros(pos.shape[0], device=pos.device, dtype=torch.bool)
        
        # Replace dropped values with null
        null_pos = torch.tensor(self.null_position, device=pos.device, dtype=pos.dtype).unsqueeze(0)
        pos = torch.where(drop_ids.unsqueeze(-1), null_pos, pos)
        pos = torch.clamp(pos, min=-100.0, max=100.0)
        
        # Get embeddings
        if self.embedding_type == "sinusoidal":
            pos_freq = self.position_embedding(pos, self.frequency_embedding_size)
        elif self.embedding_type == "linear":
            pos_freq = self.coord_proj(pos)
        
        # Replace null embeddings
        if drop_ids.any() and self.null_embedding is not None:
            null_emb = self.null_embedding.unsqueeze(0).expand(drop_ids.sum(), -1)
            pos_freq[drop_ids] = null_emb
        elif drop_ids.any() and self.null_embedding_type == "zero":
            pos_freq[drop_ids] = 0.0
        
        return self.mlp(pos_freq)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Based on the original DiT implementation.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class ShapeEmbedder(LabelEmbedder):
    """
    Embeds categorical shape values into vector representations.
    Uses shape names like "circle", "square", "triangle", "diamond".
    """
    def __init__(self, hidden_size, num_shapes=4, dropout_prob=0.0):
        super().__init__(num_shapes, hidden_size, dropout_prob)
        self.num_shapes = num_shapes


class ColorEmbedder(LabelEmbedder):
    """
    Embeds categorical color values into vector representations.
    Uses color names like "red", "blue", "green", etc.
    """
    def __init__(self, hidden_size, num_colors=8, dropout_prob=0.0):
        super().__init__(num_colors, hidden_size, dropout_prob)
        self.num_colors = num_colors


class CountEmbedder(PropertyEmbedder):
    """
    Embeds object count values using linear embedding for extrapolation.
    Changed from LabelEmbedder to PropertyEmbedder to enable count extrapolation.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="linear",
                 max_period=10, dropout_prob=0.0, null_count=1,
                 null_embedding_type="learnable"):
        super().__init__(hidden_size, frequency_embedding_size, dropout_prob, null_embedding_type)
        self.max_period = max_period
        self.embedding_type = embedding_type
        self.null_count = null_count
        
        if embedding_type == "sinusoidal":
            pass  # No additional parameters
        elif embedding_type == "linear":
            self.count_proj = nn.Linear(1, frequency_embedding_size)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def count_embedding(self, count, dim, max_period=10):
        """Create sinusoidal count embeddings."""
        # Treat count as a continuous value for embedding
        return RadiusEmbedder.radius_embedding(count.float(), dim, max_period)
    
    def forward(self, count, train=False, force_drop_ids=None):
        # Handle dropout
        drop_ids = self.token_drop(count.shape[0], count.device, force_drop_ids) if (train and self.dropout_prob > 0) or (force_drop_ids is not None) else torch.zeros(count.shape[0], device=count.device, dtype=torch.bool)
        
        # Replace dropped values with null
        count = torch.where(drop_ids, self.null_count, count)
        count = torch.clamp(count, min=1.0, max=10.0)  # Reasonable bounds for count
        
        # Get embeddings
        if self.embedding_type == "sinusoidal":
            count_freq = self.count_embedding(count, self.frequency_embedding_size)
        elif self.embedding_type == "linear":
            count_expanded = count.unsqueeze(-1).float()
            count_freq = self.count_proj(count_expanded).squeeze(-1)
        
        # Replace null embeddings
        if drop_ids.any() and self.null_embedding is not None:
            null_emb = self.null_embedding.unsqueeze(0).expand(drop_ids.sum(), -1)
            count_freq[drop_ids] = null_emb
        elif drop_ids.any() and self.null_embedding_type == "zero":
            count_freq[drop_ids] = 0.0
        
        return self.mlp(count_freq)


class RotationEmbedder(PropertyEmbedder):
    """Embeds rotation angles into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, embedding_type="sinusoidal",
                 max_period=360, dropout_prob=0.0, null_rotation=0.0,
                 null_embedding_type="learnable"):
        super().__init__(hidden_size, frequency_embedding_size, dropout_prob, null_embedding_type)
        self.max_period = max_period
        self.embedding_type = embedding_type
        self.null_rotation = null_rotation
        
        if embedding_type == "sinusoidal":
            pass  # No additional parameters
        elif embedding_type == "linear":
            self.angle_proj = nn.Linear(1, frequency_embedding_size)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
    
    def rotation_embedding(self, rot, dim, max_period=360):
        """Create sinusoidal rotation embeddings."""
        # Convert degrees to radians for better periodicity
        rot_rad = rot * (2 * np.pi / 360)
        return RadiusEmbedder.radius_embedding(rot_rad, dim, max_period=2*np.pi)
    
    def forward(self, rot, train=False, force_drop_ids=None):
        # Handle dropout
        drop_ids = self.token_drop(rot.shape[0], rot.device, force_drop_ids) if (train and self.dropout_prob > 0) or (force_drop_ids is not None) else torch.zeros(rot.shape[0], device=rot.device, dtype=torch.bool)
        
        # Replace dropped values with null
        rot = torch.where(drop_ids, self.null_rotation, rot)
        rot = torch.clamp(rot, min=0.0, max=360.0)
        
        # Get embeddings
        if self.embedding_type == "sinusoidal":
            rot_freq = self.rotation_embedding(rot, self.frequency_embedding_size)
        elif self.embedding_type == "linear":
            rot_expanded = rot.unsqueeze(-1)
            rot_freq = self.angle_proj(rot_expanded).squeeze(-1)
        
        # Replace null embeddings
        if drop_ids.any() and self.null_embedding is not None:
            null_emb = self.null_embedding.unsqueeze(0).expand(drop_ids.sum(), -1)
            rot_freq[drop_ids] = null_emb
        elif drop_ids.any() and self.null_embedding_type == "zero":
            rot_freq[drop_ids] = 0.0
        
        return self.mlp(rot_freq)


class DiTBlock(nn.Module):
    """A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
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
    """Standard transformer block without conditioning (for concat method)."""
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
    """The final layer of DiT."""
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
    """Standard final layer without conditioning (for concat method)."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiT_Compositional(nn.Module):
    """
    Diffusion model with a Transformer backbone and compositional conditioning on
    radius, position, shape, and color.
    Only creates embedders for active properties.
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
        # Property embedding settings
        radius_embedding_type="linear",
        position_embedding_type="linear", 
        # Conditioning settings
        conditioning_method="concat",  # "adaln" or "concat"
        # CFG settings
        property_dropout_prob=0.0,
        null_radius=0.0,
        null_position=(0.0, 0.0),
        null_shape_id=0,
        null_color_id=0,
        null_embedding_type="learnable",
        # Vocabulary sizes
        num_shapes=4,  # circle, square, triangle, diamond
        num_colors=8,  # red, blue, green, yellow, magenta, cyan, orange, purple
        max_count=4,   # maximum number of objects (1-4)
        # Property selection
        active_properties=None,  # List of active properties, e.g., ['shape', 'color']
        # Additional property settings for count and rotation
        rotation_embedding_type="sinusoidal",
        null_count=1,
        null_rotation=0.0
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.conditioning_method = conditioning_method
        
        # Set active properties - default to all if not specified
        if active_properties is None:
            active_properties = ['radius', 'position', 'shape', 'color']
        self.active_properties = active_properties
        self.max_count = max_count

        # Image embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        
        # Always create timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Only create embedders for active properties
        if 'radius' in active_properties:
            self.radius_embedder = RadiusEmbedder(
                hidden_size, 
                embedding_type=radius_embedding_type, 
                dropout_prob=property_dropout_prob, 
                null_radius=null_radius,
                null_embedding_type=null_embedding_type
            )
        else:
            self.radius_embedder = None
            
        if 'position' in active_properties:
            self.position_embedder = PositionEmbedder(
                hidden_size,
                embedding_type=position_embedding_type,
                dropout_prob=property_dropout_prob,
                null_position=null_position,
                null_embedding_type=null_embedding_type
            )
        else:
            self.position_embedder = None
            
        if 'shape' in active_properties:
            self.shape_embedder = ShapeEmbedder(
                hidden_size,
                num_shapes=num_shapes,
                dropout_prob=property_dropout_prob
            )
        else:
            self.shape_embedder = None
            
        if 'color' in active_properties:
            self.color_embedder = ColorEmbedder(
                hidden_size,
                num_colors=num_colors,
                dropout_prob=property_dropout_prob
            )
        else:
            self.color_embedder = None
        
        if 'count' in active_properties:
            self.count_embedder = CountEmbedder(
                hidden_size,
                embedding_type="linear",  # Use linear instead of lookup for extrapolation
                dropout_prob=property_dropout_prob,
                null_count=null_count,
                null_embedding_type=null_embedding_type
            )
        else:
            self.count_embedder = None
        
        if 'rotation' in active_properties:
            self.rotation_embedder = RotationEmbedder(
                hidden_size,
                embedding_type=rotation_embedding_type,
                dropout_prob=property_dropout_prob,
                null_rotation=null_rotation,
                null_embedding_type=null_embedding_type
            )
        else:
            self.rotation_embedder = None
        
        num_patches = self.x_embedder.num_patches
        
        # Position embeddings for patches
        if conditioning_method == "concat":
            # Dynamic conditioning tokens based on active properties
            num_property_tokens = len(self.active_properties)
            total_conditioning_tokens = 1 + num_property_tokens  # 1 for time + properties
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + total_conditioning_tokens, hidden_size), requires_grad=False)
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

    def detect_conditioning_tokens_from_checkpoint(self, state_dict):
        """
        Detect the number of conditioning tokens from a checkpoint's pos_embed shape.
        This enables loading checkpoints trained with different active_properties.
        """
        if 'pos_embed' in state_dict and self.conditioning_method == "concat":
            # pos_embed shape: [1, num_patches + num_conditioning_tokens, hidden_size]
            pos_embed_shape = state_dict['pos_embed'].shape
            total_tokens = pos_embed_shape[1]
            num_patches = self.x_embedder.num_patches
            detected_conditioning_tokens = total_tokens - num_patches
            
            print(f"Detected {detected_conditioning_tokens} conditioning tokens from checkpoint")
            print(f"Current model expects {1 + len(self.active_properties)} conditioning tokens")
            
            # If there's a mismatch, we need to resize pos_embed
            expected_tokens = 1 + len(self.active_properties)
            if detected_conditioning_tokens != expected_tokens:
                print(f"Resizing pos_embed from {detected_conditioning_tokens} to {expected_tokens} conditioning tokens")
                return detected_conditioning_tokens
        
        return None
    
    def resize_pos_embed_for_checkpoint(self, state_dict, detected_tokens):
        """Resize pos_embed to match current model architecture."""
        if 'pos_embed' in state_dict and self.conditioning_method == "concat":
            old_pos_embed = state_dict['pos_embed']  # [1, old_total_tokens, hidden_size]
            
            # Extract image patch embeddings (always at the end)
            num_patches = self.x_embedder.num_patches
            patch_embeddings = old_pos_embed[:, -num_patches:, :]  # [1, num_patches, hidden_size]
            
            # Create new conditioning token embeddings (same device as old_pos_embed)
            expected_tokens = 1 + len(self.active_properties)
            new_conditioning_embeddings = torch.zeros(1, expected_tokens, self.hidden_size, 
                                                     device=old_pos_embed.device, dtype=old_pos_embed.dtype)
            
            # Copy as many old conditioning embeddings as possible
            old_conditioning_embeddings = old_pos_embed[:, :detected_tokens, :]
            copy_tokens = min(detected_tokens, expected_tokens)
            new_conditioning_embeddings[:, :copy_tokens, :] = old_conditioning_embeddings[:, :copy_tokens, :]
            
            # Combine new conditioning + patch embeddings
            new_pos_embed = torch.cat([new_conditioning_embeddings, patch_embeddings], dim=1)
            state_dict['pos_embed'] = new_pos_embed
            
            print(f"Resized pos_embed from {old_pos_embed.shape} to {new_pos_embed.shape}")
    
    def load_state_dict_with_resize(self, state_dict, strict=True):
        """Load state dict with automatic pos_embed resizing if needed."""
        detected_tokens = self.detect_conditioning_tokens_from_checkpoint(state_dict)
        
        if detected_tokens is not None:
            # Resize pos_embed to match current architecture
            self.resize_pos_embed_for_checkpoint(state_dict, detected_tokens)
        
        # Now load normally
        return super().load_state_dict(state_dict, strict=strict)

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
            # Add extra learned positions for active conditioning tokens
            num_property_tokens = len(self.active_properties)
            total_conditioning_tokens = 1 + num_property_tokens  # 1 for time + properties
            extra_tokens = torch.zeros(total_conditioning_tokens, self.hidden_size)
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
        
        # Initialize property embedding MLPs (only for embedders that exist)
        if self.radius_embedder is not None and hasattr(self.radius_embedder, 'mlp'):
            nn.init.normal_(self.radius_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.radius_embedder.mlp[2].weight, std=0.02)
            
        if self.position_embedder is not None and hasattr(self.position_embedder, 'mlp'):
            nn.init.normal_(self.position_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.position_embedder.mlp[2].weight, std=0.02)
        
        # Initialize categorical embedding tables (only for embedders that exist):
        if self.shape_embedder is not None and hasattr(self.shape_embedder, 'embedding_table'):
            nn.init.normal_(self.shape_embedder.embedding_table.weight, std=0.02)
            
        if self.color_embedder is not None and hasattr(self.color_embedder, 'embedding_table'):
            nn.init.normal_(self.color_embedder.embedding_table.weight, std=0.02)
        
        if self.count_embedder is not None and hasattr(self.count_embedder, 'mlp'):
            nn.init.normal_(self.count_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.count_embedder.mlp[2].weight, std=0.02)
        
        if self.rotation_embedder is not None and hasattr(self.rotation_embedder, 'mlp'):
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

    def forward(self, x, t, radius=None, position=None, shape=None, color=None, count=None, rotation=None, **kwargs):
        """
        Forward pass of DiT with compositional conditioning.
        Only uses embedders that exist (based on active_properties).
        """
        # Handle kwargs for backward compatibility
        if radius is None and 'radius' in kwargs:
            radius = kwargs['radius']
        if position is None and 'position' in kwargs:
            position = kwargs['position']
        if shape is None and 'shape' in kwargs:
            shape = kwargs['shape']
        if color is None and 'color' in kwargs:
            color = kwargs['color']
        if count is None and 'count' in kwargs:
            count = kwargs['count']
        if rotation is None and 'rotation' in kwargs:
            rotation = kwargs['rotation']
        
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Embed inputs
        x = self.x_embedder(x)  # (N, num_patches, D)
        t_emb = self.t_embedder(t)  # (N, D)
        
        # Only compute embeddings for active properties
        if self.radius_embedder is not None and radius is not None:
            radius_emb = self.radius_embedder(radius, self.training)  # (N, D)
        else:
            radius_emb = torch.zeros_like(t_emb)
            
        if self.position_embedder is not None and position is not None:
            position_emb = self.position_embedder(position, self.training)  # (N, D)
        else:
            position_emb = torch.zeros_like(t_emb)
            
        if self.shape_embedder is not None and shape is not None:
            shape_emb = self.shape_embedder(shape, self.training)  # (N, D)
        else:
            shape_emb = torch.zeros_like(t_emb)
            
        if self.color_embedder is not None and color is not None:
            color_emb = self.color_embedder(color, self.training)  # (N, D)
        else:
            color_emb = torch.zeros_like(t_emb)
        
        if self.count_embedder is not None and count is not None:
            count_emb = self.count_embedder(count, self.training)  # (N, D)
        else:
            count_emb = torch.zeros_like(t_emb)
        
        if self.rotation_embedder is not None and rotation is not None:
            rotation_emb = self.rotation_embedder(rotation, self.training)  # (N, D)
        else:
            rotation_emb = torch.zeros_like(t_emb)
        
        if self.conditioning_method == "adaln":
            # AdaLN approach - combine all property embeddings for conditioning
            x = x + self.pos_embed  # (N, T, D)
            c = t_emb + radius_emb + position_emb + shape_emb + color_emb + count_emb + rotation_emb  # (N, D)
            
            for block in self.blocks:
                x = block(x, c)  # (N, T, D)
            
            x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
            
        else:  # concat
            # In-context conditioning approach with separate property tokens
            batch_size = x.shape[0]
            
            # Start with time token (always included)
            conditioning_tokens = [t_emb.unsqueeze(1)]  # [(N, 1, D)]
            
            # Add only active property tokens
            if 'radius' in self.active_properties and self.radius_embedder is not None:
                conditioning_tokens.append(radius_emb.unsqueeze(1))
            if 'position' in self.active_properties and self.position_embedder is not None:
                conditioning_tokens.append(position_emb.unsqueeze(1))
            if 'shape' in self.active_properties and self.shape_embedder is not None:
                conditioning_tokens.append(shape_emb.unsqueeze(1))
            if 'color' in self.active_properties and self.color_embedder is not None:
                conditioning_tokens.append(color_emb.unsqueeze(1))
            if 'count' in self.active_properties and self.count_embedder is not None:
                conditioning_tokens.append(count_emb.unsqueeze(1))
            if 'rotation' in self.active_properties and self.rotation_embedder is not None:
                conditioning_tokens.append(rotation_emb.unsqueeze(1))
            
            # Concatenate all conditioning tokens with image tokens
            num_conditioning_tokens = len(conditioning_tokens)
            conditioning_tensor = torch.cat(conditioning_tokens, dim=1)  # (N, num_conditioning_tokens, D)
            x = torch.cat([conditioning_tensor, x], dim=1)  # (N, num_conditioning_tokens + num_patches, D)
            
            # Add position embeddings
            x = x + self.pos_embed  # (N, num_conditioning_tokens + num_patches, D)
            
            # Pass through transformer blocks
            for block in self.blocks:
                x = block(x)  # (N, num_conditioning_tokens + num_patches, D)
            
            # Remove conditioning tokens
            x = x[:, num_conditioning_tokens:, :]  # (N, num_patches, D)
            
            # Final layer
            x = self.final_layer(x)  # (N, num_patches, patch_size ** 2 * out_channels)
        
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, radius, position, shape, color, count=None, rotation=None, cfg_scale=1.0):
        """
        Forward pass with classifier-free guidance.
        All property inputs should be doubled: [conditional, unconditional]
        """
        batch_size = x.size(0)
        
        # Double the input for conditional and unconditional
        x_doubled = torch.cat([x, x], dim=0)
        
        # Handle timesteps - duplicate for conditional and unconditional
        if t.size(0) == batch_size:
            t_doubled = torch.cat([t, t], dim=0)
        else:
            t_doubled = t
        
        # Properties should already be doubled with [conditional, unconditional] values
        model_out = self.forward(x_doubled, t_doubled, radius, position, shape, color, count, rotation)
        
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


# Model configurations with compositional conditioning
def DiT_XL_2_compositional(**kwargs):
    return DiT_Compositional(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_L_2_compositional(**kwargs):
    return DiT_Compositional(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_B_2_compositional(**kwargs):
    return DiT_Compositional(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_S_2_compositional(**kwargs):
    return DiT_Compositional(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


DiT_models_compositional = {
    'DiT-XL/2': DiT_XL_2_compositional,
    'DiT-L/2':  DiT_L_2_compositional,
    'DiT-B/2':  DiT_B_2_compositional,
    'DiT-S/2':  DiT_S_2_compositional,
}