#!/usr/bin/env python
"""
Sampling and size metrics for the continuous radius DiT model.
Supports classifier-free guidance (CFG).
"""
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Import model components
from vgl.models import DiT_models_continuous as DiT_models
from vgl.unet_models import UNet_models
from diffusers.models import AutoencoderKL


MIN_CLASS_GAP = 20.0  # grey levels; GT renders: noise-only Otsu splits < 6, disk/background splits > 40
SMALL_MASK_PX = 20    # closed masks below this (radius < ~2.5 px) skip the 3x3 opening


def frame_border(array):
    """Pixels on the outermost ring of the frame (the background for every VGL render)."""
    return np.concatenate([array[0, :], array[-1, :], array[1:-1, 0], array[1:-1, -1]])


def extract_circle_mask(image, threshold_method='otsu'):
    """Extract binary mask of the circle from an image."""
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply thresholding to get binary mask
    if threshold_method == 'otsu':
        # Otsu's thresholding
        thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Tiny disks: with a radius 1-2 disk on a noisy background Otsu can split the
        # background noise instead of separating the disk; its two classes are then only a few
        # grey levels apart. Re-threshold halfway between the background (border median) and the
        # grey level farthest from it.
        low, high = gray[gray <= thresh], gray[gray > thresh]
        if not (low.size and high.size) or float(high.mean()) - float(low.mean()) < MIN_CLASS_GAP:
            bg = float(np.median(frame_border(gray)))
            far = float(gray.min()) if bg - float(gray.min()) >= float(gray.max()) - bg else float(gray.max())
            _, mask = cv2.threshold(gray, (bg + far) / 2.0, 255, cv2.THRESH_BINARY)
        # Invert if needed (ensure circle is white)
        # Polarity: decide by the frame border, which is background, not by the majority;
        # a centred disk with r > ~25.3 px covers more than half of a 64x64 frame.
        if np.mean(frame_border(mask)) > 127:  # If the border is white, invert
            mask = 255 - mask
    else:
        # Adaptive thresholding
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Convert to binary (0 or 1)
    mask = (mask > 127).astype(np.uint8)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Radius 1-2: the 3x3 opening erases a radius-1 disk (about 2x2 px) and turns a
    # radius-2 disk into a 3x3 square or nothing, depending on anti-aliasing; skip it for
    # masks smaller than SMALL_MASK_PX (radius < ~2.5). Larger masks are cleaned as before.
    if mask.sum() >= SMALL_MASK_PX:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def calculate_iou(mask1, mask2):
    """Calculate Intersection over Union between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return iou


def calculate_circle_metrics(generated_img, ground_truth_img, gt_radius):
    """
    Calculate various metrics between generated and ground truth circles.
    
    Returns:
        dict with IoU, precision, recall, estimated radius, and radius errors
    """
    # Extract masks
    gen_mask = extract_circle_mask(generated_img)
    gt_mask = extract_circle_mask(ground_truth_img)
    
    # Calculate IoU
    iou = calculate_iou(gen_mask, gt_mask)
    
    # Calculate precision and recall
    intersection = np.logical_and(gen_mask, gt_mask).sum()
    gen_pixels = gen_mask.sum()
    gt_pixels = gt_mask.sum()
    
    precision = intersection / gen_pixels if gen_pixels > 0 else 0
    recall = intersection / gt_pixels if gt_pixels > 0 else 0
    
    # Estimate radius from generated mask
    if gen_pixels > 0:
        # Approximate radius assuming circle
        estimated_area = gen_pixels
        estimated_radius = np.sqrt(estimated_area / np.pi)
    else:
        estimated_radius = 0
    
    # Calculate radius errors
    radius_error_raw = estimated_radius - gt_radius
    radius_error_percentage = (radius_error_raw / gt_radius * 100) if gt_radius > 0 else 0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'estimated_radius': estimated_radius,
        'gt_radius': gt_radius,
        'radius_error_raw': radius_error_raw,
        'radius_error_percentage': radius_error_percentage,
        'gen_mask': gen_mask,
        'gt_mask': gt_mask
    }


def load_model(config, device):
    """Load the trained DiT model."""
    # Determine input size and channels
    input_size = config['image_size']
    in_channels = 3  # Default
    
    # Load checkpoint first to check model configuration
    print(f"Loading checkpoint...")
    checkpoint = torch.load(config['ckpt_path'], map_location=device, weights_only=False)
    
    # Get the state dict to check
    state_dict = None
    if 'ema' in checkpoint:
        state_dict = checkpoint['ema']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Check if this is a VAE model (latent diffusion)
    # Be careful not to match "novae" or "no_vae" (no VAE) - only match "vae" as a standalone word
    folder_lower = config['folder_name'].lower()
    is_vae_model = ('_vae_' in folder_lower or 
                   folder_lower.startswith('vae_') or 
                   folder_lower.endswith('_vae') or 
                   folder_lower == 'vae') and 'novae' not in folder_lower and 'no_vae' not in folder_lower
    # run_config records this explicitly and is authoritative; the folder-name
    # heuristic above is a fallback for older checkpoints.
    if config.get('use_latent_diffusion'):
        is_vae_model = True

    if is_vae_model:
        config['use_latent_diffusion'] = True
        print(f"VAE model detected - using latent diffusion")
        # For latent diffusion, we work in 8x8 latent space for 64x64 images
        input_size = input_size // 8
        print(f"Adjusted input size for latent diffusion: {input_size}x{input_size}")
        
        # VAE models use 4 input channels (latent space) and different output dimensions
        # Check the actual checkpoint to get the correct dimensions
        for key in state_dict.keys():
            if 'x_embedder.proj.weight' in key:
                checkpoint_in_channels = state_dict[key].shape[1]
                print(f"VAE checkpoint has {checkpoint_in_channels} input channels")
                in_channels = checkpoint_in_channels
                break
        
        # Check output dimensions from final layer
        if 'final_layer.linear.weight' in state_dict:
            checkpoint_out_features = state_dict['final_layer.linear.weight'].shape[0]
            print(f"VAE checkpoint expects {checkpoint_out_features} output features")
            # This affects the patch size calculation for the final layer
    
    else:
        # For DiT/UNet, keep existing logic but also check for channel detection
        for key in state_dict.keys():
            if 'x_embedder.proj.weight' in key:  # DiT patch embedding
                detected_channels = state_dict[key].shape[1]
                print(f"Detected {detected_channels} input channels from DiT patch embedding")
                in_channels = detected_channels
                break
            elif 'input_conv.weight' in key:  # UNet input conv
                detected_channels = state_dict[key].shape[1]
                print(f"Detected {detected_channels} input channels from UNet input conv")
                in_channels = detected_channels
                break
    
    # Check positional embedding size to infer conditioning method (for DiT models only)
    if 'pos_embed' in state_dict and config.get('architecture') == 'dit':
        pos_embed_shape = state_dict['pos_embed'].shape
        model_name = config.get('model_name', 'DiT-S/2')
        
        # Determine number of patches based on model name and input size.
        # Parse the patch size off the end of the name rather than substring-matching:
        # 'S/1' is a substring of 'S/16', which silently gave patch-16 models a
        # patch-1 patch count.
        try:
            patch_size = int(model_name.rsplit('/', 1)[1])
        except (IndexError, ValueError):
            patch_size = 2
        num_patches = (input_size // patch_size) ** 2
        
        expected_adaln_positions = num_patches
        expected_concat_positions = num_patches + 2  # +2 for time and radius tokens
        
        actual_positions = pos_embed_shape[1]
        
        if actual_positions == expected_concat_positions:
            print(f"Detected concat conditioning from pos_embed shape: {pos_embed_shape}")
            print(f"Expected {expected_concat_positions} positions (patches={num_patches} + 2 tokens), got {actual_positions}")
            config['conditioning_method'] = 'concat'
        elif actual_positions == expected_adaln_positions:
            print(f"Detected adaln conditioning from pos_embed shape: {pos_embed_shape}")
            print(f"Expected {expected_adaln_positions} positions (patches={num_patches}), got {actual_positions}")
            config['conditioning_method'] = 'adaln'
        else:
            print(f"Warning: Unexpected pos_embed shape {pos_embed_shape}. Expected {expected_adaln_positions} (adaln) or {expected_concat_positions} (concat)")
            print(f"Model: {model_name}, Input size: {input_size}, Patches: {num_patches}")
            print(f"Keeping original conditioning method: {config['conditioning_method']}")
    
    # Initialize model - select by inferred model name and architecture
    model_key = config.get('model_name', 'DiT-S/2')
    architecture = config.get('architecture', 'dit')
    
    # For VAE models, we need to use the checkpoint dimensions directly
    if config.get('use_latent_diffusion', False):
        print(f"Initializing VAE model with checkpoint-derived dimensions")
        print(f"  Input size: {input_size}x{input_size}")
        print(f"  Input channels: {in_channels}")
    
    if architecture == 'unet':
        if model_key not in UNet_models:
            print(f"Warning: model '{model_key}' not found. Falling back to 'UNet-S'.")
            model_key = "UNet-S"
        model = UNet_models[model_key](
            input_size=input_size,
            in_channels=in_channels,
            radius_embedding_type=config['radius_embedding_type'],
            conditioning_method=config['conditioning_method'],
            radius_dropout_prob=config['radius_dropout_prob'],
            null_radius=config['null_radius'],
            null_embedding_type=config['null_embedding_type'],
        ).to(device)
    else:
        if model_key not in DiT_models:
            print(f"Warning: model '{model_key}' not found. Falling back to 'DiT-S/2'.")
            model_key = "DiT-S/2"
        model = DiT_models[model_key](
            input_size=input_size,
            in_channels=in_channels,
            radius_embedding_type=config['radius_embedding_type'],
            conditioning_method=config['conditioning_method'],
            radius_dropout_prob=config['radius_dropout_prob'],
            null_radius=config['null_radius'],
            null_embedding_type=config['null_embedding_type'],
            radius_text_table=config.get('radius_text_table')
        ).to(device)
    
    # For DiT/UNet, use strict loading
    if 'ema' in checkpoint:
        model.load_state_dict(checkpoint['ema'], strict=False)
        print("Loaded EMA weights")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded weights directly")
    
    model.eval()
    return model


def generate_samples(
    model,
    diffusion,
    radius_values,
    config,
    device,
    num_samples=3,
    cfg_scale=1.0,
    batch_size=1,
    num_sampling_steps=250,
):
    """Generate samples for given radius values with optional CFG."""
    samples_dict = {}
    
    input_size = config['image_size']
    use_cfg = config.get('use_cfg', False) and cfg_scale > 1.0
    use_latent_diffusion = config.get('use_latent_diffusion', False)
    
    # Load VAE if needed
    vae = None
    if use_latent_diffusion:
        try:
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
            vae.eval()
            print("Loaded VAE for latent diffusion decoding")
            # For latent diffusion, we sample in latent space (8x8 for 64x64 images)
            input_size = input_size // 8
            print(f"Using latent space size: {input_size}x{input_size}")
        except Exception as e:
            print(f"Warning: Could not load VAE: {e}. Falling back to direct pixel sampling.")
            use_latent_diffusion = False
            vae = None
    
    if use_cfg:
        print(f"Using CFG with scale {cfg_scale}")
    else:
        print(f"Not using CFG (cfg_scale={cfg_scale})")
    
    batch_size = max(1, batch_size)
    
    for radius in tqdm(radius_values, desc="Generating samples"):
        samples_list = []
        
        sample_index = 0
        while sample_index < num_samples:
            current_batch = min(batch_size, num_samples - sample_index)
            
            # Determine input channels
            in_channels = getattr(model, 'in_channels', 3)
            
            # Build deterministic noise batch
            noise_batches = []
            for j in range(current_batch):
                gen = torch.Generator(device=device)
                gen.manual_seed(42 + sample_index + j)
                noise_batches.append(
                    torch.randn(
                        (1, in_channels, input_size, input_size),
                        device=device,
                        generator=gen,
                    )
                )
            z = torch.cat(noise_batches, dim=0)
            z_shape = z.shape
            
            # Check if model has forward_with_cfg method
            has_forward_with_cfg = hasattr(model, 'forward_with_cfg')
            
            # Sample with or without CFG
            with torch.no_grad():
                # Check if we're using flow matching
                is_flow_matching = config.get('use_flow_matching', False)
                
                if is_flow_matching:
                    # Flow matching sampling
                    r = torch.full((current_batch,), radius, device=device, dtype=torch.float32)

                    if use_cfg and has_forward_with_cfg:
                        # Flow matching with CFG
                        sample = diffusion.sample_with_cfg(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            cfg_scale=cfg_scale,
                            model_kwargs=dict(r=r),
                            device=device,
                        )
                    else:
                        # Flow matching without CFG
                        sample = diffusion.sample(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            model_kwargs=dict(r=r),
                            device=device,
                        )
                else:
                    # Diffusion sampling. The predicted clean image is clipped to [-1, 1] in pixel
                    # space only: scaled VAE latents often exceed 1, so latent models are not
                    # clipped (as in DiT's sample.py and eval_rotation.py).
                    if use_cfg and has_forward_with_cfg:
                        # Use the model's forward_with_cfg method
                        # Create doubled radius tensor: [conditional, unconditional]
                        r_cond = torch.full(
                            (current_batch,),
                            radius,
                            device=device,
                            dtype=torch.float32,
                        )
                        r_uncond = torch.full(
                            (current_batch,),
                            config['null_radius'],
                            device=device,
                            dtype=torch.float32,
                        )
                        r_doubled = torch.cat([r_cond, r_uncond], 0)
                        
                        # Custom forward function that uses forward_with_cfg
                        def cfg_forward(x, t, **kwargs):
                            # Note: x is NOT doubled, forward_with_cfg handles that
                            return model.forward_with_cfg(x, t, r_doubled, cfg_scale)
                        
                        sample = diffusion.p_sample_loop(
                            cfg_forward,
                            z_shape,
                            noise=z,
                            clip_denoised=not use_latent_diffusion,
                            model_kwargs={},
                            progress=False,
                            device=device,
                        )
                    else:
                        # No CFG - standard sampling
                        r = torch.full(
                            (current_batch,),
                            radius,
                            device=device,
                            dtype=torch.float32,
                        )
                        sample = diffusion.p_sample_loop(
                            model,
                            z_shape,
                            noise=z,
                            clip_denoised=not use_latent_diffusion,
                            model_kwargs=dict(r=r),
                            progress=False,
                            device=device,
                        )
                
                # Handle VAE decoding if using latent diffusion
                if use_latent_diffusion and vae is not None:
                    # Decode from latent space to pixel space
                    sample = sample / 0.18215  # Reverse the scaling applied during training
                    sample = vae.decode(sample).sample
            
            # Convert to image - handle channel conversion
            if sample.shape[1] == 4:
                sample = sample[:, :3, :, :]
            elif sample.shape[1] != 3:
                raise ValueError(f"Unexpected number of output channels: {sample.shape[1]}")
            
            sample = torch.clamp(127.5 * sample + 128.0, 0, 255)
            sample = sample.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            for img in sample:
                samples_list.append(img)
                sample_index += 1
                if sample_index >= num_samples:
                    break
        
        samples_dict[radius] = samples_list
    
    return samples_dict
