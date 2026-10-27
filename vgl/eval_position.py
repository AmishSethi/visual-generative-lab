#!/usr/bin/env python
"""
Sampling and position metrics for the continuous position DiT model.
Measures position error against the conditioning target and reports the distance
to the nearest training position.
"""
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from scipy import ndimage
from scipy.spatial.distance import cdist

# Import model components
from vgl.models_position import DiT_models_position as DiT_models
from vgl.unet_models_position import UNet_models_position as UNet_models
from diffusers.models import AutoencoderKL

# Grey levels between Otsu's two classes below which a frame is blank (no object), as
# COUNT_MIN_CONTRAST in paper_runs/table2/evaluate_table2.py.
POSITION_MIN_CONTRAST = 20.0


def find_nearest_training_position(test_pos, training_positions):
    """
    Find the nearest training position to a test position.
    Returns the nearest position and the distance.
    """
    test_pos_array = np.array([test_pos])
    train_pos_array = np.array(training_positions)
    
    distances = cdist(test_pos_array, train_pos_array)[0]
    nearest_idx = np.argmin(distances)
    
    return training_positions[nearest_idx], distances[nearest_idx]


def extract_circle_center_debug(image, expected_radius=8, debug=False):
    """
    Extract the center position of a circle in the image with optional debugging.
    Returns (x, y) coordinates relative to image center and debug info.
    """
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
    thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Blank frame: Otsu always splits the frame in two, so a frame with no object still gives a
    # mask. Require an absolute contrast between the two classes, as the count detector does.
    dark, light = gray[gray <= thresh], gray[gray > thresh]
    if not (dark.size and light.size) or float(light.mean()) - float(dark.mean()) < POSITION_MIN_CONTRAST:
        return 0.0, 0.0, False, {'mask_pixels': 0.0, 'mask': None}

    # Invert if needed (ensure circle is white)
    if np.mean(mask) > 127:
        mask = 255 - mask
    
    # Find center of mass
    mask_binary = (mask > 127).astype(np.float32)
    
    debug_info = {
        'mask_pixels': np.sum(mask_binary),
        'mask': mask_binary if debug else None
    }
    
    if np.sum(mask_binary) > 10:  # Minimum pixels to consider valid
        # Calculate center of mass
        # Note: ndimage returns (row, col) which is (y, x) in image coordinates
        cy, cx = ndimage.center_of_mass(mask_binary)
        
        # Convert to coordinates relative to image center. center_of_mass puts pixel i at i, so the
        # center of a W-pixel frame is (W - 1) / 2, where the renderers place offset (0, 0).
        img_center_x = (img_array.shape[1] - 1) / 2
        img_center_y = (img_array.shape[0] - 1) / 2
        
        # x_rel: positive = right of center, negative = left of center
        # y_rel: positive = below center, negative = above center
        x_rel = cx - img_center_x
        y_rel = cy - img_center_y
        
        debug_info.update({
            'cx_pixel': cx,
            'cy_pixel': cy,
            'img_center_x': img_center_x,
            'img_center_y': img_center_y,
            'x_rel': x_rel,
            'y_rel': y_rel
        })
        
        return x_rel, y_rel, True, debug_info
    else:
        return 0.0, 0.0, False, debug_info


def calculate_position_metrics_enhanced(generated_img, expected_pos, training_positions, image_size=64, debug=False):
    """
    Calculate position-specific metrics with nearest training point info.
    """
    # Extract circle center from generated image
    detected_x, detected_y, valid, debug_info = extract_circle_center_debug(generated_img, debug=debug)
    
    # Find nearest training position
    nearest_train_pos, distance_to_nearest = find_nearest_training_position(expected_pos, training_positions)
    
    if not valid:
        return {
            'position_error': float('inf'),
            'position_error_x': float('inf'),
            'position_error_y': float('inf'),
            'relative_error': float('inf'),
            'detection_success': False,
            'detected_position': (0.0, 0.0),
            'expected_position': expected_pos,
            'nearest_train_position': nearest_train_pos,
            'distance_to_nearest_train': distance_to_nearest,
            'debug_info': debug_info
        }
    
    # Calculate errors
    expected_x, expected_y = expected_pos
    error_x = detected_x - expected_x
    error_y = detected_y - expected_y
    position_error = np.sqrt(error_x**2 + error_y**2)
    
    # Calculate relative error
    max_distance = np.sqrt(2) * (image_size / 2)
    relative_error = (position_error / max_distance) * 100
    
    return {
        'position_error': position_error,
        'position_error_x': error_x,
        'position_error_y': error_y,
        'relative_error': relative_error,
        'detection_success': True,
        'detected_position': (detected_x, detected_y),
        'expected_position': expected_pos,
        'nearest_train_position': nearest_train_pos,
        'distance_to_nearest_train': distance_to_nearest,
        'debug_info': debug_info
    }


def load_model(config, device):
    """Load the trained model (DiT or UNet)."""
    input_size = config['image_size']
    in_channels = 3  # Default
    architecture = config.get('architecture', 'dit')
    
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
    ckpt_path_lower = str(config['ckpt_path']).lower()
    # Trust the training run's own flag when the caller supplies it; the path
    # heuristic below only fires when it is absent, and misses directory names
    # like ".../hires128_vae/position/...".
    is_vae_model = bool(config.get('use_latent_diffusion')) or (
                  ('_vae_' in ckpt_path_lower or 
                   ckpt_path_lower.endswith('_vae') or 
                   '/vae_' in ckpt_path_lower or
                   '/vae/' in ckpt_path_lower) and 'novae' not in ckpt_path_lower and 'no_vae' not in ckpt_path_lower)

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
        # For non-VAE models, keep standard settings
        config['use_latent_diffusion'] = False
    
    if architecture == 'unet':
        model = UNet_models[config['model_type']](
            input_size=input_size,
            in_channels=in_channels,
            position_embedding_type=config['position_embedding_type'],
            conditioning_method=config['conditioning_method'],
            position_dropout_prob=config['position_dropout_prob'],
            max_position_value=config['train_max_pos']  # Max position value for the dataset
        ).to(device)
    else:
        model = DiT_models[config['model_type']](
            input_size=input_size,
            in_channels=in_channels,
            position_embedding_type=config['position_embedding_type'],
            conditioning_method=config['conditioning_method'],
            position_dropout_prob=config['position_dropout_prob'],
            null_position=config['null_position'],
            null_embedding_type=config['null_embedding_type']
        ).to(device)
    
    if 'ema' in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint['ema'], strict=False)
        print("Loaded EMA weights")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    elif 'model' in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        print("Loaded model weights")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print("Loaded weights directly")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    
    model.eval()
    return model


def generate_samples(
    model,
    diffusion,
    position_values,
    config,
    device,
    num_samples=3,
    cfg_scale=1.0,
    batch_size=1,
    num_sampling_steps=250,
):
    """Generate samples for given position values with optional CFG."""
    samples_dict = {}
    
    input_size = config['image_size']
    use_cfg = config.get('use_cfg', False) and cfg_scale > 1.0
    use_latent_diffusion = config.get('use_latent_diffusion', False)
    
    # Load VAE if needed
    vae = None
    if use_latent_diffusion:
        try:
            from diffusers.models import AutoencoderKL
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
    
    for pos in tqdm(position_values, desc="Generating samples"):
        samples_list = []
        
        sample_index = 0
        while sample_index < num_samples:
            current_batch = min(batch_size, num_samples - sample_index)
            
            in_channels = getattr(model, 'in_channels', 3)
            
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
                # Convert position tuple to tensor
                pos_tensor = torch.tensor(
                    [pos] * current_batch, device=device, dtype=torch.float32
                )
                
                # Check if we're using flow matching
                is_flow_matching = config.get('use_flow_matching', False)
                
                if is_flow_matching:
                    # Flow matching sampling
                    if use_cfg and has_forward_with_cfg:
                        # Flow matching with CFG
                        sample = diffusion.sample_with_cfg(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            cfg_scale=cfg_scale,
                            model_kwargs=dict(pos=pos_tensor),
                            device=device,
                        )
                    else:
                        # Flow matching without CFG
                        sample = diffusion.sample(
                            model,
                            z_shape,
                            num_steps=num_sampling_steps,
                            model_kwargs=dict(pos=pos_tensor),
                            device=device,
                        )
                else:
                    # Diffusion sampling. The predicted clean image is clipped to [-1, 1] in pixel
                    # space only: scaled VAE latents often exceed 1, so latent models are not
                    # clipped (as in DiT's sample.py).
                    sample = diffusion.p_sample_loop(
                        model,
                        z_shape,
                        noise=z,
                        clip_denoised=not use_latent_diffusion,
                        model_kwargs=dict(pos=pos_tensor),
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
        
        samples_dict[pos] = samples_list
    
    return samples_dict
