#!/usr/bin/env python
"""
A minimal training script for DiT using PyTorch DDP.
Modified for continuous rotation angle conditioning with checkpoint resuming and signal handling.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import signal
import sys
import json
from datetime import datetime

# Import rotation models
from vgl.models_rotation import DiT_models_rotation as DiT_models
from vgl.unet_models_rotation import UNet_models_rotation as UNet_models
from vgl.unet_models_song_rotation import SongUNet_Rotation_models as SongUNet_models
from vgl.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
# NEW: Import flow matching utilities
from vgl.flow_matching import create_loss_function, add_flow_matching_args
from vgl.reproducibility_utils import (
    configure_reproducibility,
    make_data_loader_generator,
    make_worker_init_fn,
    write_run_config,
)


# Global variable for graceful shutdown
should_stop = False
checkpoint_on_signal = None


def signal_handler(signum, frame):
    """Handle signals gracefully by setting a flag to stop training."""
    global should_stop, checkpoint_on_signal
    if dist.get_rank() == 0:
        print(f"\nReceived signal {signum}. Preparing to save checkpoint and exit gracefully...")
    should_stop = True
    checkpoint_on_signal = signum


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination
signal.signal(signal.SIGHUP, signal_handler)   # Hangup (terminal disconnect)


class ImageFolderWithRotation(ImageFolder):
    """Wrapper around ImageFolder that converts folder names to rotation angles."""
    def __init__(self, root, transform=None, selected_angles=None, max_samples_per_angle=None):
        super().__init__(root, transform)
        
        # Parse folder names to extract rotation angles
        # Expected format: "angle" like "45p0" for 45.0 degrees or "neg30p5" for -30.5 degrees
        self.angle_mapping = {}
        for i, folder_name in enumerate(sorted(self.classes)):
            # Convert from folder format back to angle
            angle_str = folder_name.replace('neg', '-').replace('p', '.')
            try:
                angle = float(angle_str)
                self.angle_mapping[i] = angle
            except ValueError:
                print(f"Warning: Could not parse folder name {folder_name}")
                self.angle_mapping[i] = 0.0  # Default to 0 degrees
        
        # Filter dataset to only include selected angles
        if selected_angles is not None:
            # Convert selected_angles to strings for matching
            selected_folders = []
            for angle in selected_angles:
                angle_str = f"{angle:.1f}".replace('-', 'neg').replace('.', 'p')
                selected_folders.append(angle_str)
            
            # Find indices of selected classes
            selected_indices = [i for i, cls in enumerate(self.classes) if cls in selected_folders]
            
            # Filter samples to only include those from selected classes
            filtered_samples = []
            for path, target in self.samples:
                if target in selected_indices:
                    filtered_samples.append((path, target))
            
            self.samples = filtered_samples
            self.targets = [target for _, target in filtered_samples]
            
            # Create a mapping from old target indices to new target indices
            old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
            
            # Update samples with new target indices
            updated_samples = []
            for path, old_target in self.samples:
                new_target = old_to_new[old_target]
                updated_samples.append((path, new_target))
            self.samples = updated_samples
            self.targets = [target for _, target in updated_samples]
            
            # Update angle mapping to only include selected angles
            filtered_angle_mapping = {}
            for new_idx, old_idx in enumerate(selected_indices):
                filtered_angle_mapping[new_idx] = self.angle_mapping[old_idx]
            self.angle_mapping = filtered_angle_mapping
        
        # Limit samples per angle if specified
        if max_samples_per_angle is not None:
            # Group samples by angle class
            samples_by_class = {}
            for path, target in self.samples:
                if target not in samples_by_class:
                    samples_by_class[target] = []
                samples_by_class[target].append((path, target))
            
            # Limit samples per angle
            limited_samples = []
            for target, samples in samples_by_class.items():
                limited_samples.extend(samples[:max_samples_per_angle])
            
            self.samples = limited_samples
            self.targets = [target for _, target in limited_samples]
            
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Convert label index to rotation angle
        angle = self.angle_mapping[label]
        rotation = torch.tensor(angle, dtype=torch.float32)
        return image, rotation


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix=""):
    """Save a checkpoint with optional prefix."""
    checkpoint = {
        "model": model.module.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "train_steps": train_steps,
        "epoch": epoch,
        "timestamp": datetime.now().isoformat()
    }
    
    if prefix:
        checkpoint_path = f"{checkpoint_dir}/{prefix}_{train_steps:07d}.pt"
    else:
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    
    # Save to temporary file first, then rename (atomic operation)
    temp_path = checkpoint_path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, checkpoint_path)
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def find_resume_checkpoint(checkpoint_dir, resume_from=None):
    """Find the latest checkpoint to resume from."""
    if resume_from and os.path.exists(resume_from):
        return resume_from
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Find all checkpoints
    checkpoints = glob(f"{checkpoint_dir}/[0-9]*.pt")
    if not checkpoints:
        return None
    
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    return checkpoints[-1]


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model with rotation angle conditioning.
    """
    global should_stop, checkpoint_on_signal
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    configure_reproducibility(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        
        if args.resume_from:
            # Extract experiment directory from checkpoint path
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_from))
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-rotation"  # Create an experiment folder
            checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        write_run_config(
            experiment_dir,
            args,
            extra={
                "rank": rank,
                "resolved_seed": seed,
                "world_size": dist.get_world_size(),
            },
        )
    else:
        logger = create_logger(None)
        if args.resume_from:
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_from))
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # Create model:
    # Determine input size and channels based on diffusion mode
    if args.use_latent_diffusion:
        # Latent diffusion: use VAE compressed latents
        input_size = args.image_size // 8
        in_channels = 4  # VAE latent channels
        logger.info(f"Using latent diffusion: input_size={input_size}, in_channels={in_channels}")
    else:
        # Direct pixel diffusion: use full image
        input_size = args.image_size
        in_channels = 3  # RGB channels
        logger.info(f"Using direct pixel diffusion: input_size={input_size}, in_channels={in_channels}")
    
    # Create rotation-conditioned model
    # Select model based on architecture
    if args.architecture == "dit":
        model = DiT_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            rotation_embedding_type=args.rotation_embedding_type,
            rotation_text_table=args.rotation_text_table,
            conditioning_method=args.conditioning_method,
            rotation_dropout_prob=args.rotation_dropout_prob,
            null_angle=args.null_angle,
            null_embedding_type=args.null_embedding_type
        )
    elif args.architecture == "unet":
        model = UNet_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            rotation_embedding_type=args.rotation_embedding_type,
            conditioning_method=args.conditioning_method,
            class_dropout_prob=args.rotation_dropout_prob
        )
    elif args.architecture == "songunet":
        model = SongUNet_models[args.model](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=in_channels,  # SongUNet doesn't learn sigma by default
            # learn_sigma defaults to False in SongUNet to match original implementation
            rotation_embedding_type=args.rotation_embedding_type,
            conditioning_method=args.conditioning_method,
            rotation_dropout_prob=args.rotation_dropout_prob,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    # Note that parameter initialization is done within the model constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # Use find_unused_parameters=True for SongUNet to avoid DDP errors
    if args.architecture == "songunet":
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    else:
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    # Create loss function (either diffusion or flow matching)
    if args.architecture == "songunet":
        # SongUNet doesn't learn sigma by default
        loss_fn = create_loss_function(args, timestep_respacing="", learn_sigma=False)
    else:
        # DiT and UNet models learn sigma
        loss_fn = create_loss_function(args, timestep_respacing="", learn_sigma=True)
    
    # Log which objective we're using
    objective_type = "Flow Matching" if getattr(args, 'use_flow_matching', False) else "Diffusion"
    logger.info(f"Using {objective_type} training objective")
    
    # Load VAE if using latent diffusion
    vae = None
    if args.use_latent_diffusion:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        logger.info("Loaded VAE for latent diffusion")
    else:
        logger.info("Using direct pixel diffusion (no VAE)")
    
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Initialize training state
    train_steps = 0
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume_from:
        resume_checkpoint = find_resume_checkpoint(checkpoint_dir, args.resume_from)
        if resume_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            checkpoint = torch.load(resume_checkpoint, map_location=f'cuda:{device}')

            model.module.load_state_dict(checkpoint["model"])
            ema.load_state_dict(checkpoint["ema"])
            opt.load_state_dict(checkpoint["opt"])
            train_steps = checkpoint.get("train_steps", 0)
            start_epoch = checkpoint.get("epoch", 0) + 1  # Start from next epoch
            
            logger.info(f"Resumed from epoch {start_epoch}, step {train_steps}")
        else:
            logger.warning(f"Could not find checkpoint to resume from: {args.resume_from}")
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Use ImageFolderWithRotation
    dataset = ImageFolderWithRotation(
        args.data_path, 
        transform=transform, 
        selected_angles=args.selected_angles, 
        max_samples_per_angle=args.max_samples_per_angle
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True
    )
    per_device_batch_size = args.micro_batch_size or int(args.global_batch_size // dist.get_world_size())
    if args.global_batch_size % (per_device_batch_size * dist.get_world_size()) != 0:
        raise ValueError(
            f"global_batch_size={args.global_batch_size} must be divisible by "
            f"micro_batch_size*world_size={per_device_batch_size * dist.get_world_size()}"
        )
    grad_accum_steps = args.global_batch_size // (per_device_batch_size * dist.get_world_size())
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=make_worker_init_fn(seed),
        generator=make_data_loader_generator(seed),
        pin_memory=False,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Number of rotation classes: {len(dataset.angle_mapping)}")
    logger.info(
        "Effective global batch size: %d (micro batch %d x world_size %d x grad_accum_steps %d)",
        args.global_batch_size,
        per_device_batch_size,
        dist.get_world_size(),
        grad_accum_steps,
    )
    if args.selected_angles is not None:
        logger.info(f"Training on selected angles: {args.selected_angles[:5]}..." if len(args.selected_angles) > 5 else f"Training on selected angles: {args.selected_angles}")
    else:
        logger.info("Training on all available angles")

    # Prepare models for training:
    if train_steps == 0:  # Only reset EMA if starting fresh
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    epoch = start_epoch  # Initialize epoch variable

    logger.info(f"Training for {args.epochs} epochs (starting from epoch {start_epoch})...")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            if should_stop:
                logger.info("Stopping training due to signal...")
                break
                
            sampler.set_epoch(epoch)
            logger.info(f"Beginning epoch {epoch}...")
            opt.zero_grad()
            accum_loss = 0.0
            accum_steps = 0
            
            for batch_idx, (x, rotation) in enumerate(loader):
                if should_stop:
                    logger.info("Stopping training due to signal...")
                    break
                    
                x = x.to(device)
                x = x.contiguous()
                rotation = rotation.to(device)  # (N,) rotation angles
                
                # Handle input based on diffusion mode
                if args.use_latent_diffusion:
                    # Latent diffusion: encode to VAE latents
                    with torch.no_grad():
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                else:
                    # Direct pixel diffusion: use images directly (already normalized to [-1, 1])
                    pass
                if not args.use_flow_matching:
                    # Diffusion: sample timesteps
                    t = torch.randint(0, loss_fn.num_timesteps, (x.shape[0],), device=device)
                else:
                    # Flow matching: timesteps are sampled internally, pass dummy timesteps
                    t = torch.zeros(x.shape[0], device=device)
                
                # Pass rotation in model_kwargs
                model_kwargs = dict(rotation=rotation)
                
                # Use training_losses function (works for both diffusion and flow matching)
                loss_dict = loss_fn.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {train_steps}, skipping batch")
                    opt.zero_grad()
                    accum_loss = 0.0
                    accum_steps = 0
                    continue
                
                # Check rotation values
                if torch.isnan(rotation).any() or torch.isinf(rotation).any():
                    logger.warning(f"NaN/Inf in rotation values at step {train_steps}")
                    opt.zero_grad()
                    accum_loss = 0.0
                    accum_steps = 0
                    continue
                
                micro_loss = loss.item()
                (loss / grad_accum_steps).backward()
                accum_loss += micro_loss
                accum_steps += 1
                
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(f"NaN/Inf gradient in {name} at step {train_steps}")
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    logger.warning("Skipping step due to NaN/Inf gradients")
                    opt.zero_grad()  # Clear the bad gradients
                    accum_loss = 0.0
                    accum_steps = 0
                    continue

                if accum_steps < grad_accum_steps:
                    continue
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
                accum_steps = 0
                
                # Force synchronization to catch CUDA errors early
                torch.cuda.synchronize()
                update_ema(ema, model.module)

                # Log loss values:
                running_loss += accum_loss / grad_accum_steps
                accum_loss = 0.0
                log_steps += 1
                train_steps += 1
                
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger)
                    dist.barrier()
                
                # Periodic memory cleanup
                if train_steps % 500 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Save final checkpoint if training completed normally
        if not should_stop and rank == 0:
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix="final")
            
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        if rank == 0:
            # Save emergency checkpoint
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix="emergency")
        raise
    
    finally:
        # Save checkpoint if stopped by signal
        if should_stop and rank == 0:
            logger.info(f"Saving checkpoint due to signal {checkpoint_on_signal}...")
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix=f"signal_{checkpoint_on_signal}")
        
        model.eval()  # important! This disables randomized embedding dropout
        logger.info("Done!")
        cleanup()


def parse_angle_list(angle_str):
    """Parse angle string into list of float angles."""
    if angle_str is None:
        return None
    
    angles = []
    parts = angle_str.split(',')
    for part in parts:
        angles.append(float(part))
    return angles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_rotation")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()) + list(UNet_models.keys()) + list(SongUNet_models.keys()), default="DiT-S/2")
    parser.add_argument("--architecture", type=str, choices=["dit", "unet", "songunet"], default="dit", help="Model architecture to use")
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Optional per-device micro batch size used with gradient accumulation")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--resume-from", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--selected-angles", type=str, default=None,
                        help="Comma-separated list of angles to train on")
    parser.add_argument("--max-samples-per-angle", type=int, default=None,
                        help="Maximum number of samples to use per angle")
    parser.add_argument("--rotation-text-table", type=str, default=None,
                        help="Precomputed CLIP text-embedding table; required for --rotation-embedding-type text")
    parser.add_argument("--rotation-embedding-type", type=str, 
                        choices=["circular_sinusoidal", "sinusoidal", "linear", "rotary", "text"], 
                        default="linear",
                        help="Type of rotation embedding to use") 
    parser.add_argument("--conditioning-method", type=str, choices=["adaln", "concat"], default="concat",
                        help="Conditioning method: adaln (adaptive layer norm) or concat (in-context)")
    parser.add_argument("--rotation-dropout-prob", type=float, default=0.0,
                        help="Rotation dropout probability for classifier-free guidance (0.0 to disable CFG)")
    parser.add_argument("--null-angle", type=float, default=180.0,
                        help="Null angle for unconditional generation in CFG")
    parser.add_argument("--null-embedding-type", type=str, choices=["zero", "learnable"], default="zero",
                        help="Type of null embedding to use for unconditional generation in CFG")
    parser.add_argument("--use-latent-diffusion", action="store_true", default=False,
                        help="Use VAE latent diffusion. Use --no-use-latent-diffusion for direct pixel diffusion")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="Choice of VAE model (doesn't affect training)")
    
    # Add flow matching arguments
    parser = add_flow_matching_args(parser)
    
    args = parser.parse_args()
    
    # Parse selected angles
    if args.selected_angles:
        args.selected_angles = parse_angle_list(args.selected_angles)
    
    main(args)
