# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
Modified for continuous position (x,y) conditioning with checkpoint resuming and signal handling.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import functools, torch, torch.distributed as dist

def _wrap_collective(fn):
    @functools.wraps(fn)
    def _sync(*args, **kw):
        tensor = args[0] if args else kw.get('tensor', None)
        name   = fn.__name__

        # ------------ diagnostics ------------
        if isinstance(tensor, torch.Tensor):
            assert tensor.is_contiguous(), \
                   f"[rank {dist.get_rank()}] non-contiguous tensor to {name}"
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        # ------------ real call --------------
        # force asynchronous mode so we *always* get a Work handle
        kw['async_op'] = True
        work = fn(*args, **kw)
        work.wait()          # blocks until the NCCL kernel finishes
        return work
    return _sync

for _name in ("all_reduce", "all_gather", "reduce_scatter"):
    if hasattr(dist, _name):
        setattr(dist, _name, _wrap_collective(getattr(dist, _name)))
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset  
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

# Import position models instead of radius models
from vgl.models_position import DiT_models_position as DiT_models
from vgl.unet_models_position import UNet_models_position as UNet_models
from vgl.unet_models_song_position import SongUNet_Position_models as SongUNet_models
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


class ImageFolderWithPosition(ImageFolder):
    """Wrapper around ImageFolder that converts folder names to position coordinates."""
    def __init__(self, root, transform=None, selected_positions=None, max_samples_per_position=None):
        super().__init__(root, transform)
        
        # Parse folder names to extract (x,y) coordinates
        # Expected format: "1p0_2p0" for (1.0, 2.0), or "neg1p0_2p0" for (-1.0, 2.0)
        self.position_mapping = {}
        for i, folder_name in enumerate(sorted(self.classes)):
            x_str, y_str = folder_name.split('_')
            
            # Parse x coordinate
            x_str = x_str.replace('neg', '-').replace('p', '.')
            x = float(x_str)
            
            # Parse y coordinate  
            y_str = y_str.replace('neg', '-').replace('p', '.')
            y = float(y_str)
            
            self.position_mapping[i] = (x, y)
        
        # Filter dataset to only include selected positions
        if selected_positions is not None:
            # Convert selected_positions to strings for matching
            selected_folders = []
            for x, y in selected_positions:
                # Convert (x,y) back to folder name format
                x_str = f"{x:.1f}".replace('-', 'neg').replace('.', 'p')
                y_str = f"{y:.1f}".replace('-', 'neg').replace('.', 'p')
                folder_name = f"{x_str}_{y_str}"
                selected_folders.append(folder_name)
            
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
            
            # Update position mapping to only include selected positions
            filtered_position_mapping = {}
            for new_idx, old_idx in enumerate(selected_indices):
                filtered_position_mapping[new_idx] = self.position_mapping[old_idx]
            self.position_mapping = filtered_position_mapping
        
        # Limit samples per position if specified
        if max_samples_per_position is not None:
            # Group samples by position class
            samples_by_class = {}
            for path, target in self.samples:
                if target not in samples_by_class:
                    samples_by_class[target] = []
                samples_by_class[target].append((path, target))
            
            # Limit samples per position
            limited_samples = []
            for target, samples in samples_by_class.items():
                limited_samples.extend(samples[:max_samples_per_position])
            
            self.samples = limited_samples
            self.targets = [target for _, target in limited_samples]
            
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Convert label index to (x,y) position coordinates
        x, y = self.position_mapping[label]
        position = torch.tensor([x, y], dtype=torch.float32)
        return image, position


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
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
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
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
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
    Trains a new DiT model with position conditioning.
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
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-position"  # Create an experiment folder
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
    
    # Create position-conditioned model
    # Select model based on architecture
    if args.architecture == "dit":
        model = DiT_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            position_embedding_type=args.position_embedding_type,
            conditioning_method=args.conditioning_method,
            position_dropout_prob=args.position_dropout_prob,
            null_position=args.null_position,
            null_embedding_type=args.null_embedding_type
        )
    elif args.architecture == "unet":
        model = UNet_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            position_embedding_type=args.position_embedding_type,
            conditioning_method=args.conditioning_method,
            position_dropout_prob=args.position_dropout_prob,
            max_position_value=1.0  # Assuming normalized positions
        )
    elif args.architecture == "songunet":
        model = SongUNet_models[args.model](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=in_channels,  # SongUNet doesn't learn sigma by default
            # learn_sigma defaults to False in SongUNet to match original implementation
            position_embedding_type=args.position_embedding_type,
            conditioning_method=args.conditioning_method,
            position_dropout_prob=args.position_dropout_prob,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    # Note that parameter initialization is done within the model constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    # Use find_unused_parameters=True for SongUNet to avoid DDP errors
    # Also needed for learnable null embedding which isn't used when dropout_prob=0
    needs_unused = args.architecture == "songunet" or args.null_embedding_type == "learnable"
    if needs_unused:
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
    print(f"Using {objective_type} training objective")
    
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
            
            # Extract from filename if needed
            if train_steps == 0:
                import re
                filename = os.path.basename(resume_checkpoint)
                match = re.search(r'(\d{7,})\.pt', filename)
                if match:
                    train_steps = int(match.group(1))
                    steps_per_epoch = 62  # Approximate
                    start_epoch = train_steps // steps_per_epoch
                    logger.info(f"Extracted from filename: step {train_steps}, calculated epoch {start_epoch}")
            
            start_epoch = start_epoch + 1  # Start from next epoch
            logger.info(f"Resumed from epoch {start_epoch}, step {train_steps}")
        else:
            logger.warning(f"Could not find checkpoint to resume from: {args.resume_from}")
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Use ImageFolderWithPosition instead of ImageFolderWithRadius
    dataset = ImageFolderWithPosition(
        args.data_path, 
        transform=transform, 
        selected_positions=args.selected_positions, 
        max_samples_per_position=args.max_samples_per_position
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
    mapping_preview = list(dataset.position_mapping.items())[:5]
    logger.info(
        "Position mapping preview: %s ... (%d total positions)",
        mapping_preview,
        len(dataset.position_mapping),
    )
    logger.info(
        "Effective global batch size: %d (micro batch %d x world_size %d x grad_accum_steps %d)",
        args.global_batch_size,
        per_device_batch_size,
        dist.get_world_size(),
        grad_accum_steps,
    )
    if args.selected_positions is not None:
        logger.info(f"Training on selected positions: {args.selected_positions}")
    else:
        logger.info("Training on all available positions")

    # Prepare models for training:
    if train_steps == 0:  # Only reset EMA if starting fresh
        update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    epoch = start_epoch  # Initialize epoch variable to avoid UnboundLocalError

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
            
            # Rename pos variable to match the position conditioning
            for batch_idx, (x, pos) in enumerate(loader):
                if should_stop:
                    logger.info("Stopping training due to signal...")
                    break
                    
                x = x.to(device)
                x = x.contiguous()

                pos = pos.to(device)  # This is now position coordinates (N, 2)
                
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
                
                # Pass pos (position) instead of r (radius) in model_kwargs
                model_kwargs = dict(pos=pos)
                
                # Use training_losses function (works for both diffusion and flow matching)
                loss_dict = loss_fn.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {train_steps}, skipping batch")
                    opt.zero_grad()
                    accum_loss = 0.0
                    accum_steps = 0
                    continue
                
                # Check position values
                if torch.isnan(pos).any() or torch.isinf(pos).any():
                    logger.warning(f"NaN/Inf in position values at step {train_steps}")
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
                
                opt.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def parse_position_list(position_str):
    """Parse position string into list of (x,y) tuples."""
    if position_str is None:
        return None
    
    positions = []
    pairs = position_str.split(',')
    for pair in pairs:
        x, y = map(float, pair.split('_'))
        positions.append((x, y))
    return positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_position")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()) + list(UNet_models.keys()) + list(SongUNet_models.keys()), default="DiT-S/2")
    parser.add_argument("--architecture", type=str, choices=["dit", "unet", "songunet"], default="dit", help="Model architecture to use")
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Per-device micro batch size. If unset, uses global_batch_size/world_size.")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--resume-from", type=str, default=None, 
                        help="Path to checkpoint to resume from, or 'latest' to find the latest checkpoint")
    parser.add_argument("--selected-positions", type=str, default=None,
                        help="Comma-separated list of positions to train on (format: 'x1_y1,x2_y2,...')")
    parser.add_argument("--max-samples-per-position", type=int, default=None,
                        help="Maximum number of samples to use per position (for faster training)")
    parser.add_argument("--position-embedding-type", type=str, choices=["sinusoidal", "linear", "rotary"], default="sinusoidal",
                        help="Type of position embedding to use") 
    parser.add_argument("--conditioning-method", type=str, choices=["adaln", "concat"], default="concat",
                        help="Conditioning method: adaln (adaptive layer norm) or concat (in-context)")
    parser.add_argument("--position-dropout-prob", type=float, default=0.1,
                        help="Position dropout probability for classifier-free guidance (0.0 to disable CFG)")
    parser.add_argument("--null-position", type=float, nargs=2, default=[0.0, 0.0],
                        help="Null position values for unconditional generation in CFG (x y)")
    parser.add_argument("--null-embedding-type", type=str, choices=["zero", "learnable"], default="learnable",
                        help="Type of null embedding to use for unconditional generation in CFG")
    parser.add_argument("--use-latent-diffusion", action="store_true", default=False,
                        help="Use VAE latent diffusion. Use --no-use-latent-diffusion for direct pixel diffusion")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema",
                        help="Choice of VAE model (doesn't affect training)")
    
    # Add flow matching arguments
    parser = add_flow_matching_args(parser)
    
    args = parser.parse_args()
    
    # Parse selected positions
    if args.selected_positions:
        args.selected_positions = parse_position_list(args.selected_positions)
    
    # Convert null_position list to tuple
    args.null_position = tuple(args.null_position)
    
    main(args) 
