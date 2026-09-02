# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
Modified for compositional conditioning on all 6 properties: radius, position, shape, color, count, rotation.
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

# Import compositional models
from vgl.models_compositional import DiT_models_compositional as DiT_models
from vgl.unet_models_song_compositional import CompositionalSongUNet_models as SongUNet_models
from vgl.diffusion import create_diffusion
# NEW: Import flow matching utilities
from vgl.flow_matching import create_loss_function, add_flow_matching_args

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


class ImageFolderWithComposition(ImageFolder):
    """Wrapper around ImageFolder that extracts compositional properties from folder names."""
    def __init__(self, root, transform=None, max_samples_per_combination=None,
                 include_properties=None, num_shapes=4, num_colors=8):
        super().__init__(root, transform)

        # Default to all properties if not specified
        if include_properties is None:
            include_properties = ['radius', 'position', 'shape', 'color']
        self.include_properties = include_properties

        # Parse folder names to extract compositional properties
        self.property_mapping = {}
        # Ids 0-3 are the paper's four shapes; 4-7 extend the vocabulary for the
        # coverage-vs-combination-count experiment.
        all_shapes = ['circle', 'square', 'triangle', 'diamond',
                      'pentagon', 'hexagon', 'star', 'cross']
        self.shape_to_id = {name: i for i, name in enumerate(all_shapes[:num_shapes])}
        # Ids 0-7 are the paper's eight colours; 8-15 extend the vocabulary.
        all_colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan', 'orange', 'purple',
                      'lime', 'teal', 'pink', 'brown', 'navy', 'maroon', 'olive', 'azure']
        self.color_to_id = {name: i for i, name in enumerate(all_colors[:num_colors])}
        
        for i, folder_name in enumerate(sorted(self.classes)):
            try:
                parts = folder_name.split('_')
                
                # Initialize properties with defaults
                properties = {
                    'radius': 10.0,
                    'position': (0.0, 0.0),
                    'shape': 0,     # circle
                    'color': 0,     # red
                    'count': 1,     # single object
                    'rotation': 0.0 # no rotation
                }
                
                # Detect format: NEW binary format (radius_10_position_0_5) vs OLD format (r6p0_xn5p0_y5p0)
                if 'radius' in folder_name or 'position' in folder_name or 'shape' in folder_name or 'color' in folder_name or 'count' in folder_name or 'rotation' in folder_name:
                    # NEW BINARY FORMAT: property_value_property_value
                    # Example: radius_10_position_0_5 or shape_circle_color_red
                    part_idx = 0
                    while part_idx < len(parts):
                        prop_name = parts[part_idx]
                        
                        if prop_name == 'radius' and part_idx + 1 < len(parts):
                            properties['radius'] = float(parts[part_idx + 1])
                            part_idx += 2
                        elif prop_name == 'position' and part_idx + 2 < len(parts):
                            x_str = parts[part_idx + 1].replace('n', '-')
                            y_str = parts[part_idx + 2].replace('n', '-')
                            properties['position'] = (float(x_str), float(y_str))
                            part_idx += 3
                        elif prop_name == 'shape' and part_idx + 1 < len(parts):
                            shape_val = parts[part_idx + 1]
                            properties['shape'] = self.shape_to_id.get(shape_val, 0)
                            part_idx += 2
                        elif prop_name == 'color' and part_idx + 1 < len(parts):
                            color_val = parts[part_idx + 1]
                            properties['color'] = self.color_to_id.get(color_val, 0)
                            part_idx += 2
                        elif prop_name == 'count' and part_idx + 1 < len(parts):
                            properties['count'] = int(parts[part_idx + 1])
                            part_idx += 2
                        elif prop_name == 'rotation' and part_idx + 1 < len(parts):
                            properties['rotation'] = float(parts[part_idx + 1])
                            part_idx += 2
                        else:
                            part_idx += 1
                else:
                    # OLD FORMAT: r6p0_xn5p0_y5p0_circle_red (backwards compatibility)
                    part_idx = 0
                    
                    # Parse radius (r6p0 -> 6.0, r7p6 -> 7.6)
                    if part_idx < len(parts) and parts[part_idx].startswith('r'):
                        radius_str = parts[part_idx][1:]  # Remove 'r'
                        radius_str = radius_str.replace('p', '.')
                        properties['radius'] = float(radius_str)
                        part_idx += 1
                    
                    # Parse x and y position (xn5p0_y5p0 -> (-5.0, 5.0))
                    if part_idx < len(parts) - 1 and parts[part_idx].startswith('x') and parts[part_idx + 1].startswith('y'):
                        x_str = parts[part_idx][1:]  # Remove 'x'
                        x_str = x_str.replace('n', '-').replace('p', '.')
                        x = float(x_str)
                        
                        y_str = parts[part_idx + 1][1:]  # Remove 'y'
                        y_str = y_str.replace('n', '-').replace('p', '.')
                        y = float(y_str)
                        
                        properties['position'] = (x, y)
                        part_idx += 2
                    
                    # Parse shape (circle -> 0)
                    if part_idx < len(parts) and parts[part_idx] in self.shape_to_id:
                        shape_str = parts[part_idx]
                        properties['shape'] = self.shape_to_id[shape_str]
                        part_idx += 1
                    
                    # Parse color (red -> 0)
                    if part_idx < len(parts) and parts[part_idx] in self.color_to_id:
                        color_str = parts[part_idx]
                        properties['color'] = self.color_to_id[color_str]
                        part_idx += 1
                    
                    # Parse count (cnt3 -> 3)
                    if part_idx < len(parts) and parts[part_idx].startswith('cnt'):
                        count_str = parts[part_idx][3:]  # Remove 'cnt'
                        properties['count'] = int(count_str)
                        part_idx += 1
                    
                    # Parse rotation (rot45 -> 45.0)
                    if part_idx < len(parts) and parts[part_idx].startswith('rot'):
                        rot_str = parts[part_idx][3:]  # Remove 'rot'
                        properties['rotation'] = float(rot_str)
                        part_idx += 1
                
                self.property_mapping[i] = properties
                
                # Debug: Print examples of successful parses
                if i < 3:
                    print(f"✓ Parsed folder '{folder_name}': {properties}")
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse folder name '{folder_name}': {e}")
                # Use default values
                self.property_mapping[i] = {
                    'radius': 10.0,
                    'position': (0.0, 0.0),
                    'shape': 0,     # circle
                    'color': 0,     # red  
                    'count': 1,     # single object
                    'rotation': 0.0 # no rotation
                }
        
        # Limit samples per combination if specified
        if max_samples_per_combination is not None:
            # Group samples by class
            samples_by_class = {}
            for path, target in self.samples:
                if target not in samples_by_class:
                    samples_by_class[target] = []
                samples_by_class[target].append((path, target))
            
            # Limit samples per class
            limited_samples = []
            for target, samples in samples_by_class.items():
                limited_samples.extend(samples[:max_samples_per_combination])
            
            self.samples = limited_samples
            self.targets = [target for _, target in limited_samples]

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Convert label index to compositional properties
        props = self.property_mapping[label]
        
        # Return only the properties that are being used in this experiment
        property_tensors = {}
        
        if 'radius' in self.include_properties:
            property_tensors['radius'] = torch.tensor(props['radius'], dtype=torch.float32)
        if 'position' in self.include_properties:
            property_tensors['position'] = torch.tensor(props['position'], dtype=torch.float32)
        if 'shape' in self.include_properties:
            property_tensors['shape'] = torch.tensor(props['shape'], dtype=torch.long)
        if 'color' in self.include_properties:
            property_tensors['color'] = torch.tensor(props['color'], dtype=torch.long)
        if 'count' in self.include_properties:
            property_tensors['count'] = torch.tensor(props['count'], dtype=torch.long)
        if 'rotation' in self.include_properties:
            property_tensors['rotation'] = torch.tensor(props['rotation'], dtype=torch.float32)
        
        return image, property_tensors


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """End DDP training."""
    dist.destroy_process_group()


def create_logger(logging_dir):
    """Create a logger that writes to a log file and stdout."""
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
    """Center cropping implementation from ADM."""
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
    """Trains a new DiT model with compositional conditioning."""
    global should_stop, checkpoint_on_signal
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        
        if args.resume_from:
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_from))
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-compositional"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Active properties: {args.include_properties}")
    else:
        logger = create_logger(None)
        if args.resume_from:
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_from))
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # Create model:
    # Use pixel diffusion for simplicity (no VAE)
    input_size = args.image_size
    in_channels = 3  # RGB channels
    logger.info(f"Using direct pixel diffusion: input_size={input_size}, in_channels={in_channels}")
    
    # Create compositional model
    # Select model based on architecture
    if args.architecture == "dit":
        model = DiT_models[args.model](
            input_size=input_size,
            in_channels=in_channels,
            radius_embedding_type=args.radius_embedding_type,
            position_embedding_type=args.position_embedding_type,
            conditioning_method=args.conditioning_method,
            property_dropout_prob=args.property_dropout_prob,
            null_radius=args.null_radius,
            null_position=tuple(args.null_position),
            null_shape_id=args.null_shape_id,
            null_color_id=args.null_color_id,
            null_count=args.null_count,
            null_rotation=args.null_rotation,
            null_embedding_type=args.null_embedding_type,
            num_shapes=args.num_shapes,
            num_colors=args.num_colors,
            active_properties=args.include_properties  # Pass active properties to model
        )
    elif args.architecture == "songunet":
        model = SongUNet_models[args.model](
            img_resolution=input_size,
            in_channels=in_channels,
            out_channels=in_channels,  # SongUNet doesn't learn sigma by default
            active_properties=args.include_properties,
            conditioning_method=args.conditioning_method,
            radius_embedding_type=args.radius_embedding_type,
            position_embedding_type=args.position_embedding_type,
            property_dropout_prob=args.property_dropout_prob,
            num_shapes=args.num_shapes,
            num_colors=args.num_colors,
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    ema = deepcopy(model).to(device)
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
    
    logger.info("Using direct pixel diffusion (no VAE)")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
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
            start_epoch = checkpoint.get("epoch", 0) + 1
            
            if train_steps == 0:
                import re
                filename = os.path.basename(resume_checkpoint)
                match = re.search(r'(\d{7,})\.pt', filename)
                if match:
                    train_steps = int(match.group(1))
                    steps_per_epoch = 100  # Approximate
                    start_epoch = train_steps // steps_per_epoch
                    logger.info(f"Extracted from filename: step {train_steps}, calculated epoch {start_epoch}")
            
            start_epoch = start_epoch + 1
            logger.info(f"Resumed from epoch {start_epoch}, step {train_steps}")
        else:
            logger.warning(f"Could not find checkpoint to resume from: {args.resume_from}")
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    dataset = ImageFolderWithComposition(
        args.data_path, 
        transform=transform,
        max_samples_per_combination=args.max_samples_per_combination,
        include_properties=args.include_properties,
        num_shapes=args.num_shapes,
        num_colors=args.num_colors
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Property mapping example: {list(dataset.property_mapping.items())[:3]}")

    # Prepare models for training:
    if train_steps == 0:
        update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    epoch = start_epoch

    logger.info(f"Training for {args.epochs} epochs (starting from epoch {start_epoch})...")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            if should_stop:
                logger.info("Stopping training due to signal...")
                break
                
            sampler.set_epoch(epoch)
            logger.info(f"Beginning epoch {epoch}...")
            
            for batch_idx, (x, property_dict) in enumerate(loader):
                if should_stop:
                    logger.info("Stopping training due to signal...")
                    break
                    
                x = x.to(device)
                x = x.contiguous()

                # Move property tensors to device and build model_kwargs dynamically
                model_kwargs = {}
                
                # Only include properties that are both active and present in the data
                if 'radius' in args.include_properties and 'radius' in property_dict:
                    model_kwargs['radius'] = property_dict['radius'].to(device)
                
                if 'position' in args.include_properties and 'position' in property_dict:
                    model_kwargs['position'] = property_dict['position'].to(device)
                
                if 'shape' in args.include_properties and 'shape' in property_dict:
                    model_kwargs['shape'] = property_dict['shape'].to(device)
                
                if 'color' in args.include_properties and 'color' in property_dict:
                    model_kwargs['color'] = property_dict['color'].to(device)
                
                if 'count' in args.include_properties and 'count' in property_dict:
                    model_kwargs['count'] = property_dict['count'].to(device)
                
                if 'rotation' in args.include_properties and 'rotation' in property_dict:
                    model_kwargs['rotation'] = property_dict['rotation'].to(device)
                
                # Use images directly (direct pixel diffusion)
                if not args.use_flow_matching:
                    # Diffusion: sample timesteps
                    t = torch.randint(0, loss_fn.num_timesteps, (x.shape[0],), device=device)
                else:
                    # Flow matching: timesteps are sampled internally, pass dummy timesteps
                    t = torch.zeros(x.shape[0], device=device)
                
                # Use training_losses function (works for both diffusion and flow matching)
                loss_dict = loss_fn.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {train_steps}, skipping batch")
                    continue
                
                # Check for NaN/Inf in properties
                for prop_name, prop_val in model_kwargs.items():
                    if torch.isnan(prop_val).any() or torch.isinf(prop_val).any():
                        logger.warning(f"NaN/Inf in {prop_name} at step {train_steps}")
                        continue
                
                opt.zero_grad()
                loss.backward()
                
                # Check gradients
                has_nan_grad = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.warning(f"NaN/Inf gradient in {name} at step {train_steps}")
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    logger.warning("Skipping step due to NaN/Inf gradients")
                    opt.zero_grad()
                    continue
                
                opt.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                torch.cuda.synchronize()
                update_ema(ema, model.module)

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                
                if train_steps % args.log_every == 0:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    
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
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix="emergency")
        raise
    
    finally:
        # Save checkpoint if stopped by signal
        if should_stop and rank == 0:
            logger.info(f"Saving checkpoint due to signal {checkpoint_on_signal}...")
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix=f"signal_{checkpoint_on_signal}")
        
        model.eval()
        logger.info("Done!")
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results_compositional")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()) + list(SongUNet_models.keys()), default="DiT-S/2")
    parser.add_argument("--architecture", type=str, choices=["dit", "songunet"], default="dit", help="Model architecture to use")
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-samples-per-combination", type=int, default=None)
    
    # Property embedding settings
    parser.add_argument("--radius-embedding-type", type=str, choices=["sinusoidal", "linear"], default="linear")
    parser.add_argument("--position-embedding-type", type=str, choices=["sinusoidal", "linear"], default="linear")
    parser.add_argument("--rotation-embedding-type", type=str, choices=["sinusoidal", "linear"], default="linear")

    
    # Conditioning settings
    parser.add_argument("--conditioning-method", type=str, choices=["adaln", "concat"], default="concat")
    
    # CFG settings
    parser.add_argument("--property-dropout-prob", type=float, default=0.0)
    parser.add_argument("--null-radius", type=float, default=0.0)
    parser.add_argument("--null-position", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--num-colors", type=int, default=8,
                        help="Size of the colour vocabulary (8 = paper default, 16 = extended).")
    parser.add_argument("--num-shapes", type=int, default=4,
                        help="Size of the shape vocabulary (4 = paper default, 8 = extended vocabulary).")
    parser.add_argument("--null-shape-id", type=int, default=0)
    parser.add_argument("--null-color-id", type=int, default=0)
    parser.add_argument("--null-count", type=int, default=1)
    parser.add_argument("--null-rotation", type=float, default=0.0)
    parser.add_argument("--null-embedding-type", type=str, choices=["learnable", "fixed", "none"], default="learnable")
    parser.add_argument("--include-properties", type=str, nargs='+', 
                       choices=['radius', 'position', 'shape', 'color', 'count', 'rotation'],
                       default=['radius', 'position', 'shape', 'color'],
                       help='Properties to include in training (minimum 2 required)')
    
    # Add flow matching arguments
    parser = add_flow_matching_args(parser)
    
    args = parser.parse_args()
    
    # Validate minimum properties
    if len(args.include_properties) < 2:
        parser.error("Must include at least 2 properties")
    
    main(args)