"""
Training script for unconditional compositional DiT model on bimodal compositional dataset.
Trains without any property conditioning to learn compositional structure implicitly.
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

# Import unconditional compositional models
from vgl.models_compositional_unconditional import DiT_models_compositional_unconditional as DiT_models
from vgl.diffusion import create_diffusion
from diffusers.models import AutoencoderKL


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


def analyze_dataset_distribution(dataset, logger):
    """Analyze the compositional distribution in the dataset."""
    try:
        # Count samples per class
        class_counts = {}
        for _, label in dataset.samples:
            class_name = dataset.classes[label]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # Parse class names to extract property values
        radius_values = []
        position_x_values = []
        position_y_values = []
        
        for class_name in class_counts.keys():
            try:
                # Parse class names like "r8_xn12_y12" or "r18_x12_yn12"
                parts = class_name.split('_')
                for part in parts:
                    if part.startswith('r') and len(part) > 1:
                        radius_val = float(part[1:])
                        radius_values.append(radius_val)
                    elif part.startswith('x'):
                        x_str = part[1:].replace('n', '-')
                        x_val = float(x_str)
                        position_x_values.append(x_val)
                    elif part.startswith('y'):
                        y_str = part[1:].replace('n', '-')
                        y_val = float(y_str)
                        position_y_values.append(y_val)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse class name: {class_name}")
                continue
        
        # Calculate distribution statistics
        distribution = {
            'total_samples': len(dataset.samples),
            'unique_classes': len(class_counts),
            'class_counts': class_counts,
        }
        
        if radius_values:
            distribution['radius'] = {
                'min': min(radius_values),
                'max': max(radius_values),
                'mean': np.mean(radius_values),
                'std': np.std(radius_values),
                'unique_values': sorted(list(set(radius_values)))
            }
        
        if position_x_values:
            distribution['position_x'] = {
                'min': min(position_x_values),
                'max': max(position_x_values),
                'mean': np.mean(position_x_values),
                'std': np.std(position_x_values),
                'unique_values': sorted(list(set(position_x_values)))
            }
        
        if position_y_values:
            distribution['position_y'] = {
                'min': min(position_y_values),
                'max': max(position_y_values),
                'mean': np.mean(position_y_values),
                'std': np.std(position_y_values),
                'unique_values': sorted(list(set(position_y_values)))
            }
        
        return distribution
        
    except Exception as e:
        logger.warning(f"Could not analyze dataset distribution: {e}")
        return {
            'total_samples': len(dataset.samples),
            'error': str(e)
        }


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains an unconditional compositional DiT model.
    """
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
            # Extract experiment directory from checkpoint path
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_from))
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            # Create descriptive experiment name
            dataset_name = os.path.basename(args.data_path.rstrip('/'))
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-compositional-unconditional-{dataset_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        logger.info(f"Training UNCONDITIONAL COMPOSITIONAL model")
        logger.info(f"Dataset: {args.data_path}")
    else:
        logger = create_logger(None)
        if args.resume_from:
            experiment_dir = os.path.dirname(os.path.dirname(args.resume_from))
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # Create model:
    if args.use_latent_diffusion:
        assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        input_size = args.image_size // 8
        in_channels = 4
        logger.info(f"Using latent diffusion: input_size={input_size}, in_channels={in_channels}")
    else:
        input_size = args.image_size
        in_channels = 3
        logger.info(f"Using direct pixel diffusion: input_size={input_size}, in_channels={in_channels}")
    
    # Create unconditional compositional model
    model_kwargs = {
        'input_size': input_size,
        'in_channels': in_channels,
    }
    
    # Add architectural options
    if args.use_deeper_network:
        model_kwargs['use_deeper_network'] = True
        logger.info("Using deeper network variant for compositional learning")
    
    if args.use_positional_bias:
        model_kwargs['use_positional_bias'] = True
        logger.info("Using positional bias for spatial relationships")
    
    model = DiT_models[args.model](**model_kwargs)
    
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    
    # Load VAE if using latent diffusion
    vae = None
    if args.use_latent_diffusion:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        logger.info("Loaded VAE for latent diffusion")
    else:
        logger.info("Skipping VAE loading for direct pixel diffusion")
    
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
            logger.info(f"Resumed from epoch {start_epoch}, step {train_steps}")
        else:
            logger.warning(f"Could not find checkpoint to resume from: {args.resume_from}")
    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    # Use standard ImageFolder (we don't need property values for unconditional training)
    dataset = ImageFolder(args.data_path, transform=transform)
    
    # Limit samples per class if specified
    if args.max_samples_per_class is not None:
        samples_by_class = {}
        for path, target in dataset.samples:
            if target not in samples_by_class:
                samples_by_class[target] = []
            samples_by_class[target].append((path, target))
        
        limited_samples = []
        for target, samples in samples_by_class.items():
            limited_samples.extend(samples[:args.max_samples_per_class])
        
        dataset.samples = limited_samples
        dataset.targets = [target for _, target in limited_samples]
    
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
    
    # Analyze and save dataset distribution
    if rank == 0:
        distribution = analyze_dataset_distribution(dataset, logger)
        logger.info(f"Dataset analysis:")
        logger.info(f"  Total samples: {distribution.get('total_samples', len(dataset))}")
        logger.info(f"  Unique classes: {distribution.get('unique_classes', 'N/A')}")
        
        if 'radius' in distribution:
            r_stats = distribution['radius']
            logger.info(f"  Radius range: {r_stats['min']:.1f} to {r_stats['max']:.1f} (mean: {r_stats['mean']:.1f}, std: {r_stats['std']:.1f})")
            logger.info(f"  Unique radius values: {len(r_stats['unique_values'])}")
        
        if 'position_x' in distribution:
            x_stats = distribution['position_x']
            logger.info(f"  Position X range: {x_stats['min']:.1f} to {x_stats['max']:.1f} (mean: {x_stats['mean']:.1f}, std: {x_stats['std']:.1f})")
        
        if 'position_y' in distribution:
            y_stats = distribution['position_y']
            logger.info(f"  Position Y range: {y_stats['min']:.1f} to {y_stats['max']:.1f} (mean: {y_stats['mean']:.1f}, std: {y_stats['std']:.1f})")
        
        # Save distribution info
        with open(os.path.join(experiment_dir, 'dataset_distribution.json'), 'w') as f:
            json.dump(distribution, f, indent=2, default=str)
        
        # Save training configuration
        training_config = {
            'model': args.model,
            'dataset_path': args.data_path,
            'image_size': args.image_size,
            'global_batch_size': args.global_batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'use_latent_diffusion': args.use_latent_diffusion,
            'use_deeper_network': args.use_deeper_network,
            'use_positional_bias': args.use_positional_bias,
            'experiment_type': 'unconditional_compositional',
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(experiment_dir, 'training_config.json'), 'w') as f:
            json.dump(training_config, f, indent=2)

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

    logger.info(f"Training UNCONDITIONAL COMPOSITIONAL model for {args.epochs} epochs (starting from epoch {start_epoch})...")
    logger.info(f"Model will learn compositional structure implicitly without property conditioning")
    
    try:
        for epoch in range(start_epoch, args.epochs):
            if should_stop:
                logger.info("Stopping training due to signal...")
                break
                
            sampler.set_epoch(epoch)
            logger.info(f"Beginning epoch {epoch}...")
            
            for batch_idx, (x, _) in enumerate(loader):  # Ignore labels - this is unconditional
                if should_stop:
                    logger.info("Stopping training due to signal...")
                    break
                    
                x = x.to(device)
                x = x.contiguous()
                
                # Handle input based on diffusion mode
                if args.use_latent_diffusion:
                    with torch.no_grad():
                        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                
                # No model_kwargs for unconditional training
                model_kwargs = dict()
                
                # Calculate loss
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss detected at step {train_steps}, skipping batch")
                    continue
                
                opt.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                opt.step()
                
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

                # Save checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger)
                    dist.barrier()
                
                # Periodic memory cleanup
                if train_steps % 500 == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Save final checkpoint
        if not should_stop and rank == 0:
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix="final")
            
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        if rank == 0:
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix="emergency")
        raise
    
    finally:
        if should_stop and rank == 0:
            logger.info(f"Saving checkpoint due to signal {checkpoint_on_signal}...")
            save_checkpoint(checkpoint_dir, model, ema, opt, train_steps, epoch, logger, prefix=f"signal_{checkpoint_on_signal}")
        
        model.eval()
        logger.info("Done!")
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to the bimodal compositional dataset")
    parser.add_argument("--results-dir", type=str, default="results_compositional_unconditional")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument("--resume-from", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max-samples-per-class", type=int, default=None,
                        help="Maximum number of samples to use per class")
    parser.add_argument("--use-latent-diffusion", action="store_true", default=False,
                        help="Use VAE latent diffusion")
    
    # Optimization parameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    
    # Architectural options for compositional learning
    parser.add_argument("--use-deeper-network", action="store_true", default=False,
                        help="Use deeper network variant for complex compositional patterns")
    parser.add_argument("--use-positional-bias", action="store_true", default=False,
                        help="Use positional bias for spatial relationships")
    
    args = parser.parse_args()
    main(args)

