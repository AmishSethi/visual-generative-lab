"""
Flow Matching implementation for PyTorch.
Based on the JAX implementation in dit_flow/flow.py and dit_flow/dit_trainer.py.

This module provides:
1. Flow matching loss function
2. Flow matching sampling utilities
3. Integration with existing diffusion framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, Callable, Tuple


class FlowMatching:
    """
    Flow matching implementation for continuous normalizing flows.
    
    This class provides the flow matching loss and sampling methods
    that can be used as an alternative to diffusion training.
    """
    
    def __init__(self, 
                 sigma_min: float = 0.0,
                 sigma_data: float = 1.0,
                 use_sigmoid_time: bool = True):
        """
        Initialize flow matching.
        
        Args:
            sigma_min: Minimum noise level (typically 0.0 for flow matching)
            sigma_data: Data scaling factor
            use_sigmoid_time: Whether to apply sigmoid to time samples
        """
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.use_sigmoid_time = use_sigmoid_time
        
    def flow_loss(self, 
                  model: nn.Module, 
                  x_1: torch.Tensor, 
                  model_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        Args:
            model: The neural network model (DiT, UNet, etc.)
            x_1: Clean data samples (target)
            model_kwargs: Additional arguments for the model (conditioning, etc.)
            
        Returns:
            Flow matching loss
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        batch_size = x_1.shape[0]
        device = x_1.device
        
        # Sample noise (source distribution)
        x_0 = torch.randn_like(x_1)
        
        # Sample time uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)
        
        # Apply sigmoid to time if specified (as in JAX implementation)
        if self.use_sigmoid_time:
            t = torch.sigmoid(t)
        
        # Expand time to match data dimensions for broadcasting
        # Shape: [batch_size, 1, 1, 1] for 4D tensors (images)
        t_expanded = t.view(batch_size, *([1] * (len(x_1.shape) - 1)))
        
        # Linear interpolation between noise and data
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # True velocity field (vector field we want to learn)
        v_true = x_1 - x_0
        
        # Model prediction of velocity field
        model_output = model(x_t, t, **model_kwargs)
        
        # Handle models that output both prediction and sigma (like DiT with learn_sigma=True)
        if model_output.shape[1] > v_true.shape[1]:
            # Take only the first channels that match the target (ignore sigma prediction)
            v_pred = model_output[:, :v_true.shape[1]]
        else:
            v_pred = model_output
        
        # Mean squared error loss
        loss = F.mse_loss(v_pred, v_true, reduction='none')
        
        # Average over all dimensions except batch
        loss = loss.view(batch_size, -1).mean(dim=1)
        
        # Average over batch
        return loss.mean()
    
    def flow_step(self, 
                  model: nn.Module,
                  x_t: torch.Tensor,
                  t_start: float,
                  t_end: float,
                  model_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Perform a single flow step using Euler's method with midpoint evaluation.
        
        Args:
            model: The neural network model
            x_t: Current state
            t_start: Start time
            t_end: End time
            model_kwargs: Additional arguments for the model
            
        Returns:
            Next state x_{t+dt}
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Midpoint time
        t_mid = (t_start + t_end) / 2.0
        
        # Create time tensor for batch
        batch_size = x_t.shape[0]
        t_mid_tensor = torch.full((batch_size,), t_mid, device=x_t.device)
        
        # Evaluate vector field at midpoint
        with torch.no_grad():
            model_output = model(x_t, t_mid_tensor, **model_kwargs)
            
            # Handle models that output both prediction and sigma
            if model_output.shape[1] > x_t.shape[1]:
                # Take only the first channels that match input (ignore sigma prediction)
                v_mid = model_output[:, :x_t.shape[1]]
            else:
                v_mid = model_output
        
        # Euler step
        dt = t_end - t_start
        x_next = x_t + dt * v_mid
        
        return x_next
    
    def sample(self, 
               model: nn.Module,
               shape: Tuple[int, ...],
               num_steps: int = 50,
               model_kwargs: Optional[Dict[str, Any]] = None,
               device: Optional[torch.device] = None,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Generate samples using flow matching.
        
        Args:
            model: The neural network model
            shape: Shape of samples to generate
            num_steps: Number of integration steps
            model_kwargs: Additional arguments for the model
            device: Device to generate samples on
            generator: Random number generator
            
        Returns:
            Generated samples
        """
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
            
        # Initialize from noise (source distribution)
        x_t = torch.randn(shape, device=device, generator=generator)
        
        # Time steps from 0 to 1
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        
        model.eval()
        with torch.no_grad():
            for k in tqdm(range(num_steps), desc="Flow matching sampling"):
                t_start = time_steps[k].item()
                t_end = time_steps[k + 1].item()
                
                x_t = self.flow_step(
                    model=model,
                    x_t=x_t,
                    t_start=t_start,
                    t_end=t_end,
                    model_kwargs=model_kwargs
                )
        
        return x_t
    
    def sample_with_cfg(self,
                       model: nn.Module,
                       shape: Tuple[int, ...],
                       num_steps: int = 50,
                       cfg_scale: float = 1.0,
                       model_kwargs: Optional[Dict[str, Any]] = None,
                       device: Optional[torch.device] = None,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Generate samples using flow matching with classifier-free guidance.
        
        Args:
            model: The neural network model
            shape: Shape of samples to generate
            num_steps: Number of integration steps
            cfg_scale: Classifier-free guidance scale
            model_kwargs: Additional arguments for the model (should contain conditional and unconditional kwargs)
            device: Device to generate samples on
            generator: Random number generator
            
        Returns:
            Generated samples
        """
        if model_kwargs is None:
            model_kwargs = {}
        if device is None:
            device = next(model.parameters()).device
            
        # Initialize from noise
        x_t = torch.randn(shape, device=device, generator=generator)
        
        # Time steps from 0 to 1
        time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        
        model.eval()
        with torch.no_grad():
            for k in tqdm(range(num_steps), desc="Flow matching sampling with CFG"):
                t_start = time_steps[k].item()
                t_end = time_steps[k + 1].item()
                
                # Midpoint time
                t_mid = (t_start + t_end) / 2.0
                batch_size = x_t.shape[0]
                t_mid_tensor = torch.full((batch_size,), t_mid, device=device)
                
                if cfg_scale == 1.0 or not hasattr(model, 'forward_with_cfg'):
                    # No CFG or model doesn't support CFG
                    model_output = model(x_t, t_mid_tensor, **model_kwargs)
                else:
                    # Use CFG if model supports it
                    model_output = model.forward_with_cfg(x_t, t_mid_tensor, cfg_scale=cfg_scale, **model_kwargs)
                
                # Handle models that output both prediction and sigma
                if model_output.shape[1] > x_t.shape[1]:
                    # Take only the first channels that match input (ignore sigma prediction)
                    v_mid = model_output[:, :x_t.shape[1]]
                else:
                    v_mid = model_output
                
                # Euler step
                dt = t_end - t_start
                x_t = x_t + dt * v_mid
        
        return x_t


def create_flow_matching(sigma_min: float = 0.0,
                        sigma_data: float = 1.0,
                        use_sigmoid_time: bool = True) -> FlowMatching:
    """
    Create a FlowMatching instance with specified parameters.
    
    Args:
        sigma_min: Minimum noise level
        sigma_data: Data scaling factor  
        use_sigmoid_time: Whether to apply sigmoid to time samples
        
    Returns:
        FlowMatching instance
    """
    return FlowMatching(
        sigma_min=sigma_min,
        sigma_data=sigma_data,
        use_sigmoid_time=use_sigmoid_time
    )


class FlowMatchingLosses:
    """
    A wrapper class that provides the same interface as diffusion losses
    but uses flow matching internally. This allows easy integration with
    existing training code.
    """
    
    def __init__(self, flow_matching: FlowMatching):
        self.flow_matching = flow_matching
        self.num_timesteps = 1000  # Dummy value for compatibility
        
    def training_losses(self, 
                       model: nn.Module,
                       x_start: torch.Tensor,
                       t: torch.Tensor,  # This will be ignored in flow matching
                       model_kwargs: Optional[Dict[str, Any]] = None,
                       noise: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute training losses using flow matching.
        
        This method has the same signature as diffusion training_losses
        to maintain compatibility with existing training code.
        
        Args:
            model: The neural network model
            x_start: Clean data samples
            t: Timesteps (ignored in flow matching)
            model_kwargs: Additional model arguments
            noise: Noise samples (ignored, we sample fresh noise)
            
        Returns:
            Dictionary containing the loss
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Compute flow matching loss
        loss = self.flow_matching.flow_loss(model, x_start, model_kwargs)
        
        return {"loss": loss}


# Utility function to check if we should use flow matching
def should_use_flow_matching(args) -> bool:
    """Check if flow matching should be used based on arguments."""
    return getattr(args, 'use_flow_matching', False)


# Factory function for creating the appropriate loss function
def create_loss_function(args, **kwargs):
    """
    Create either a flow matching or diffusion loss function based on arguments.
    
    Args:
        args: Arguments object containing use_flow_matching flag
        **kwargs: Additional arguments for loss function creation
        
    Returns:
        Loss function object with training_losses method
    """
    if should_use_flow_matching(args):
        # Create flow matching loss
        flow_matching = create_flow_matching(
            sigma_min=getattr(args, 'flow_sigma_min', 0.0),
            sigma_data=getattr(args, 'flow_sigma_data', 1.0),
            use_sigmoid_time=getattr(args, 'flow_use_sigmoid_time', True)
        )
        return FlowMatchingLosses(flow_matching)
    else:
        # Use existing diffusion creation
        from vgl.diffusion import create_diffusion
        return create_diffusion(**kwargs)


# Integration utilities
def add_flow_matching_args(parser):
    """Add flow matching arguments to argument parser."""
    parser.add_argument("--use-flow-matching", action="store_true", default=False,
                       help="Use flow matching objective instead of diffusion")
    parser.add_argument("--flow-sigma-min", type=float, default=0.0,
                       help="Minimum noise level for flow matching")
    parser.add_argument("--flow-sigma-data", type=float, default=1.0,
                       help="Data scaling factor for flow matching")
    parser.add_argument("--flow-use-sigmoid-time", action="store_true", default=True,
                       help="Apply sigmoid to time samples in flow matching")
    parser.add_argument("--flow-num-steps", type=int, default=50,
                       help="Number of steps for flow matching sampling")
    return parser
