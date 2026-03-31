import torch
from torchtitan.tools.logging import logger

class NumericalHealthMonitor:
    """Monitor for numerical stability and training health."""
    def __init__(self, grad_norm_threshold: float = 100.0, reward_std_threshold: float = 1e-6):
        self.grad_norm_threshold = grad_norm_threshold
        self.reward_std_threshold = reward_std_threshold
        self.alerts = []

    def check(self, step: int, grad_norm: float, rewards: torch.Tensor = None):
        """
        Perform health checks on training metrics.
        Returns a dict of health status.
        """
        status = {"healthy": True, "issues": []}
        
        # 1. Gradient Norm Check
        if not torch.isfinite(torch.tensor(grad_norm)):
            status["healthy"] = False
            status["issues"].append("Non-finite gradient norm detected")
            logger.error(f"Step {step}: CRITICAL - Non-finite gradient norm!")
        elif grad_norm > self.grad_norm_threshold:
            status["issues"].append(f"High gradient norm: {grad_norm:.2f}")
            logger.warning(f"Step {step}: Warning - High gradient norm ({grad_norm:.2f} > {self.grad_norm_threshold})")

        # 2. Reward Distribution Check
        if rewards is not None and rewards.numel() > 1:
            r_std = rewards.std().item()
            r_max = rewards.max().item()
            r_min = rewards.min().item()
            
            if not torch.isfinite(rewards).all():
                status["healthy"] = False
                status["issues"].append("Non-finite rewards detected")
                logger.error(f"Step {step}: CRITICAL - Non-finite rewards in batch!")
            
            if r_std < self.reward_std_threshold:
                status["issues"].append(f"Collapsed reward std: {r_std:.8f}")
                logger.warning(f"Step {step}: Warning - Collapsed reward distribution (std={r_std:.8f})")
                
            if abs(r_max) > 1e4 or abs(r_min) > 1e4:
                status["issues"].append(f"Extreme rewards: [{r_min:.2f}, {r_max:.2f}]")
                logger.warning(f"Step {step}: Warning - Extreme rewards detected")

        return status
