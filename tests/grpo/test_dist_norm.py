
import torch
import numpy as np
from torchtitan.grpo.utils import distributed_reward_norm

def test_reward_norm_local():
    # Test local fallback logic
    rewards = torch.tensor([1.0, 2.0, 10.0, 20.0], dtype=torch.float32)
    prompt_indices = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    norm_rewards = distributed_reward_norm(rewards, prompt_indices)
    
    # Group 0: mean=1.5, std=0.7071
    # (1.0 - 1.5) / 0.7071 = -0.7071
    # (2.0 - 1.5) / 0.7071 = 0.7071
    
    # Group 1: mean=15.0, std=7.071
    # (10.0 - 15.0) / 7.071 = -0.7071
    # (20.0 - 15.0) / 7.071 = 0.7071
    
    expected = torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071], dtype=torch.float32)
    torch.testing.assert_close(norm_rewards, expected, atol=1e-4, rtol=1e-4)
    print("Local reward norm test passed!")

def test_single_item_group():
    # Test group with only one item (should be 0)
    rewards = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
    prompt_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    
    norm_rewards = distributed_reward_norm(rewards, prompt_indices)
    assert norm_rewards[2] == 0.0
    print("Single item group test passed!")

if __name__ == "__main__":
    test_reward_norm_local()
    test_single_item_group()
