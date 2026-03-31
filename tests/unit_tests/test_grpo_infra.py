import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
def setup_grpo_imports():
    from torchtitan.grpo.utils import distributed_reward_norm
    from torchtitan.grpo.health import NumericalHealthMonitor
    from torchtitan.grpo.data_handling import OnlineDataHandler
    return distributed_reward_norm, NumericalHealthMonitor, OnlineDataHandler

class TestGRPOInfra(unittest.TestCase):
    def setUp(self):
        self.distributed_reward_norm, self.NumericalHealthMonitor, self.OnlineDataHandler = setup_grpo_imports()
        self.device = "cpu" # Use CPU for unit tests
        if torch.cuda.is_available():
            self.device = "cuda"

    def test_numerical_health_monitor(self):
        monitor = NumericalHealthMonitor(grad_norm_threshold=10.0, reward_std_threshold=1e-3)
        
        # Test 1: Healthy
        rewards = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        status = monitor.check(step=1, grad_norm=1.0, rewards=rewards)
        self.assertTrue(status["healthy"])
        self.assertEqual(len(status["issues"]), 0)
        
        # Test 2: Collapsed rewards
        rewards_collapsed = torch.tensor([1.0, 1.00001, 1.0], device=self.device)
        status = monitor.check(step=2, grad_norm=1.0, rewards=rewards_collapsed)
        self.assertTrue(status["healthy"]) # std warning doesn't make it unhealthy usually, just a warning
        self.assertIn("Collapsed reward std", status["issues"][0])
        
        # Test 3: Non-finite rewards
        rewards_nan = torch.tensor([1.0, float('nan'), 3.0], device=self.device)
        status = monitor.check(step=3, grad_norm=1.0, rewards=rewards_nan)
        self.assertFalse(status["healthy"])
        self.assertIn("Non-finite rewards", status["issues"][0])
        
        # Test 4: High gradient norm
        status = monitor.check(step=4, grad_norm=100.0, rewards=rewards)
        self.assertTrue(status["healthy"]) # warning only
        self.assertIn("High gradient norm", status["issues"][0])

    @patch("torch.distributed.all_reduce")
    @patch("torch.distributed.get_rank")
    def test_distributed_reward_norm_logic(self, mock_get_rank, mock_all_reduce):
        mock_get_rank.return_value = 0
        
        # Mock all_reduce to simulate global sum
        # In this test, we'll assume 2 ranks, each with 1 prompt group, 2 samples
        # Rank 0: prompt_id=0, rewards=[1.0, 2.0]
        # Rank 1: prompt_id=0, rewards=[3.0, 4.0]
        # Global: prompt_id=0, rewards=[1.0, 2.0, 3.0, 4.0], mean=2.5, std=1.118...
        
        rewards = torch.tensor([1.0, 2.0], device=self.device)
        prompt_ids = torch.tensor([0, 0], device=self.device)
        
        # The function calls all_reduce twice (sum of rewards, sum of squared rewards, count)
        # Actually in my implementation it does one sum of packed tensor [sum, sum_sq, count] per prompt group?
        # No, it does one reduction for ALL prompt groups at once.
        
        def side_effect(tensor, op, group=None):
            # Simulate Rank 1's contribution [sum=7.0, sum_sq=25.0, count=2]
            # Rank 0's local is [sum=3.0, sum_sq=5.0, count=2]
            # Global should be [10.0, 30.0, 4.0]
            if tensor.shape[0] == 3: # Packed per-group stats
                tensor[0] += 7.0 
                tensor[1] += 25.0
                tensor[2] += 2.0
            return None

        mock_all_reduce.side_effect = side_effect
        
        # Run norm
        norm_rewards = distributed_reward_norm(rewards, prompt_ids)
        
        # Expected mean = 10 / 4 = 2.5
        # Expected var = (30 / 4) - (2.5^2) = 7.5 - 6.25 = 1.25
        # Expected std = sqrt(1.25) = 1.11803
        # Rank 0 normalized: (1.0 - 2.5) / 1.11803 = -1.3416, (2.0 - 2.5) / 1.11803 = -0.4472
        
        self.assertAlmostEqual(norm_rewards[0].item(), -1.3416, places=4)
        self.assertAlmostEqual(norm_rewards[1].item(), -0.4472, places=4)

    @patch("torch.distributed.broadcast")
    @patch("torch.distributed.get_rank")
    def test_online_data_handler_tensor_broadcast(self, mock_get_rank, mock_broadcast):
        mock_get_rank.return_value = 0
        handler = OnlineDataHandler(metrics_rank=0)
        
        # Simulate rank 0 having data and rank 1 wanting it
        # Data: list of (input_ids, labels, masks, inf_logps, rewards, prompt_ids)
        dummy_data = [
            (
                torch.tensor([1, 2, 3]), 
                torch.tensor([4, 5, 6]), 
                torch.tensor([1, 1, 0]), 
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([1.0]),
                torch.tensor([42])
            )
        ]
        
        # Mock broadcast to simulate sending/receiving
        # On rank 0, broadcast sends. On others, it receives into the tensor.
        # Here we just verify the handler calls it with expected shapes.
        
        handler.data_handling(dummy_data, rank=0, group=None)
        
        self.assertTrue(mock_broadcast.called)
        # Check that metadata tensor (first broadcast) is correct
        args, kwargs = mock_broadcast.call_args_list[0]
        metadata_tensor = args[0]
        self.assertEqual(metadata_tensor[0], 1) # num_items

    def test_distributed_reward_norm_cp_prevention(self):
        # Verify that if mesh is passed, it uses it.
        # This is harder to test without a real mesh, but we can check if it tries to use it.
        rewards = torch.tensor([1.0], device=self.device)
        prompt_ids = torch.tensor([0], device=self.device)
        mesh = MagicMock()
        mesh.get_group.return_value = "dummy_group"
        
        with patch("torch.distributed.all_reduce") as mock_reduce:
            distributed_reward_norm(rewards, prompt_ids, mesh=mesh)
            # Check if all_reduce was called with the Correct group
            # Actually, depending on CP, it might divide by CP degree first
            self.assertTrue(mock_reduce.called)
            # group = mesh.get_group() should be passed
            # Wait, in my utils.py: dist.all_reduce(packed_stats, group=group)
            mock_reduce.assert_called_with(unittest.mock.ANY, group="dummy_group")

if __name__ == "__main__":
    unittest.main()
