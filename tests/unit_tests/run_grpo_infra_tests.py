import sys
from unittest.mock import MagicMock

# Mock problematic modules
sys.modules["tyro"] = MagicMock()
sys.modules["tqdm"] = MagicMock()
sys.modules["wandb"] = MagicMock()
sys.modules["requests"] = MagicMock()

# Now run the test
import unittest
from tests.unit_tests.test_grpo_infra import TestGRPOInfra

if __name__ == "__main__":
    unittest.main()
