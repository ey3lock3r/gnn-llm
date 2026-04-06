import torch
import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aptp_gnn import APTPGNN_LLM
from data_pipeline import GNNDataPipeline

class TestAPTPIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d_model = 128 # Smaller for integration testing
        cls.depth = 2
        cls.vocab_size = 50257
        cls.model = APTPGNN_LLM(cls.vocab_size, depth=cls.depth, d_model=cls.d_model)
        
    def test_train_step_convergence_logic(self):
        """Verify train_step correctly updates internal states and returns loss."""
        # Create a single batch
        batch = torch.randint(0, self.vocab_size, (2, 64))
        labels = torch.roll(batch, -1, dims=1)
        
        # Initial weight snapshot
        initial_w = self.model.blocks[0].W.clone()
        
        loss = self.model.train_step(batch, labels, lr=1e-1)
        
        # Check if loss is a scalar tensor
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0)
        
        # Check if weights actually changed (Plasticity)
        updated_w = self.model.blocks[0].W
        self.assertFalse(torch.allclose(initial_w, updated_w), "Weights did not update after train_step!")

    def test_generation_pipeline(self):
        """Verify the model can produce token sequences."""
        prompt = torch.randint(0, self.vocab_size, (1, 10))
        output = self.model.generate(prompt, max_new_tokens=5)
        self.assertEqual(output.shape, (1, 15))

if __name__ == "__main__":
    unittest.main()
