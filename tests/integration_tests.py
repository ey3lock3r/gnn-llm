import unittest
import torch
import os
import sys

# Add parent dir for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aptp_gnn import GigaGraph_3B
from data_pipeline import GigaDataPipeline

class TestGigaIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small constants for integration check
        cls.d_model = 128
        cls.depth = 4
        cls.vocab_size = 128256 # Llama-3 size but small hidden
        cls.seq_len = 16
        cls.batch_size = 2
        cls.model = GigaGraph_3B(cls.vocab_size, depth=cls.depth, d_model=cls.d_model)

    def test_forward_backward_flow(self):
        """Verify the full APTP-GigaGraph train step produces a valid loss."""
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        y = torch.roll(x, -1, dims=1)
        
        # Initial loss
        loss = self.model.train_step(x, y, lr=1e-4)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() > 0, "Loss should be positive")
        
        # Second step should (usually) decrease loss or remain stable
        loss2 = self.model.train_step(x, y, lr=1e-2)
        self.assertIsInstance(loss2, torch.Tensor)

    def test_pipeline_streaming(self):
        """Verify the Llama-3 data pipeline delivers correct shapes."""
        pipeline = GigaDataPipeline()
        loader = pipeline.get_dataloader(batch_size=self.batch_size, seq_len=self.seq_len)
        
        batch = next(loader)
        self.assertEqual(batch.shape, (self.batch_size, self.seq_len))
        self.assertEqual(batch.dtype, torch.long)

if __name__ == "__main__":
    unittest.main()
