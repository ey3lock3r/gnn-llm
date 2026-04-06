import torch
import unittest
import os
import sys

# Add parent dir for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aptp_gnn import APTPBlockV8, EntropyBalancedAttention, GigaGraph_3B

class TestGigaComponents(unittest.TestCase):
    def setUp(self):
        self.d_model = 256
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 1000

    def test_attention_normalization(self):
        """Verify EBA correctly normalizes scores (Softmax across neighbors)."""
        eba = EntropyBalancedAttention(tau=0.1)
        h_i = torch.randn(self.batch_size, self.d_model)
        h_j = torch.randn(self.batch_size, self.d_model)
        
        # Output is [batch_size, batch_size] where rows/cols sum to 1.0 (softmax(dim=0))
        scores = eba(h_i, h_j)
        # Check column-wise sum (each node competes for neighbor attention)
        column_sums = torch.sum(scores, dim=0)
        for s in column_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=4)

    def test_block_zero_grad(self):
        """Verify APTPBlock has no requires_grad (including LayerNorm)."""
        block = APTPBlockV8(self.d_model)
        for param in block.parameters():
            self.assertFalse(param.requires_grad, f"Parameter has requires_grad=True")

    def test_block_shapes(self):
        """Verify P1 forward pass shape consistency."""
        block = APTPBlockV8(self.d_model)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = block.forward_pass(h)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_zero_backprop_giga(self):
        """Verify the 3B model (sharded) has all Gradients Locked."""
        model = GigaGraph_3B(self.vocab_size, depth=4, d_model=self.d_model)
        for name, param in model.named_parameters():
            self.assertFalse(param.requires_grad, f"Model parameter {name} has active gradients!")

if __name__ == "__main__":
    unittest.main()
