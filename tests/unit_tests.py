import torch
import unittest
import os
import sys

# Add parent dir for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aptp_gnn import APTPBlockV7_1, EntropyBalancedAttention, APTPGNN_LLM

class TestAPTPComponents(unittest.TestCase):
    def setUp(self):
        self.d_model = 256
        self.batch_size = 4
        self.seq_len = 32
        self.vocab_size = 1000

    def test_attention_normalization(self):
        """Verify EBA correctly normalizes scores."""
        eba = EntropyBalancedAttention(tau=0.1)
        # Mock hidden states
        h_i = torch.randn(self.batch_size, self.d_model)
        h_j = torch.randn(self.batch_size, self.d_model)
        
        scores = eba(h_i, h_j)
        # Check sum for batch indices (competitive neighborhood)
        # A softmax across dim=0 should sum to 1.0
        self.assertAlmostEqual(torch.sum(scores).item(), 1.0, places=4)

    def test_block_zero_grad(self):
        """Verify APTPBlock has no requires_grad."""
        block = APTPBlockV7_1(self.d_model, self.d_model)
        for param in block.parameters():
            self.assertFalse(param.requires_grad, f"Parameter {param} has requires_grad=True")

    def test_block_shapes(self):
        """Verify P1 forward pass shape consistency."""
        block = APTPBlockV7_1(self.d_model, self.d_model)
        h = torch.randn(self.batch_size, self.seq_len, self.d_model)
        out = block.forward_pass(h)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_zero_backprop_llm(self):
        """Verify the full LLM has all Gradients Locked."""
        model = APTPGNN_LLM(self.vocab_size, depth=4, d_model=self.d_model)
        for name, param in model.named_parameters():
            self.assertFalse(param.requires_grad, f"Model parameter {name} has active gradients!")

if __name__ == "__main__":
    unittest.main()
