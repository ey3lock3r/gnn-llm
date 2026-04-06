import unittest
import torch
import os
import shutil
from aptp_gnn import GigaGraph_3B

class TestGigaCheckpointing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = "./tmp_checkpoints"
        os.makedirs(cls.test_dir, exist_ok=True)
        # Small model for fast testing
        cls.vocab_size = 1000
        cls.depth = 2
        cls.d_model = 128
        cls.checkpoint_path = os.path.join(cls.test_dir, "test_checkpoint.pt")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_save_load_parity(self):
        """Verify that weights are identical after a save/load cycle."""
        model = GigaGraph_3B(self.vocab_size, self.depth, self.d_model)
        original_step = 42
        
        # Save
        model.save_checkpoint(self.checkpoint_path, original_step)
        
        # Create a fresh model and load
        new_model = GigaGraph_3B(self.vocab_size, self.depth, self.d_model)
        loaded_step = new_model.load_checkpoint(self.checkpoint_path)
        
        self.assertEqual(original_step, loaded_step, "Step metadata mismatch")
        
        # Compare weights
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), new_model.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), f"Weight mismatch in parameter: {n1}")
            
    def test_missing_checkpoint(self):
        """Verify graceful handling of missing checkpoint files."""
        model = GigaGraph_3B(self.vocab_size, self.depth, self.d_model)
        step = model.load_checkpoint("non_existent.pt")
        self.assertEqual(step, 0, "Should return step 0 for missing checkpoint")

    def test_device_consistency(self):
        """Verify that sharded devices are restored correctly after load."""
        model = GigaGraph_3B(self.vocab_size, self.depth, self.d_model)
        model.save_checkpoint(self.checkpoint_path, 1)
        
        new_model = GigaGraph_3B(self.vocab_size, self.depth, self.d_model)
        new_model.load_checkpoint(self.checkpoint_path)
        
        # Check sharding (even if on CPU, logic should be sound)
        self.assertEqual(new_model.embeddings.weight.device.type, new_model.device0.split(':')[0])
        self.assertEqual(new_model.lm_head.weight.device.type, new_model.device1.split(':')[0])

if __name__ == "__main__":
    unittest.main()
