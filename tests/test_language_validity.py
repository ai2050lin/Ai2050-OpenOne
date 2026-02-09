import math
import os
import sys
import unittest

import torch
import torch.nn as nn

# Add parent directory to path to import structure_analyzer (now in server/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from structure_analyzer import LanguageValidity


class MockConfig:
    def __init__(self):
        self.n_layers = 2
        self.d_model = 16
        self.n_heads = 4
        self.d_head = 4

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = MockConfig()
        self.vocab_size = 100
        
        # Simple embeddings for testing anisotropy
        self.W_E = nn.Parameter(torch.randn(self.vocab_size, self.cfg.d_model))
        
    def to_tokens(self, text):
        # Mock tokenizer: simple hash or fixed mapping
        # Just return dummy tensor
        return torch.tensor([[1, 2, 3, 4]]) # Batch size 1, seq len 4
        
    def forward(self, tokens, return_type=None):
        batch, seq = tokens.shape
        if return_type == "loss":
            # Return dummy loss
            return torch.tensor(2.5) # exp(2.5) ~= 12.18
        
        # Return logits [batch, seq, vocab]
        return torch.randn(batch, seq, self.vocab_size)
    
    def run_with_cache(self, tokens):
        logits = self(tokens)
        
        # Mock cache
        cache = {}
        batch, seq = tokens.shape
        d_model = self.cfg.d_model
        
        # Create perfect isotropic or anisotropic data for testing
        # Layer 0: Random (likely low anisotropy)
        cache["blocks.0.hook_resid_post"] = torch.randn(batch, seq, d_model)
        
        # Layer 1: Collapsed (high anisotropy) - all vectors same direction
        v = torch.randn(1, 1, d_model)
        cache["blocks.1.hook_resid_post"] = v.repeat(batch, seq, 1)
        
        return logits, cache
    
    def hooks(self, *args, **kwargs):
        # Context manager mock
        class DummyContext:
            def __enter__(self): return None
            def __exit__(self, *args): pass
        return DummyContext()

class TestLanguageValidity(unittest.TestCase):
    def test_perplexity(self):
        model = MockModel()
        analyzer = LanguageValidity(model)
        
        ppl = analyzer.compute_perplexity("test text")
        # Expected: exp(2.5)
        expected = math.exp(2.5)
        self.assertTrue(abs(ppl - expected) < 1e-4)

    def test_entropy_profile(self):
        model = MockModel()
        analyzer = LanguageValidity(model)
        
        stats = analyzer.compute_entropy_profile("test text")
        
        self.assertIn("mean_entropy", stats)
        self.assertIn("variance_entropy", stats)
        self.assertTrue(stats["mean_entropy"] >= 0)

    def test_anisotropy(self):
        model = MockModel()
        analyzer = LanguageValidity(model)
        
        # Layer 0: Random -> Expect low cosine similarity (around 0)
        anisotropy_0 = analyzer.compute_anisotropy("test", layer_idx=0)
        self.assertTrue(abs(anisotropy_0) < 0.5) # Generous bound for random high-dim vectors
        
        # Layer 1: Collapsed -> Expect high cosine similarity (1.0)
        anisotropy_1 = analyzer.compute_anisotropy("test", layer_idx=1)
        self.assertTrue(abs(anisotropy_1 - 1.0) < 1e-4)

    def test_holistic_validity(self):
        model = MockModel()
        analyzer = LanguageValidity(model)
        
        results = analyzer.analyze_holistic_validity("test text", target_layers=[0, 1])
        
        self.assertIn("perplexity", results)
        self.assertIn("geometric_stats", results)
        self.assertIn("layer_0_anisotropy", results["geometric_stats"])

if __name__ == '__main__':
    unittest.main()
