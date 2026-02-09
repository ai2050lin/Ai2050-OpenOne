
import os
import sys
import unittest

import torch

# Add parent directory to path to import structure_analyzer (now in server/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server"))

from structure_analyzer import CompositionalAnalysis


class MockModel:
    def __init__(self, d_model=64):
        self.d_model = d_model
        
    def to_tokens(self, text):
        return torch.tensor([1, 2, 3]) # Dummy tokens
        
    def run_with_cache(self, tokens):
        # Return dummy cache
        batch_size = 1
        seq_len = len(tokens)
        
        # Create perfect additive data for testing
        # Black: [1, 0, ...]
        # Cat: [0, 1, ...]
        # Black Cat: [1, 1, ...]
        
        cache = {
            "blocks.0.hook_resid_post": torch.zeros(batch_size, seq_len, self.d_model)
        }
        
        # Hacky way to return different vectors based on 'text'
        # But here we don't know the text, only tokens.
        # So we will rely on the test to patch the cache function or just return random
        return None, cache

class TestCompositionality(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.analyzer = CompositionalAnalysis(self.model)
        
    def test_perfect_compositionality(self):
        # Override get_token_activation to return controlled vectors
        def mock_get_activation(text, layer_idx):
            if text == "A": return torch.tensor([1.0, 0.0, 0.0])
            if text == "B": return torch.tensor([0.0, 1.0, 0.0])
            if text == "AB": return torch.tensor([1.0, 1.0, 0.0]) # Perfect addition
            return torch.zeros(3)
            
        # Monkey patch
        self.analyzer.get_token_activation = mock_get_activation
        
        phrases = [("A", "B", "AB")]
        result = self.analyzer.analyze_compositionality(phrases, layer_idx=0)
        
        print("Perfect Case Result:", result)
        self.assertAlmostEqual(result['r2_score'], 1.0, places=5)
        self.assertAlmostEqual(result['cosine_similarity'], 1.0, places=5)

    def test_noise_compositionality(self):
        # Override get_token_activation to return noisy vectors
        def mock_get_activation(text, layer_idx):
            if text == "A": return torch.tensor([1.0, 0.0, 0.0])
            if text == "B": return torch.tensor([0.0, 1.0, 0.0])
            if text == "AB": return torch.tensor([0.8, 0.0, 0.0]) # Bad addition
            return torch.zeros(3)
            
        self.analyzer.get_token_activation = mock_get_activation
        
        phrases = [("A", "B", "AB")]
        result = self.analyzer.analyze_compositionality(phrases, layer_idx=0)
        
        print("Noisy Case Result:", result)
        self.assertTrue(result['r2_score'] < 1.0)

if __name__ == '__main__':
    unittest.main()
