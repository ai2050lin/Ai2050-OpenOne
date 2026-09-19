"""Architecture registration for the actual local GlmModel checkpoint.

The existing qualified Qwen wave engine is unchanged. GlmModel's inspected
native forward has the same embedding/mask/RoPE/layer/norm scheduling interface;
every operation remains a call to the actual installed GlmModel modules.
No model_type alias is changed and no Glm4Model stand-in is used.
"""
import sys
from rdc_formation_microbatch import NativeWave, LayerWeights


class GlmNativeWave(NativeWave):
    def __init__(self, model, device='cuda'):
        self.model = model
        self.core = model.model
        self.device = device
        self.config = model.config
        self.depth = len(self.core.layers)
        assert self.config.model_type == 'glm'
        assert self.core.__class__.__name__ == 'GlmModel'
        assert not getattr(self.core, 'has_sliding_layers', False)
        self.module = sys.modules[self.core.__class__.__module__]
        self.weights = LayerWeights(device)
        self.forward_waves = 0
        self.layer_calls = 0
