"""Original complete-vocabulary B1 readout, without another full CUDA model."""
from rdc_formation_common import *


CUDA_FORMATION = {'phase2747_rdc_training.py', 'phase2747_rdc_gradient_radius.py',
    'phase2747_rdc_calibration.py', 'phase2747_rdc_transfer_readout.py',
    'phase2747_rdc_parameter_propagation.py', 'phase2747_rdc_parameter_capture.py', 'phase2747_rdc_own_history.py',
    'phase2747_rdc_program_own_history.py', 'phase2747_rdc_own_wave.py'}


def head_parameter_name():
    folder = ROOT/'models/hf'/MODELS['qwen4']
    mapping = read(folder/'model.safetensors.index.json')['weight_map']
    if 'lm_head.weight' in mapping:
        return 'lm_head.weight'
    assert read(folder/'config.json').get('tie_word_embeddings') is True
    assert 'model.embed_tokens.weight' in mapping
    return 'model.embed_tokens.weight'


class Readout:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer
        from rdc_native_tail import checkpoint_tensor, cuda_singleton
        cuda_singleton(CUDA_FORMATION)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.parameter_name = head_parameter_name()
        self.weight = checkpoint_tensor(self.parameter_name, torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS['qwen4'], local_files_only=True)
        self.V, self.D = self.weight.shape
        assert (self.V, self.D) == (151936, 2560)

    def logits(self, hidden):
        import torch
        from torch.nn.functional import linear
        a = unbits(hidden) if hidden.dtype == np.uint16 else hidden
        x = torch.tensor(a, device='cuda', dtype=torch.bfloat16).reshape(1, self.D)
        # B1 is deliberately not silently replaced with a larger GEMM shape.
        return linear(x, self.weight).float()[0].double()

    def token(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) == 1, (text, ids)
        return ids[0]

    def close(self):
        import torch
        self.weight = None
        gc.collect()
        torch.cuda.empty_cache()


def checked_arrays(receipt):
    path = ROOT/receipt['field_path']
    assert sha(path) == receipt['field_sha256']
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
