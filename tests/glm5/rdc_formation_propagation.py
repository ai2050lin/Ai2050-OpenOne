"""Whole-prefix parameter effects; original earlier history is fixed, not upper KV."""
from rdc_formation_common import *
from rdc_formation_direction import NAMES

PROP = OUT/'parameter_propagation'
SCALES = [.1, .3, 1.]


def runs():
    p = read(OUT/'training/protocol.json')
    return [condition+'_'+str(seed) for seed in p['seeds'] for condition in p['conditions']]


def material(pilot=False):
    rows = gzread(OUT/'followup/material.json.gz')['parameter_differential']
    assert len(rows) == 64
    return [rows[0], rows[1], rows[24], rows[25]] if pilot else rows


def bf16bits(value):
    value = np.asarray(value, np.float32)
    result = (value.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(unbits(result), value)
    return result


def freeze():
    path = PROP/'protocol.json'
    if path.exists(): return read(path)
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'rows': [r['sample_id'] for r in material()],
        'runs': runs(), 'scales': SCALES, 'training_block': 16, 'original_parameter_scalars': 74711040,
        'native_capture': 'Original unquantized BF16 full-prefix preMLP16 residual and normalized input, actual cos/sin; all current-position H/Q/gate/up/product/MLPinput/write baseline. Every nativeBF16 finite deployment collects allH; scale1 also all current-position unit/query fields.',
        'smooth_initial_state': 'Promote originalBF16 r16,x16 toFP32. Actual full matrices W16 -> r16+MLP(W16,x16) at ALL prefix positions. Earlier computations are independent of W16 and fixed.',
        'smooth_suffix': 'All19 same-valued originalFP32 blocks17..35 with complete causal prefix attention, nativeRMSNorm and fullFP32 original-weight LMhead. No frozen upper-layer KV and no skipped block.',
        'parameter_tangent': 'ExactFP64 reconstructed endpoint difference rounded once toFP32 for forward AD. Finite smooth parameters useFP32(original + scale*tangent), distinct from nativeBF16 deployments.',
        'history_control': 'For each of6directions propagate fullprefix tangent and matched lastposition-only initial tangent. Local current MLP derivative is identical, earlier-position H17 tangents are set tozero ONLY in control.',
        'adjoint': 'Per-expression fixed hash-dense fullV covector, reverse through complete suffix and MLP16, then all74711040scalar innerproducts with each direction. Known-calculus identity admission, not semantic uniqueness.',
        'field_retention': 'All chosen coordinates/units in each saved axis. Baseline and12tangent current-position fields each layer; 6scale1 smooth finite unitfields; other scales fullH and fullpostnorm plus completeV summary/recompute. Fullprefix baseline intermediates held inRAM during adjoint; initialprefix, weights/config and code permit recomputation.',
        'native_scoring': 'Gold only for NLL/accuracy. Baseline-centered logit derivative may use original model endpoint as a disclosed reference, not an early-only language predictor.',
        'pilot': '4predeclared prefixes (2natural+2controlled), first2directions, all3scales; original nativecapture and fullprefix JVP/VJP before64x6main.',
        'budget': 'Physical4GiB safety reserve; layerwise loading and CPU state staging, no arbitrary elapsed cutoff.'}
    immutable(path, value)
    return value


def tensor(a):
    import torch
    return torch.tensor(unbits(a) if a.dtype == np.uint16 else a, device='cuda', dtype=torch.float32)


def causal_inputs(receipt):
    import torch
    from rdc_formation_readout import checked_arrays
    arrays = checked_arrays(receipt)
    length = arrays['prefix_residual_BF16'].shape[0]
    mask = torch.zeros((1, 1, length, length), device='cuda', dtype=torch.float32)
    mask.masked_fill_(torch.ones(length, length, device='cuda', dtype=torch.bool).triu(1), torch.finfo(torch.float32).min)
    return arrays, tensor(arrays['prefix_residual_BF16'])[None], tensor(arrays['prefix_MLP_input_BF16'])[None], mask, tensor(arrays['cos_BF16'])[None], tensor(arrays['sin_BF16'])[None]


def mlp_function(x, residual):
    import torch.nn.functional as F
    def call(wg, wu, wd):
        g, u = F.linear(x, wg), F.linear(x, wu)
        product = F.silu(g)*u
        write = F.linear(product, wd)
        return residual+write, g[0, -1], u[0, -1], product[0, -1], write[0, -1]
    return call


class FullBlock:
    def __init__(self, layer):
        self.layer = layer; self.values = {}; self.handles = []
        for name, module in [('Q', layer.self_attn.q_norm), ('gate', layer.mlp.gate_proj), ('up', layer.mlp.up_proj),
                             ('MLP_input', layer.post_attention_layernorm), ('MLP_write', layer.mlp)]:
            self.handles.append(module.register_forward_hook(lambda m, a, o, name=name: self.values.__setitem__(name, o[0, -1].reshape(-1))))
        self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: self.values.__setitem__('product', a[0][0, -1])))
    def call(self, x, mask, cos, sin):
        self.values.clear()
        h = self.layer(hidden_states=x, attention_mask=mask, position_embeddings=(cos, sin), use_cache=False)
        return (h,)+tuple(self.values[name] for name in ['Q', 'gate', 'up', 'product', 'MLP_input', 'MLP_write'])
    def close(self):
        for handle in self.handles: handle.remove()


def last_tuple(output):
    return [output[0][0, -1]]+list(output[1:])


def cpu_tuple(output):
    return [v.detach().cpu().numpy().copy() for v in last_tuple(output)]


def finite_metrics(base, derivative, current, scale):
    import torch
    actual = current-base
    error = actual-scale*derivative
    return [float(actual.double().square().mean()), float(error.double().square().mean()),
        float((scale*derivative).double().square().mean())]
