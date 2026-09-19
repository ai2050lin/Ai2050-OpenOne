"""Complete native gated-MLP operators, conditional prototypes, and two-branch tangents.

The native arithmetic identity is an oracle, not a candidate discovered language law.
All fitted prototypes use training inputs/outputs; prediction receives only current x
and explicit causal token-prefix labels. No donor displacement is applied to a model.
"""
from collections import Counter
from rdc_operator_common import *


def weights(block, device='cuda:0'):
    from safetensors import safe_open
    checkpoint = ROOT / 'models/hf/qwen3-4b'
    index = read(checkpoint / 'model.safetensors.index.json')['weight_map']
    out = {}
    for key, name in [('g', 'gate_proj'), ('u', 'up_proj'), ('d', 'down_proj')]:
        name = f'model.layers.{block}.mlp.{name}.weight'
        with safe_open(str(checkpoint / index[name]), framework='pt', device='cpu', backend='pread') as f:
            out[key] = f.get_tensor(name).float().to(device)
    return out


def metadata(selected, mode='main'):
    result = []
    for row in selected:
        with np.load(BASE / 'capture' / mode / 'energies' / (row['sample_id']+'.npz')) as z:
            for anchor, p in enumerate(row['anchors']):
                result.append({'sample_id': row['sample_id'], 'source_group': row['source_group'], 'language': row['language'],
                    'split': row['split'], 'anchor': anchor, 'position': p, 'token_id': row['prompt_ids'][p],
                    'token': row['tokens'][p], 'piece': int(z['token_class'][p]), 'cue': int(z['prefix_cue_mask'][p]),
                    'position_bin': min(3, p//32),
                    'event': bool(z['H_energy'][23,p] > 10 and z['H_energy'][23,p]/max(z['H_energy'][12,p], 1e-20) >= 10),
                    'native_next_NLL': float(z['next_NLL'][p]), 'next_token_id': row['prompt_ids'][p+1]})
    return result


def load_block(selected, block, mode='main'):
    fields = {k: [] for k in ('x', 'gate', 'up', 'activation', 'mlp')}
    for row in selected:
        with np.load(BASE / 'capture' / mode / 'factors' / (row['sample_id']+'.npz')) as z:
            for key in fields:
                fields[key].append(unbits(z[f'L{block}_{key}']))
    return {key: np.concatenate(value).astype(np.float32) for key, value in fields.items()}


def label(row, group):
    if group == 'global':
        return 'all'
    if group == 'token':
        return row['language'] + '/' + str(row['token_id'])
    if group == 'position':
        return row['language'] + '/' + str(row['position_bin'])
    return row['language'] + '/' + str(row[group])


def silu_np(x):
    # Stable sigmoid for ordinary and rare large values without clipping their native input.
    sigmoid = np.exp(-np.logaddexp(0, -x))
    return x * sigmoid


def fit_bank(data, meta):
    ix = np.array([i for i, r in enumerate(meta) if r['split'] == 'train'])
    assert len(ix) == 2560
    bank, info = {}, {}
    phi = silu_np(data['gate'])
    for group in ('global', 'piece', 'cue', 'token', 'position'):
        labels = [label(meta[i], group) for i in ix]
        keys = sorted(set(labels))
        counts = Counter(labels)
        if group == 'token':
            keys = [k for k in keys if counts[k] >= 8]
        centers = {}
        for key in keys:
            chosen = ix[np.asarray(labels) == key]
            centers[key] = {'count': len(chosen), **{k: data[k][chosen].astype(np.float64).mean(0).astype(np.float32)
                            for k in ('x', 'up', 'mlp')}, 'phi': phi[chosen].astype(np.float64).mean(0).astype(np.float32)}
        bank[group] = centers
        info[group] = {'groups': len(keys), 'retained_keys': keys, 'counts': [counts[k] for k in keys],
                       'extra_learned_float_scalars': len(keys) * (2*2560+2*9728),
                       'scope': 'All coordinates/units, no truncation; token groups with<8train anchors fall back to global. Condition families differ in parameter count.'}
    # A same-coordinate baseline preserves each native coordinate, without a dense learned rotation.
    x, y = data['x'][ix].astype(np.float64), data['mlp'][ix].astype(np.float64)
    xm, ym = x.mean(0), y.mean(0)
    slope = ((x-xm)*(y-ym)).sum(0) / np.maximum(((x-xm)**2).sum(0), 1e-12)
    bank['diagonal'] = {'slope': slope.astype(np.float32), 'intercept': (ym-slope*xm).astype(np.float32)}
    info['diagonal'] = {'extra_learned_float_scalars': 5120, 'fit': 'Unregularized per-coordinate training least squares with1e-12variance floor'}
    return bank, info


def save_bank(path, bank):
    arrays, index = {}, {}
    for group in ('global', 'piece', 'cue', 'token', 'position'):
        keys = sorted(bank[group])
        index[group] = keys
        if keys:
            for name in ('x', 'up', 'mlp', 'phi'):
                arrays[group+'_'+name] = np.stack([bank[group][key][name] for key in keys])
            arrays[group+'_count'] = np.array([bank[group][key]['count'] for key in keys])
    arrays['diagonal_slope'] = bank['diagonal']['slope']
    arrays['diagonal_intercept'] = bank['diagonal']['intercept']
    npz(path.with_suffix('.npz'), **arrays)
    save(path.with_suffix('.json'), index)


def load_bank(path):
    index = read(path.with_suffix('.json'))
    bank = {}
    with np.load(path.with_suffix('.npz')) as z:
        for group, keys in index.items():
            bank[group] = {key: {'count': int(z[group+'_count'][i]), **{name: z[group+'_'+name][i] for name in ('x','up','mlp','phi')}} for i, key in enumerate(keys)}
        bank['diagonal'] = {'slope': z['diagonal_slope'], 'intercept': z['diagonal_intercept']}
    return bank


def prototypes(bank, meta, group, shrink=32.):
    global_center = bank['global']['all']
    selected = [bank[group].get(label(row, group), global_center) for row in meta]
    shrinkages = np.array([1. if group == 'global' else item['count']/(item['count']+shrink) for item in selected], dtype=np.float32)
    return {key: np.stack([item[key] for item in selected]) * shrinkages[:,None] + global_center[key][None]*(1-shrinkages[:,None])
            for key in ('x', 'up', 'mlp', 'phi')}


NAMES = ['constant_global', 'coordinate_affine', 'frozen_gate_global', 'frozen_gate_piece', 'frozen_gate_cue',
         'frozen_gate_token', 'frozen_gate_position', 'tangent_global', 'tangent_piece', 'tangent_cue', 'quadratic_global']


def apply_operator(name, x, meta, bank, w):
    """x is real current post-attention-normalized input, never the target MLP output."""
    import torch
    global_center = bank['global']['all']
    if name == 'constant_global':
        return torch.as_tensor(global_center['mlp'], device=x.device)[None].expand(len(x), -1)
    if name == 'coordinate_affine':
        return x * torch.as_tensor(bank['diagonal']['slope'], device=x.device) + torch.as_tensor(bank['diagonal']['intercept'], device=x.device)
    group = name.rsplit('_', 1)[1]
    center = {k: torch.as_tensor(v, device=x.device) for k, v in prototypes(bank, meta, group).items()}
    u = x @ w['u'].T
    if name.startswith('frozen_gate'):
        return (center['phi'] * u) @ w['d'].T
    x0 = center['x']
    g0, u0 = x0 @ w['g'].T, x0 @ w['u'].T
    dg, du = (x-x0) @ w['g'].T, (x-x0) @ w['u'].T
    sig = torch.sigmoid(g0)
    phi = g0 * sig
    derivative = sig * (1 + g0*(1-sig))
    value = phi*u0 + phi*du + derivative*dg*u0
    if name.startswith('quadratic'):
        second = sig*(1-sig)*(2 + g0*(1-2*sig))
        value = value + derivative*dg*du + .5*second*dg.square()*u0
    return value @ w['d'].T


def exact_decomposition(x, meta, bank, w):
    """Exact FP32 finite product expansion with frozen TRAIN means, not a Taylor approximation."""
    import torch
    center = {k: torch.as_tensor(v, device=x.device) for k, v in prototypes(bank, meta, 'global').items()}
    g, u = x @ w['g'].T, x @ w['u'].T
    phi = torch.nn.functional.silu(g)
    a0, u0 = center['phi'], center['up']
    terms = [a0*u0, a0*(u-u0), (phi-a0)*u0, (phi-a0)*(u-u0)]
    writes = torch.stack([a @ w['d'].T for a in terms], 1)
    native32 = (phi*u) @ w['d'].T
    return writes, native32, terms
