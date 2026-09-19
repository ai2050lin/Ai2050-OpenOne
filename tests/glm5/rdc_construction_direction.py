"""Prospective full-scalar direction controls at matched actual BF16 radii."""
from rdc_construction_common import *

LEARNED = ['natural_target_2742', 'within_cohort_permuted_target_2742',
           'natural_target_2743', 'within_cohort_permuted_target_2743']
RADII = [.05, .10, .18]


def radius_name(radius):
    return f'{radius:.2f}'.replace('.', 'p')


def variants():
    result = [{'name': 'native', 'kind': 'reuse', 'old_variant': 'native'}]
    result += [{'name': 'original_'+d, 'kind': 'reuse', 'old_variant': d} for d in LEARNED]
    for direction in LEARNED:
        for radius in RADII:
            result.append({'name': 'matched_'+direction+'_r'+radius_name(radius), 'kind': 'matched',
                           'direction': direction, 'transform': 'identity', 'radius': radius})
    for seed in [2742, 2743]:
        for radius in [.10, .18]:
            result.append({'name': f'coordinate_shuffle_{seed}_r'+radius_name(radius), 'kind': 'matched',
                'direction': f'natural_target_{seed}', 'transform': 'coordinate_shuffle', 'radius': radius,
                'permutation_seed': 2745000+seed})
        result.append({'name': f'reverse_{seed}_r0p10', 'kind': 'matched',
            'direction': f'natural_target_{seed}', 'transform': 'reverse', 'radius': .10})
    assert len(result) == 23
    return result


class Direction:
    def __init__(self, model):
        import torch
        self.model = model
        self.target = model.model.layers[16].mlp
        self.original = {n: p.detach().float().clone() for n, p in self.target.named_parameters()}
        self.delta = None
        self.active_key = None
        self.direction_audit = None
        assert sum(v.numel() for v in self.original.values()) == 74711040
        assert all(torch.equal(p, self.original[n].to(torch.bfloat16)) for n, p in self.target.named_parameters())

    def activate(self, variant):
        import torch
        key = (variant['direction'], variant['transform'])
        if self.active_key == key:
            return self.direction_audit
        self.delta = None
        torch.cuda.empty_cache()
        folder = OLD / 'formation' / variant['direction']
        file = folder / 'parameter_delta_FP32.npz'
        recorded = read(folder / 'result.json')
        assert sha(file) == recorded['delta_sha256']
        delta, entries = {}, []
        rng = np.random.default_rng(variant.get('permutation_seed', 0))
        with np.load(file) as z:
            for name, original in self.original.items():
                values = z[name].copy()
                before = identity(values)
                if variant['transform'] == 'reverse':
                    values = -values
                elif variant['transform'] == 'coordinate_shuffle':
                    order = rng.permutation(values.size)
                    values = values.reshape(-1)[order].reshape(values.shape)
                    del order
                assert values.shape == tuple(original.shape) and np.isfinite(values).all()
                delta[name] = torch.from_numpy(values).to('cuda')
                entries.append({'parameter': name, 'original_direction_identity': before,
                                'transformed_identity': identity(values), 'all_scalar_count': values.size})
                del values
        self.delta = delta
        self.active_key = key
        self.direction_audit = {'source_delta_sha256': recorded['delta_sha256'], 'direction': variant['direction'],
            'transform': variant['transform'], 'permutation_seed': variant.get('permutation_seed'),
            'parameters': entries, 'transformed_FP32_direction_norm': float(torch.stack([v.double().square().sum() for v in delta.values()]).sum().sqrt())}
        return self.direction_audit

    def measure(self, alpha):
        import torch
        total = torch.zeros((), device='cuda', dtype=torch.float64)
        norms = {}
        for name, original in self.original.items():
            value = (original + alpha*self.delta[name]).to(torch.bfloat16).float()
            squared = (value-original).double().square().sum()
            total += squared
            norms[name] = float(squared.sqrt())
            del value, squared
        return float(total.sqrt()), norms

    def match(self, variant):
        import torch
        audit = self.activate(variant)
        desired = variant['radius']
        lo, hi = 0., 1.
        norm, _ = self.measure(hi)
        trace = [{'alpha': hi, 'actual_BF16_norm': norm}]
        while norm < desired:
            hi *= 2
            assert hi <= 128, 'Unexpectedly tiny learned direction; do not extrapolate without review'
            norm, _ = self.measure(hi)
            trace.append({'alpha': hi, 'actual_BF16_norm': norm})
        best = (abs(norm-desired), hi, norm)
        for _ in range(48):
            alpha = (lo+hi)/2
            norm, _ = self.measure(alpha)
            trace.append({'alpha': alpha, 'actual_BF16_norm': norm})
            best = min(best, (abs(norm-desired), alpha, norm))
            if abs(norm-desired) <= desired*.0005:
                break
            if norm < desired:
                lo = alpha
            else:
                hi = alpha
        _, alpha, _ = best
        measured, parameter_norms = self.measure(alpha)
        relative = abs(measured-desired)/desired
        assert relative <= .001, ('Actual BF16 norm not matched', variant, measured, alpha)
        with torch.no_grad():
            for name, p in self.target.named_parameters():
                p.copy_((self.original[name]+alpha*self.delta[name]).to(torch.bfloat16))
        check = float(torch.stack([(p.float()-self.original[n]).double().square().sum()
                                   for n, p in self.target.named_parameters()]).sum().sqrt())
        assert abs(check-measured) < 1e-12
        changed, dot = {}, torch.zeros((), device='cuda', dtype=torch.float64)
        for name, p in self.target.named_parameters():
            actual = p.float()-self.original[name]
            changed[name] = int(torch.count_nonzero(actual))
            dot += (actual.double()*self.delta[name].double()).sum()
        return {'timestamp': stamp(), 'variant': variant, 'direction_audit': audit,
            'chosen_alpha': alpha, 'desired_BF16_norm': desired, 'actual_BF16_norm': measured,
            'relative_norm_error': relative, 'actual_parameter_norms': parameter_norms,
            'actual_changed_scalars_by_parameter': changed, 'actual_changed_scalars': sum(changed.values()),
            'effective_BF16_vs_intended_direction_cosine': float(dot)/(measured*audit['transformed_FP32_direction_norm']),
            'target_FP32_norm_before_BF16_cast': alpha*audit['transformed_FP32_direction_norm'],
            'search_trace': trace, 'all_scalars_used': 74711040,
            'scope': 'All scalar coordinates retained; global deployed displacement norm matched, not every coordinate displacement or parameter-matrix norm. The original checkpoint remains unchanged.'}

    def restore(self):
        import torch
        with torch.no_grad():
            for name, p in self.target.named_parameters():
                p.copy_(self.original[name].to(torch.bfloat16))
        assert all(torch.equal(p, self.original[n].to(torch.bfloat16)) for n, p in self.target.named_parameters())
