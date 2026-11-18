"""Export native parameter objects and descriptive maps, without causal promotion.

Only discovery chooses indices. Native parameter slices are structural provenance;
correlation/readout, measured attention and local predictions have separate cards.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
TMP = ROOT / 'tests/glm5_temp/conditional_lookup_review_mpl'
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MPLCONFIGDIR=str(TMP))
import numpy as np
from deploy.storage_policy import checked_dir, check_write
from conditional_lookup_stage_audit import read, sha, bits_to_float, SOURCE, OUTPUT


def write_json(path, value):
    assert not path.exists(), str(path)
    check_write(path, 1 << 20, operation='CLT feature cards')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def draw_maps():
    checked_dir(TMP, 16 << 20, operation='CLT plot cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    with np.load(OUTPUT / 'discovery-full-coordinate-maps.npz') as maps:
        for standardized in (False, True):
            suffix = 'standardized_contrast' if standardized else 'contrast_mean'
            h = np.stack([maps[f'h{i:02}_{suffix}'] for i in range(41)], axis=1)
            bound = max(float(np.abs(h).max()), 1e-8)
            norm = SymLogNorm(linthresh=max(bound / 1000, 1e-8), vmin=-bound, vmax=bound)
            fig, axes = plt.subplots(3, 1, figsize=(16, 8), layout='constrained')
            for axis, field, label in zip(axes, h, ['r0', 'r1', 'r0 * r1']):
                im = axis.imshow(field, aspect='auto', interpolation='nearest', cmap='RdBu_r', norm=norm)
                axis.set_ylabel(f'{label}\nResidual boundary')
                axis.set_yticks([0, 10, 20, 30, 40])
            axes[-1].set_xlabel('Native residual coordinate (original order, 0..5119)')
            unit = 'state standard deviations' if standardized else 'raw activation units'
            fig.colorbar(im, ax=axes, label=f'Balanced factorial coefficient; {unit}; symmetric log color')
            fig.suptitle('Discovery observations: all native residual coordinates; no causal claim')
            path = OUTPUT / ('residual-standardized-map.png' if standardized else 'residual-raw-map.png')
            assert not path.exists()
            fig.savefig(path, dpi=140)
            plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(17, 8), layout='constrained')
        for row, kind in enumerate(('g', 'u', 'a')):
            h = np.stack([maps[f'{kind}{i}_contrast_mean'] for i in (20, 21, 22)], axis=1)
            bound = max(float(np.abs(h).max()), 1e-8)
            norm = SymLogNorm(linthresh=max(bound / 1000, 1e-8), vmin=-bound, vmax=bound)
            for col, label in enumerate(('r0', 'r1', 'r0*r1')):
                im = axes[row, col].imshow(h[col], aspect='auto', interpolation='nearest', cmap='RdBu_r', norm=norm)
                axes[row, col].set_title(f'{kind.upper()}: {label}')
                axes[row, col].set_yticks([0, 1, 2], labels=['B20', 'B21', 'B22'])
                axes[row, col].set_xlabel('Native MLP unit, 0..17407')
            fig.colorbar(im, ax=axes[row, :], label='Raw coefficient; symmetric log')
        fig.suptitle('All acquired native MLP units; three acquired blocks only; descriptive observations')
        assert not (OUTPUT / 'mlp-gua-raw-map.png').exists()
        fig.savefig(OUTPUT / 'mlp-gua-raw-map.png', dpi=140)
        plt.close(fig)


def main():
    import torch
    torch.set_num_threads(4)
    from source_binding_runtime import Runtime
    checked_dir(OUTPUT, 64 << 20, operation='CLT native parameter cards')
    seal = read(SOURCE / 'candidate.json')
    for name, digest in seal['files'].items():
        assert sha(SOURCE / name) == digest
    discovery = read(SOURCE / 'discovery-analysis.json')
    pre = read(SOURCE / 'prereg.json')
    model = Runtime('qwen14')  # metadata and CPU weight reading only; no CUDA call.
    layer, head = seal['head']['layer'], seal['head']['head']
    kv_head = head // (model.heads // model.kv)
    slices = {}
    native_index = {}
    for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        weight_name = f'model.layers.{layer}.self_attn.{key}.weight'
        weight = model.weight(weight_name, 'cpu')
        hh = head if key in ('q_proj', 'o_proj') else kv_head
        lo, hi = hh * model.hd, (hh + 1) * model.hd
        piece = weight[:, lo:hi] if key == 'o_proj' else weight[lo:hi]
        slices[key] = piece.clone().contiguous().view(torch.uint16).numpy()
        native_index[key] = dict(parameter=weight_name, original_shape=list(weight.shape),
                                 sliced_axis=1 if key == 'o_proj' else 0, start=lo, stop=hi,
                                 stored_dtype='uint16 bit representation of original BF16')
        del piece, weight
    for key in ('q_norm', 'k_norm'):
        weight_name = f'model.layers.{layer}.self_attn.{key}.weight'
        slices[key] = model.weight(weight_name, 'cpu').contiguous().view(torch.uint16).numpy().copy()
        native_index[key] = dict(parameter=weight_name, original_shape=[model.hd], full=True)
    weight_name = f'model.layers.{layer}.input_layernorm.weight'
    slices['input_norm'] = model.weight(weight_name, 'cpu').contiguous().view(torch.uint16).numpy().copy()
    native_index['input_norm'] = dict(parameter=weight_name, full=True)
    path = OUTPUT / 'observed-head-native-parameters.npz'
    assert not path.exists()
    np.savez_compressed(path, **slices)
    print(json.dumps(dict(stage='native_head_parameters_exported', layer=layer, head=head,
                          kv_head=kv_head, scalar_count=sum(a.size for a in slices.values()))), flush=True)

    # Follow the already frozen 32 units to the scalar terms in their true inputs.
    meta = seal['parameter']
    b = meta['layer']
    original = np.load(SOURCE / 'native-parameters.npz')
    rows = read(SOURCE / 'discovery-materials.json')
    keep = [i for i, r in enumerate(rows) if r['late'] and r['b'] == 0]
    xs, gs, us = [], [], []
    for name in sorted(read(SOURCE / 'discovery-collection.json')['shards']):
        with np.load(SOURCE / name) as data:
            xs.append(bits_to_float(data[f'z{b}']))
            gs.append(bits_to_float(data[f'g{b}'])[:, original['units']])
            us.append(bits_to_float(data[f'u{b}'])[:, original['units']])
    x, true_g, true_u = [np.concatenate(chunks)[keep] for chunks in (xs, gs, us)]
    cards = []
    for j, unit in enumerate(original['units']):
        branches = {}
        for branch in ('gate', 'up'):
            weights = original[branch][j]
            terms = x * weights[None]
            term_var = terms.var(0)
            indices = np.argsort(-term_var, kind='stable')[:8]
            branches[branch] = dict(
                parameter=f'model.layers.{b}.mlp.{"gate_proj" if branch == "gate" else "up_proj"}.weight',
                unit_row=int(unit), scalar_terms=[dict(input_coordinate=int(k), weight=float(weights[k]),
                     mean_product=float(terms[:, k].mean()), sd_product=float(terms[:, k].std())) for k in indices],
                selection='largest discovery per-term variance among all coordinates; observation only; covariance prevents interpreting it as additive explained variance')
        down = original['down'][:, j]
        largest = np.argsort(-np.abs(down), kind='stable')[:8]
        cards.append(dict(id=f"{pre['model_snapshot']}:B{b}:mlp:A:{int(unit)}", layer=b, unit=int(unit),
                          readout='post-attention RMSNorm -> gate/up -> SiLU(G)*U -> down',
                          provenance=pre['provenance'], branches=branches,
                          down=dict(parameter=f'model.layers.{b}.mlp.down_proj.weight', unit_column=int(unit),
                                    largest_absolute_coefficients=[dict(residual_coordinate=int(k), weight=float(down[k])) for k in largest],
                                    selection='absolute parameter magnitude, not semantic attribution'),
                          readout_selection='supervised discovery-only row-information ranking, maximum32 units',
                          prediction_scope='same-position 32-input native-parameter surrogate; original parameter file referenced below',
                          discovery_prediction=discovery['parameter'], causal_tested=False,
                          upstream='actual z at same position; full original gate/up rows retained',
                          downstream='original down column; later-layer causal readers untested',
                          counterexample_scope='random coordinates also carry row information; omitted inputs remain material',
                          evidence_level='native structural source and discovery information observation'))
    g = x @ original['gate'].T
    u = x @ original['up'].T
    audit = dict(gate_fp32_dot_vs_native_bf16_max_abs=float(np.abs(g - true_g).max()),
                 up_fp32_dot_vs_native_bf16_max_abs=float(np.abs(u - true_u).max()),
                 scope='mixed-precision numerical diagnostic, no new calibrated pass threshold and no prediction claim')
    model_bits = np.load(SOURCE / 'readout-models.npz')
    residual = dict(id=f"{pre['model_snapshot']}:{seal['row_key']}:entity:readout32",
                    layer_boundary=seal['row_key'], coordinates=model_bits['row_idx'].tolist(),
                    parameter_file=str(SOURCE / 'readout-models.npz'),
                    discovery_readout=discovery['chosen'], strongest_random_advantage=discovery['advantage'],
                    evidence_level='selected discovery information readout', causal_tested=False,
                    caveat='a probe coefficient is not an original model parameter or causal role')
    head_card = dict(id=f"{pre['model_snapshot']}:B{layer}:attention:H{head}", layer=layer, head=head,
                     kv_head=kv_head, head_dim=model.hd, parameters=native_index,
                     parameters_sha256=sha(path), observed_sources=discovery['routing']['chosen'],
                     allowed_interpretation='observed QK routing and raw local OV accounting; original parameters exported',
                     evidence_level='native structural source plus discovery response association',
                     upstream_prediction_tested=False, causal_tested=False,
                     normalization='Q/K per-head RMSNorm and RoPE still required; exported weights alone do not determine context-specific attention',
                     downstream='OV writes enter residual and downstream layers; final-logit causal effects untested')
    write_json(OUTPUT / 'discovery-native-feature-cards.json',
               dict(schema='clt_native_feature_cards.v1', kind='real', provenance=pre['provenance'],
                    candidate_sha256=sha(SOURCE / 'candidate.json'), exporter_sha256=sha(Path(__file__)),
                    native_mlp_parameters_sha256=sha(SOURCE / 'native-parameters.npz'),
                    scalar_term_selection_scope='discovery only; exploratory, not an additional confirmed candidate',
                    residual=residual, attention=head_card, mlp=cards, numerical_audit=audit,
                    mechanism_accepted=False, causal_tested=False))
    draw_maps()
    print('EXPORTED native feature cards and full-coordinate maps; no causal promotion', flush=True)


if __name__ == '__main__':
    main()
