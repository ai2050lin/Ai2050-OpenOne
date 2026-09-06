"""CPU-only checkpoint address audit for every Q14 global-gate MLP candidate.

Post-discovery descriptive windows; not new model forwards or a held-out test.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import hashlib, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT, read, save, sha
OUT = RESULT/'phase2683_crossmodel_function_atlas/qwen14'
CHECKPOINT = ROOT/'models/hf/Qwen3-14B'


def unbits(value):
    assert value.dtype == np.uint16
    return (value.astype(np.uint32) << 16).view(np.float32)


def main():
    audit = read(OUT/'analysis/independent_protocol_audit.json')
    assert audit['all_checks_passed'] and read(OUT/'analysis/completion.json')['cases'] == 512
    destination = OUT/'weights'; destination.mkdir(exist_ok=True)
    index_path = CHECKPOINT/'model.safetensors.index.json'
    index_hash = sha(index_path); mapping = read(index_path)['weight_map']
    touched = {}; vectors = {}; metadata = {}; examples = []; contributions = {}; links = []
    def native_slice(name, selection):
        path = (CHECKPOINT/mapping[name]).resolve()
        assert path.parent == CHECKPOINT.resolve()
        touched.setdefault(str(path), (path.stat().st_size, path.stat().st_mtime_ns))
        with safe_open(path, framework='pt', device='cpu') as file:
            tensor = file.get_slice(name)[selection]
            assert tensor.device.type == 'cpu' and tensor.dtype == torch.bfloat16
            value = tensor.float().numpy().copy()
        return value
    candidates = audit['all_global64_task_gate_addresses']
    for candidate in candidates['a']:
        layer, unit = candidate['layer_or_checkpoint'], candidate['coordinate']
        for kind in ('gate','up','down'):
            name = f'model.layers.{layer}.mlp.{kind}_proj.weight'
            selection = (slice(None), unit) if kind == 'down' else (unit, slice(None))
            vector = native_slice(name, selection)
            assert vector.shape == (5120,) and np.isfinite(vector).all()
            key = f'L{layer}_J{unit}_{kind}'
            vectors[key] = vector
            metadata[key] = {'checkpoint_tensor':name, 'selection':'all_output_rows, unit_column' if kind=='down' else 'unit_row, all_input_columns',
                             'unit':unit, 'layer':layer, 'kind':kind, 'native_dtype':'BF16',
                             'storage_dtype':'float32_exact_BF16_values', 'vector_sha256':hashlib.sha256(vector.tobytes()).hexdigest()}
        for hidden in candidates['h']:
            if hidden['layer_or_checkpoint'] != layer+1: continue
            coordinate = hidden['coordinate']; weight = float(vectors[f'L{layer}_J{unit}_down'][coordinate])
            unit_sign = 1 if candidate['positive_base_groups'] == 64 else -1 if candidate['negative_base_groups'] == 64 else None
            hidden_sign = 1 if hidden['positive_base_groups'] == 64 else -1 if hidden['negative_base_groups'] == 64 else None
            predicted = int(np.sign(weight))*unit_sign if unit_sign is not None else None
            links.append({'MLP_layer':layer, 'unit':unit, 'H_checkpoint':layer+1, 'coordinate':coordinate,
                          'actual_Wdown':weight, 'observed_unit_delta_sign_all64':unit_sign,
                          'single_unit_projection_delta_sign':predicted, 'observed_H_delta_sign_all64':hidden_sign,
                          'same_sign_descriptive_only':predicted == hidden_sign if None not in (predicted,hidden_sign) else None})
    cases = [r for r in read(OUT/'material/cases.json') if r['published']]
    embeddings = {}; comparisons = 0
    for row in cases:
        case = row['case_index']
        with np.load(OUT/f'field/case_{case:04d}.npz') as data:
            h = unbits(data['h']); a = unbits(data['a']); e = unbits(data['full__h'][0])
        for token, token_id in enumerate(row['prompt_ids']):
            if token_id not in embeddings:
                embeddings[token_id] = native_slice('model.embed_tokens.weight', (slice(token_id,token_id+1),slice(None)))[0]
            assert np.array_equal(embeddings[token_id], e[token]); comparisons += 1
        for candidate in candidates['a']:
            layer, unit = candidate['layer_or_checkpoint'], candidate['coordinate']
            for q in (0,1):
                value = float(a[layer,q,unit]); vector = vectors[f'L{layer}_J{unit}_down'].astype(np.float64)*value
                name = f'case{case:04d}_L{layer}_J{unit}_{("body","task")[q]}'
                contributions[name] = vector
                examples.append({'case_index':case,'case_id':row['case_id'],'layer':layer,'unit':unit,
                                 'query_boundary':('body','task')[q],'native_a':value,'term_array':name,
                                 'native_H_at_same_layer_candidate_coordinates':{str(p['coordinate']):float(h[layer+1,q,p['coordinate']])
                                    for p in candidates['h'] if p['layer_or_checkpoint'] == layer+1}})
    assert len(cases) == 2 and len(vectors) == 3*len(candidates['a'])
    np.savez_compressed(destination/'candidate_native_vectors.npz', **vectors)
    np.savez_compressed(destination/'candidate_single_unit_terms.npz', **contributions)
    np.savez_compressed(destination/'published_native_embeddings.npz', **{f'token_{k}':v for k,v in embeddings.items()})
    checks = {'all_gate_candidate_units_addressed':len(vectors)==3*len(candidates['a']),
              'all5120_coordinates_each_vector':all(v.shape==(5120,) for v in vectors.values()),
              'all_published_embedding_tokens_match_checkpoint':comparisons == sum(len(r['prompt_ids']) for r in cases),
              'checkpoint_index_unchanged':sha(index_path)==index_hash,
              'read_shard_metadata_unchanged':all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==before for p,before in touched.items()),
              'no_CUDA_context_or_model_instance':not torch.cuda.is_initialized()}
    report = {'all_checks_passed':all(checks.values()),'checks':checks,'checkpoint_index_sha256':index_hash,
              'checkpoint_shard_metadata_before':touched, 'vectors':metadata, 'direct_same_layer_links':links,
              'actual_published_examples':examples, 'embedding_token_occurrences':comparisons,'unique_embedding_tokens':len(embeddings),
              'scope':'CPU-only checkpoint slices, native BF16 values stored exactly inFP32; no model instance, forward, donor or weight modification. Everyglobal64-gate MLP unit included, all5120incoming/outgoing coordinates retained. Backgroundallcoordinatecharts unchanged.',
              'limits':'Post-discovery address bookkeeping on two predetermined chronologyEN v0 examples only. No v1rawhere, so no new paired differential/cause/necessity test. FP64 single-unit Wdown*a is one ideal projection term, not whole nativeMLP/HiddenState or its rounded sum. Matching direction cannot establish mediation; other units, attention and residual can reinforce/cancel. Shard metadata checks are not whole-shard content hashes. Gate/up rows are learnedweights, not measuredg/u activations.'}
    save(OUT/'analysis/candidate_weight_audit.json', report)
    print({'checks':checks,'direct_same_layer_links':links,'embedding_occurrences':comparisons,'unique_tokens':len(embeddings)},flush=True)
    assert report['all_checks_passed']


if __name__ == '__main__': main()
