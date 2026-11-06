"""Frozen Qwen3-4B B35 corpus and native recursive-depth collection."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np

import mlp_factor_steer_common as C
from mlp_functional_trace import bit_array, load_model
from shared.dist_protocol.mechanism_trace import GpuLease


OUT = C.ROOT / 'tests/glm5/result/b35_recursive_v2'
PUBLIC = C.ROOT / 'research/b35_recursive/v2'
RUN = 'QWEN3-4B-B35-RECURSIVE-20261105-v2'
SEED = 511052804
BLOCK = 35
WIDTH = 9728
PARTS = {'pilot': (24, 6), 'discovery': (128, 3), 'confirm': (160, 3), 'lockbox': (224, 6)}


def identity():
    base = C.identity()
    return {**base, 'run_id': RUN, 'provenance': {**base['provenance'], 'seed': SEED}}


def freeze():
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'inputs.json.gz'
    if path.exists():
        return
    tok = AutoTokenizer.from_pretrained(C.MODEL, local_files_only=True)
    docs = C.read_gz(C.OUT / 'documents.json.gz')['docs']
    alphabet = 'ABCDEFGHJKLMNPQR'
    code_ids = {ch: tok.encode(' ' + ch, add_special_tokens=False) for ch in alphabet}
    assert all(len(v) == 1 for v in code_ids.values()) and len(set(v[0] for v in code_ids.values())) == len(alphabet)
    code_ids = {ch: v[0] for ch, v in code_ids.items()}
    start = tok.encode('<|im_start|>user\n', add_special_tokens=False)
    tail = '\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nPath:'
    sets = {}
    offset = 600
    cycle_set = set()
    for part, (count, depth) in PARTS.items():
        items = []
        for i, doc in enumerate(docs[offset:offset + count]):
            rng = random.Random(SEED + offset + i)
            labels = alphabet[:8] if i % 2 == 0 else alphabet[8:]
            frame = i // 2 % 2
            while True:
                key = (labels[0], *rng.sample(list(labels[1:]), len(labels)-1))
                if key not in cycle_set:
                    cycle_set.add(key)
                    break
            order = list(key)
            rotate = rng.randrange(8)
            order = order[rotate:] + order[:rotate]
            entries = [(order[j], order[(j + 1) % 8]) for j in range(8)]
            rng.shuffle(entries)
            if frame:
                rules = '\n'.join(f'The next locker after {a} is {b}.' for a, b in entries)
                task = 'The preceding article is unrelated. Use only this next-locker table.'
            else:
                rules = '\n'.join(f'{a} -> {b}' for a, b in entries)
                task = 'Ignore the unrelated article above. Use only the directed links below.'
            text = (f'\n\n{task}\nExample: with T -> W, V -> U, W -> V and start T, the path after three moves is T W V U.\n'
                    f'Current table:\n{rules}\nStart at {order[0]} and follow the links exactly {depth} times. '
                    f'The answer prefix already contains the starting node {order[0]}. Continue with one destination '
                    'per move, in order. Use one space then one uppercase letter per move.' + tail + ' ' + order[0])
            end = tok.encode(text, add_special_tokens=False)
            n = 1024 - len(start) - len(end)
            assert n >= 820
            items.append({'id': f'{part}:{i}', 'frame': frame, 'depth': depth, 'labels': labels,
                          'rule': order, 'gold': order[1:depth+1], 'foil': order[2:depth+2],
                          'input_ids': start + doc['tokens'][:n] + end,
                          'source': {k: v for k, v in doc.items() if k != 'tokens'}, 'background_tokens': n})
        sets[part] = items
        offset += count
    C.write_gz(path, {'sets': sets, 'code_ids': code_ids, **identity()})
    C.save(OUT/'prereg.json', {**identity(), 'schema': 'b35_recursive_prereg.v1', 'input_sha256': C.file_sha(path),
        'parts': PARTS, 'capture': 'Native B35 down_proj input A and B34 A at each actual prediction position. Native BF16 bits retained.',
        'revision': 'v1 pilot failed: 0/24 exact first tokens, many trajectories included or skipped the start. v2 supplies the start as a prefix and a worked example. New backgrounds at offset 600, no reuse of v1 cases.',
        'task': 'Two directed-cycle formulations. Discovery and confirmation depth 3, independent lockbox depth 6, steps 4–6 primary. Separate teacher forcing and greedy own-history.',
        'pilot_gate': 'At least 50% pooled actual prefix accuracy at step 3 and 35% at step 6. Pilot never enters feature fitting or inference.',
        'behavior_gate': 'Confirm strict sequence accuracy >=.50 at 3; lockbox >=.35 at 6, with each frame >=.30. Score exact intended token sequence.',
        'feature_primary': 'Signs of local gold-versus-next-position-foil logit contribution fit on discovery documents. Positive oriented document-mean contribution in confirm and at lockbox steps 4–6 tested against .5 via binomial with Bonferroni .05/9728. Require >=.65 same-sign and absolute mean standardized by RMS >=.10 in both. Shuffled foil polarity is a negative control. This is a local association, no individual causal claim.',
        'predictive_control': 'Compare same B35 unit with an equal treatment foil direction and its B34-to-B35 candidate single-input linear prediction fitted on discovery. Lockbox late steps separate gold-prefix and own-history. No future generated output is used in the observed prefix.',
        'source_sha256': C.file_sha(__file__)})
    print('frozen', {k: len(v) for k, v in sets.items()}, 'min_background', min(r['background_tokens'] for v in sets.values() for r in v), flush=True)


def capture(part):
    import torch
    torch.set_num_threads(4)
    result = OUT / f'{part}-result.json.gz'
    if result.exists():
        return
    path = OUT / 'inputs.json.gz'
    plan = C.read(OUT/'prereg.json')
    assert C.file_sha(path) == plan['input_sha256']
    data = C.read_gz(path)
    rows, ids = data['sets'][part], data['code_ids']
    depth = rows[0]['depth']
    n = len(rows)
    C.save(OUT/f'{part}-execution.json', {'part': part, 'model_snapshot': identity()['model_snapshot'],
          'source_sha256': C.file_sha(__file__), 'design_sha256': C.file_sha(OUT/'prereg.json')})
    arrays = {key: np.empty((n, depth, WIDTH), np.uint16) for key in ['teacher_A','teacher_B34','free_A']}
    local = np.empty((n, depth, WIDTH), np.float32)
    reports = []
    t0 = time.time()
    with GpuLease(), torch.inference_mode():
        model, tok, _ = load_model()
        assert len(model.model.layers) == 36 and model.config.intermediate_size == WIDTH
        slot = {}
        def observe(key):
            def hook(_m, args):
                slot[key] = args[0][:,-1].detach().clone()
            return hook
        handles = [model.model.layers[b].mlp.down_proj.register_forward_pre_hook(observe(f'A{b}')) for b in [34,35]]
        handles.append(model.model.norm.register_forward_pre_hook(observe('residual')))
        w = model.model.layers[35].mlp.down_proj.weight.detach().float()
        norm = model.model.norm
        eps = norm.variance_epsilon
        weights = model.get_output_embeddings().weight.detach()
        code = {k: v for k,v in ids.items()}
        for offset in range(0, n, 4):
            batch = rows[offset:offset+4]
            x = torch.tensor([r['input_ids'] for r in batch],device='cuda')
            gold = torch.tensor([[code[ch] for ch in r['gold']] for r in batch],device='cuda')
            foil = torch.tensor([[code[ch] for ch in r['foil']] for r in batch],device='cuda')
            res = model(input_ids=x,use_cache=True,logits_to_keep=1)
            teacher_cache = res.past_key_values
            free_cache = copy.deepcopy(teacher_cache)
            tlog = flog = res.logits[:, -1]
            batch_out = [{'id': r['id'], 'teacher': [], 'free': []} for r in batch]
            area = slice(offset, offset+len(batch))
            initial = {k: v for k,v in slot.items()}
            for step in range(depth):
                if step == 0:
                    slot.update(initial)
                else:
                    tlog = model(input_ids=gold[:,step-1:step],past_key_values=teacher_cache,use_cache=True,logits_to_keep=1).logits[:,-1]
                a, resid = slot['A35'], slot['residual'].float()
                arrays['teacher_A'][area,step] = bit_array(a)
                arrays['teacher_B34'][area,step] = bit_array(slot['A34'])
                y, f = gold[:,step], foil[:,step]
                z = (weights[y].float()-weights[f].float())*norm.weight.float()
                rms = (resid.square().mean(-1,keepdim=True)+eps).sqrt()
                gradient = z/rms - (z*resid).mean(-1,keepdim=True)*resid/rms.pow(3)
                local[area,step] = (a.float()*(gradient@w)).cpu().numpy()
                pred = tlog.argmax(-1)
                ii = torch.arange(len(batch),device='cuda')
                teacher_rows = list(zip(pred.tolist(), (tlog[ii,y].float()-tlog[ii,f].float()).tolist()))
                if step > 0:
                    flog = model(input_ids=fp[:,None],past_key_values=free_cache,use_cache=True,logits_to_keep=1).logits[:,-1]
                arrays['free_A'][area,step] = bit_array(slot['A35'])
                fp = flog.argmax(-1)
                free_rows = list(zip(fp.tolist(), (flog[ii,y].float()-flog[ii,f].float()).tolist()))
                for j, (t, f_) in enumerate(zip(teacher_rows,free_rows)):
                    yy = code[batch[j]['gold'][step]]
                    for key, (token,gap) in [('teacher',t),('free',f_)]:
                        batch_out[j][key].append({'token':token,'text':tok.decode([token]),'correct':token==yy,'gap':gap})
            reports.extend(batch_out)
            if (offset+len(batch))%32==0 or offset+len(batch)==n:
                print(part,offset+len(batch),'/',n,round(time.time()-t0,1),flush=True)
        for hook in handles:
            hook.remove()
        if part=='pilot':
            a = [sum(all(t['correct'] for t in r['free'][:s]) for r in reports)/n for s in [3,6]]
            passed = a[0]>=.5 and a[1]>=.35
        else:
            a, passed = [], None
        archive = OUT / f'{part}-native.npz'
        if part != 'pilot':
            np.savez(archive, **arrays, local=local)
        C.write_gz(result,{**identity(),'part':part,'rows':reports,'pilot_prefix_accuracy':a,'pilot_passed':passed,
                          'native_sha256':C.file_sha(archive) if part!='pilot' else None})
        if part=='pilot':
            print('pilot',a,passed,[(r['id'],[v['text'] for v in r['free']]) for r in reports[:8]],flush=True)
        if part=='discovery':
            # One-hop static parameter alignment to B34. Both gate and value projections are preserved.
            source = model.model.layers[34].mlp.down_proj.weight.detach().float()
            source /= source.norm(dim=0,keepdim=True).clamp_min(1e-12)
            layer = model.model.layers[35].mlp
            candidates = {}
            for key, matrix in [('gate',layer.gate_proj.weight),('value',layer.up_proj.weight)]:
                values, indices = [], []
                for block in matrix.detach().split(256):
                    block = torch.nn.functional.normalize(block.float(),dim=-1)
                    corr = block@source
                    _, idx = corr.abs().topk(3,dim=-1)
                    values.append(corr.gather(-1,idx).cpu().numpy())
                    indices.append(idx.cpu().numpy().astype(np.int16))
                candidates[key+'_cosine'] = np.concatenate(values)
                candidates[key+'_unit'] = np.concatenate(indices)
            np.savez(OUT/'upstream-candidates.npz', **candidates)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('freeze',*PARTS))
    part = ap.parse_args().stage
    if part == 'freeze':
        freeze()
    else:
        if part!='pilot':
            assert C.read_gz(OUT/'pilot-result.json.gz')['pilot_passed'], 'pilot behavior gate failed'
        capture(part)
