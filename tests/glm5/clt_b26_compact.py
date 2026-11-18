"""Native B26/H36 read-before-operation prediction, with discovery and sealed confirmation."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'result/clt_b26_compact_v1'
TMP = ROOT / 'tests/glm5_temp/clt_b26_compact_v1'
sys.path[:0] = [str(ROOT), str(HERE)]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TEMP=str(TMP), TMP=str(TMP), TMPDIR=str(TMP))
import numpy as np
from deploy.storage_policy import check_write, checked_dir
from source_binding_design import MODELS, WORDS, git, read, sha
from tools.power_analysis import check_phase

SEED = 202611171157
BLOCK, HEAD, KV, HD = 26, 36, 7, 128
HEAD_SLICE = slice(HEAD * HD, (HEAD + 1) * HD)
KV_SLICE = slice(KV * HD, (KV + 1) * HD)
COUNT = {'discovery': 64, 'confirm': 128, 'causal': 128}
ARMS = ('native32', 'random32', 'native128', 'random128', 'mean', 'zero', 'full')
CELLS = list(itertools.product(range(2), repeat=4))
CAP = .065419


def commit(name, obj):
    path = OUT / name
    assert not path.exists(), path
    check_write(path, 8 << 20, operation='CLT compact metadata')
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    partial.replace(path)


def archive(name, arrays):
    path = OUT / name
    assert not path.exists(), path
    check_write(path, (8 << 20) + sum(x.nbytes for x in arrays.values()), operation='CLT compact native arrays')
    partial = path.with_suffix('.partial')
    with partial.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    partial.replace(path)


def material(tokenizer, split, family, names, words):
    titles = ['Orion', 'Vega'][::1 if family % 2 == 0 else -1]
    codes = [tokenizer.encode(w, add_special_tokens=False) for w in words]
    assert all(len(v) == 1 for v in codes)
    rows = []
    for cell, (r0, r1, identity, condition) in enumerate(CELLS):
        content = 'Two independent archives assign codes to the same entities. Return only the requested code word.\n'
        spans = []
        for block, order in enumerate((r0, r1)):
            content += f'Archive {titles[block]}:\n'
            for line in range(2):
                start = len(content)
                content += f'{names[line ^ order]} has code {words[2 * block + line]}.'
                spans.append((start, len(content)))
                content += '\n'
        content += 'Entity to look up: '
        left = len(content)
        content += names[identity]
        entity_span = left, len(content)
        content += f'\nArchive to use: {titles[condition]}\nReturn only the code word.'
        prompt = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False) + 'Answer:\n'
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offset = prompt.index(content)

        def positions(span):
            start, stop = span
            return [i for i, (a, b) in enumerate(enc['offset_mapping']) if a < offset + stop and b > offset + start]

        records = list(map(positions, spans))
        assert all(records) and len(set(sum(records, []))) == len(sum(records, []))
        slot = 2 * condition + ((r0, r1)[condition] ^ identity)
        rows.append(dict(id=f'{split}-{family:03}-{cell:02}', family=family, cell=cell,
                         b=condition, names=names, tokens=enc['input_ids'], entity_pos=max(positions(entity_span)),
                         records=records, target_record=slot, gold_id=codes[slot][0], choice_ids=[i[0] for i in codes],
                         prompt=prompt))
    for a, b in zip(rows[::2], rows[1::2]):
        stop = a['entity_pos'] + 1
        assert a['entity_pos'] == b['entity_pos'] and a['tokens'][:stop] == b['tokens'][:stop]
    return rows


def freeze():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot

    checked_dir(OUT, 700 << 20, operation='CLT compact campaign')
    checked_dir(TMP)
    assert not (OUT / 'prereg.json').exists()
    model = MODELS['qwen14']
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    old = HERE / 'result'
    revisions = [read(old / 'conditional_lookup_native_v1/prereg.json')['model_snapshot'],
                 read(old / 'clt_b26_native_v1/prereg.json')['provenance']['model_revision']]
    revision = snapshot(model)
    assert revisions == [revision, revision]
    used = {name for folder, splits in [('conditional_lookup_native_v1', ('pilot', 'discovery', 'confirm')),
                                       ('clt_b26_native_v1', ('discovery', 'confirm'))]
            for split in splits for row in read(old / folder / f'{split}-materials.json') for name in row['names']}
    rng = np.random.default_rng(SEED)
    materials = {}
    for split, count in COUNT.items():
        rows = []
        for family in range(count):
            while True:
                names = [''.join(rng.choice(list('bdfgklmnprstvz')) + rng.choice(list('aeiou'))
                                 for _ in range(3)).capitalize() for _ in range(2)]
                if len(set(names)) == 2 and not used.intersection(names) and len({
                    len(tokenizer.encode(' ' + n, add_special_tokens=False)) for n in names
                }) == 1:
                    break
            used.update(names)
            rows.extend(material(tokenizer, split, family, names, list(map(str, rng.choice(WORDS, 4, replace=False)))))
        name = f'{split}-materials.json'
        commit(name, rows)
        materials[split] = dict(families=count, inputs=len(rows), sha=sha(OUT / name))
    power = check_phase(.15, 128, p_d=.25, alpha=.025)
    assert power['pass']
    assert check_write(OUT, 650 << 20)
    deps = [Path(__file__), HERE / 'clt_b26_compact_prereg.md', HERE / 'source_binding_runtime.py',
            HERE / 'source_binding_design.py', HERE / 'source_binding_resources.py',
            ROOT / 'deploy/storage_policy.py', ROOT / 'deploy/storage_policy.json',
            ROOT / 'shared/dist_protocol/mechanism_trace.py', ROOT / 'tools/power_analysis.py']
    commit('prereg.json', dict(utc=datetime.now(timezone.utc).isoformat(), materials=materials,
                              code={p.relative_to(ROOT).as_posix(): sha(p) for p in deps}, power=power,
                              provenance=dict(model_path=str(model), model_revision=revision, seed=SEED,
                                              git_sha=git('rev-parse', 'HEAD'), git_dirty=True),
                              scope='single local BF16 model; no independent model node'))
    print('Frozen new discovery, confirmation and intervention families', flush=True)


def checked():
    prereg = read(OUT / 'prereg.json')
    for name, digest in prereg['code'].items():
        assert sha(ROOT / name) == digest, name
    return prereg


def candidate():
    c = read(OUT / 'candidate.json')
    assert c['prereg_sha'] == sha(OUT / 'prereg.json')
    for key, digest in c['files'].items():
        assert sha(OUT / key) == digest, key
    return c


def behavior(rows, scores):
    groups = []
    for cell in range(16):
        sample = [s for r, s in zip(rows, scores, strict=True) if r['cell'] == cell and r['id'] == s['id']]
        assert len(sample) == len(rows) // 16
        groups.append(dict(n=len(sample), strict=sum(s['gold_margin'] > 0 for s in sample),
                           top1=sum(s['top1_correct'] for s in sample)))
    return groups, all(min(g['strict'], g['top1']) / g['n'] >= .95 for g in groups)


def collect(split):
    import torch as t
    import torch.nn.functional as F
    import transformers
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_runtime import Runtime
    from source_binding_resources import resources

    prereg = checked()
    seal = candidate() if split == 'confirm' else None
    materials = OUT / f'{split}-materials.json'
    assert sha(materials) == prereg['materials'][split]['sha']
    rows = read(materials)
    assert not (OUT / f'{split}-collection.json').exists()
    checked_dir(TMP)
    clock = time.monotonic()
    batch_count = (len(rows) + 127) // 128

    def progress(shard, block, completed, stage):
        elapsed = time.monotonic() - clock
        event = dict(split=split, stage=stage, shard=shard, block=block, completed=completed,
                     total=batch_count * 40, elapsed=elapsed,
                     eta=elapsed / completed * (batch_count * 40 - completed) if completed else None,
                     cuda_peak=t.cuda.max_memory_allocated(), **resources())
        with (OUT / f'{split}-progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
        assert elapsed < 45 * 60 and (event.get('shared_gpu_bytes') or 0) < 1 << 30
        assert event['ram_available_bytes'] > 2 << 30 and event['commit_available_bytes'] > 4 << 30

    progress(0, -1, 0, 'snapshot')
    assert snapshot(MODELS['qwen14']) == prereg['provenance']['model_revision']
    rt = Runtime('qwen14')
    assert (rt.heads, rt.kv, rt.hd) == (40, 8, HD)
    t.backends.cuda.matmul.allow_tf32 = False
    all_scores, shards, max_errors, prefix_checks = [], {}, [], []
    selected = t.tensor(seal['units'], device='cuda') if seal else None
    if seal:
        with np.load(OUT / 'predictor.npz') as fitted:
            means = {kind: t.tensor(fitted['mu_' + kind], device='cuda') for kind in ('q', 'k', 'v')}
            indices = {name: t.tensor(fitted[name], device='cuda') for name in fitted.files if '_idx_' in name}
    for shard, start in enumerate(range(0, len(rows), 128)):
        name = f'{split}-{shard:02}'
        scores_file = OUT / (name + '.json')
        if scores_file.exists():
            summary = read(scores_file)
            assert sha(OUT / (name + '.npz')) == summary['array_sha']
            assert summary['prereg_sha'] == sha(OUT / 'prereg.json')
            assert summary['seal_sha'] == (sha(OUT / 'candidate.json') if seal else None)
        else:
            check_write(OUT, 80 << 20, operation='CLT compact shard')
            rr = rows[start:start+128]
            collected, visits = {}, {}
            totals = np.zeros((2, 2, rt.dim), dtype=np.float64)
            counts = np.zeros(2, dtype=np.int64)
            unit_sse = np.zeros(rt.width, dtype=np.float64)

            def emit(key, tensor):
                cpu = tensor.detach().cpu().contiguous()
                if cpu.dtype == t.bfloat16:
                    cpu = cpu.view(t.uint16)
                collected.setdefault(key, []).append(cpu.numpy())

            def block(b):
                layer = Runtime.block(rt, b)
                if b != BLOCK:
                    return layer
                att = layer.self_attn
                w = dict(q=att.q_proj.weight[HEAD_SLICE], k=att.k_proj.weight[KV_SLICE],
                         v=att.v_proj.weight[KV_SLICE], o=att.o_proj.weight[:, HEAD_SLICE])
                gate, up = layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight
                if seal:
                    gate, up = gate[selected], up[selected]
                state = {'offset': 0}
                compiled = {}
                if seal:
                    for arm in ARMS[:5]:
                        compiled[arm] = {}
                        for kind in ('q', 'k', 'v'):
                            ids = indices.get(f'{arm}_idx_{kind}')
                            weight = w[kind].float()
                            compiled[arm][kind] = (ids, weight @ means[kind],
                                                   weight[:, ids].contiguous() if ids is not None else None)

                def remember_residual(_, inputs):
                    state['res'] = inputs[0][:, -1].clone()

                def prepare(_, args, kwargs):
                    z = kwargs['hidden_states']
                    state.update(z=z, mask=kwargs['attention_mask'][:, 0, -1], pe=kwargs['position_embeddings'])

                def capture_branch(kind, chosen):
                    def hook(_, args, output):
                        state[kind] = output[..., chosen].contiguous()
                    return hook

                def route(q, k, v):
                    cos, sin = state['pe']
                    q, k = rt.mod.apply_rotary_pos_emb(att.q_norm(q).unsqueeze(1), att.k_norm(k).unsqueeze(1), cos, sin)
                    logits = (q[:, 0, -1, None].float() * k[:, 0].float()).sum(-1) * att.scaling
                    probabilities = logits.masked_fill(~state['mask'], float('-inf')).softmax(-1)
                    return probabilities, (probabilities[..., None] * v.float()).sum(1)

                def mlp_at(write):
                    # The other-head O projection was independently evaluated with H36's slot zero.
                    after_attention = (state['other'].float() + write).to(t.bfloat16)
                    x = layer.post_attention_layernorm(state['res'] + after_attention)
                    g, u = F.linear(x, gate), F.linear(x, up)
                    a = layer.mlp.act_fn(g) * u
                    return t.stack([g.float(), u.float(), a.float()], dim=1)

                def observe_head(_, inputs):
                    raw = inputs[0][:, -1]
                    z = state['z']
                    off = state['offset']
                    batch = rr[off:off+len(z)]
                    width = z.shape[1]
                    record = t.zeros((len(z), 4, width), dtype=t.float32, device='cuda')
                    for j, row in enumerate(batch):
                        pad = width - len(row['tokens'])
                        for k, sites in enumerate(row['records']):
                            record[j, k, [pad + p for p in sites]] = 1
                    real_probs, rebuilt = route(state['q'], state['k'], state['v'])
                    raw_head = raw[:, HEAD_SLICE].float()
                    actual_write = F.linear(raw_head, w['o'].float())
                    errors = (rebuilt - raw_head).abs().amax(-1)
                    emit('head_audit', errors)
                    emit('mass_actual', (record * real_probs[:, None]).sum(-1))
                    other = raw.clone()
                    other[:, HEAD_SLICE] = 0
                    state['other'] = F.linear(other, att.o_proj.weight)
                    if not seal:
                        for i, tensor in enumerate((z[:, -1], z[state['mask']])):
                            y = tensor.double()
                            totals[i, 0] += y.sum(0).cpu().numpy()
                            totals[i, 1] += y.square().sum(0).cpu().numpy()
                            counts[i] += len(y)
                        ep = [width - len(row['tokens']) + row['entity_pos'] for row in batch]
                        emit('entity_z', z[t.arange(len(z), device='cuda'), ep])
                        emit('final_z', z[:, -1])
                        state['predictions'] = {'zero': mlp_at(t.zeros_like(actual_write))}
                    else:
                        state['predictions'] = {}
                        emit('write_actual', actual_write)
                        emit('head_actual', raw_head)
                        for arm in ARMS:
                            if arm == 'zero':
                                write = t.zeros_like(actual_write)
                            elif arm == 'full':
                                write = actual_write
                            else:
                                projections = {}
                                for kind, (idx, intercept, weight) in compiled[arm].items():
                                    x = z[:, -1:] if kind == 'q' else z
                                    v = (F.linear(x[..., idx].float() - means[kind][idx], weight) + intercept
                                         if idx is not None else intercept.expand(*x.shape[:-1], -1))
                                    projections[kind] = v.to(t.bfloat16)
                                probs, values = route(projections['q'], projections['k'], projections['v'])
                                write = F.linear(values.to(t.bfloat16).float(), w['o'].float())
                                emit('mass_' + arm, (record * probs[:, None]).sum(-1))
                                emit('head_' + arm, values.to(t.bfloat16).float())
                            state['predictions'][arm] = mlp_at(write)
                            if arm not in ('full', 'zero'):
                                emit('write_se_' + arm, (write - actual_write).square().mean(-1))
                        if off == 0:
                            emit('replay_z_example', z[:1])
                            emit('replay_background_example', t.stack([state['res'][:1], state['other'][:1]], dim=1))
                    state['offset'] += len(z)

                def observe_mlp(kind):
                    def hook(_, args, output=None):
                        tensor = (args[0] if kind == 'a' else output)[:, -1]
                        state[kind] = tensor[:, selected] if seal else tensor
                        if kind == 'a':
                            actual = t.stack([state[key].float() for key in ('g', 'u', 'a')], dim=1)
                            if seal:
                                emit('mlp_actual', actual)
                                for arm, prediction in state['predictions'].items():
                                    emit('mlp_' + arm, prediction)
                            else:
                                unit_sse[:] += (actual[:, 2] - state['predictions']['zero'][:, 2]).square().sum(0).cpu().numpy()
                            state['predictions'].clear()
                    return hook

                layer.register_forward_pre_hook(remember_residual)
                att.register_forward_pre_hook(prepare, with_kwargs=True)
                att.q_proj.register_forward_hook(capture_branch('q', HEAD_SLICE))
                att.k_proj.register_forward_hook(capture_branch('k', KV_SLICE))
                att.v_proj.register_forward_hook(capture_branch('v', KV_SLICE))
                att.o_proj.register_forward_pre_hook(observe_head)
                layer.mlp.gate_proj.register_forward_hook(observe_mlp('g'))
                layer.mlp.up_proj.register_forward_hook(observe_mlp('u'))
                layer.mlp.down_proj.register_forward_pre_hook(observe_mlp('a'))
                return layer

            def callback(b, layer, x, y, batch, width, offset=0, context=None):
                if b >= 0 and offset + len(batch) == len(rr) and (b % 5 == 4 or b == BLOCK):
                    progress(shard, b, shard * 40 + b + 1, 'native_forward')
                if b == BLOCK - 1:
                    ep = [width - len(row['tokens']) + row['entity_pos'] for row in batch]
                    tensor = y[t.arange(len(batch), device='cuda'), ep].contiguous().view(t.uint16)
                    visits[offset] = int((tensor[::2] == tensor[1::2]).all(-1).sum())

            rt.block = block
            with GpuLease(), t.inference_mode():
                progress(shard, -1, shard * 40, 'shard_start')
                _, logits = rt.forward(rr, callback)
                scores = rt.score(logits, rr)
                del logits
            arrays = {k: np.concatenate(v) for k, v in collected.items()}
            if not seal:
                arrays.update(totals=totals, counts=counts, unit_sse=unit_sse)
            archive(name + '.npz', arrays)
            summary = dict(array_sha=sha(OUT / (name + '.npz')), prereg_sha=sha(OUT / 'prereg.json'),
                           seal_sha=sha(OUT / 'candidate.json') if seal else None,
                           head_max=float(arrays['head_audit'].max()), prefix_equal=sum(visits.values()), scores=scores)
            commit(name + '.json', summary)
        max_errors.append(summary['head_max'])
        prefix_checks.append(summary['prefix_equal'])
        all_scores.extend(summary['scores'])
        shards[name + '.npz'] = summary['array_sha']
        assert summary['head_max'] <= CAP and summary['prefix_equal'] == min(128, len(rows)-start)//2, summary['head_max']
    groups, passes = behavior(rows, all_scores)
    commit(f'{split}-collection.json', dict(provenance=prereg['provenance'], shards=shards, cells=groups,
                                         behavior_pass=passes, max_head_rebuild_abs=max(max_errors),
                                         equivalent_prefix_pairs=sum(prefix_checks), elapsed=time.monotonic()-clock,
                                         precision='BF16, FP32 native head accounting',
                                         torch_version=t.__version__, transformers_version=transformers.__version__))
    print(f'Completed {split}: {len(rows)} inputs; behavior={passes}', flush=True)


def select():
    from source_binding_runtime import Runtime
    prereg = checked()
    discovery = read(OUT / 'discovery-collection.json')
    assert discovery['behavior_pass'] and not (OUT / 'candidate.json').exists()
    totals, counts, unit_sse = [], [], []
    for name, digest in discovery['shards'].items():
        assert sha(OUT / name) == digest
        with np.load(OUT / name) as data:
            totals.append(data['totals'])
            counts.append(data['counts'])
            unit_sse.append(data['unit_sse'])
    totals = np.sum(totals, axis=0)
    counts = np.sum(counts, axis=0)
    mean = totals[:, 0] / counts[:, None]
    variance = np.maximum(totals[:, 1] / counts[:, None] - mean**2, 0)
    sensitivity = np.sum(unit_sse, axis=0)
    units = np.argsort(-sensitivity, kind='stable')[:32]
    rt = Runtime('qwen14')
    prefix = f'model.layers.{BLOCK}.'
    frozen = dict(unit_sse=sensitivity, units=units)
    rng = np.random.default_rng(SEED + 17)
    for kind, region in [('q', HEAD_SLICE), ('k', KV_SLICE), ('v', KV_SLICE)]:
        part = rt.weight(prefix + f'self_attn.{kind}_proj.weight', 'cpu')[region].contiguous().clone()
        i = int(kind != 'q')
        score = variance[i] * part.float().numpy().astype(np.float64).__pow__(2).sum(0)
        rank = np.argsort(-score, kind='stable')
        random = rng.permutation(rt.dim)
        frozen['score_' + kind] = score
        frozen['mu_' + kind] = mean[i].astype(np.float32)
        for n in (32, 128):
            frozen[f'native{n}_idx_{kind}'] = rank[:n]
            frozen[f'random{n}_idx_{kind}'] = random[:n]
        frozen['param_' + kind] = part.view(rt.t.uint16).numpy()
    for key, name, axis, idx in [('o', 'self_attn.o_proj.weight', 1, HEAD_SLICE),
                                 ('gate', 'mlp.gate_proj.weight', 0, units), ('up', 'mlp.up_proj.weight', 0, units),
                                 ('down', 'mlp.down_proj.weight', 1, units)]:
        weight = rt.weight(prefix + name, 'cpu')
        part = (weight[:, idx] if axis else weight[idx]).contiguous()
        frozen['param_' + key] = part.view(rt.t.uint16).numpy().copy()
    for key, name in [('q_norm', 'self_attn.q_norm.weight'), ('k_norm', 'self_attn.k_norm.weight'),
                      ('post_norm', 'post_attention_layernorm.weight')]:
        frozen['param_' + key] = rt.weight(prefix + name, 'cpu').view(rt.t.uint16).numpy().copy()
    archive('predictor.npz', frozen)
    commit('candidate.json', dict(utc=datetime.now(timezone.utc).isoformat(), prereg_sha=sha(OUT / 'prereg.json'),
                                  files={p: sha(OUT / p) for p in ('predictor.npz', 'discovery-collection.json')},
                                  units=units.tolist(), block=BLOCK, head=HEAD, kv=KV,
                                  provenance=prereg['provenance'], decision_budget=128))
    print('Sealed coordinate sets and downstream units before confirmation', flush=True)


def evaluate():
    prereg = checked()
    seal = candidate()
    collection = read(OUT / 'confirm-collection.json')
    rows = read(OUT / 'confirm-materials.json')
    assert sha(OUT / 'confirm-materials.json') == prereg['materials']['confirm']['sha']
    amounts = {}
    for name, digest in collection['shards'].items():
        assert sha(OUT / name) == digest
        summary = read(OUT / name.replace('.npz', '.json'))
        assert summary['seal_sha'] == sha(OUT / 'candidate.json')
        with np.load(OUT / name) as data:
            for key in data.files:
                if not key.startswith('replay_'):
                    amounts.setdefault(key, []).append(data[key])
    arrays = {key: np.concatenate(value) for key, value in amounts.items()}
    families = np.array([r['family'] for r in rows])
    gold = np.array([r['target_record'] for r in rows])
    boot = np.random.default_rng(SEED + 29).integers(0, COUNT['confirm'], (10000, COUNT['confirm']))

    def compare(diff):
        vector = np.bincount(families, weights=diff) / 16
        return dict(mean=float(vector.mean()), ci975=np.quantile(vector[boot].mean(1), [.0125, .9875]).tolist())

    observed = arrays['mass_actual'].argmax(-1)
    report = {}
    target = arrays['mlp_actual'].astype(np.float64)
    masses = arrays['mass_actual']
    ii = np.arange(len(rows))

    def margin(a):
        return np.log(np.maximum(a[ii, gold], 1e-20)) - np.log(np.maximum(a[ii, gold ^ 1], 1e-20))

    for arm in ARMS:
        item = dict(mlp_mse_by_GUA=np.mean((arrays['mlp_' + arm] - target)**2, axis=(0, 2)).tolist())
        if arm not in ('zero', 'full'):
            m = arrays['mass_' + arm]
            item.update(route_agreement=float((m.argmax(-1) == observed).mean()),
                        task_gold_agreement=float((m.argmax(-1) == gold).mean()),
                        write_mse=float(arrays['write_se_' + arm].mean()),
                        same_archive_margin_rmse=float(np.sqrt(np.mean((margin(m)-margin(masses))**2))),
                        by_cell_route_agreement=((m.argmax(-1) == observed).reshape(-1, 16).mean(0)).tolist())
        report[arm] = item
    model = 'native128'
    gates = dict(selection={}, write={}, downstream={})
    for base in ('random128', 'mean'):
        a = (arrays['mass_' + model].argmax(-1) == observed)
        b = (arrays['mass_' + base].argmax(-1) == observed)
        summary = compare(a.astype(float) - b)
        gates['selection'][base] = dict(**summary, passes=bool(a.mean() >= .90 and summary['mean'] >= .10 and summary['ci975'][0] > 0))
        x, y = [arrays['write_se_' + arm] for arm in (model, base)]
        summary = compare(y-x)
        gain = 1 - float(x.mean() / max(y.mean(), 1e-20))
        gates['write'][base] = dict(**summary, fraction_gain=gain, passes=gain >= .20 and summary['ci975'][0] > 0)
    for base in ('random128', 'mean', 'zero'):
        x, y = [np.mean((arrays['mlp_' + arm][:, 2] - target[:, 2])**2, axis=-1) for arm in (model, base)]
        summary = compare(y-x)
        gain = 1 - float(x.mean() / max(y.mean(), 1e-20))
        gates['downstream'][base] = dict(**summary, fraction_gain=gain, passes=gain >= .20 and summary['ci975'][0] > 0)
    admitted = collection['behavior_pass'] and all(result['passes'] for gate in gates.values() for result in gate.values())
    failures = np.flatnonzero(arrays['mass_native128'].argmax(-1) != observed)
    result = dict(provenance=seal['provenance'], candidate_sha=sha(OUT/'candidate.json'),
                  collection_sha=sha(OUT/'confirm-collection.json'), inputs=len(rows), families=COUNT['confirm'],
                  behavior_pass=collection['behavior_pass'], natural_gold_route=float((observed == gold).mean()),
                  arms=report, gates=gates, causal_admitted=admitted, causal_tested=False,
                  example_failures=[dict(id=rows[i]['id'], actual_mass=masses[i].tolist(),
                                         prediction=arrays['mass_native128'][i].tolist()) for i in failures[:16]],
                  scope='local conditional prediction with real operation-before states and other-head background')
    commit('evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'discovery', 'select', 'confirm', 'evaluate'))
    stage = parser.parse_args().stage
    if stage in ('discovery', 'confirm'):
        collect(stage)
    else:
        {'freeze': freeze, 'select': select, 'evaluate': evaluate}[stage]()
