"""Frozen, grouped prediction of style x negation interactions (Phase 2752).

No holdout double-conditioned state or answer label is a predictor input.
Material, input contract, resource allocation and code are sealed before capture.
Only last-position fields are collected; this is not an all-token mechanism claim.
"""
import argparse
import gc
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/rdc_context_interaction_20260923'
FAMILIES = ('category', 'role', 'spatial', 'containment')
COHORTS = dict(train=32, validation=8, entity=12, joint_wording=12, role_order=12, depth=12)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def snapshot(path):
    dest = OUT / 'code_snapshots' / (sha(path) + path.suffix)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(path, dest)
    assert sha(dest) == sha(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path), snapshot=str(dest.relative_to(ROOT)))


def make_world(family, names, role, truth, depth, wording):
    """Finite counterfactual input graphs, with explicit truth used only in scoring."""
    p = names[:depth + 1]
    if family == 'category':
        facts = [f'{p[0]} is a {p[1]}.'] + [f'Every {p[k]} is a {p[k+1]}.' for k in range(1, depth)]
        facts += [f'No {p[1]} is a {names[4]}.']
        target = p[-1] if truth else names[4]
        proposition = f'{p[0]} belongs to the {target} category' if role else f'{p[0]} is a {target}'
        if wording == 1:
            facts = [f'{p[0]} belongs to {p[1]}.'] + [f'The {p[k]} category is included in {p[k+1]}.' for k in range(1, depth)] + [f'The {p[1]} and {names[4]} categories do not overlap.']
    elif family == 'role':
        facts = [f'{p[k]} gave the key to {p[k+1]}.' for k in range(depth)]
        facts += ['These transfers happened in the listed order. All people named here are different.']
        chosen = (p[-1] if role else p[0]) if truth else (p[0] if role else p[-1])
        proposition = f'{chosen} was the ' + ('final recipient' if role else 'initial giver')
        if wording == 1:
            facts = [f'The key was given to {p[k+1]} by {p[k]}.' for k in range(depth)] + facts[-1:]
        # Event order must not be reversed when surface order changes: index events.
        facts = [f'Event {k+1}: {s}' if k < depth else s for k, s in enumerate(facts)]
        facts[-1] = 'Events happened in numerical order. All people named here are different.'
    elif family == 'spatial':
        facts = [f'{p[k]} is to the left of {p[k+1]}.' for k in range(depth)]
        facts += ['Left and right are opposite directions on one line.']
        subject, obj = (p[-1], p[0]) if role else (p[0], p[-1])
        direction = 'right' if bool(role) == bool(truth) else 'left'
        proposition = f'{subject} is to the {direction} of {obj}'
        if wording == 1:
            facts = [f'{p[k+1]} is to the right of {p[k]}.' for k in range(depth)] + facts[-1:]
    else:
        facts = [f'{p[k]} is inside {p[k+1]}.' for k in range(depth)]
        facts += ['These are strictly nested containers. No container is inside itself.']
        subject, obj = (p[0], p[-1]) if truth else (p[-1], p[0])
        proposition = f'{obj} contains {subject}' if role else f'{subject} is inside {obj}'
        if wording == 1:
            facts = [f'{p[k+1]} contains {p[k]}.' for k in range(depth)] + facts[-1:]
    return facts, proposition


def prepare():
    path = OUT / 'material.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    tokenizers = {s: AutoTokenizer.from_pretrained(ROOT / 'models/hf' / m, local_files_only=True)
                  for s, m in [('4B', 'qwen3-4b'), ('14B', 'Qwen3-14B')]}
    rng = np.random.default_rng(2752001)
    syllables = ('ba', 'ce', 'di', 'fo', 'gu', 'ha', 'ji', 'ko', 'lu', 'me', 'ni', 'po', 'ra', 'se', 'ti', 'vu')
    name_pool = [(''.join((syllables[i//256], syllables[(i//16)%16], syllables[i%16])) + 'x').capitalize() for i in range(4096)]
    rng.shuffle(name_pool)
    rows, worlds = [], []
    windex = 0
    for family in FAMILIES:
        for cohort, count in COHORTS.items():
            for j in range(count):
                names = name_pool[5*windex:5*windex+5]
                wid = f'{family}_{cohort}_{j:02d}'
                role, order = (1, 1) if cohort == 'role_order' else ((0, 0), (0, 1), (1, 0))[j % 3]
                depth = 3 if cohort == 'depth' else 1 + (j//6) % 2
                truth = (j//3) % 2 == 0
                wordings = (2, 3) if cohort == 'joint_wording' else (0, 1)
                if cohort == 'train':
                    wordings = (0, 1, 2)  # wording 2 is HELD OUT, never fit/tune.
                worlds.append(dict(id=wid, family=family, cohort=cohort, index=j, entities=names,
                                   role=role, order=order, depth=depth, relation_truth=truth))
                for wording in wordings:
                    split = 'wording' if cohort == 'train' and wording == 2 else cohort
                    facts, prop = make_world(family, names, role, truth, depth, wording)
                    if order:
                        facts = facts[:-1][::-1] + facts[-1:]
                    facts_text = ' '.join(facts)
                    if wording == 2:
                        facts_text = 'Given statements: ' + ' ; '.join(s.rstrip('.') for s in facts) + '.'
                    elif wording == 3:
                        facts_text = 'Consider the following information. ' + facts_text
                    qbase = [f'Is it {{word}} true that {prop}?', f'Would it {{word}} be correct to say that {prop}?',
                             f'According to these statements, is it {{word}} true that {prop}?', f'Can we {{word}} conclude that {prop}?'][wording]
                    for style in (0, 1):
                        for neg in (0, 1):
                            lead = 'Use a formal tone. ' if style else 'Use an ordinary tone. '
                            question = qbase.format(word='not' if neg else 'really')
                            text = lead + facts_text + '\nQuestion: ' + question + '\nAnswer yes or no.\nAnswer:'
                            # Only syntax/graph metadata and source-prompt token positions enter features.
                            token_meta = {}
                            for side, tok in tokenizers.items():
                                ids = tok.encode(text, add_special_tokens=False)
                                token_meta[side] = dict(token_ids=ids, length=len(ids),
                                    question_start=len(tok.encode(lead + facts_text + '\nQuestion: ', add_special_tokens=False)))
                            rows.append(dict(id=f'{wid}_t{wording}_s{style}n{neg}', group=f'{wid}_t{wording}', world=wid,
                                family=family, split=split, cohort=cohort, world_index=j, wording=wording, role=role,
                                order=order, depth=depth, fact_count=len(facts), cond=2*style+neg, style=style, negation=neg,
                                text=text, tokenization=token_meta, expected='yes' if truth != bool(neg) else 'no',
                                replication14B=(wording == (2 if cohort == 'joint_wording' else 0) and
                                    j < (8 if cohort == 'train' else 2 if cohort == 'validation' else 3)) or
                                    (cohort == 'train' and wording == 2 and j < 3)))
                windex += 1
    assert len(worlds) == 352 and len(rows) == 3328
    all_names = [n for w in worlds for n in w['entities']]
    assert len(set(all_names)) == len(all_names)
    for side in tokenizers:
        groups = {}
        for row in rows:
            groups.setdefault(row['group'], []).append(row)
        assert all(len({r['tokenization'][side]['length'] for r in rs}) == 1 for rs in groups.values()), 'Within-quartet position mismatch'
    material = dict(created_utc=now(), rows=rows, worlds=worlds, sampling_unit='world; wordings and 4 conditions are repeated measurements',
        annotation_scope='Generator-provided typed input graph, not automatic natural-language parsing. Names are unique experimental entities, not claims about training corpus novelty.')
    write(path, material)
    write(OUT / 'design.json', dict(created_utc=now(), material_sha256=sha(path), seed=2752001,
        target='I_l = h11_l - h10_l - h01_l + h00_l; predict double state from h10+h01-h00 + predicted I',
        input_contract=dict(graph='family, query-role, fact-order, path depth, question syntax/context, length/position; no answer or world ID',
                            state='baseline h00 or three source states at SAME layer allowed; no heldout h11, future token or interaction',
                            training='Only train targets fit; only validation targets select alpha. Wording test shares training worlds, all other tests use new disjoint entities.'),
        candidates=['zero/additive', 'family mean', 'surface ridge', 'graph-only ridge', 'graph+surface ridge',
                    'base-coordinate ridge', 'source-coordinate product ridge', 'graph+surface residual + source-coordinate product ridge'],
        alpha_grid=[0.01, 0.1, 1.0, 10.0], selection='mean validation final-layer interaction relative error, fixed per candidate; no refit after selection',
        primary='Heldout interaction relative L2 error; total combined-change relative L2 error secondary; zero interaction denominator rows excluded and counted',
        uncertainty='Paired mean-error bootstrap over worlds, 2000 draws, by split. Fixed 2 train and 2 novel template constructions; no template-population significance.',
        resource='4B pilot16 then3328; 14B pilot16 then frozen reduced subset; sequential CUDA models, 14B auto CPU offload10GiB. Full14B cap2100s.',
        count4B=len(rows), count14B=sum(r['replication14B'] for r in rows),
        scope='Last-position native all-coordinate field at every returned hidden boundary, raw final residual separately; BF16 batch1 eager no-cache, no padding/truncation.',
        generation='First-token accuracy and yes/no mass only, not full free-generation ability.',
        limitations=['synthetic English input graphs', 'style x question negation only', 'known families', 'state methods need three source forwards',
                     '14B reduced subset has only8 train worlds/family and one training wording; replication is limited and not model-size isolation']),
        )
    write(OUT / 'pre_capture_seal.json', dict(created_utc=now(), files=[snapshot(Path(__file__))],
        material_sha256=sha(path), design_sha256=sha(OUT/'design.json')))
    print(json.dumps(dict(rows=len(rows), worlds=len(worlds), rows14B=sum(r['replication14B'] for r in rows)), indent=2), flush=True)
    return material


def collect(side, pilot=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mat = prepare()
    rows = [r for r in mat['rows'] if side == '4B' or r['replication14B']]
    if pilot:
        rows = [next(r for r in rows if r['family'] == f and r['cond'] == c and r['wording'] == 0) for f in FAMILIES for c in range(4)]
    dest = OUT / (side + ('_pilot' if pilot else ''))
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'capture_done.json').exists():
        print('Completed capture reused:', dest, flush=True)
        return
    assert not list(dest.glob('chunk_*.npz')), 'Preserve partial captures; explicit resume required.'
    torch.set_num_threads(8)
    torch.manual_seed(2752001)
    torch.backends.cuda.matmul.allow_tf32 = False
    mdir = ROOT/'models/hf'/('qwen3-4b' if side == '4B' else 'Qwen3-14B')
    start = time.time()
    execution = dict(created_utc=now(), source=snapshot(Path(__file__)), material_sha256=sha(OUT/'material.json'),
        design_sha256=sha(OUT/'design.json'), torch=torch.__version__, dtype='bfloat16', attention='eager', batch=1,
        cache=False, padding=False, truncation=False, selected_ids=[r['id'] for r in rows], model_config_sha256=sha(mdir/'config.json'))
    write(dest/'execution.json', execution)
    opts = dict(dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True)
    if side == '14B':
        opts.update(device_map='auto', max_memory={0:'10GiB', 'cpu':'48GiB'})
    model = AutoModelForCausalLM.from_pretrained(mdir, **opts).eval()
    if side == '4B':
        model.to('cuda')
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True)
    execution['device_map'] = {k:str(v) for k,v in getattr(model, 'hf_device_map', {'':'cuda'}).items()}
    execution['load_seconds'] = time.time() - start
    write(dest/'execution.json', execution)
    # Single-token yes/no sets explicitly retained; not silently equated to whole answers.
    yesno = {a:sorted({tok.encode(s, add_special_tokens=False)[0] for s in (a, a.capitalize(), ' '+a, ' '+a.capitalize()) if len(tok.encode(s, add_special_tokens=False)) == 1}) for a in ('yes','no')}
    write(dest/'answer_token_sets.json', {a:[dict(id=i, text=tok.decode([i])) for i in ids] for a,ids in yesno.items()})
    last = model.model.layers[-1]
    store = {}
    def rec(key, pre=False):
        def hook(mod, args, output=None):
            v = args[0] if pre else output
            store[key] = v[0] if isinstance(v, tuple) else v
        return hook
    hooks = [last.register_forward_pre_hook(rec('h0', True)), last.self_attn.register_forward_hook(rec('a')),
             last.mlp.register_forward_hook(rec('m')), last.register_forward_hook(rec('h2')),
             model.model.norm.register_forward_pre_hook(rec('pre', True))]
    bank, meta, anchors, chunk = [], [], [], 0
    def flush():
        nonlocal bank, meta, chunk
        if not meta:
            return
        np.savez(dest/f'chunk_{chunk:03d}.npz', hidden=np.stack(bank))
        write(dest/f'chunk_{chunk:03d}.json', meta)
        bank, meta = [], []
        chunk += 1
    try:
        for i,row in enumerate(rows):
            ids = torch.tensor([row['tokenization'][side]['token_ids']], device=model.get_input_embeddings().weight.device)
            store.clear()
            with torch.inference_mode():
                out = model(input_ids=ids, use_cache=False, output_hidden_states=True, logits_to_keep=1)
                err = float(((store['h0'] + store['a']) + store['m'] - store['h2']).abs().max())
                err_norm = float((model.model.norm(store['pre']) - out.hidden_states[-1]).abs().max())
                if err != 0 or err_norm != 0:
                    raise RuntimeError(f'Boundary anchor failed: {err}, {err_norm}')
                # Add raw last residual AFTER final-norm item; boundary metadata is explicit.
                hidden = np.stack([h[0,-1].float().cpu().numpy() for h in out.hidden_states] + [store['pre'][0,-1].float().cpu().numpy()])
                bank.append(hidden)
                logits = out.logits[0,-1].float()
                lp = logits.log_softmax(-1)
                pred = int(logits.argmax())
                vals, idx = torch.topk(logits, 20)
                meta.append(dict(id=row['id'], group=row['group'], world=row['world'], split=row['split'],
                    prediction_id=pred, prediction_text=tok.decode([pred]),
                    yesno_logmass={a:float(torch.logsumexp(lp[toks], 0)) for a,toks in yesno.items()},
                    top20_ids=idx.cpu().tolist(), top20_logits=vals.cpu().tolist(), logsumexp=float(logits.logsumexp(0))))
                anchors.append(dict(id=row['id'], residual_max=err, norm_max=err_norm))
                del out, hidden, logits, lp
            if (i+1) % 64 == 0:
                flush()
                print(f'{side} {i+1}/{len(rows)} {time.time()-start:.1f}s', flush=True)
            if time.time()-start > (600 if pilot else 2100):
                flush()
                write(dest/'partial.json', dict(count=i+1, reason='bounded time cap', elapsed=time.time()-start))
                return
        flush()
        write(dest/'anchors.json', anchors)
        write(dest/'capture_done.json', dict(created_utc=now(), count=len(rows), chunks=chunk, elapsed=time.time()-start,
            last_norm_index=len(model.model.layers), raw_last_index=len(model.model.layers)+1,
            layers=len(model.model.layers), hidden_size=model.config.hidden_size,
            precision='FP32 storage is lossless embedding of native BF16 activations', peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            boundary='index0 embedding;1..L-1 intermediate residual outputs;L final norm;L+1 raw final residual',
            code_sha256=sha(Path(__file__))))
    finally:
        for h in hooks:
            h.remove()
        del model, store
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare','collect'])
    p.add_argument('--model', choices=['4B','14B'], default='4B')
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    prepare() if a.mode == 'prepare' else collect(a.model, a.pilot)
