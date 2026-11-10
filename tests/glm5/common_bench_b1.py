"""Stage B (Phase 384): common two-factor benchmark B1, Qwen3-4B BF16.

Factors: answer source S (document record / parametric knowledge) x requested output language L
(English / Chinese), crossed with the queried slot q.  Five colours, entity-level family split.
Stages:  draft  -> pilot (GPU, discovery only) -> baselines (CPU, discovery only) -> freeze.
Confirm and lockbox materials are generated deterministically but never passed through the model here.
"""
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import argparse, hashlib, json, random, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'tests/glm5/result/common_bench_b1_v3'
MODEL = ROOT / 'models/hf/qwen3-4b'
RUN, SEED = 'COMMON-BENCH-B1-20261110-v3', 202611100388
SUFFIX = 'Answer:\n'
AMENDMENT = {
    'supersedes': 'COMMON-BENCH-B1-20261110-v1 and -v2 (both pilots failed their behaviour gates; kept in '
                  'tests/glm5/result/common_bench_b1 and common_bench_b1_v2)',
    'reasons': [
        'v1 ended the prompt with "Answer:"; the model then starts a Chinese answer with a space-plus-partial-byte token '
        'shared by several colours, so first-token scoring could not see correct Chinese answers '
        '(free generation was right). v2 ends with "Answer:" and a line break, after which the first token is the colour word.',
        'v1 discovery entities "rubber ducks" and "hockey pucks" scored 0 on parametric colour; manufactured objects '
        'without one canonical colour were replaced by natural kinds in all splits. Decided from discovery behaviour only.',
        'v2 fixed the Chinese scoring (document cells 1.00, no language errors, free generation exact 1.00) but failed the '
        'per-frame gate inside the parametric cells: 12 items per frame and cell, and one entity ("baby chicks", 0.41) '
        'caused most errors. v3 replaces that entity and states the gate in two steps, written before the v3 pilot: '
        'first the entity screen, then cell and frame thresholds on the remaining items, frames pooled over cells.'],
    'note': 'v1 confirm and lockbox materials were never run through the model; v2 regenerates every split with a new seed'}
sys.stdout.reconfigure(encoding='utf-8')

COLORS = ['red', 'yellow', 'green', 'white', 'black']
ZH = {'red': '红', 'yellow': '黄', 'green': '绿', 'white': '白', 'black': '黑'}
ENTITIES = {
    'red': ['fresh blood', 'ripe strawberries', 'ripe tomatoes', 'ripe cherries', 'rubies', 'poppies',
            'ladybugs', 'ripe raspberries', 'watermelon flesh', 'cooked lobsters', 'cranberries', 'pomegranate seeds'],
    'yellow': ['ripe bananas', 'lemons', 'sunflower petals', 'egg yolks', 'canaries', 'dandelion flowers',
               'sweet corn kernels', 'butter', 'buttercups', 'mustard', 'daffodils',
               'pineapple flesh'],
    'green': ['fresh grass', 'emeralds', 'spinach leaves', 'broccoli', 'limes', 'cucumbers', 'lettuce',
              'garden peas', 'mint leaves', 'pine needles', 'basil leaves', 'parsley'],
    'white': ['fresh snow', 'milk', 'table salt', 'cotton balls', 'polar bears', 'swans', 'cooked rice',
              'refined sugar', 'wheat flour', 'cauliflower', 'coconut flesh', 'whipped cream'],
    'black': ['coal', 'crows', 'ravens', 'charcoal', 'tar', 'soot', 'obsidian', 'crude oil', 'squid ink',
              'onyx', 'the night sky', 'ebony wood'],
}
ENTITY_SPLIT = {'discovery': 4, 'confirm': 5, 'lockbox': 3}     # entities per colour
NOUNS = ['box', 'parcel', 'folder', 'bag', 'crate', 'envelope', 'tray', 'basket']
BASE = {'discovery': 30000, 'confirm': 40000, 'lockbox': 50000, 'lockbox_frame': 60000}
NOTES = ['The room has a wooden shelf.', 'The clerk checked the date.', 'The record is stored in a cabinet.',
         'A visitor left a short note.', 'The door was closed at noon.', 'The table stands near the window.']
DOC = ['Record: {a} is {x}; {b} is {y}.',
       'The label on {a} says {x}. The label on {b} says {y}.',
       'Inventory entry: {a}, color {x}; {b}, color {y}.',
       'The inspector recorded {a} as {x} and {b} as {y}.']
MENTION = ['The shopping note also lists {e1} and {e2}.', 'A picture on the wall shows {e1} next to {e2}.']
Q_DOC = ['Which color does the file give for {t}?', 'According to the file, what color is {t}?',
         'What color is recorded for {t} in the file?', 'State the color that the file assigns to {t}.']
Q_PAR = ['What is the usual real-world color of {t}?', 'In everyday life, what color is typical for {t}?',
         'Ignoring the file, what is the normal color of {t}?', 'State the typical real-world color of {t}.']
INSTR = ['Answer in {A}, not {B}. Give only the color word.',
         'Use {A} for your answer; {B} is excluded. Write just the color.',
         'Reply with a single color word in {A}, not in {B}.',
         'The answer must be one color word written in {A}; avoid {B}.']
HELD = {'doc': ['The color assigned to {a} is {x}, whereas {y} is the color assigned to {b}.',
                'Consult the two labels: {x} belongs to {a}; {y} belongs to {b}.',
                'In the final description, {a} has the {x} coating, and {b} has the {y} coating.',
                'The signed form pairs {a} with {x} and pairs {b} with {y}.'],
        'mention': ['Someone scribbled the words {e1} and {e2} in the margin.'],
        'q_doc': ['Look up {t} in the file and report its color.', 'What color did the file note down for {t}?'],
        'q_par': ['Thinking of the real world rather than the file, what is the common color of {t}?',
                  'Outside this file, what color would people expect for {t}?'],
        'instr': ['Write one color word, and write it in {A} rather than {B}.',
                  'Your reply should be in {A} (not {B}) and contain only the color.']}
CLASSES = [f'{lang}_{c}' for lang in ('en', 'zh') for c in COLORS]
PREREG = {
    'behavior_gate': {'metric': '10-way first-token forced choice (5 colours x 2 languages)',
                      'step_1_entity_screen': 'parametric forced-choice accuracy per entity; below 0.75 the entity and every '
                                              'family containing it are excluded; more than 20 % excluded entities fails the gate',
                      'max_excluded_entity_fraction': 0.20,
                      'step_2': 'on the remaining items: every source x language cell and every frame (pooled over cells)',
                      'min_cell_accuracy': 0.90, 'min_frame_accuracy': 0.90, 'min_overall_top1': 0.80,
                      'discovery_entity_rule': 'a discovery entity whose parametric forced-choice accuracy in the pilot '
                                               'is below 0.75 is dropped from discovery fits; nothing is dropped by internals',
                      'confirm_rule': 'confirm / lockbox items are scored by base behaviour only; failures stay in the '
                                      'denominator of behaviour and are excluded from mechanism positives'},
    'combo_holdout': {'cells': [{'source': 'document', 'language': 'zh', 'answer': 'green'},
                                {'source': 'parametric', 'language': 'en', 'answer': 'white'}],
                      'rule': 'stage C fits on discovery must exclude these cells; they are predicted, then checked on confirm'},
    'baseline_ranking': 'oracle predictors (given S, q, L or colour labels) and text-only predictors are ranked in separate tables',
    'metrics': ['internal state and increment prediction (raw and per-coordinate standardised)',
                'full-vocabulary KL and task answer margins', 'intervention effect prediction and capability retention',
                'generalisation scope, parameter count and compute cost'],
}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, operation='阶段 B 基准产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1,
                            default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(stage, n, total, start, **extra):
    sec = time.monotonic() - start
    d = {'time': datetime.now().astimezone().isoformat(), 'stage': stage, 'done': n, 'total': total,
         'seconds': round(sec, 1), 'eta_seconds': round(sec * (total - n) / n, 1) if n else None, **extra}
    try:
        import torch
        if torch.cuda.is_initialized():
            d.update(allocated_MiB=round(torch.cuda.memory_allocated() / 2 ** 20),
                     peak_MiB=round(torch.cuda.max_memory_allocated() / 2 ** 20))
            cmd = ("(Get-Counter '\\GPU Process Memory(*)\\Shared Usage' -ErrorAction Stop).CounterSamples | "
                   "Where-Object {$_.InstanceName -like 'pid_" + str(os.getpid()) + "_*'} | "
                   "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum")
            p = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True, timeout=8)
            d['shared_gpu_MiB'] = round(float(p.stdout.strip()) / 2 ** 20, 2) if p.returncode == 0 and p.stdout.strip() else None
    except Exception:
        pass
    print(json.dumps(d, ensure_ascii=False), flush=True)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(d, ensure_ascii=False) + '\n')


def entity_splits():
    rng = random.Random(SEED)
    out = {k: {} for k in ENTITY_SPLIT}
    for c in COLORS:
        names = ENTITIES[c][:]
        assert not any(w in n.lower() for n in names for w in COLORS), c
        rng.shuffle(names)
        i = 0
        for split, k in ENTITY_SPLIT.items():
            out[split][c] = names[i:i + k]
            i += k
    return out


def frame_of(split, f):
    """16 seen frames (doc 4 x question 4); held-out frames use unseen sentences only."""
    if split == 'lockbox_frame':
        i, j = f % 4, (f // 4) % 2
        return {'id': 100 + i + 4 * j, 'doc': HELD['doc'][i], 'mention': HELD['mention'][0], 'q_doc': HELD['q_doc'][j],
                'q_par': HELD['q_par'][j], 'instr': HELD['instr'][(i + j) % 2], 'mention_first': bool(i % 2)}
    i, j = f % 4, (f // 4) % 4
    return {'id': i + 4 * j, 'doc': DOC[i], 'mention': MENTION[(i + j) % 2],
            'q_doc': Q_DOC[j], 'q_par': Q_PAR[j], 'instr': INSTR[(i + j) % 4], 'mention_first': bool((i + j // 2) % 2)}


def families(split):
    ents = entity_splits()['lockbox' if split.startswith('lockbox') else split]
    pairs = [(e1, c1, e2, c2) for c1 in COLORS for c2 in COLORS if c1 != c2 for e1 in ents[c1] for e2 in ents[c2]]
    random.Random(SEED + sum(map(ord, split))).shuffle(pairs)
    if split.startswith('lockbox'):
        pairs = pairs[0::2] if split == 'lockbox' else pairs[1::2]
    return pairs


def material(tok, split, f, pair):
    e1, p1, e2, p2 = pair
    rng = random.Random(SEED * 31 + BASE[split] + f)
    d1, d2 = rng.sample([c for c in COLORS if c not in (p1, p2)], 2)
    noun = NOUNS[(f // 16) % len(NOUNS)]
    a, b = f'{noun} {BASE[split] + 2 * f}', f'{noun} {BASE[split] + 2 * f + 1}'
    fr = frame_of(split, f)
    doc, mention = fr['doc'].format(a=a, b=b, x=d1, y=d2), fr['mention'].format(e1=e1, e2=e2)
    body = f'{mention}\n{doc}' if fr['mention_first'] else f'{doc}\n{mention}'
    head = (f'File {split[0]}{f:05d}. Some questions concern the file, others concern general knowledge.\n'
            f"{' '.join(rng.sample(NOTES, 3))}\n{body}")
    rows = []
    for s in (1, -1):            # +1 document, -1 parametric
        for q in (1, -1):        # +1 first slot, -1 second slot
            for l in (1, -1):    # +1 English, -1 Chinese
                target = (a if q == 1 else b) if s == 1 else (e1 if q == 1 else e2)
                gold = (d1 if q == 1 else d2) if s == 1 else (p1 if q == 1 else p2)
                rival = (p1 if q == 1 else p2) if s == 1 else (d1 if q == 1 else d2)   # same slot, other source
                question = (fr['q_doc'] if s == 1 else fr['q_par']).format(t=target)
                A, B = ('English', 'Chinese') if l == 1 else ('Chinese', 'English')
                text = f"{head}\n{question}\n{fr['instr'].format(A=A, B=B)}"
                prompt = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) + SUFFIX
                ids = tok.encode(prompt, add_special_tokens=False)
                cuts = [prompt.index(body) + len(body), prompt.index(question) + len(question)]
                pos = []
                for c in cuts:
                    assert prompt[c] == '\n'    # include the line break: '.\n' can be a single token
                    prefix = tok.encode(prompt[:c + 1], add_special_tokens=False)
                    assert ids[:len(prefix)] == prefix
                    pos.append(len(prefix) - 1)
                rows.append({'id': f'{split}-{f:04d}-{s}{q}{l}', 'split': split, 'family': f, 'frame': fr['id'],
                             'source': s, 'query': q, 'language': l, 'gold_color': gold, 'rival_color': rival,
                             'gold': f"{'en' if l == 1 else 'zh'}_{gold}", 'entities': [e1, e2],
                             'parametric_colors': [p1, p2], 'document_colors': [d1, d2], 'noun': noun,
                             'prompt': prompt, 'tokens': ids, 'positions': {'doc_end': pos[0], 'question_end': pos[1],
                                                                            'answer_prefix': len(ids) - 1}})
    assert len({tuple(r['tokens'][:r['positions']['doc_end'] + 1]) for r in rows}) == 1
    assert len({r['tokens'][-1] for r in rows}) == 1
    return rows


def choice_ids(tok):
    ids = {}
    for c in COLORS:
        ids[f'en_{c}'] = sorted({tok.encode(v, add_special_tokens=False)[0] for v in (' ' + c, c, ' ' + c.capitalize(), c.capitalize())})
        ids[f'zh_{c}'] = sorted({tok.encode(v, add_special_tokens=False)[0] for v in (ZH[c] + '色', ZH[c])})
    flat = [i for v in ids.values() for i in v]
    assert len(flat) == len(set(flat)), 'answer first tokens overlap'
    return ids


def draft():
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 64 << 20, operation='阶段 B 基准产物')
    assert not (OUT / 'draft.json').exists(), 'draft exists; a changed design needs a new run id'
    start = time.monotonic()
    snap = snapshot(MODEL)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    info = {}
    for split in BASE:
        pairs = families(split)
        rows = [r for f, pair in enumerate(pairs) for r in material(tok, split, f, pair)]
        assert len({tuple(r['tokens']) for r in rows}) == len(rows)
        save(f'{split}-materials.json', rows)
        info[split] = {'sha256': sha(OUT / f'{split}-materials.json'), 'families': len(pairs), 'rows': len(rows),
                       'frames': sorted({r['frame'] for r in rows}),
                       'max_tokens': max(len(r['tokens']) for r in rows)}
    save('draft.json', {'schema': 'common_bench_draft.v1', 'run_id': RUN, 'seed': SEED, 'model_snapshot': snap,
                        'code_sha256': sha(__file__), 'created_at': datetime.now().astimezone().isoformat(),
                        'materials': info, 'entity_splits': entity_splits(), 'choice_ids': choice_ids(tok),
                        'classes': CLASSES, 'prereg': PREREG, 'amendment': AMENDMENT, 'answer_prefix': SUFFIX,
                        'custody': 'confirm, lockbox and lockbox_frame are generated here and not run through the model in stage B'})
    log('draft', 1, 1, start, **{k: v['rows'] for k, v in info.items()})


def checked():
    d = read(OUT / 'draft.json')
    assert d['code_sha256'] == sha(__file__), 'source changed after the draft was written'
    for split, v in d['materials'].items():
        assert sha(OUT / f'{split}-materials.json') == v['sha256'], split
    return d


def pilot_rows(d):
    """Discovery only: 96 families = 6 per frame."""
    rows = read(OUT / 'discovery-materials.json')
    fams = sorted({r['family'] for r in rows})[:96]
    return [r for r in rows if r['family'] in fams]



def pilot():
    import torch
    from transformers import AutoTokenizer
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    sys.path.insert(0, str(Path(__file__).parent))
    import mlp_condition_design as D, mlp_condition_run as R, mlp_condition_stream_load as SL
    d = checked()
    assert not (OUT / 'pilot-behavior.json').exists()
    assert snapshot(MODEL) == d['model_snapshot']
    rows = pilot_rows(d)
    ids = [d['choice_ids'][k] for k in CLASSES]
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    torch.manual_seed(SEED)
    start = time.monotonic()
    out = []
    with GpuLease(), torch.inference_mode():
        D.checked = lambda: d
        R.save = lambda name, body, _: save(name, body)
        m = SL.load(MODEL, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
        n = len(rows) // 8
        log('pilot', 0, n, start)
        for f in range(n):
            rs = rows[f * 8:f * 8 + 8]
            batch = tok.pad({'input_ids': [r['tokens'] for r in rs]}, return_tensors='pt').to('cuda')
            logits = m(**batch, use_cache=False, logits_to_keep=1).logits[:, 0].float()
            for r, lg in zip(rs, logits):
                v = [float(lg[i].max()) for i in ids]
                gold, pred = CLASSES.index(r['gold']), int(np.argmax(v))
                rival = CLASSES.index(r['gold'].split('_')[0] + '_' + r['rival_color'])
                out.append({'id': r['id'], 'family': r['family'], 'frame': r['frame'], 'source': r['source'],
                            'query': r['query'], 'language': r['language'], 'gold': r['gold'], 'pred': CLASSES[pred],
                            'forced_correct': pred == gold, 'top1': int(lg.argmax()),
                            'top1_correct': int(lg.argmax()) in ids[gold], 'top1_text': tok.decode([int(lg.argmax())]),
                            'margin_vs_best_wrong': v[gold] - max(x for i, x in enumerate(v) if i != gold),
                            'margin_vs_rival_source': v[gold] - v[rival],
                            'entity': r['entities'][0 if r['query'] == 1 else 1] if r['source'] == -1 else None,
                            'entities': r['entities']})
            if (f + 1) % 16 == 0 or f + 1 == n:
                log('pilot', f + 1, n, start)
        gen_rows = [r for r in rows if r['family'] < 16]          # one family per frame, all 8 conditions
        gens = []
        for i in range(0, len(gen_rows), 8):
            rs = gen_rows[i:i + 8]
            batch = tok.pad({'input_ids': [r['tokens'] for r in rs]}, return_tensors='pt').to('cuda')
            g = m.generate(**batch, max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
            for r, seq in zip(rs, g):
                text = tok.decode(seq[batch['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                want = r['gold_color'] if r['language'] == 1 else ZH[r['gold_color']] + '色'
                gens.append({'id': r['id'], 'text': text, 'expected': want,
                             'exact': text.lower().rstrip('.。') in (want, ZH[r['gold_color']])})
        log('generation', len(gens), len(gens), start)
    acc = lambda sel, key='forced_correct': float(np.mean([x[key] for x in sel])) if sel else None
    gate = PREREG['behavior_gate']
    ent = {}
    for x in out:
        if x['entity']:
            ent.setdefault(x['entity'], []).append(x['forced_correct'])
    ent = {k: {'n': len(v), 'forced': float(np.mean(v))} for k, v in sorted(ent.items())}
    dropped = [k for k, v in ent.items() if v['forced'] < 0.75]
    all_items, out = out, [x for x in out if not set(x['entities']) & set(dropped)]
    frames = {str(fr): acc([x for x in out if x['frame'] == fr]) for fr in sorted({x['frame'] for x in out})}
    cells = {}
    for s in (1, -1):
        for l in (1, -1):
            sel = [x for x in out if x['source'] == s and x['language'] == l]
            name = f"{'document' if s == 1 else 'parametric'}_{'en' if l == 1 else 'zh'}"
            cells[name] = {'n': len(sel), 'forced': acc(sel), 'top1': acc(sel, 'top1_correct')}
    wrong = [x for x in out if not x['forced_correct']]
    passed = len(dropped) <= gate['max_excluded_entity_fraction'] * len(ent)         and all(c['forced'] >= gate['min_cell_accuracy'] for c in cells.values())         and min(frames.values()) >= gate['min_frame_accuracy'] and acc(out, 'top1_correct') >= gate['min_overall_top1']
    save('pilot-items.json', all_items)
    save('pilot-behavior.json', {
        'schema': 'common_bench_pilot.v2', 'run_id': RUN, 'draft_sha256': sha(OUT / 'draft.json'),
        'scope': 'discovery only; confirm and lockbox not read', 'families': len(rows) // 8, 'items': len(all_items),
        'before_entity_screen': {'forced': acc(all_items), 'top1': acc(all_items, 'top1_correct')},
        'overall': {'items_after_screen': len(out), 'forced': acc(out), 'top1': acc(out, 'top1_correct')},
        'cells': cells, 'frames_pooled': frames, 'parametric_entities': ent, 'dropped_discovery_entities': dropped,
        'errors': {'n': len(wrong), 'language_wrong': sum(x['pred'][:2] != x['gold'][:2] for x in wrong),
                   'rival_source_color': sum(x['margin_vs_rival_source'] < 0 for x in wrong),
                   'examples': wrong[:12]},
        'generation': {'n': len(gens), 'exact': float(np.mean([g['exact'] for g in gens])), 'samples': gens},
        'gate': gate, 'passed': bool(passed), 'peak_MiB': round(torch.cuda.max_memory_allocated() / 2 ** 20),
        'items_sha256': sha(OUT / 'pilot-items.json'), 'elapsed_seconds': round(time.monotonic() - start, 1)})
    print(json.dumps({'passed': bool(passed), 'overall': acc(out), 'cells': {k: v['forced'] for k, v in cells.items()}}), flush=True)


def baselines():
    """Text-only predictors, discovery only, leave-entities-out folds.  No model output is used as input."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    d = checked()
    rows = read(OUT / 'discovery-materials.json')
    ents = d['entity_splits']['discovery']
    start = time.monotonic()
    y = np.array([CLASSES.index(r['gold']) for r in rows])
    docs = [' '.join(map(str, r['tokens'])) for r in rows]
    res = {name: [] for name in ('majority', 'last_color_word', 'unigram_logreg', 'bigram_logreg')}
    for k in range(ENTITY_SPLIT['discovery']):
        held = {ents[c][k] for c in COLORS}
        test = np.array([set(r['entities']) <= held for r in rows])
        train = np.array([not (set(r['entities']) & held) for r in rows])
        maj = np.bincount(y[train], minlength=10).argmax()
        for i in np.flatnonzero(test):
            r = rows[i]
            words = [w.strip('.,;:?') for w in r['prompt'].split()]
            last = [w for w in words if w in COLORS][-1]
            res['majority'].append((i, maj))
            res['last_color_word'].append((i, CLASSES.index('en_' + last)))
        for name, ngram in (('unigram_logreg', (1, 1)), ('bigram_logreg', (1, 2))):
            vec = CountVectorizer(token_pattern=r'\S+', ngram_range=ngram, binary=False)
            X = vec.fit_transform([docs[i] for i in np.flatnonzero(train)])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X, y[train])
            pred = clf.predict(vec.transform([docs[i] for i in np.flatnonzero(test)]))
            res[name] += list(zip(np.flatnonzero(test), pred))
        log('baselines', k + 1, ENTITY_SPLIT['discovery'], start)
    table = {}
    for name, pairs in res.items():
        idx, pred = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        src = np.array([rows[i]['source'] for i in idx])
        table[name] = {'n': len(idx), 'accuracy_10way': float((pred == y[idx]).mean()),
                       'document': float((pred == y[idx])[src == 1].mean()), 'parametric': float((pred == y[idx])[src == -1].mean()),
                       'language_only': float(((pred // 5) == (y[idx] // 5)).mean()),
                       'color_only': float(((pred % 5) == (y[idx] % 5)).mean())}
    save('text-baselines.json', {'schema': 'common_bench_text_baselines.v1', 'run_id': RUN, 'scope': 'discovery only',
                                 'folds': 'leave one entity per colour out; test families use held-out entities only',
                                 'kind': 'text-only (tokens of the prompt); no labels of S, q, L, no model state',
                                 'chance_10way': 0.1, 'results': table})
    print(json.dumps(table, indent=1), flush=True)


def freeze():
    from tools import provenance
    from tools.power_analysis import check_phase, n_required
    d = checked()
    assert not (OUT / 'plan.json').exists()
    pilot_result, base = read(OUT / 'pilot-behavior.json'), read(OUT / 'text-baselines.json')
    assert pilot_result['draft_sha256'] == sha(OUT / 'draft.json')
    n_items = {k: v['rows'] for k, v in d['materials'].items()}
    power = {'unit': 'paired binary outcome per item (McNemar), p_d = 0.10, alpha = 0.05',
             'expected_delta': 0.03, 'n_required': n_required(0.03), 'confirm_items': n_items['confirm'],
             'check': check_phase(0.03, n_items['confirm']),
             'per_cell_check': check_phase(0.03, n_items['confirm'] // 4),
             'clustering_caveat': 'confirm has 25 parametric entities and 500 families; items are not independent, '
                                  'so entity-clustered intervals must accompany any item-level test'}
    per_input = 37 * 2560 * 2 * 3
    plan = {'schema': 'common_bench_plan.v1', 'kind': 'real', 'run_id': RUN, 'seed': SEED,
            'created_at': datetime.now().astimezone().isoformat(), 'model_snapshot': d['model_snapshot'],
            'model_path': str(MODEL), 'code_sha256': sha(__file__), 'draft_sha256': sha(OUT / 'draft.json'),
            'materials': d['materials'], 'choice_ids': d['choice_ids'], 'classes': CLASSES, 'prereg': d['prereg'], 'amendment': d['amendment'], 'answer_prefix': d['answer_prefix'],
            'pilot': {'sha256': sha(OUT / 'pilot-behavior.json'), 'passed': pilot_result['passed'],
                      'overall': pilot_result['overall'], 'before_entity_screen': pilot_result['before_entity_screen'], 'dropped_discovery_entities': pilot_result['dropped_discovery_entities']},
            'text_baselines': {'sha256': sha(OUT / 'text-baselines.json'), 'results': base['results']},
            'power': power,
            'storage_estimate': {'residual_bytes_three_positions_all_layers': per_input * sum(n_items.values()),
                                 'note': 'BF16, 37 states x 3 readout positions x 2560; attention-side residual doubles it'},
            'design': {'factors': {'source': 'document (+1) / parametric (-1)', 'language': 'English (+1) / Chinese (-1)',
                                   'query': 'first (+1) / second (-1) slot'},
                       'colors': COLORS, 'readouts': ['doc_end', 'question_end', 'answer_prefix'],
                       'visibility': 'doc_end prefix is identical within a family; the final token is identical in all prompts',
                       'legal_combinations': 'all source x language x query cells; the four colours of a family are distinct',
                       'split': 'by parametric entity (4 / 5 / 3 per colour); lockbox_frame uses unseen sentences only',
                       'input_language': 'all prompts are written in English; only the requested output language varies'},
            'limits': ['one model, one physical node', 'five colours, 60 parametric entities written by the author',
                       'rule-derived attributes are not covered', 'style and syntax factors are not covered',
                       'first-token forced choice; free generation checked on a small discovery sample only'],
            'formal_evidence_promotion': 0, 'provenance': provenance.provenance(model_path=MODEL, seed=SEED, tool=RUN)}
    plan['provenance']['model_revision'] = d['model_snapshot']
    save('plan.json', plan)
    print(json.dumps({'frozen': True, 'plan_sha256': sha(OUT / 'plan.json'), 'power': power['check']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['draft', 'pilot', 'baselines', 'freeze'])
    globals()[parser.parse_args().stage]()
