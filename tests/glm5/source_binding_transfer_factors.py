"""Paired Qwen3-14B source-binding transfer experiment. See the frozen design JSON."""
import argparse
from collections import defaultdict
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write, checked_dir

OUT = ROOT / 'tests/glm5/result/source_binding_transfer_factors_v1'
RUN = 'SOURCE-BINDING-TRANSFER-FACTORS-20261113-v1'
SEED = 202611130428
BUDGET = 2**30
CHUNK = 128
FACTORS = ('language', 'name_tokens', 'name_distance', 'answer_cue')
CONDITIONS = tuple(itertools.product((0, 1), repeat=4))
RELATIONS = ((1, 0), (0, 1), (1, 1), (0, 0))
LAYERS = (23, 24, 25)
SITES = ('qsrc', 'qent', 'names')
POOLS = {
    'pilot': (['Grace', 'Julia', 'Harry', 'Robin'], ['camera', 'flower', 'shell', 'forest']),
    'holdout': (['Theo', 'Nora', 'Felix', 'Aaron', 'Helen', 'Olive', 'Gavin', 'Frank'],
                ['garden', 'violin', 'rabbit', 'puzzle', 'stone', 'river', 'lake', 'clock']),
}
FAMILIES = {'pilot': 4, 'holdout': 64}
VALUES = ('red', 'blue', 'green', 'black', 'white', 'pink', 'gold', 'grey', 'brown', 'purple', 'orange', 'yellow')
FRAMES = (
    dict(intro='Each list gives a value for each item. Use the specified source and item.\n',
         question='Given source ( {src} ) and item ( {ent} ), what is the exact value?\n',
         aside='All four records are in the two lists above.\n', tail='Reply with just the recorded value.'),
    dict(intro='每个来源都列出这两件物品的值。请找出指定的值。\n',
         question='请找出 ( {src} ) 为 ( {ent} ) 写下的对应值。\n',
         aside='两份名单已包含全部四条记录。\n', tail='请只输出记录里的值。'),
)
CUES = ('Answer:\n', '答案：\n')
BEHAVIOR = 0.95
NUMERICAL_TOLERANCE = 1.0


def tag(levels):
    return ''.join(map(str, levels))


def setup():
    checked_dir(OUT, BUDGET, operation=RUN)
    tmp = checked_dir(OUT / 'tmp', operation=RUN)
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[key] = str(tmp)
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_MODULES_CACHE', 'HF_DATASETS_CACHE',
                'TORCH_HOME', 'XDG_CACHE_HOME', 'TRANSFORMERS_CACHE'):
        os.environ[key] = str(OUT / 'cache' / key)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def save(name, obj):
    path = check_write(OUT / name, BUDGET, operation=RUN)
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('w', encoding='utf-8') as stream:
        json.dump(obj, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    partial.replace(path)


def read(name):
    return json.loads((OUT / name).read_bytes())


def two_tokens(tok, vocabulary, original):
    piece = tok.convert_ids_to_tokens(original)
    surface = tok.decode([original], clean_up_tokenization_spaces=False)
    for cut in sorted(range(2, len(piece) - 1), key=lambda cut: abs(cut - len(piece) / 2)):
        parts = piece[:cut], piece[cut:]
        if all(p in vocabulary for p in parts):
            ids = [vocabulary[p] for p in parts]
            if tok.decode(ids, clean_up_tokenization_spaces=False) == surface:
                return ids
    raise ValueError(f'No spelling-preserving split for {surface!r}')


def row_for(tok, vocabulary, split, family, setting, sources, items, values, s, e, order):
    language, double, far, cue = setting
    frame = FRAMES[language]
    order_bit = order ^ (family % 2)
    table = []
    for i in (order, 1 - order):
        table.append(f'[{sources[i]}]\n')
        first = i ^ order_bit
        table.extend(f'{items[j]} = {values[2*i+j]}\n' for j in (first, 1-first))
    prefix = frame['intro'] + ''.join(table) + ('' if far else frame['aside'])
    question = frame['question'].format(src=sources[s], ent=items[e])
    content = prefix + question + (frame['aside'] if far else '') + frame['tail']
    prompt = tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False) + CUES[cue]
    encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    original = encoded['input_ids']
    q0 = prompt.index(content) + len(prefix)
    sites = {}
    for name, word in (('qsrc', sources[s]), ('qent', items[e])):
        lo = q0 + question.index(word)
        hit = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > lo and a < lo + len(word)]
        assert len(hit) == 1 and tok.decode([original[hit[0]]]).strip() == word, (word, hit)
        sites[name] = hit[0]
    choice = []
    for value in values:
        extension = tok.encode(prompt + value, add_special_tokens=False)
        assert extension[:-1] == original
        choice.append(extension[-1])
    assert len(set(choice)) == 4
    site_at = {pos: key for key, pos in sites.items()}
    ids, positions, up = [], [], {}
    for pos, token in enumerate(original):
        key = site_at.get(pos)
        piece = two_tokens(tok, vocabulary, token) if key and double else [token]
        if key:
            up[key] = list(range(len(ids), len(ids) + len(piece)))
        ids.extend(piece)
        positions.extend([pos] * len(piece))
    assert tok.decode(ids, clean_up_tokenization_spaces=False) == tok.decode(original, clean_up_tokenization_spaces=False)
    assert max(up['qsrc'] + up['qent']) < len(ids) - 1
    group = f'{split}-{family:03d}-{tag(setting)}'
    return dict(id=f'{group}-S{s}E{e}O{order}', family=group, pair_family=family, condition=tag(setting),
                source=s, entity=e, order=order, item_order_bit=order_bit, sources=sources, items=items, values=values,
                target_record=2*s+e, choice_ids=choice, gold_id=choice[2*s+e], prompt=prompt, tokens=ids, positions=positions,
                up=up, original_names=sites, name_distances={k: len(original)-1-v for k, v in sites.items()})


def make_materials(split):
    import numpy as np
    import source_binding_relay as relay

    tok = relay.tokenizer()
    vocabulary = tok.get_vocab()
    source_pool, item_pool = POOLS[split]
    designs = list(itertools.product(itertools.combinations(source_pool, 2), itertools.combinations(item_pool, 2)))
    rng = np.random.default_rng(SEED + (split == 'holdout'))
    rows = []
    for family, idx in enumerate(rng.permutation(len(designs))[:FAMILIES[split]]):
        sources, items = designs[idx]
        if rng.integers(2):
            sources = sources[::-1]
        if rng.integers(2):
            items = items[::-1]
        values = [str(v) for v in rng.choice(VALUES, 4, replace=False)]
        for setting in CONDITIONS:
            rows.extend(row_for(tok, vocabulary, split, family, setting, sources, items, values, s, e, order)
                        for s, e, order in itertools.product((0, 1), repeat=3))
    audit_materials(rows, tok)
    return rows


def audit_materials(rows, tok):
    groups = defaultdict(dict)
    for row in rows:
        groups[row['pair_family'], row['source'], row['entity'], row['order']][row['condition']] = row
    for variants in groups.values():
        assert len(variants) == 16
        for setting in CONDITIONS:
            a = variants[tag(setting)]
            for axis, bit in enumerate(setting):
                if bit:
                    continue
                other = list(setting)
                other[axis] = 1
                b = variants[tag(other)]
                assert all(a[k] == b[k] for k in ('choice_ids', 'sources', 'items', 'values', 'target_record'))
                if axis != 1:
                    assert len(a['tokens']) == len(b['tokens'])
                if axis != 2:
                    assert a['name_distances'] == b['name_distances']
                else:
                    assert {b['name_distances'][k] - a['name_distances'][k] for k in a['up']} == {10}
                if axis == 0:
                    assert a['up'] == b['up']
                    assert all([a['tokens'][p] for p in a['up'][k]] == [b['tokens'][p] for p in b['up'][k]] for k in a['up'])
                    assert a['tokens'][-2:] == b['tokens'][-2:]
                if axis == 1:
                    assert a['prompt'] == b['prompt'] and a['positions'] == sorted(set(b['positions']))
                    assert all(a['tokens'][i] == t for i, t in zip(b['positions'], b['tokens']) if i not in a['original_names'].values())
                if axis == 2:
                    assert sorted(a['tokens']) == sorted(b['tokens']) and a['tokens'][-2:] == b['tokens'][-2:]
                if axis == 3:
                    assert a['tokens'][:-2] == b['tokens'][:-2] and a['tokens'][-2:] != b['tokens'][-2:]


def design():
    import source_binding_design as d
    import source_binding_readheads as rh
    from shared.dist_protocol.natural_register_runner import snapshot

    assert not (OUT / 'design.json').exists(), 'Frozen design already exists'
    for split in FAMILIES:
        rows = make_materials(split)
        save(f'{split}-materials.json', rows)
        print(split, len(rows), 'rows', flush=True)
    identity = snapshot(d.MODELS['qwen14'])
    assert identity == d.read(d.OUT / 'qwen14-identity.json')['model_snapshot']
    dependencies = sorted(Path(__file__).parent.glob('source_binding_*.py'))
    frozen = dict(run=RUN, created=datetime.now().astimezone().isoformat(), seed=SEED, model='Qwen3-14B BF16',
                  model_snapshot=identity, factors=dict(zip(FACTORS, (['English', 'Chinese'], ['canonical one', 'spelling-preserving split, same RoPE position'],
                                                                     ['near', '10 tokens farther, move identical sentence'], list(CUES)))),
                  layout='Common neutral four-record table, identical choice tokens, shared Latin names and item labels. Two complementary record layouts per family.',
                  split='Pilot and holdout disjoint by source and item label; answer vocabulary shared.',
                  heads={k: rh.head_sets()[k] for k in ('H16', 'R0')}, layers=LAYERS, sites=SITES,
                  relations='Flip source only, item only, both at same layout; fourth control keeps query and swaps the layout.',
                  arms='wide, H16, all, R0 on first three donor relations. wide and H16 also on fourth. Self uses H16.',
                  outcome='Report total agreement and agreement conditional on wide choosing a different record. Paired inference uses both-changed cases and absolute gain over unchanged prediction (all cases). Also report direction-only common results and marginal single-axis edges.',
                  inference='Bootstrap family clusters across paired variants. 98.75% percentile intervals for four averaged factor contrasts. Compare effect size and direction on both metrics. Attribute old gap only to extent neutral bridge reproduces it. No graph promotion.',
                  behavior_min=BEHAVIOR, noop_max=NUMERICAL_TOLERANCE, chunk=CHUNK,
                  materials={s: d.sha(OUT / f'{s}-materials.json') for s in FAMILIES},
                  code={p.name: d.sha(p) for p in dependencies if p == Path(__file__) or p.name in (
                      'source_binding_sites.py', 'source_binding_runtime.py', 'source_binding_readheads.py', 'source_binding_relay.py', 'source_binding_design.py', 'source_binding_resources.py', 'source_binding_transfer_factors.py')})
    save('design.json', frozen)


def plan_arms(rows):
    import source_binding_readheads as rh

    index = {(r['family'], r['source'], r['entity'], r['order']): i for i, r in enumerate(rows)}
    arms = {'self': {'donor': list(range(len(rows))), 'don': rh.spec_heads('names', 23, rh.H16)}}
    random_heads = rh.head_sets()['R0']
    for rep, (ds, de) in enumerate(RELATIONS):
        donors = [index[r['family'], r['source'] ^ ds, r['entity'] ^ de, r['order'] ^ (rep == 3)] for r in rows]
        for site, layer in itertools.product(SITES, LAYERS):
            suffix = f'{site}@{layer}#{rep}'
            arms[f'wide|{suffix}'] = {'donor': donors, 'swap': [site, layer]}
            arms[f'H16|{suffix}'] = {'donor': donors, 'don': rh.spec_heads(site, layer, rh.H16)}
            if rep < 3:
                arms[f'all|{suffix}'] = {'donor': donors, 'don': rh.spec_all(site, layer)}
                arms[f'R0|{suffix}'] = {'donor': donors, 'don': rh.spec_heads(site, layer, random_heads)}
    return arms


def engine(rows, arms, stage):
    import source_binding_sites as sites
    from source_binding_runtime import Runtime

    class FixedPositionRuntime(Runtime):
        def context(self, rows):
            ids, pos, pe, mask = super().context(rows)
            for i, row in enumerate(rows):
                logical = row.get('positions')
                if logical is not None:
                    pos[i, -len(logical):] = self.t.tensor(logical, device='cuda')
            rotary = self.Rotary(self.cfg, device='cuda')
            pe = rotary(self.t.zeros((*ids.shape, 1), dtype=self.t.bfloat16, device='cuda'), pos)
            return ids, pos, pe, mask

    sites.OUT = OUT
    sites.Runtime = FixedPositionRuntime
    sites.CHUNK = CHUNK
    return sites.engine(rows, arms, stage)


def run(split, mode):
    import source_binding_design as d
    import torch
    import transformers
    from source_binding_resources import resources

    frozen = read('design.json')
    for name, digest in frozen['code'].items():
        assert d.sha(Path(__file__).parent / name) == digest, name
    assert d.sha(OUT / f'{split}-materials.json') == frozen['materials'][split]
    rows = read(f'{split}-materials.json')
    manifest = f'{split}-{mode}.json'
    if (OUT / manifest).exists():
        return
    print(json.dumps(dict(stage=manifest, rows=len(rows), preflight=resources(), gpu=torch.cuda.get_device_name(),
                          cuda_free_MiB=torch.cuda.mem_get_info()[0] // 2**20, torch=torch.__version__,
                          transformers=transformers.__version__)), flush=True)
    if mode == 'gate':
        results, meta = engine(rows, None, f'{split}-gate')
        cells = defaultdict(list)
        for row, outcome in zip(rows, results['base']):
            assert row['id'] == outcome['id']
            cells[row['condition']].append(outcome)
        summary = {k: dict(n=len(v), forced=sum(x['forced_correct'] for x in v)/len(v), top1=sum(x['top1_correct'] for x in v)/len(v))
                   for k, v in cells.items()}
        save(f'{split}-gate-items.json', results['base'])
        save(manifest, dict(meta=meta, summary=summary, passed=all(v['forced'] >= BEHAVIOR for v in summary.values())))
        print(summary, flush=True)
        return
    assert read(f'{split}-gate.json')['passed'], 'Behavior gate failed'
    if split == 'holdout':
        assert read('pilot-edits.json')['passed'], 'Pilot numerical controls failed'
    gate = read(f'{split}-gate-items.json')
    shards = []
    started = time.monotonic()
    for start in range(0, len(rows), CHUNK):
        name = f'{split}-edits-{start//CHUNK:03d}.json'
        if not (OUT / name).exists():
            check_write(OUT / name, BUDGET, operation=RUN)
            chunk = rows[start:start+CHUNK]
            arms = plan_arms(chunk)
            results, meta = engine(chunk, arms, f'{split}-edits-{start//CHUNK+1}/{(len(rows)+CHUNK-1)//CHUNK}')
            paired = list(zip(results['base'], gate[start:start+CHUNK]))
            assert all(x['id'] == y['id'] for x, y in paired)
            meta['gate_replay_max_difference'] = max(abs(x-y) for a, b in paired for x, y in zip(a['scores'], b['scores']))
            meta['gate_replay_choices_equal'] = all(a['pred_record'] == b['pred_record'] for a, b in paired)
            donor_map = {k.split('|', 1)[1]: v['donor'] for k, v in arms.items() if k.startswith('wide|')}
            compact = {k: [[x['pred_record'], *x['scores']] for x in v] for k, v in results.items()}
            save(name, dict(meta=meta, ids=[r['id'] for r in chunk], donors=donor_map, items=compact))
        else:
            meta = read(name)['meta']
        shards.append({'file': name, 'sha256': d.sha(OUT / name), 'meta': meta})
        passed = meta['controls_ok'] and meta['gate_replay_choices_equal'] and meta['gate_replay_max_difference'] <= NUMERICAL_TOLERANCE
        print(json.dumps(dict(stage=manifest, finished=min(start+CHUNK,len(rows)), total=len(rows),
                              seconds=round(time.monotonic()-started, 1), passed=passed)), flush=True)
        if not passed:
            break
    save(manifest, dict(passed=passed and len(shards)*CHUNK >= len(rows), shards=shards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('design', 'gate', 'edits'))
    parser.add_argument('--split', choices=tuple(FAMILIES), default='holdout')
    args = parser.parse_args()
    setup()
    if args.action == 'design':
        design()
    else:
        run(args.split, args.action)


if __name__ == '__main__':
    main()
