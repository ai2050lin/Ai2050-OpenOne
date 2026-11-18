"""Confirm whether answer-cue wording or its final punctuation changes H16 transfer.

All comparisons are within one original Phase 426 prompt. Candidate IDs, names,
positions, donor relations and all other tokens stay identical.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import itertools
import json
from pathlib import Path

import source_binding_transfer_factors as common

ROOT = common.ROOT
OUT = ROOT / 'tests/glm5/result/source_binding_transfer_cue_v1'
HISTORICAL = ROOT / 'tests/glm5/result/source_binding_transfer_v1'
TASKS = ('colours', 'zh')
CELLS = tuple(itertools.product((0, 1), repeat=2))
WORDS = ('Answer', '答案')
PUNCTUATION = (':\n', '：\n')
WINDOW = (23, 24, 25)
SITES = ('qsrc', 'qent', 'names')
REPS = (0, 1, 2, 3)
FAMILIES = 16
GROUP = 3
CHUNK = 128


def make_materials():
    import numpy as np
    import source_binding_design as d
    import source_binding_relay as relay
    import source_binding_readheads as rh

    tok = relay.tokenizer()
    rng = np.random.default_rng(common.SEED+1)
    old_report = d.read(HISTORICAL/'report.json')
    records = []
    for task in TASKS:
        source = HISTORICAL/f'{task}-holdout-materials.json'
        report = HISTORICAL/f'{task}-items.json'
        assert d.sha(report) == old_report['items_sha256'][task]
        rows = d.read(source)
        donors, _ = rh.donors(rows, f'SOURCE-BINDING-TRANSFER-20261113-v1|{task}|holdout', 202611130426)
        families = list(dict.fromkeys(r['family'] for r in rows))
        selected = sorted(rng.choice(families,FAMILIES,replace=False))
        for w,p in CELLS:
            cue = WORDS[w]+PUNCTUATION[p]
            original = int(task=='zh')
            for idx,r in enumerate(rows):
                if r['family'] not in selected:
                    continue
                old_cue = WORDS[original]+PUNCTUATION[original]
                assert r['prompt'].endswith(old_cue)
                prompt = r['prompt'][:-len(old_cue)]+cue
                ids = tok.encode(prompt,add_special_tokens=False)
                assert ids[:-2] == r['tokens'][:-2]
                assert all(tok.encode(prompt+v,add_special_tokens=False)==ids+[choice] for v,choice in zip(r['values'],r['choice_ids']))
                family = f'{r["family"]}-cue{w}{p}'
                records.append({**r,'id':r['id'].replace(r['family'],family),'family':family,
                                'original_id':r['id'],'original_family':r['family'],'original_index':idx,
                                'cue':f'{w}{p}','task':task,'prompt':prompt,'tokens':ids,'positions':list(range(len(ids))),
                                'original_donors':[rows[donors[k][idx]]['id'] for k in REPS]})
    common.save('materials.json',records)
    frozen = dict(run='SOURCE-BINDING-TRANSFER-CUE-20261113-v1',time=datetime.now().astimezone().isoformat(),
                  selection='16 preselected families per task; materials from opened Phase 426. Cue counterfactuals previously unevaluated.',
                  factors=dict(word=WORDS,punctuation=PUNCTUATION), heads=rh.H16, layers=WINDOW, sites=SITES,
                  arms='wide, H16, unchanged, self. Same original donor relationships, all 4 draws.',
                  behavior='forced choice >=95% in every cell, unedited and self argmax equal and candidate score max diff <=1.',
                  audit='Original cue arm compare to Phase 426 choices and numerical margins. Primary effect paired family contrasts.',
                  materials_sha256=d.sha(OUT/'materials.json'),code=d.sha(Path(__file__)),
                  engine_code={name:d.sha(Path(__file__).with_name(name)) for name in ('source_binding_transfer_factor_engine.py',
                                                                                       'source_binding_transfer_factor_campaign.py')},
                  source_report=d.sha(HISTORICAL/'report.json'),source_items=old_report['items_sha256'])
    common.save('design.json',frozen)
    print(len(records),'cue rows',flush=True)


def arms(rows):
    import source_binding_readheads as rh

    index = {(r['cue'],r['original_id']):i for i,r in enumerate(rows)}
    result = {'self':dict(donor=list(range(len(rows))),don=rh.spec_heads('names',23,rh.H16))}
    for rep,site,layer in itertools.product(REPS,SITES,WINDOW):
        donors = [index[r['cue'],r['original_donors'][rep]] for r in rows]
        suffix = f'{site}@{layer}#{rep}'
        result[f'swap|{suffix}'] = dict(swap=[site,layer],donor=donors)
        result[f'H16|{suffix}'] = dict(don=rh.spec_heads(site,layer,rh.H16),donor=donors)
    return result


def forward(mode):
    import source_binding_design as d
    import source_binding_transfer_factor_engine as engine
    from source_binding_transfer_factor_campaign import fixed_runtime, differences, good

    frozen = common.read('design.json')
    assert frozen['code'] == d.sha(Path(__file__))
    assert frozen['materials_sha256'] == d.sha(OUT/'materials.json')
    for name, digest in frozen['engine_code'].items():
        assert d.sha(Path(__file__).with_name(name)) == digest, name
    rows = common.read('materials.json')
    chunks = [rows[i:i+CHUNK] for i in range(0,len(rows),CHUNK)]
    manifest = []
    key = 'gate' if mode=='gate' else 'items'
    if mode=='run':
        assert common.read('gate.json')['passed']
    for start in range(0,len(chunks),GROUP):
        group = chunks[start:start+GROUP]
        name = f'{key}-{start//GROUP:03d}.json'
        if not (OUT/name).exists():
            got, meta = engine.run(group,fixed_runtime(),None if mode=='gate' else arms,
                                   f'cue-{mode}-{start+1}-{start+len(group)}/{len(chunks)}',OUT,
                                   lambda: common.check_write(OUT,common.BUDGET,operation='cue-factor'))
            for rs,result in zip(group,got,strict=True):
                assert [r['id'] for r in rs]==[r['id'] for r in result['base']]
                if mode=='run':
                    assert all(good(differences(result['base'],result[k])) for k in ('noop','self'))
            common.save(name,dict(meta=meta,items=got))
        manifest.append(dict(name=name,sha256=d.sha(OUT/name)))
    result = dict(shards=manifest,passed=True)
    if mode=='gate':
        groups = defaultdict(list)
        index = {r['id']:r for r in rows}
        for record in manifest:
            for block in common.read(record['name'])['items']:
                for y in block['base']:
                    r = index[y['id']]
                    groups[r['task'],r['cue']].append(y)
        result['behavior'] = {f'{t}-{c}':dict(n=len(s),forced=sum(x['forced_correct'] for x in s)/len(s),top1=sum(x['top1_correct'] for x in s)/len(s)) for (t,c),s in groups.items()}
        result['passed'] = all(x['forced']>=common.BEHAVIOR for x in result['behavior'].values())
    common.save(f'{key}.json',result)
    if not result['passed']:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('design','gate','run'))
    args = ap.parse_args()
    common.OUT = OUT
    common.setup()
    if args.mode=='design':
        assert not (OUT/'design.json').exists()
        make_materials()
    else:
        forward(args.mode)


if __name__=='__main__':
    main()
