"""Freeze and run Q/K/V cue transfer and post-window replay in Qwen3-14B."""
from collections import defaultdict
from datetime import datetime
import argparse
import itertools
from pathlib import Path

import source_binding_transfer_factors as io
import source_binding_transfer_cue as old

OUT = io.ROOT / 'tests/glm5/result/source_binding_qkv_followup_v1'
RUN = 'SOURCE-BINDING-QKV-FOLLOWUP-20261113-v1'
SEED = 202611130429
TASKS = ('colours', 'zh')
SITES = ('qsrc', 'qent', 'names')
LAYERS = (23, 24, 25)
MAIN_FAMILIES = 24
PILOT_FAMILIES = 2
PANEL_ROWS = 64
TOL = 1.0


def make_materials():
    import numpy as np
    import source_binding_design as d
    import source_binding_relay as relay
    import source_binding_readheads as rh
    from shared.dist_protocol.natural_register_runner import snapshot

    tok = relay.tokenizer()
    old_cue = d.read(old.OUT/'materials.json')
    cue_families = {r['original_family'] for r in old_cue}
    reports = d.read(old.HISTORICAL/'report.json')
    files = {}
    for task in TASKS:
        path = old.HISTORICAL/f'{task}-holdout-materials.json'
        rows = d.read(path)
        families = sorted({r['family'] for r in rows} - cue_families)
        rng = np.random.default_rng(SEED + TASKS.index(task))
        sequence = list(rng.permutation(families))
        original_items = old.HISTORICAL/f'{task}-items.json'
        assert d.sha(original_items) == reports['items_sha256'][task]
        donors, relations = rh.donors(rows,f'SOURCE-BINDING-TRANSFER-20261113-v1|{task}|holdout',202611130426)
        by_family = defaultdict(list)
        for i,r in enumerate(rows):
            by_family[r['family']].append(i)
        for split, chosen in (('pilot',sequence[:PILOT_FAMILIES]),('holdout',sequence[PILOT_FAMILIES:PILOT_FAMILIES+MAIN_FAMILIES])):
            assert len(chosen)%2 == 0
            material = []
            for offset in range(0,len(chosen),2):
                for cue in (0,1):
                    for family in chosen[offset:offset+2]:
                        for j,i in enumerate(by_family[family]):
                            r = rows[i]
                            suffix = old.WORDS[cue]+old.PUNCTUATION[cue]
                            natural_cue = old.WORDS[TASKS.index(task)] + old.PUNCTUATION[TASKS.index(task)]
                            assert r['prompt'].endswith(natural_cue)
                            prefix = r['prompt'][:-len(natural_cue)]
                            prompt = prefix+suffix
                            ids = tok.encode(prompt,add_special_tokens=False)
                            assert len(ids)==len(r['tokens']) and ids[:-2]==r['tokens'][:-2]
                            assert all(tok.encode(prompt+v,add_special_tokens=False)==ids+[c] for v,c in zip(r['values'],r['choice_ids']))
                            rep = (j+offset//2)%4
                            material.append({**r,'original_id':r['id'],'id':f'{r["id"]}-c{cue}', 'cue':cue,
                                             'task':task,'prompt':prompt,'tokens':ids,'original_index':i,'donor':rows[donors[rep][i]]['id'],
                                             'rep':rep,'relation':relations[rep][i]})
            file = f'{split}-{task}-materials.json'
            io.save(file,material)
            files[file] = d.sha(OUT/file)
            print(task,split,len(material),flush=True)
    identity = snapshot(d.MODELS['qwen14'])
    assert identity == d.read(d.OUT/'qwen14-identity.json')['model_snapshot']
    scripts = ('source_binding_qkv_followup.py','source_binding_qkv_followup_engine.py',
               'source_binding_transfer_factor_engine.py','source_binding_runtime.py','source_binding_readheads.py')
    io.save('design.json',dict(run=RUN,time=datetime.now().astimezone().isoformat(),snapshot=identity,seed=SEED,files=files,
        code={name:d.sha(Path(__file__).with_name(name)) for name in scripts},heads=rh.H16,
        materials='Historically exposed Phase 426 families, excluding Phase 2921 cue panel. Two per task pilot, 24 disjoint per task holdout.',
        edits='Three sites and three starts. One original donor draw per recipient balanced by layout index and panel; same donor under both cues.',
        variants={'K':'donor pre-RoPE K only at named sites, natural recipient Q/V', 'V':'donor V only, natural recipient Q/K',
                  'H':'donor K/V as Phase 424','Q':'cross-cue Q only, from opposite-cue no-name-edit trajectory',
                  'QH':'H16 + Q from opposite-cue H trajectory of same donor and edit; compare H vs QH',
                  'D':'inject opposite cue H-minus-base block-28 input residual difference into natural receiver; same donor/sites',
                  'Dself':'same-cue delta identity replay','ALL':'all heads at same sites and window',
                  'Hpos':'H with edited-site K rotary distance normalized to 22 at each narrow block; same incoming tokens and candidate',
                  'Hback':'H plus H16 reads donor K/V at question suffix holders excluding names; Answer cue only'},
        outcomes='Original four-choice behavior >=.95 each task/cue. Positive control H, no edit, Dself. Report same-wide-target paired cue contrasts, candidate margins, name mass, crossed B28 transfer and English/Chinese gap by site/layer/donor target.',
        primary='Effect on fixed wide target margin on samples both cues change to same choice. Separate Q-only direct effect; Q modification cannot change upstream K/V. Bootstrap prompt families. B28 transfer includes later context and interactions.' ,
        limits='K/V from opposite cue at earlier positions must be bitwise equal. Hpos changes relative rotary position of edited K only, not literal prompt order. Finite coupled edits do not identify mediation fractions.'))


def run(split):
    import torch
    import transformers
    import source_binding_design as d
    import source_binding_qkv_followup_engine as engine

    frozen = io.read('design.json')
    for f, digest in frozen['code'].items():
        assert d.sha(Path(__file__).with_name(f))==digest,f
    if split=='holdout':
        assert io.read('pilot-manifest.json')['passed']
    print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(),flush=True)
    shards = []
    for task in TASKS:
        file = f'{split}-{task}-materials.json'
        assert d.sha(OUT/file)==frozen['files'][file]
        rows = io.read(file)
        chunks = [rows[i:i+PANEL_ROWS] for i in range(0,len(rows),PANEL_ROWS)]
        for offset in range(0,len(chunks),2):
            shard = f'{split}-{task}-{offset//2:02d}.json'
            if not (OUT/shard).exists():
                payload = engine.run(chunks[offset:offset+2], OUT, f'{split}-{task}-{offset+1}/{len(chunks)}', frozen['snapshot'])
                io.save(shard,payload)
            payload = io.read(shard)
            shards.append(dict(file=shard,sha256=d.sha(OUT/shard),controls=payload['controls']))
            print(shard, payload['controls'],flush=True)
            if not payload['controls']['passed']:
                io.save(f'{split}-manifest.json',dict(passed=False,shards=shards))
                raise SystemExit(2)
    io.save(f'{split}-manifest.json',dict(passed=True,shards=shards,design_sha256=d.sha(OUT/'design.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('design','pilot','holdout'))
    args = parser.parse_args()
    io.OUT = OUT
    io.setup()
    if args.action=='design':
        assert not (OUT/'design.json').exists()
        make_materials()
    else:
        run(args.action)


if __name__=='__main__':
    main()
