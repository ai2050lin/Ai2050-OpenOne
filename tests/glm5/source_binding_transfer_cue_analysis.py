"""Recompute paired word and punctuation effects, including historical replay checks."""
from collections import defaultdict
import itertools
import json

import numpy as np

import source_binding_transfer_cue as cue
import source_binding_transfer_factors as common
from source_binding_design import sha


def main():
    common.OUT = cue.OUT
    root = common.OUT
    design = common.read('design.json')
    assert design['materials_sha256'] == sha(root/'materials.json')
    assert design['source_report'] == sha(cue.HISTORICAL/'report.json')
    rows = {r['id']:r for r in common.read('materials.json')}
    historic = {}
    for task in cue.TASKS:
        path = cue.HISTORICAL/f'{task}-items.json'
        assert sha(path) == design['source_items'][task]
        historic[task] = json.loads(path.read_bytes())
    gate = common.read('gate.json')
    manifest = common.read('items.json')
    assert gate['passed'] and manifest['passed']
    gate_scores = {}
    for shard in gate['shards']:
        assert sha(root/shard['name']) == shard['sha256']
        for block in common.read(shard['name'])['items']:
            gate_scores.update({r['id']:r for r in block['base']})
    paired = defaultdict(dict)
    audit = defaultdict(lambda: defaultdict(lambda: [0,0,0.0]))
    for shard in manifest['shards']:
        assert sha(root/shard['name']) == shard['sha256']
        for block in common.read(shard['name'])['items']:
            for i,b in enumerate(block['base']):
                r = rows[b['id']]
                b0 = gate_scores[b['id']]
                assert b0['pred_record'] == b['pred_record'] and max(abs(x-y) for x,y in zip(b['scores'],b0['scores'])) <= 1
                group = paired[r['task'],r['original_family'],r['cue']]
                if r['cue'] == ('00' if r['task']=='colours' else '11'):
                    hist = historic[r['task']]
                    for key in ('base', *(k for k in block if k.startswith(('swap|','H16|')))):
                        x, y = hist[key][r['original_index']],block[key][i]
                        assert x['id']==r['original_id']
                        stat = audit[r['task']][key.split('|')[0]]
                        stat[0] += 1
                        stat[1] += x['pred_record'] != y['pred_record']
                        if 'scores' in x:
                            stat[2] = max(stat[2],max(abs(a-b) for a,b in zip(x['scores'],y['scores'])))
                for key,swaps in block.items():
                    if key.startswith('swap|'):
                        idx = r['original_id'], key
                        assert idx not in group
                        y = swaps[i]
                        h = block['H16|'+key[5:]][i]
                        assert y['id'] == h['id'] == r['id']
                        original, target = b['pred_record'], y['pred_record']
                        margin = lambda row: row['scores'][target]-row['scores'][original]
                        group[idx] = original, target, h['pred_record'], margin(b), margin(y), margin(h)
    report = {'design_sha256':sha(root/'design.json'),'analysis_sha256':sha(__file__),
              'items_manifest_sha256':sha(root/'items.json'),'historical_replay':audit,'tasks':{}}
    for task in cue.TASKS:
        families = sorted({family for t,family,_ in paired if t==task})
        assert len(families)==cue.FAMILIES
        tags = [f'{a}{b}' for a,b in cue.CELLS]
        data = []
        for family in families:
            keys = sorted(paired[task,family,tags[0]])
            variants = [paired[task,family,c] for c in tags]
            assert all(set(v)==set(keys) for v in variants)
            data.append([[v[k] for k in keys] for v in variants])
        data = np.asarray(data)
        assert data.shape == (cue.FAMILIES,4,16*len(cue.WINDOW)*len(cue.SITES)*len(cue.REPS),6)
        base, wide, narrow, bm, wm, hm = np.moveaxis(data,-1,0)
        changed, hit = base!=wide, wide==narrow
        gain = hit.astype(int) - ~changed
        stats = {code:dict(n=int(hit[:,i].size),changed=int(changed[:,i].sum()),hit_changed=int((changed[:,i]&hit[:,i]).sum()),
                           changed_agreement=float(hit[:,i][changed[:,i]].mean()),agreement=float(hit[:,i].mean()),
                           net_gain=float(gain[:,i].mean())) for i,code in enumerate(tags)}
        draws = np.random.default_rng(common.SEED).multinomial(len(families),np.full(len(families),1/len(families)),8000)

        def delta(left,right):
            same = changed[:,left] & changed[:,right] & (wide[:,left]==wide[:,right])
            both = changed[:,left] & changed[:,right]
            p = hit[:,right].astype(int)-hit[:,left]

            def pack(values, mask):
                d,n = (values*mask).sum(axis=1),mask.sum(axis=1)
                nb = draws@n
                b = (draws@d)[nb>0]/nb[nb>0]
                return dict(delta=float(d.sum()/n.sum()),n=int(n.sum()),ci95=np.quantile(b,[.025,.975]).tolist())

            common_choice = same & (base[:,left] == base[:,right])
            return dict(same_target=pack(p,same),both_changed=pack(p,both),gain=pack(gain[:,right]-gain[:,left],np.ones_like(same)),
                        margins_descriptive=dict(
                            base=pack(bm[:,right]-bm[:,left],common_choice),
                            wide_edit_uplift=pack((wm-bm)[:,right]-(wm-bm)[:,left],common_choice),
                            H16_edit_uplift=pack((hm-bm)[:,right]-(hm-bm)[:,left],common_choice)))

        edges = {f'{tags[i]}-{tags[j]}':delta(i,j) for i,j in itertools.combinations(range(4),2)}
        report['tasks'][task] = dict(conditions=stats,edges=edges)
        print(task,'old replay',dict(audit[task]),flush=True)
        for k,v in stats.items():
            print(k,v,flush=True)
        for k,v in edges.items():
            print(k,v,flush=True)
    common.save('report.json',report)


if __name__=='__main__':
    main()
