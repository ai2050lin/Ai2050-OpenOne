"""Batched execution of the already frozen factor design, with serial replay validation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import source_binding_transfer_factors as experiment


def fixed_runtime():
    from source_binding_runtime import Runtime

    class FixedPositionRuntime(Runtime):
        def context(self, rows):
            ids, pos, _, mask = super().context(rows)
            logical = self.t.tensor([[0]*(ids.shape[1]-len(r['positions']))+r['positions'] for r in rows], device='cuda')
            rotary = self.Rotary(self.cfg, device='cuda')
            pe = rotary(self.t.zeros((*ids.shape,1), dtype=self.t.bfloat16, device='cuda'), logical)
            return ids, logical, pe, mask

    return FixedPositionRuntime


def differences(expected, actual):
    assert len(expected) == len(actual)
    assert all(a['id']==b['id'] for a,b in zip(expected,actual))
    return dict(max_score_difference=max(abs(x-y) for a,b in zip(expected,actual) for x,y in zip(a['scores'],b['scores'])),
                choices_equal=all(a['pred_record']==b['pred_record'] for a,b in zip(expected,actual)))


def good(comparison):
    return comparison['choices_equal'] and comparison['max_score_difference'] <= experiment.NUMERICAL_TOLERANCE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('equivalence', 'gate', 'edits'))
    ap.add_argument('--group', type=int, default=8)
    ap.add_argument('--families', type=int, choices=(24,64), default=64)
    args = ap.parse_args()
    assert 1 <= args.group <= 12
    experiment.setup()
    import torch
    import transformers
    import source_binding_transfer_factor_engine as fast
    import source_binding_design as d
    from source_binding_resources import resources

    root = experiment.OUT
    frozen = experiment.read('design.json')
    for name, digest in frozen['code'].items():
        assert d.sha(Path(__file__).parent/name) == digest, name
    runtime = fixed_runtime()
    implementation = {p.name:d.sha(p) for p in (Path(__file__),Path(fast.__file__))}
    print(json.dumps(dict(mode=args.mode, ram=resources(), gpu=torch.cuda.get_device_name(),
                          torch=torch.__version__, transformers=transformers.__version__, group=args.group)),flush=True)

    def disk():
        experiment.check_write(root, experiment.BUDGET, operation=experiment.RUN)

    if args.mode == 'equivalence':
        rows = experiment.read('pilot-materials.json')
        sample_path = root/'pilot-edits-000.json'
        sample = {'file':sample_path.name,'sha256':d.sha(sample_path)}
        result, meta = fast.run([rows[:128]],runtime,experiment.plan_arms,'equivalence',root,disk)
        old = experiment.read(sample['file'])
        assert old['meta']['controls_ok'] and old['meta']['gate_replay_choices_equal']
        assert old['ids'] == [r['id'] for r in rows[:128]]
        assert set(old['items']) == set(result[0])
        delta = {k: differences([dict(id=s,pred_record=v[0],scores=v[1:]) for s,v in zip(old['ids'],old['items'][k])],new)
                 for k,new in result[0].items()}
        passed = all(good(x) for x in delta.values())
        experiment.save('engine-equivalence.json', dict(passed=passed, reference=sample, code=implementation,
                                                       design_sha256=d.sha(root/'design.json'), meta=meta, comparisons=delta))
        print('equivalence',passed,max(v['max_score_difference'] for v in delta.values()),flush=True)
        if not passed:
            raise SystemExit(2)
        return

    verified = experiment.read('engine-equivalence.json')
    assert verified['passed'] and verified['code']==implementation and verified['design_sha256']==d.sha(root/'design.json')
    assert d.sha(root/'holdout-materials.json') == frozen['materials']['holdout']
    rows = experiment.read('holdout-materials.json')[:args.families*experiment.CHUNK]
    chunks = [rows[i:i+experiment.CHUNK] for i in range(0,len(rows),experiment.CHUNK)]
    if args.mode == 'gate':
        n = len(chunks)
        result, metas = [], []
        for start in range(0,n,args.group):
            group = chunks[start:start+args.group]
            got, meta = fast.run(group,runtime,None,f'holdout-gate-{start+1}-{start+len(group)}/{n}',root,disk)
            result.extend(x for arm in got for x in arm['base'])
            metas.append(meta)
        cells = defaultdict(list)
        for row,outcome in zip(rows,result,strict=True):
            assert row['id']==outcome['id']
            cells[row['condition']].append(outcome)
        summary = {k:dict(n=len(v),forced=sum(x['forced_correct'] for x in v)/len(v),top1=sum(x['top1_correct'] for x in v)/len(v))
                   for k,v in cells.items()}
        experiment.save('holdout-gate-items.json',result)
        passed = all(v['forced']>=experiment.BEHAVIOR for v in summary.values())
        experiment.save('holdout-gate.json',dict(passed=passed,summary=summary,meta=metas,code=implementation,families=args.families,
                                               items_sha256=d.sha(root/'holdout-gate-items.json')))
        print(summary,flush=True)
        if not passed:
            raise SystemExit(2)
        return

    gate_manifest = experiment.read('holdout-gate.json')
    assert gate_manifest['passed'] and gate_manifest['code']==implementation
    assert gate_manifest['families']==args.families
    assert d.sha(root/'holdout-gate-items.json')==gate_manifest['items_sha256']
    gate = experiment.read('holdout-gate-items.json')
    shards = []
    started = time.monotonic()
    for start in range(0,len(chunks),args.group):
        indices = list(range(start,min(start+args.group,len(chunks))))
        files = {i:f'holdout-edits-{i:03d}.json' for i in indices}
        pending = [i for i in indices if not (root/files[i]).exists()]
        if pending:
            got, meta = fast.run([chunks[i] for i in pending],runtime,experiment.plan_arms,
                                 f'holdout-edits-{pending[0]+1}-{pending[-1]+1}/{len(chunks)}',root,disk)
            for i,outcomes in zip(pending,got,strict=True):
                chunk = chunks[i]
                ctl = {name:differences(outcomes['base'],outcomes[name]) for name in ('noop','self')}
                ref = differences(gate[i*experiment.CHUNK:i*experiment.CHUNK+len(chunk)],outcomes['base'])
                ctl['behavior_replay'] = ref
                arms = experiment.plan_arms(chunk)
                donors = {k.split('|',1)[1]:v['donor'] for k,v in arms.items() if k.startswith('wide|')}
                items = {name:[[r['pred_record'],*r['scores']] for r in samples] for name,samples in outcomes.items()}
                experiment.save(files[i],dict(ids=[r['id'] for r in chunk],donors=donors,items=items,
                                             meta=dict(**meta,controls=ctl,passed=all(good(x) for x in ctl.values())),code=implementation))
        for i in indices:
            saved = experiment.read(files[i])
            assert saved['code']==implementation and saved['ids']==[r['id'] for r in chunks[i]]
            assert saved['meta']['passed'], (i,saved['meta'])
            shards.append(dict(file=files[i],sha256=d.sha(root/files[i]),meta=saved['meta']))
        print(json.dumps(dict(stage='holdout-edits',done=len(shards),total=len(chunks),seconds=time.monotonic()-started)),flush=True)
    experiment.save('holdout-edits.json',dict(passed=True,shards=shards,code=implementation,families=args.families))


if __name__ == '__main__':
    main()
