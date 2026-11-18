"""Retest frozen items with a physically adjacent control and single-edge restoration."""
import argparse
from pathlib import Path

import source_binding_qkv_low_margin as exp
import source_binding_qkv_low_margin_analysis as analysis
import source_binding_transfer_factors as io
from source_binding_design import sha


def run(panel):
    from source_binding_qkv_suffix_token_engine import run as engine

    root=exp.OUT.parent/panel
    io.OUT,exp.OUT=root,root
    # A replay saves only candidate scores for six arms and at most 640 rows
    # per shard. Reserve space for the final files and one atomic write.
    io.BUDGET=32<<20
    io.setup()
    exp.CHUNK=io.read('design.json')['chunk']
    cohort=io.read('selected.json')
    test=analysis.read_run('test')
    chunks=exp.batches(io.read('materials.json'))
    ids={x['id'] for x in cohort['rows']}
    indices=sorted({test[x][2] for x in ids})

    def make_arms(rows):
        arms=exp.make_arms(rows)
        key=arms['mask_key']
        for r in rows:
            region=r['suffix']
            i=region.index(r['key'])
            near=region[i+1 if i+1<len(region) else i-1]
            assert abs(near-r['key'])==1
            r['groups']['near']=[near]
            r['groups']['except_near']=[p for p in region if p!=near]
        def arm(actions):
            return dict(donor=key['donor'],heads=key['heads'],actions=actions)
        return {'repeat_mask_key':key,'restore_mask_key':arm([]),'mask_near':arm([('mask','near')]),
                'reopen_near':arm([('mask','except_near')]),'restore_near':arm([('zero','full'),('own_v','near')])}

    entries={}
    details=[]
    for start in range(0,len(indices),10):
        group=indices[start:start+10]
        batch=[chunks[i] for i in group]
        file=f'single-edge-replay-{start//10:02d}.json'
        if not (root/file).exists():
            saved,meta=engine(batch,make_arms,file,root,lambda:io.check_write(root,io.BUDGET,operation=file))
            io.save(file,dict(indices=group,items=saved,meta=meta))
        payload=io.read(file)
        assert payload['indices']==group
        entries[file]=sha(root/file)
        for results in payload['items']:
            for i,b in enumerate(results['base']):
                if b['id'] not in ids:
                    continue
                r,old,_=test[b['id']]
                assert b==old['base']
                assert results['repeat_mask_key'][i]==old['mask_key']
                restore=results['restore_mask_key'][i]
                offset=r['suffix'].index(r['key'])
                near_offset=offset+1 if offset+1<len(r['suffix']) else offset-1
                if near_offset==offset+1:
                    assert all(results[a][i]==old[c] for a,c in (('mask_near','mask_adjacent'),('reopen_near','reopen_adjacent'),('restore_near','restore_adjacent')))
                details.append(dict(id=r['id'],family=r['family'],frame=r['frame'],target_record=r['target_record'],values=r['values'],
                                    key=r['parts'][offset],near=r['parts'][near_offset],original=b,restore=restore,
                                    outputs={**{k:results[k][i] for k in ('repeat_mask_key','mask_near','reopen_near','restore_near')},
                                             **{k:old[k] for k in ('mask_all','reopen_key','zero_all','restore_key','self_all','noop','self')}}))
    assert {r['id'] for r in details}==ids
    flips=[x for x in details if not x['outputs']['repeat_mask_key']['forced_correct']]
    io.save('single-edge-replay.json',dict(ids=sorted(ids),passed=True,complete=len(details),
                 flips=len(flips),recovered=sum(x['restore']['forced_correct'] for x in flips),records=details,files=entries,
                 manifest_sha256=sha(root/'test-manifest.json'),script_sha256=sha(__file__),
                 method='All frozen items, same original batch shape. A physical neighbor on the right when available, otherwise immediately on the left. Restore only the suppressed key edge group at all five blocks.'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('panel',choices=('low-margin','low-margin-expanded','low-margin-conflict'))
    run(ap.parse_args().panel)
