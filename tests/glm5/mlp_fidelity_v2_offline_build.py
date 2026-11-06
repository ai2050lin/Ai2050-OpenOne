"""Rebuild only the research extra after its ordinary worker skipped a stale marker."""
import sys
import time

import mlp_fidelity_v2_plan as P
sys.path.insert(0,str(P.ROOT/'deploy'))
from deploy import datapack as DP
from deploy.research_library import library


def main():
    data=P.ROOT/'distributed_data'
    target='research-library'
    path=DP.extra_file(data,target)
    version=library().fingerprint()
    existing=DP.read_extras(data).get(target,{})
    if existing.get('version')==version and path.is_file():
        print(f'{target} is already current: {version}',flush=True)
        return
    stage=P.OUT/'offline-stage'/path.name
    assert not stage.exists(),stage
    entries,notes=DP.research_entries()
    print(f'Building {len(entries)} entries, version {version}',flush=True)
    created=time.time()
    header={'schema':DP.EXTRA_SCHEMA,'id':target,'version':version,'created_at':created,'notes':notes}
    size,raw=DP._write_extra_gz(stage,header,entries)
    assert library().fingerprint()==version,'Library changed during the snapshot; staged package left intact'
    meta=DP.read_extras(data)
    assert meta.get(target,{})==existing,'Research pack changed during staging; staged package left intact'
    marker=data/'snapshots/_running.json'
    assert not marker.is_file() or P.read(marker).get('stage')!='extra:research-library','Existing worker is writing the same pack'
    stage.replace(path)
    meta[target]={'id':target,'kind':target,'label':'持续研究库 · 产物与图谱全版本','version':version,'created_at':created,'n_entries':len(entries),'bytes':size,'raw_bytes':raw,'notes':notes[:20]}
    DP._write_extras(data,meta)
    print(f'{target}: {len(entries)} entries, {size} compressed bytes',flush=True)


if __name__=='__main__':
    main()
