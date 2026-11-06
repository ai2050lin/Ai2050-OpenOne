"""Check the actual asynchronously rebuilt offline main pack contains completed workpackage state."""
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import phase2768_knowledge_loop as L


def main():
    path=ROOT/'distributed_data/datapack/datapack.json.gz'
    with gzip.open(path,'rt',encoding='utf-8') as f:pack=json.load(f)
    packages=pack['entries']['/api/research-library/workpackages']['workpackages']
    run=next(w for w in packages if w['body']['run_id']==L.RUN)
    assert run['body']['stages']['client_acceptance']['state']=='done'
    assert run['body']['stages']['validation']['state']=='not_supported'
    result={'kind':'software_offline_pack_check','passed':True,'run_id':L.RUN,'version':pack['version'],
        'created_at':pack['created_at'],'workpackage_revision':run['sha'],'client_acceptance':'done',
        'research_validation':'not_supported','automatic_background_refresh':True,
        'pack_errors':pack.get('errors',[])}
    assert not result['pack_errors']
    L.save(L.OUT/'offline-pack-check.json',result,True)
    print(json.dumps({k:v for k,v in result.items() if k!='version'},ensure_ascii=False))


if __name__=='__main__':main()
