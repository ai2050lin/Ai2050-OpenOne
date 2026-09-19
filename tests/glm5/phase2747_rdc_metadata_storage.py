"""Audit task-owned metadata relocation; numerical archives and model files stay put."""
import argparse
from rdc_formation_common import *

NAMES=['calibration','parameter_propagation','training_analysis','radius_analysis','transfer']
DEST=Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2747_metadata')
MANIFEST=OUT/'engineering/metadata_relocation_preflight.json'


def main(verify=False):
    if not verify:
        assert not MANIFEST.exists()
        for name in NAMES:
            source=OUT/name
            assert source.is_dir() and not source.is_junction() and source.resolve().is_relative_to(OUT.resolve())
            assert not (DEST/name).exists()
            assert all(not p.is_symlink() and not p.is_junction() for p in source.rglob('*'))
        for relative in ['calibration/result.json','parameter_propagation/native/result.json','parameter_propagation/smooth/result.json',
                         'training_analysis/result.json','radius_analysis/result.json','transfer/preparation.json']:
            assert read(OUT/relative)['all_passed']
        entries=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
                 for name in NAMES for p in sorted((OUT/name).rglob('*')) if p.is_file()]
        immutable(MANIFEST,{'timestamp':stamp(),'source':snapshot(__file__),'source_directory':str(OUT),'destination_directory':str(DEST),
            'subdirectories':NAMES,'files':entries,'bytes':sum(r['bytes'] for r in entries),'free_D_before':shutil.disk_usage(ROOT).free,
            'reason':'Preserve completed task-owned metadata while maintaining existing logical paths. Original model files, prior Phases and all full numerical fields untouched.'})
        print('FORMATION_METADATA_PREFLIGHT',len(entries),sum(r['bytes'] for r in entries),flush=True)
    else:
        spec=read(MANIFEST);count=0
        for name in NAMES:
            assert (OUT/name).is_junction() and (OUT/name).resolve()==(DEST/name).resolve()
            actual={p.relative_to(DEST).as_posix() for p in (DEST/name).rglob('*') if p.is_file()}
            assert actual=={r['path'] for r in spec['files'] if r['path'].split('/')[0]==name}
        for entry in spec['files']:
            path=OUT/entry['path'];assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'];count+=1
        immutable(OUT/'engineering/metadata_relocation_verified.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
            'files_verified':count,'bytes_retained':spec['bytes'],'preflight_sha256':sha(MANIFEST),'free_D_after':shutil.disk_usage(ROOT).free,
            'old_queries_preserved':True,'data_deleted':False,'original_models_changed':False,'physical_directory':str(DEST)})
        print('FORMATION_METADATA_VERIFIED',count,spec['bytes'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');main(p.parse_args().verify)
