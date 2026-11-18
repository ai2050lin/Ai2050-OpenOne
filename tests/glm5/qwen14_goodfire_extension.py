"""Read-only reuse of B28--35 and prospective native B36--39 extension.

All writes are in the D: result folder. Frozen historical runners are reused
through explicit runtime configuration; their source files are never edited.
"""
import argparse
import gc
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
import qwen14_mlp_atlas as C
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import save_json as raw_save
from tools import provenance
from source_binding_resources import resources

OLD = C.OUT
OUT = ROOT / 'tests/glm5/result/qwen14_mlp_b28_b39_v2'
TEMP = ROOT / 'tests/glm5_temp/qwen14_mlp_b28_b39_v2'
RUN = 'QWEN14-GOODFIRE-B28-B39-20261113-v2'
NEW = list(range(36, 40))
ALL = list(range(28, 40))
SNAP = C.read(OLD/'plan.json')['model_snapshot']
SEED = 20261113


def save(path, body):
    path = check_write(path)
    assert path.parent == OUT.resolve(), path
    helper = provenance.write_result('qwen14_ext_'+path.stem,
        {'run_id': RUN, 'result_path': str(path), 'payload_sha256': C.digest(body)},
        model_path=C.MODEL, seed=SEED, tool='qwen14_goodfire_extension')
    prov = C.read(helper)['provenance']
    assert not provenance.check_provenance({'provenance': prov})
    raw_save(path, {**body, 'kind': 'real', 'run_id': RUN, 'model_snapshot': SNAP,
                   'provenance': prov, 'provenance_helper': str(helper.relative_to(ROOT))})


def log(stage, done=0, total=0, start=None, **kw):
    elapsed = time.monotonic()-start if start else 0
    # The historical analysis routines hard-code eight-block progress totals.
    if stage in ('profiles', 'parameter_readout'):
        done -= 8*C.WIDTH; total = 4*C.WIDTH
    if stage == 'conditional_prediction': done -= 24; total = 12
    row = dict(stage=stage, done=done, total=total, seconds=round(elapsed, 2),
               eta_seconds=round(elapsed/done*(total-done), 1) if done else None,
               **resources(), **kw)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t = sys.modules['torch']
        row.update(cuda_allocated=t.cuda.memory_allocated(), cuda_reserved=t.cuda.memory_reserved(),
                   cuda_peak=t.cuda.max_memory_allocated())
    check_write(OUT)
    if row.get('shared_gpu_bytes') is not None:
        assert row['shared_gpu_bytes'] < 2**30, 'GPU spill exceeds frozen 1 GiB limit'
    print(__import__('json').dumps(row, ensure_ascii=False), flush=True)


def configure():
    check_write(OUT); check_write(TEMP); check_write(ROOT/'results')
    OUT.mkdir(parents=True, exist_ok=True); TEMP.mkdir(parents=True, exist_ok=True)
    for key in ['TEMP','TMP','TMPDIR','HF_HOME','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        os.environ[key] = str(TEMP)
    os.environ.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='safe.directory',
                      GIT_CONFIG_VALUE_0=str(ROOT), PYTHONUTF8='1')
    C.OUT=OUT; C.STORE=OUT; C.RUN=RUN; C.BLOCKS=NEW; C.save_json=save; C.log=log


def source(block):
    return OLD if block < 36 else OUT


def data(block, field):
    return np.load(source(block)/f'B{block}-{field}.npy', mmap_mode='r').reshape(-1,C.WIDTH)


def prepare():
    assert not (OUT/'plan.json').exists(), 'Frozen plan already exists'
    check_write(OUT, 12*2**30, operation='Goodfire extension peak budget')
    start=time.monotonic(); log('preflight', start=start)
    identity=C.read(OLD/'model-identity.json')
    for i,row in enumerate(identity['files']):
        p=C.MODEL/row['path']
        assert p.stat().st_size==row['bytes'] and C.file_sha(p)==row['sha256'], p
        log('model_identity', i+1,len(identity['files']),start)
    # Revalidate every historical dependency being reused, not package copies.
    manifest=C.read(OLD/'manifest.json'); refs=[]
    for row in manifest['files']:
        name=row['path']
        if name.startswith('B') or name in ['documents.json','tokens.npy','prefix-extrema.npz','report.json','collection.json']:
            p=OLD/name
            assert p.stat().st_size==row['bytes'] and C.file_sha(p)==row['sha256'], p
            refs.append({**row,'path':str(p)})
    save(OUT/'reused-artifacts.json', {'files':refs,'manifest_sha256':C.file_sha(OLD/'manifest.json'),
        'read_only':True,'historical_admission_limitations':C.read(OLD/'report.json')['limitations']})
    import shutil
    for name in ['tokens.npy','documents.json','model-identity.json','tokenizer-audit.json']:
        if name.endswith('.json'):
            save(OUT/name, {**C.read(OLD/name),'reused_source_sha256':C.file_sha(OLD/name)})
        else: shutil.copyfile(OLD/name, OUT/name)
    plan={**C.read(OLD/'plan.json'), 'schema':'qwen14_goodfire_extension_plan.v2',
          'blocks':NEW,'combined_blocks':ALL,'units':4*C.WIDTH,'combined_units':12*C.WIDTH,
          'collector_sha256':C.file_sha(__file__),
          'tokens_sha256':C.file_sha(OUT/'tokens.npy'),'documents_sha256':C.file_sha(OUT/'documents.json'),
          'resources':{'new_disk_GiB_max':12,'free_disk_GiB_min':20,'gpu_wall_minutes_max':120,
                       'document_chunk':256,'forward_batch':2,'shared_GPU_GiB_max':1},
          'retention':{'native_GVA_and_parameters':'Reusable native neuron archive for all-unit queries, analysis and reproducibility.',
                       'review_date':'2026-12-13','peak_GiB':12,'old_files':'read only references; no duplicated archive'},
          'exposure':'Same licensed 1024 document families as v1. Existing B28-B35 responses exposed. B36-B39 fit only discovery. No new global blind corpus.',
          'power_scope':'Descriptive census and continuous prediction screening. No paired binary behavior claim until separate intervention power gate.',
          'regimes':'All 12 blocks, per split: G/V quadrants, signed A mass, A energy, negative-gate derivative. Descriptive, not semantic confirmation.',
          'replay':'Compare all B35 G/V/A readouts with historical BF16 field during new forward; fail at relative L2 > 0.02.',
          'next_steps':'New four-block profiles/readouts/prediction; four new adjacent boundaries; selected final-token doses and capability controls.'}
    save(OUT/'plan.json',plan)
    import qwen14_mlp_analysis as A
    A.C.save_json=save
    A.freeze()
    save(OUT/'source-binding.json',{'files':[{ 'path':str(p),'sha256':C.file_sha(p)} for p in [Path(__file__),
        Path(C.__file__),Path(A.__file__),ROOT/'tests/glm5/qwen14_mlp_intervene.py',
        ROOT/'tests/glm5/source_binding_resources.py',ROOT/'deploy/storage_policy.py']]})
    log('preflight_complete',1,1,start)


def collect():
    original=C.load_block; counts={}; comparisons=[]
    old={f:np.load(OLD/f'B35-{f}.npy',mmap_mode='r') for f in ['G','V','A']}
    def loader(b,cfg):
        layer=original(b,cfg)
        if b==35:
            for field,module,pre in [('G',layer.mlp.gate_proj,False),('V',layer.mlp.up_proj,False),('A',layer.mlp.down_proj,True)]:
                def hook(m,a,y=None,field=field):
                    tensor=a[0] if field=='A' else y
                    z=tensor[:,C.POSITIONS].contiguous().view(__import__('torch').uint16).cpu().numpy()
                    lo=counts.get(field,0); ref=old[field][lo:lo+len(z)]
                    rel=float(np.linalg.norm(C.unbits(z)-C.unbits(ref))/max(np.linalg.norm(C.unbits(ref)),1e-20))
                    comparisons.append({'field':field,'first_document':lo,'documents':len(z),'relative_L2':rel,'bitwise_equal':bool(np.array_equal(z,ref))})
                    assert rel<=.02, 'Historical B35 reference drift'
                    counts[field]=lo+len(z)
                if pre: module.register_forward_pre_hook(hook)
                else: module.register_forward_hook(hook)
        return layer
    C.load_block=loader
    C.collect()
    assert counts=={'G':1024,'V':1024,'A':1024},counts
    save(OUT/'historical-replay.json',{'rows':comparisons,'all_bitwise_equal':all(x['bitwise_equal'] for x in comparisons),
        'max_relative_L2':max(x['relative_L2'] for x in comparisons),'counts':counts})


def analysis(stage):
    import qwen14_mlp_analysis as A
    C.save_json=save
    getattr(A,stage)()


def intervention(stage):
    import qwen14_mlp_intervene as I
    I.save_result=save
    if stage=='freeze_intervention':
        I.freeze(); I.select()
        save(OUT/'intervention-execution-binding.json',{'source_sha256':C.file_sha(I.__file__),
            'wrapper_sha256':C.file_sha(__file__),'scope':'Reuse exact query-shape-preserving replay; only new B36-B39 units. Power uses conservative 16-comparison correction.'})
    elif stage=='calibrate':
        I.run('calibrate_exact')
        rows=C.read(OUT/'calibrate_exact-effects.json')['replay']
        passed=all(x['restore_bitwise'] and x['full_vs_kv_kl']<=.001 for x in rows)
        save(OUT/'exact-calibration-admission.json',{'passed':passed,'rows':rows,'threshold':.001})
        assert passed
    else:I.run(stage+'_exact')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage')
    stage=parser.parse_args().stage;configure()
    if stage=='prepare':prepare()
    elif stage=='pilot':C.pilot()
    elif stage=='collect':collect()
    elif stage in ['profiles','predict','readout']:analysis(stage)
    elif stage in ['freeze_intervention','calibrate','natural','capability']:intervention(stage)
    else:raise ValueError(stage)


if __name__=='__main__':main()
