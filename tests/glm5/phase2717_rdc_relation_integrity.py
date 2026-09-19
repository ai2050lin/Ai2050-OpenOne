"""Evidence hashes, native coverage, exclusions and independent frozen prediction checks."""
import hashlib,argparse
from rdc_relation_common import *


def main(final=False):
    start=time.monotonic();checks={};counts={};files=[]
    prefix=read(BASE/'memo_prefix.json');memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_bytes()
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256'];checks['original_memo_byte_prefix_preserved']=True
    for r in read(BASE/'review.json')['original_sources']:assert sha(ROOT/r['path'])==r['sha256'],r['path']
    checks['all_reviewed_old_results_unchanged']=True
    frozen=read(BASE/'frozen.json')
    for path,digest in frozen['files'].items():assert sha(BASE/path)==digest,path
    checks['frozen_inputs_models_and_selection_unchanged']=True
    for scope,fresh in [('main',False),('fresh',True)]:
        rr=rows(fresh);assert len(rr)==(128 if fresh else 512);tokens=0
        for r in rr:
            c=read(BASE/scope/f'commits/{r["sample_id"]}.json')
            assert sha(BASE/c['protocol'])==c['protocol_sha']
            for path,digest in c['files'].items():assert sha(BASE/path)==digest,path
            z=load_field(r,fresh);n=len(r['prompt_ids']);tokens+=n
            for k in ('h12','h23'):assert z[k].shape==(n,2560) and z[k].dtype==np.uint16
            for k in ('h24','h36','postnorm'):assert z[k].shape==(6,2560) and z[k].dtype==np.uint16
            assert np.array_equal(z['positions'],r['positions'])
            for k in ('h12','h23','h24','h36','postnorm'):assert np.isfinite(unbits(z[k])).all()
        counts[scope]={'sources':len(rr),'tokens':tokens}
    for name in ('native','generation'):
        cc=list((BASE/name/'commits').glob('*.json'));assert len(cc)==(128 if name=='native' else 64)
        for path in cc:
            r=read(path);assert sha(BASE/name/f'fields/{r["sample_id"]}.npz')==r['field_sha']
        counts[name]=len(cc)
    old=old_material();allnew=rows()+rows(True)
    oldgroups={r['source_group'] for r in old};assert not oldgroups.intersection(r['source_group'] for r in allnew)
    assert len({r['source_group'] for r in allnew})==len(allnew)
    for transform in (normalized,canonical):
        oldset={transform(r['text']) for r in old};new=[transform(r['text']) for r in allnew];assert len(new)==len(set(new)) and not oldset.intersection(new)
    sk=[construction(r) for r in allnew];assert len(sk)==len(set(sk)) and not set(sk).intersection(construction(r) for r in old)
    checks['whole_source_text_numeric_and_explicit_skeleton_exclusions']=True
    source={r['sample_id']:r for r in rows()};n=0
    for e in read(BASE/'relation_atlas/pair_index.json'):
        r=source[e['sample_id']];dep,head=e['dependent'],e['head'];assert dep!=head
        for kind,pairs in e['controls'].items():
            for i,j in pairs:
                assert 0<=i<len(r['prompt_ids']) and 0<=j<len(r['prompt_ids']) and i!=j and i-j==dep-head
                if kind.endswith('same_dependent_id'):assert r['prompt_ids'][i]==r['prompt_ids'][dep]
                n+=1
    checks['all_control_index_distance_and_strict_identity_bounds']=n
    # This does not repeat fitting and does not use a target state in the predictor.
    from rdc_relation_inference import CurrentRule
    rr=rows(True);meta=read(BASE/'confirmation/rows.json');model=CurrentRule();errors=[]
    with np.load(BASE/'confirmation/current_full_quadratic.npz') as z:saved=z['prediction']
    for r in [rr[0],rr[1],rr[-2],rr[-1]]:
        field=load_field(r,True);p=model(unbits(field['h12'][r['anchors']]))
        idx=[i for i,m in enumerate(meta) if m['sample_id']==r['sample_id']];errors.append(float(np.max(np.abs(p-saved[idx]))))
    assert max(errors)<1e-4;checks['independent_frozen_current_recompute_max_abs']=max(errors)
    if final:
        for key in ('qwen4','qwen14','glm4'):
            folder=BASE/'scale'/key;result=read(folder/'result.json');assert result['fresh_sources']==64 and not read(folder/'runtime.json')['quantized']
            assert len(list((folder/'rows').glob('*.json')))==224
            assert len(list((folder/'generation').glob('*.json')))==64
            for path in (folder/'rows').glob('*.json'):
                r=read(path)
                if key!='qwen4':assert sha(folder/f'fields/{r["sample_id"]}.npz')==r['field_sha']
                else:assert sha(BASE/r['origin']/f'fields/{r["sample_id"]}.npz')==r['full_field_sha']
            counts['scale_'+key]={'sources':224,'generation_sources':64,'quantized':False}
    # Read each compressed member, including all low-amplitude and far-end coordinates.
    for path in BASE.rglob('*.npz'):
        if not final and 'scale' in path.relative_to(BASE).parts:continue
        with np.load(path,allow_pickle=False) as z:
            for name in z.files:
                a=z[name]
                if a.dtype.kind in 'fc':assert np.isfinite(a).all(),(path,name)
        files.append({'path':str(path.relative_to(BASE)),'bytes':path.stat().st_size,'sha256':sha(path)})
    # Previous complete1670-file audit measured305328 bytes;750KiB covers the expanded manifest.
    guard(750*1024);out=BASE/'verification'/('final_integrity.json' if final else 'core_integrity.json')
    save(out,{'timestamp':stamp(),'passed':True,'final_scale_included':final,'checks':checks,'counts':counts,'all_npz_count':len(files),'files':files,
      'elapsed_seconds':time.monotonic()-start,'usage_bytes':usage(),'new_ceiling_bytes':CEILING,'physical_free_bytes':shutil.disk_usage(ROOT).free,'source':snapshot(Path(__file__)),
      'limits':'Hash/numeric/coverage and bounded recomputation audit; does not prove semantic-family independence or recover a unique mechanism.'})
    print('RELATION_INTEGRITY_PASS',final,counts,len(files),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
