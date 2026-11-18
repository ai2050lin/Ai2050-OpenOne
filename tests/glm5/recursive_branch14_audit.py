"""Read-only scientific artifact audit; does not load a CUDA model."""
import gc
from pathlib import Path
import time
import numpy as np
import recursive_branch14 as R
from tools.provenance import check_provenance


def run():
    R.configure();plan=R.admission();assert not check_provenance(plan)
    start=time.monotonic();checks=[];files=[];allrows=R.materials()
    identity=R.read(R.OUT/'model-identity.json')['files'];model_checks=[]
    from shared.dist_protocol.natural_register_runner import PATTERNS
    current_names={p.name for pattern in PATTERNS for p in R.MODEL.glob(pattern) if p.is_file()}
    assert current_names=={item['path'] for item in identity},'Model/tokenizer file set changed'
    for i,item in enumerate(identity):
        path=R.MODEL/item['path']
        assert path.stat().st_size==item['bytes'] and R.file_sha(path)==item['sha256'],f'Model/tokenizer changed: {path.name}'
        model_checks.append(dict(path=item['path'],sha256=item['sha256'],passed=True))
        R.log('final_model_identity',i+1,len(identity),start,file=path.name)
    import importlib.metadata
    for name,version in plan['runtime'].items():assert importlib.metadata.version(name)==version,(name,version)
    selection=R.read(R.OUT/'selection.json');assert not check_provenance(selection)
    pilot=R.read(R.OUT/'pilot/collection.json');disc=R.read(R.OUT/'discovery/collection.json')
    repeat=[disc['row_indices'].index(i) for i in pilot['row_indices']]
    repeat_checks=[];embedding_tokens=set()
    for split in ['pilot']+R.SPLITS:
        meta=R.read(R.OUT/split/'collection.json');assert not check_provenance(meta)
        assert meta['model_snapshot']==R.SNAPSHOT and meta['plan_sha256']==R.file_sha(R.OUT/'plan.json')
        rr=[allrows[i] for i in meta['row_indices']];n=len(rr)
        assert [r['id'] for r in rr]==meta['row_ids']
        scores=np.array(meta['scores']);assert scores.shape==(n,3) and np.isfinite(scores).all()
        for b in range(R.DEPTH):
            for f in ['G','V','A']:
                path=R.OUT/split/f'B{b}-{f}.npy';x=np.load(path,mmap_mode='r')
                assert x.shape==(n,2,R.WIDTH) and x.dtype==np.uint16
                for lo in range(0,n,64):assert np.isfinite(R.unbits(x[lo:lo+64])).all()
                if split=='discovery':
                    old=np.load(R.OUT/'pilot'/f'B{b}-{f}.npy',mmap_mode='r')
                    different=0;diff=base=0.
                    for lo in range(0,len(repeat),32):
                        a=R.unbits(old[lo:lo+32]);v=R.unbits(x[repeat[lo:lo+32]])
                        different+=int(np.count_nonzero(a!=v));diff+=float(np.sum((a-v).astype(np.float64)**2));base+=float(np.sum(a.astype(np.float64)**2))
                    repeat_checks.append(dict(block=b,field=f,unequal_values=different,relative_L2=(diff/max(base,1e-20))**.5))
                    del old
                files.append(dict(path=str(path.relative_to(R.OUT)),bytes=path.stat().st_size,sha256=R.file_sha(path)))
                del x
            R.log('native_audit_'+split,b+1,R.DEPTH,start,block=b)
        hs=np.load(R.OUT/split/'hidden-all.npy',mmap_mode='r');assert hs.shape==(n,41,2,R.DIM)
        for lo in range(0,n,32):assert np.isfinite(R.unbits(hs[lo:lo+32])).all()
        assert np.all(hs[:,0,0]==hs[0,0,0]),'Current-token embedding baseline unexpectedly differs'
        embedding_tokens.update(r['tokens'][r['content_end']] for r in rr)
        for h,local_indices in meta['entrance_local_indices'].items():
            cache=np.load(R.OUT/split/f'{h}-replay-input.npy',mmap_mode='r');assert cache.shape==(len(local_indices),R.LENGTH,R.DIM)
            for lo in range(0,len(local_indices),32):
                ix=local_indices[lo:lo+32];value=np.asarray(cache[lo:lo+32])
                actual=value[np.arange(len(ix))[:,None],np.array([[rr[i]['content_end'],rr[i]['answer_pos']] for i in ix])]
                assert np.array_equal(actual,hs[ix,R.CANDIDATES[h][0]])
            del cache
        del hs
        for path in sorted((R.OUT/split).glob('*.npy')):
            rel=str(path.relative_to(R.OUT))
            if any(f['path']==rel for f in files):continue
            x=np.load(path,mmap_mode='r')
            if path.name.endswith('-all-token-GVA.npy'):
                assert x.shape==(n,R.LENGTH,3)
            elif path.name=='final-hidden.npy':assert x.shape==(n,2,R.DIM)
            elif path.name=='prefix-embedding.npy':assert x.shape==(n,R.DIM) and x.dtype==np.float32
            if x.dtype==np.uint16:
                for lo in range(0,len(x),32):assert np.isfinite(R.unbits(x[lo:lo+32])).all()
            else:assert np.isfinite(x).all()
            files.append(dict(path=rel,bytes=path.stat().st_size,sha256=R.file_sha(path)));del x
        checks.append(dict(split=split,rows=n,native_shapes=True,finite=True,entrance_hidden_consistent=True,current_token_embedding_constant=True))
        gc.collect()
    intervention_checks=[]
    for h in R.CANDIDATES:
        d=R.read(R.OUT/f'intervention-{h}.json');assert not check_provenance(d)
        assert d['selection_sha256']==R.file_sha(R.OUT/'selection.json') and d['cases']==1440
        assert len(d['rows'])==14400 and len({(r['id'],r['arm']) for r in d['rows']})==14400
        assert all(r['restore_bitwise'] and r['replay_KL_max']<=.001 and r['baseline_final_relative_L2']<=.02 for r in d['checks'])
        for r in d['rows']:
            assert all(np.isfinite(r[k]) for k in ['margin','baseline_margin','delta_margin','KL'])
            assert r['correct']==(int(np.argmax(r['scores']))==r['gold'])
            assert abs(r['delta_margin']-(r['margin']-r['baseline_margin']))<1e-9
        intervention_checks.append(dict(hypothesis=h,cases=d['cases'],arms=len(d['arms']),rows=len(d['rows']),all_replay_gates=True))
    R.log('audit_complete',1,1,start)
    R.save('native-manifest.json',dict(kind='real',files=files,total_bytes=sum(f['bytes'] for f in files),
        retention=plan['retention'],reuse_entry='tests/glm5/recursive_branch14_analysis.py; row IDs bound by each split collection.json'))
    R.save('audit.json',dict(passed=True,kind='real',checks=checks,intervention_checks=intervention_checks,
        pilot_discovery_replay=repeat_checks,content_token_ids=sorted(embedding_tokens),
        final_model_identity=model_checks,runtime_matches_freeze=True,
        source_sha256=R.file_sha(__file__),native_manifest_sha256=R.file_sha(R.OUT/'native-manifest.json'),
        limitation='Same-node artifact and arithmetic checks, not independent-node scientific replication.'))


if __name__=='__main__':run()
