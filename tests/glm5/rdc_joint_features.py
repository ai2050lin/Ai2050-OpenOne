"""Prefix-available full-coordinate current and temporal information sets."""
import hashlib
from rdc_joint_common import *
from rdc_joint_prior_rules import QueryProposal,rms_sources
from rdc_joint_native_attention import NativeAttention
from phase2715_rdc_prefix_relations import PrefixRelations
from rdc_relation_native_parameters import parameter,decode
from rdc_joint_capture import ledger


def available_current(h,ids,language,parser,native):
    h = np.asarray(h,np.float32)
    assert len(h)==len(ids) and len(h)>1
    unit = rms_sources(h[:-1])
    logscale = np.log(np.maximum(np.sqrt(np.mean(h[:-1].astype(float)**2,axis=1)),1e-8))
    prob,_ = parser.weights(ids,language)
    rel = prob[:,1:]
    message = rel.T@unit/(1+rel.sum(0))[:,None]
    source = native.context(h,ids)
    shuffled = native.context(h,ids,permuted_values=True)
    return {'current':h[-1], 'raw_mean':h[:-1].mean(0),'rms_mean':unit.mean(0),
        'direction_scale':np.concatenate([unit.mean(0),(unit*logscale[:,None]).mean(0),[logscale.mean()]]),
        'rms_relation':message, 'native_context':source,'native_permuted':shuffled}


def available_temporal(previous,embedding,past_h12,ids,proposal,native,*,actual_query=None):
    assert len(past_h12)+1==len(ids)
    query = proposal(previous,embedding)[0]
    h = np.concatenate([past_h12,query[None]],axis=0)
    result = {'previous':previous,'embedding':embedding,'query_proposal':query,
              'context':native.context(h,ids),'context_permuted':native.context(h,ids,permuted_values=True)}
    if actual_query is not None:
        hh = np.concatenate([past_h12,actual_query[None]],axis=0)
        result['context_actual_query_extra_information'] = native.context(hh,ids)
    return result


def build(store):
    fresh = store.fresh
    out = BASE/'features'/('fresh' if fresh else 'main')
    if (out/'complete.json').exists():
        report = read(out/'complete.json')
        for name,digest in report['file_sha'].items():
            assert sha(out/name)==digest
        return report
    start = time.monotonic()
    assert (PREVIOUS/'prefix_parser/frozen.json').exists()
    proposal,parser = QueryProposal(),PrefixRelations()
    native = NativeAttention()
    table = parameter(ROOT,'model.embed_tokens.weight')
    current,temporal,target,post,meta,checks = {},{},[],[],[],[]
    try:
        for ri,r in enumerate(store.material):
            z = store[r]
            h12,h23,h36 = (unbits(z[k]) for k in ('h12','h23','h36'))
            for j,p in enumerate(r['anchors']):
                ids = r['prompt_ids'][:p+1]
                c = available_current(h12[:p+1],ids,r['language'],parser,native)
                embedding = decode(table[r['prompt_ids'][p+1]])
                t = available_temporal(h36[p],embedding,h12[:p+1],r['prompt_ids'][:p+2],proposal,native,actual_query=h12[p+1])
                for name,v in c.items():current.setdefault(name,[]).append(np.asarray(v,np.float32))
                for name,v in t.items():temporal.setdefault(name,[]).append(np.asarray(v,np.float32))
                target.append(np.concatenate([h23[p],h36[p],h36[p+1],h12[p+1]]))
                post.append(np.stack([unbits(z['postnorm'][2+2*j]),unbits(z['postnorm'][3+2*j])]))
                meta.append({k:r[k] for k in ('sample_id','source_group','split','language','genre','language_mode_families')}|
                    {'anchor':j,'position':p,'known_next_token_id':r['prompt_ids'][p+1],
                     'observed_current_output_token':r['prompt_ids'][p+1],'observed_temporal_output_token':r['prompt_ids'][p+2]})
                if ri<2 and j==0:
                    checks.append(native.check_native(h12[:p+1],ids))
                    again = available_temporal(h36[p],embedding,h12[:p+1],r['prompt_ids'][:p+2],proposal,native)
                    assert set(again)==set(t)-{'context_actual_query_extra_information'}
                    assert all(np.array_equal(again[k],t[k]) for k in again)
                    checks[-1]['removing_unavailable_actual_query_does_not_change_primary_features'] = True
            if (ri+1)%64==0:print('JOINT_FEATURES',fresh,ri+1,len(store.material),flush=True)
        for category,data in [('current',current),('temporal',temporal)]:
            for name,values in data.items():
                array = np.stack(values)
                guard(array.nbytes+1024**2)
                npz(out/f'{category}_{name}.npz',value=array)
                del array
        npz(out/'targets.npz',value=np.stack(target),postnorm=np.stack(post))
        save(out/'rows.json',meta)
        save(out/'checks.json',{'timestamp':stamp(),'checks':checks,'passed':True,'source':snapshot(Path(__file__)),
            'native_source':snapshot(ROOT/'tests/glm5/rdc_joint_native_attention.py')})
        report = {'timestamp':stamp(),'fresh':fresh,'anchors':len(meta),'sources':len(store.material),
            'file_sha':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='complete.json'},
            'target_columns':{'current_H23':[0,2560],'current_H36':[2560,5120],'next_H36':[5120,7680],'next_H12_proposal_diagnostic':[7680,10240]},
            'current_inputs':'True lower-layer H12 at every prefix source, prefix-only old parser probabilities, fixed native layer12 Q/K/V. Predict current H23/H36.',
            'temporal_inputs':'Previous H36, already-known new embedding, complete past H12 at ONE selected layer; old-only proposal supplies new H12. All32x128 head-context features, no true new target state.',
            'extra_information':'context_actual_query_extra_information uses actual new H12; only diagnostic, excluded from autonomous route and primary winner selection.',
            'permutation':'Change source-value correspondence only; Q/K and actual rotary positions fixed, deterministic prefix-ID hash.',
            'limits':'Native selected-layer factors are known architecture, not an extracted whole-model mechanism. Past H12 is richer than a single H36 and is not all-layer KV sufficient state. Current true H12 already contains processed history. All coordinate blocks retained, no Tucker/PCA/Top-K.'}
        save(out/'complete.json',report)
    finally:
        native.close()
    ledger('joint_available_features',time.monotonic()-start,fresh=fresh,sources=len(store.material))
    print('JOINT_FEATURES_COMPLETE',fresh,usage(),flush=True)
    return report


def load_features(fresh=False):
    out = BASE/'features'/('fresh' if fresh else 'main')
    report = read(out/'complete.json')
    raw = {}
    for name,digest in report['file_sha'].items():
        if not name.endswith('.npz') or name=='targets.npz':continue
        assert sha(out/name)==digest
        with np.load(out/name) as z:raw[name[:-4]]=z['value']
    with np.load(out/'targets.npz') as z:y,post=z['value'],z['postnorm']
    return raw,y,post,read(out/'rows.json')
