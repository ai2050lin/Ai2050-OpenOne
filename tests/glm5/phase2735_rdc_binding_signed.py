"""Train-only signed source moments; prospective natural confirmation.

CPU-only linear algebra on existing arrays is separate from the serial CUDA
native capture. No native model weights or future states enter this fit.
"""
import argparse
from collections import Counter
from rdc_binding_common import *

def features(rows,sources,queries,embeddings,coef):
    from rdc_binding_kernels import apply_roles
    roles=apply_roles(sources,coef);packs={k:[] for k in ('position','role_position','role_shuffled')}
    for row,h,r in zip(rows,sources,roles):
        h=h.astype(np.float64);r=r.astype(np.float64);n=len(h);z=np.linspace(-1,0,n)
        pos=np.stack([np.ones(n),z,z*z],1)
        permutation=np.random.default_rng(int(ranked('signed-role/'+row.get('frozen_feature_seed_id',row['sample_id']))[:8],16)).permutation(n)
        packs['position'].append((h.T@pos/n).ravel())
        for name,role in [('role_position',r),('role_shuffled',r[permutation])]:
            tagged=(pos[:,:,None]*np.concatenate([np.ones((n,1)),role],1)[:,None,:]).reshape(n,-1)
            packs[name].append((h.T@tagged/n).ravel())
    result={k:np.array(v) for k,v in packs.items()}
    q=queries.astype(float);e=embeddings.astype(float)
    result['q']=q/np.sqrt(np.mean(q*q,-1,keepdims=True)).clip(1e-8)
    result['e']=e/np.sqrt(np.mean(e*e,-1,keepdims=True)).clip(1e-8)
    return result

def kernels(left,right):
    d=left['q'].shape[-1];base=(left['q']@right['q'].T+left['e']@right['e'].T)/(2*d)
    result={}
    for key in ('position','role_position','role_shuffled'):
        s=left[key]@right[key].T/d
        result['signed_'+key]=1+base+s+base*s
    return result

def ridge(k,y,rows):
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    counts=Counter(rows[i]['source_group'] for i in train)
    weight=np.array([1/counts[rows[i]['source_group']] for i in train]);weight/=weight.sum();sw=np.sqrt(weight)
    center=weight@y[train];scale=np.diag(k)[train].mean();kn=k/scale
    ks=kn[train][:,train]*sw[:,None]*sw[None,:];eig,vec=np.linalg.eigh(ks);positive=np.maximum(eig,0)
    lo=positive.max()*1e-14;hi=positive.max()*1e8
    for _ in range(90):
        mid=np.sqrt(lo*hi)
        if np.sum(positive/(positive+mid))>128:lo=mid
        else:hi=mid
    lam=np.sqrt(lo*hi);df=float(np.sum(positive/(positive+lam)));assert abs(df-128)<1e-5
    coefficients=sw[:,None]*(((vec/(eig+lam)[None,:])@vec.T)@(sw[:,None]*(y[train]-center)))
    prediction=kn[:,train]@coefficients+center
    return {'coefficients':coefficients,'center':center,'scale':np.array(scale),'train_indices':train},prediction,{'lambda':lam,'df':df}

def summarize(rows,pred,actual,center):
    error=np.sum((pred-actual)**2,1);den=np.sum((actual-center)**2,1)
    reports=[]
    for split,cohort in sorted({(r['split'],r['cohort']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['split'],r['cohort'])==(split,cohort)]
        reports.append({'split':split,'cohort':cohort,'rows':len(ix),
          'relative_mse':float(error[ix].sum()/den[ix].sum()),
          'source_cluster_relative_error':clustered(error[ix]/den[ix].clip(1e-12),[rows[i]['source_group'] for i in ix])})
    return reports

def freeze():
    from threadpoolctl import threadpool_limits
    from transformers import AutoTokenizer
    from phase2728_rdc_law_material import natural_candidates
    from phase2732_rdc_binding_material import connected
    from rdc_binding_kernels import source_arrays
    threadpool_limits(limits=2);start=time.monotonic();out=BASE/'signed_source'
    if (out/'frozen.json').exists():return
    guard(650*1024**2)
    assert read(BASE/'verification/source_moment_collision/result.json')['all_passed']
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    old=gzread(LAW/'material.json.gz')+gzread(LAW/'confirmation_material.json.gz')+gzread(BASE/'natural_confirmation.json.gz')
    occupied={c for r in old for c in r.get('component_ids',[])};pools={};provenance=[]
    discovery=gzread(BASE/'natural_discovery.json.gz');train_groups={r['source_group'] for r in discovery if r['split']=='train'}
    for bank in ('gum','ewt'):
        candidates=natural_candidates(tok,bank,'test',provenance)
        for r in candidates:r['connected_held']=connected(r)
        for flag in (True,False):
            selected=[];local=set(occupied)
            for r in sorted(candidates,key=lambda r:ranked('signed-next/'+r['text'])):
                if bool(r['connected_held'])!=flag or r['source_group'] in train_groups:continue
                if any(c in local for c in r['component_ids']):continue
                selected.append(r);local.update(r['component_ids'])
            pools[bank,flag]=selected
    selected_pools={};n=0
    # Connected and non-connected windows can share a neutral context sentence.
    # Enforce one global component identity set, not one set per stratum.
    for attempt in range(min(32,min(map(len,pools.values()))),15,-1):
        local=set(occupied);trial={}
        for key,pool in pools.items():
            chosen=[]
            for r in pool:
                if any(c in local for c in r['component_ids']):continue
                chosen.append(r);local.update(r['component_ids'])
                if len(chosen)==attempt:break
            trial[key]=chosen
        if all(len(v)==attempt for v in trial.values()):n=attempt;selected_pools=trial;break
    assert n>=16,{str(k):len(v) for k,v in pools.items()}
    natural=[]
    for (bank,flag),pool in selected_pools.items():
        for r in pool:
            natural.append(dict(r,split='signed_connected' if flag else 'signed_matched',cohort=bank,
              sample_id='b2735_'+ranked('signed/'+bank+'/'+r['text'])[:20],capture_mode='signed',
              anchors=sorted(set([len(r['prompt_ids'])//2,len(r['prompt_ids'])-2])),
              novelty='Prospectively frozen after algebraic collision; no component sentence in prior law main/confirmation or binding confirmation. Shared public treebanks/documents may remain.'))
    components=[c for r in natural for c in r['component_ids']]
    assert len(components)==len(set(components)) and not occupied.intersection(components)
    compressed(out/'natural_material.json.gz',natural)
    immutable(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'natural_rows':len(natural),
      'selection':'Largest feasible equal cohort/connected stratum count32down to16, with one global component exclusion set. No native outcomes available during selection.',
      'pool_counts':{str(k):len(v) for k,v in pools.items()},'actual_per_stratum':n,
      'materials_sha256':sha(out/'natural_material.json.gz'),'source_provenance':provenance,
      'fit':'Existing strict320 English natural train, validation96. New independent natural outcomes not available.',
      'kernels':['original','signed_position','signed_role_position','signed_role_shuffled','original_plus_signed'],
      'df':128,'decoder':'Direct full2560-coordinate MLP output only; no native target-layer input as feature.',
      'mixed_rule':'Equal half sum of original and signed_role_position kernels, each normalized by own training diagonal mean.',
      'novelty':'Algorithm addition is post-hoc relative to2732discovery and original128confirmation; only this new natural material and unobserved depth6 outcomes are prospective.',
      'limits':['Signed low-order moments can resolve the constructed sign collision but remain finite summaries; no sufficiency claim.',
        'New natural observations use CUDA native BF16 after other model processes finish. Fitting here is CPU-only algebra on frozen arrays.']})
    ss,q,e,_,targets,_=source_arrays(discovery)
    with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients'].astype(float)
    pack=features(discovery,ss,q,e,coef);del ss
    npz(out/'all_discovery_signed_features.npz',**pack)
    signed=kernels(pack,pack);reports=[];selected={};train=np.array([i for i,r in enumerate(discovery) if r['split']=='train'])
    validation=np.array([i for i,r in enumerate(discovery) if r['split']=='validation'])
    old_spec=read(BASE/'prediction/frozen.json')['selected'];allkernel={}
    for block in (16,35):
        with np.load(BASE/'prediction/kernels.npz') as z:original=z[old_spec[str(block)]['kernel']].astype(float)
        s=signed['signed_role_position'];mix=.5*original/np.diag(original)[train].mean()+.5*s/np.diag(s)[train].mean()
        kk={'original':original,**signed,'original_plus_signed':mix};actual=targets[block][:,:2560].astype(float)
        center=actual[train].mean(0);den=np.sum((actual[validation]-center)**2)
        for name,k in kk.items():
            bank,pred,info=ridge(k,actual,discovery)
            npz(out/'banks'/f'b{block}_{name}.npz',**bank)
            npz(out/'discovery_predictions'/f'b{block}_{name}.npz',prediction=pred)
            mse=float(np.sum((pred[validation]-actual[validation])**2)/den)
            reports.append({'block':block,'kernel':name,**info,'validation_relative_mse':mse,
              'strata':summarize(discovery,pred,actual,center)})
            allkernel[f'b{block}_{name}']=k
            print('SIGNED_SOURCE_FIT',block,name,mse,flush=True)
        selected[str(block)]=min([r for r in reports if r['block']==block],key=lambda r:r['validation_relative_mse'])['kernel']
    npz(out/'discovery_kernels.npz',**allkernel)
    save(out/'frozen.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'selected':selected,'reports':reports,
      'material_sha256':sha(out/'natural_material.json.gz'),'discovery_material_sha256':sha(BASE/'natural_discovery.json.gz'),
      'role_probe_sha256':sha(BASE/'prediction/role_probe.npz'),'features_sha256':sha(out/'all_discovery_signed_features.npz'),
      'banks_sha256':{p.name:sha(p) for p in (out/'banks').glob('*.npz')},'seconds':time.monotonic()-start,
      'scope':'Finite available-prefix full-coordinate signed moments, validation-only selection, direct MLP decoder, native confirmation pending.'})
    ledger('binding_signed_source_freeze_and_CPU_fit',time.monotonic()-start);guard()
    print('SIGNED_SOURCE_FROZEN',len(natural),selected,flush=True)

def evaluate():
    from threadpoolctl import threadpool_limits
    from rdc_binding_kernels import source_arrays,apply_roles,feature_pack,pair_kernels
    from phase2734_rdc_binding_analysis import visible_combos
    import torch
    threadpool_limits(limits=2);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic();out=BASE/'signed_source'
    if (out/'result.json').exists():return
    frozen=read(out/'frozen.json');rows=signed_rows()+gzread(BASE/'format_content/prospective_material.json.gz')
    for name,digest in frozen['banks_sha256'].items():assert sha(out/'banks'/name)==digest
    discovery=gzread(BASE/'natural_discovery.json.gz');train=[r for r in discovery if r['split']=='train']
    old_s,old_q,old_e,_,old_y,_=source_arrays(train)
    ss=[];qq=[];ee=[];targets={16:[],35:[]};visibility=[]
    for row in rows:
        p=row['anchors'][-1]
        path=(out/'fields' if row.get('capture_mode')=='signed' else BASE/'format_content/prospective_fields')/f'{row["sample_id"]}.npz'
        with np.load(path) as z:
            h=unbits(z['H12_sources'])[:p+1];ss.append(h/np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-8))
            qq.append(h[-1]);ee.append(unbits(z['embedding'])[-1])
            for b in targets:targets[b].append(unbits(z[f'L{b}_mlp'])[-1])
        if row.get('capture_mode')=='signed':
            for anchor,position in enumerate(row['anchors']):
                combos=visible_combos(row,position)
                visibility.append({'sample_id':row['sample_id'],'cohort':row['cohort'],'split':row['split'],
                  'anchor':anchor,'position':position,'combos':combos,'any_connected_visible':any(c['visible'] for c in combos)})
    compressed(out/'connected_visibility.json.gz',visibility)
    with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients'].astype(float)
    pack=features(rows,ss,np.array(qq),np.array(ee),coef)
    with np.load(out/'all_discovery_signed_features.npz') as z:
        ix=[i for i,r in enumerate(discovery) if r['split']=='train'];right={k:z[k][ix] for k in z.files}
    npz(out/'all_confirmation_signed_features.npz',**pack)
    signed=kernels(pack,right)
    # Original square/mean rules use the same existing operational CUDA kernel.
    lp=feature_pack(ss,np.array(qq),np.array(ee),apply_roles(ss,coef))
    rp=feature_pack(old_s,old_q,old_e,apply_roles(old_s,coef))
    with torch.inference_mode():originals={k:v.cpu().numpy().astype(float) for k,v in pair_kernels(lp,rp).items()}
    del lp,rp;torch.cuda.empty_cache();old_spec=read(BASE/'prediction/frozen.json')['selected']
    reports=[];pairs=[];predictions={};errors={};denominators={}
    for block in (16,35):
        original=originals[old_spec[str(block)]['kernel']]
        with np.load(BASE/'prediction/kernels.npz') as z:scale=np.diag(z[old_spec[str(block)]['kernel']])[ix].mean()
        with np.load(out/'discovery_kernels.npz') as z:scale_signed=np.diag(z[f'b{block}_signed_role_position'])[ix].mean()
        mixed=.5*original/scale+.5*signed['signed_role_position']/scale_signed
        kk={'original':original,**signed,'original_plus_signed':mixed};actual=np.array(targets[block],float)
        center=old_y[block][:,:2560].astype(float).mean(0);den=np.sum((actual-center)**2,1);denominators[block]=den
        for name,k in kk.items():
            with np.load(out/'banks'/f'b{block}_{name}.npz') as z:pred=k/float(z['scale'])@z['coefficients']+z['center']
            error=np.sum((pred-actual)**2,1);errors[block,name]=error;predictions[block,name]=pred
            npz(out/'confirmation_predictions'/f'b{block}_{name}.npz',prediction=pred,squared_error=error,baseline_squared_error=den)
            reports.append({'block':block,'kernel':name,'validation_selected':name==frozen['selected'][str(block)],
              'strata':summarize(rows,pred,actual,center)})
        for comparison in [('selected_vs_original',frozen['selected'][str(block)],'original'),
          ('signed_role_vs_original','signed_role_position','original'),
          ('signed_position_vs_original','signed_position','original'),
          ('mixed_vs_original','original_plus_signed','original'),
          ('signed_role_vs_shuffled','signed_role_position','signed_role_shuffled'),
          ('signed_role_vs_position','signed_role_position','signed_position')]:
            label,left,right_name=comparison
            for split,cohort in sorted({(r['split'],r['cohort']) for r in rows}):
                ids=[i for i,r in enumerate(rows) if (r['split'],r['cohort'])==(split,cohort)]
                delta=errors[block,left][ids]-errors[block,right_name][ids]
                pairs.append({'block':block,'comparison':label,'left':left,'right':right_name,'split':split,'cohort':cohort,'rows':len(ids),
                  'pooled_relative_error_difference':float(delta.sum()/den[ids].sum()),
                  'source_cluster':clustered(delta/den[ids].clip(1e-12),[rows[i]['source_group'] for i in ids])})
    last=[r for r in visibility if r['anchor']==1]
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'frozen_sha256':sha(out/'frozen.json'),
      'natural_rows':len(rows)-128,'natural_unique_native_inputs':127,'natural_unit':'128 source occurrences, including two identical model inputs in one source document; 127 actual native forwards plus one audited field alias',
      'identity_audit':'signed_source/identity_recovery/result.json','prospective_program_rows':128,'selected':frozen['selected'],'reports':reports,'paired':pairs,
      'visibility_counts':dict(Counter(r['cohort']+'/'+r['split']+'/'+str(r['any_connected_visible']) for r in last)),
      'seconds':time.monotonic()-start,
      'limits':['Direct MLP state predictions are not candidate digit scores or proof of improved free generation.',
        'Natural fit to prospective programs is a cross-domain test, not program-trained transfer.',
        'Original-plus-signed mixes normalized kernels at fixed equal weights, not a learned semantic composition operation.',
        'Only new natural/prospective program results are heldout for this algorithm addition; previous natural confirmation was already inspected.',
        'Observed improvements, if any, do not identify the original training cause or prove adequate full autoregressive state.']}
    save(out/'result.json',result);ledger('signed_source_frozen_prediction_confirmation',result['seconds']);guard()
    print('SIGNED_SOURCE_CONFIRMATION_COMPLETE',len(rows),frozen['selected'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    assert args.freeze;freeze()
