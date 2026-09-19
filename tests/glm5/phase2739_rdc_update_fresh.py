"""Independent new natural content positions and amplitude-matched graph controls."""
import argparse, gc
from collections import defaultdict,Counter
from rdc_update_common import *

def material():
    from transformers import AutoTokenizer
    from phase2728_rdc_law_material import natural_candidates
    from phase2732_rdc_binding_material import connected
    from phase2736_rdc_update_material import content_anchors
    out=BASE/'fresh_graph'
    if (out/'material.json.gz').exists():return gzread(out/'material.json.gz')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    old=prior_natural()+gzread(LAW/'material.json.gz')+gzread(LAW/'confirmation_material.json.gz')+gzread(BASE/'natural_material.json.gz')
    used={c for r in old for c in r.get('component_ids',[])};texts={tuple(r['prompt_ids']) for r in old}
    forbidden={r['source_group'] for r in old if r.get('split')=='train'};rows=[];manifest=[]
    for bank in ('gum','ewt'):
        candidates=[r for part in ('test','dev') for r in natural_candidates(tok,bank,part,manifest)]
        for r in candidates:r['connected_held']=connected(r)
        for want in (True,False):
            grouped=defaultdict(list)
            for r in sorted(candidates,key=lambda r:ranked('fresh2739/'+r['source_id'])):
                if bool(r['connected_held'])==want and r['source_group'] not in forbidden and content_anchors(r):grouped[r['source_group']].append(r)
            selected=[]
            for round_index in range(64):
              for group in sorted(grouped,key=ranked):
                eligible=[r for r in grouped[group] if not set(r['component_ids'])&used and tuple(r['prompt_ids']) not in texts]
                if not eligible:continue
                r=eligible[0];a=content_anchors(r);sid='u2739_'+ranked(bank+'/'+'/'.join(r['component_ids']))[:20]
                row=dict(r,sample_id=sid,cohort=bank,split='fresh_connected' if want else 'fresh_matched',capture_mode='fresh_update',
                  anchors=[x[0] for x in a],target_positions=[x[0]+1 for x in a],
                  content_boundary_annotations=[{'position':p,'next_word':w,'upos':u,'relation':rel} for p,w,u,rel in a],
                  novelty='New component IDs and native inputs against explicit prior inventory including2736; same public corpus, no global historical exposure guarantee.')
                selected.append(row);used.update(r['component_ids']);texts.add(tuple(r['prompt_ids']))
                if len(selected)==32:break
              if len(selected)==32:break
            assert len(selected)==32,(bank,want,len(selected));rows.extend(selected)
    assert len(rows)==128 and len({tuple(r['prompt_ids']) for r in rows})==128
    compressed(out/'material.json.gz',rows);save(out/'material_audit.json',{'timestamp':stamp(),'source':snapshot(__file__),'rows':128,
      'documents':len({r['source_group'] for r in rows}),'components_disjoint_against':len(old),'official_partitions':dict(Counter(r['source_key'] for r in rows)),
      'recovery':'material_recovery.json: EWT official test pool was exhausted; unused dev sources admitted before predictor freeze or outcomes.',
      'source_manifest':manifest,'sha256':sha(out/'material.json.gz')})
    return rows

def kernel(pk,right=None):
    import torch
    from rdc_update_graph import kernels
    same=right is None;right=pk if same else right
    q=(pk['q']@right['q'].T+pk['e']@right['e'].T)/(2*2560);out={'query':1+q}
    for p in (pk,right):
        if 'position_A' in p:continue
        t=p['a'].shape[-1];p['position_A']=torch.zeros_like(p['a'])
        for i,n0 in enumerate(p['length']):
            n=int(n0);z=torch.arange(n,device=q.device);scores=-(z[:,None]-z[None,:]).abs().float()/max(n/8,1)
            scores.fill_diagonal_(-torch.inf);p['position_A'][i,:n,:n]=scores.softmax(-1)
    for name in ('directed_rms','shuffled_rms','position_rms','square_rms'):out[name]=torch.empty_like(q)
    for i in range(0,len(pk['q']),8):
      for j in range(i if same else 0,len(right['q']),8):
        dot=torch.einsum('atd,bsd->abts',pk['unit'][i:i+8],right['unit'][j:j+8])/2560
        den=pk['length'][i:i+8,None]*right['length'][None,j:j+8]
        for name,aa in [('directed_rms','a'),('shuffled_rms','shuffled'),('position_rms','position_A'),('square_rms',None)]:
            term=dot if aa is None else pk[aa][i:i+8,None].transpose(-1,-2)@dot@right[aa][None,j:j+8]
            s=(term*dot).sum((-1,-2))/den;z=1+q[i:i+8,j:j+8]+s+q[i:i+8,j:j+8]*s
            out[name][i:i+8,j:j+8]=z
            if same:out[name][j:j+8,i:i+8]=z.T
    return out

def freeze():
    import torch
    from rdc_update_graph import pack
    from rdc_law_predict import ridge_lambda
    from rdc_law_native import parameter
    from phase2736_rdc_update_prediction import decoders,reports
    out=BASE/'fresh_graph';rows_fresh=material();start=time.monotonic()
    if (out/'frozen.json').exists():return
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=gzread(PRIOR/'natural_discovery.json.gz')
    with np.load(BASE/'graph/head_mapping.npz') as z:coef=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
    pk,targets,audit=pack(rows,coef);kk=kernel(pk);del pk
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train']);val=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
    ti=torch.tensor(train,device='cuda');groups=Counter(rows[i]['source_group'] for i in train)
    weight=torch.tensor([1/groups[rows[i]['source_group']] for i in train],device='cuda');weight/=weight.sum();sw=weight.sqrt();checks=[]
    for b,target in targets.items():
        yy=torch.tensor(target,device='cuda');center=weight@yy[ti];yc=yy[ti]-center
        decoder=read(BASE/'graph/frozen.json')['selected'][str(b)]['decoder']
        w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]}
        den=(yy[:,:2560]-center[:2560]).square().sum(-1).cpu().numpy()
        for name,k in kk.items():
            scale=k[ti,ti].mean();kn=k/scale;kt=(kn[ti][:,ti]*sw[:,None]*sw[None,:]).double();eig,vec=torch.linalg.eigh((kt+kt.T)*.5)
            assert eig.min()>-1e-6*eig.max();lam,df=ridge_lambda(eig,128)
            co=sw[:,None]*(((vec/(eig+lam)[None,:])@vec.T).float()@(sw[:,None]*yc));pred=decoders(kn[:,ti]@co+center,w)[decoder]
            err=(pred-yy[:,:2560]).square().sum(-1).cpu().numpy()
            npz(out/'banks'/f'b{b}_{name}.npz',coefficients=co.cpu().numpy(),center=center.cpu().numpy(),scale=np.array(float(scale)))
            checks.append({'block':b,'kernel':name,'decoder':decoder,'df':df,'lambda':lam,'validation_relative_MSE':float(err[val].sum()/den[val].sum()),'reports':reports(rows,err,den)})
        del yy,w
    immutable(out/'frozen.json',{'timestamp':stamp(),'source':snapshot(__file__),'material_sha256':sha(out/'material.json.gz'),
      'head_mapping_sha256':sha(BASE/'graph/head_mapping.npz'),'fit_material_sha256':sha(PRIOR/'natural_discovery.json.gz'),
      'banks_sha256':{p.name:sha(p) for p in (out/'banks').glob('*.npz')},'validation':checks,
      'selection':'All5 amplitude-matched candidates predeclared; df128 and decoder inherited from2736. No new heldout winner selection; original2736 selection is unchanged.',
      'seconds':time.monotonic()-start});ledger('amplitude_matched_controls_freeze',time.monotonic()-start)

def capture():
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    from phase2728_rdc_law_capture import Observer
    out=BASE/'fresh_graph';rows=material();start=time.monotonic()
    if (out/'capture_result.json').exists():return
    assert read(out/'frozen.json')['material_sha256']==sha(out/'material.json.gz')
    model,tok=load_native('qwen4');model.eval();torch.set_num_threads(2);o=Observer(model);moments={};reports=[]
    try:
      with torch.inference_mode():
        for i,row in enumerate(rows):
            p=row['anchors'];o.reset(p,False,i<2);o.enabled=True
            post=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=False).last_hidden_state;o.enabled=False
            fields={'H':np.stack([o.H[j] for j in range(37)]),'H12_sources':o.source_H12,'embedding':o.H[0],**o.factors,
              'positions':np.array(p),'postnorm':bits(post[0,p]),'token_ids':np.array(row['prompt_ids'],np.int32)}
            k=row['cohort']+'_'+row['split'];m=np.stack([o.moments[j] for j in range(37)]).astype(float);moments[k]=moments.get(k,np.zeros_like(m))+m
            path=out/'fields'/f'{row["sample_id"]}.npz';npz(path,**fields)
            rec={'sample_id':row['sample_id'],'array_sha256':sha(path),'alltoken_layer_identities':o.H_hashes,'checks':o.checks};save(out/'commits'/f'{row["sample_id"]}.json',rec);reports.append(rec)
            del post,fields;o.reset([],False);guard(5*1024**2)
            if (i+1)%32==0:print('FRESH_GRAPH_CAPTURE',i+1,128,flush=True)
        for k,m in moments.items():npz(out/'cohort_moments'/f'{k}.npz',moments=m)
        result={'timestamp':stamp(),'source':snapshot(__file__),'rows':128,'anchors':384,'seconds':time.monotonic()-start,'reports':reports}
        save(out/'capture_result.json',result);ledger('fresh_natural_content_capture',result['seconds'])
    except Exception as exc:failure(out,start,exc);raise
    finally:o.close();del model,o;gc.collect();torch.cuda.empty_cache()

def confirm():
    import torch
    from rdc_update_graph import pack,save_pack
    from rdc_law_native import parameter
    from phase2736_rdc_update_prediction import decoders,reports
    out=BASE/'fresh_graph';start=time.monotonic()
    if (out/'result.json').exists():return
    frozen=read(out/'frozen.json');assert frozen['material_sha256']==sha(out/'material.json.gz')
    for name,digest in frozen['banks_sha256'].items():assert sha(out/'banks'/name)==digest
    torch.set_num_threads(2);rows=[dict(r,anchors=[p],field_anchor=a) for r in material() for a,p in enumerate(r['anchors'])]
    train=[r for r in gzread(PRIOR/'natural_discovery.json.gz') if r['split']=='train']
    with np.load(BASE/'graph/head_mapping.npz') as z:coef=torch.tensor(z['coefficients'],device='cuda',dtype=torch.float32)
    left,targets,audit=pack(rows,coef);right,_,_=pack(train,coef);save_pack(out/'confirmation',rows,left,audit);kk=kernel(left,right);del left,right
    resultrows=[];pairs=[]
    for b,target in targets.items():
        yy=torch.tensor(target[:,:2560],device='cuda');decoder=read(BASE/'graph/frozen.json')['selected'][str(b)]['decoder']
        w={k:parameter(f'model.layers.{b}.mlp.{name}_proj.weight') for k,name in [('g','gate'),('u','up'),('d','down')]};errors={}
        for name,k in kk.items():
            with np.load(out/'banks'/f'b{b}_{name}.npz') as z:
                center=torch.tensor(z['center'],device='cuda');pred=k/float(z['scale'])@torch.tensor(z['coefficients'],device='cuda')+center
            pp=decoders(pred,w)[decoder];err=(pp-yy).square().sum(-1).cpu().numpy();den=(yy-center[:2560]).square().sum(-1).cpu().numpy();errors[name]=err
            npz(out/'errors'/f'b{b}_{name}.npz',squared_error=err,baseline_squared_error=den)
            resultrows.append({'block':b,'kernel':name,'decoder':decoder,'reports':reports(rows,err,den)})
            if name=='directed_rms':npz(out/'predictions'/f'b{b}.npz',prediction=pp.cpu().numpy(),actual=yy.cpu().numpy())
        for name in kk:
            if name=='directed_rms':continue
            for split,cohort in sorted({(r['split'],r['cohort']) for r in rows}):
                ix=[i for i,r in enumerate(rows) if (r['split'],r['cohort'])==(split,cohort)];delta=(errors['directed_rms'][ix]-errors[name][ix])/den[ix].clip(1e-12)
                pairs.append({'block':b,'control':name,'split':split,'cohort':cohort,'source_cluster_advantage':clustered(delta,[rows[i]['source_group'] for i in ix]),
                  'pooled_relative_MSE_difference':float((errors['directed_rms'][ix]-errors[name][ix]).sum()/den[ix].sum())})
        del w,yy
    result={'timestamp':stamp(),'source':snapshot(__file__),'windows':128,'boundaries':384,'frozen_sha256':sha(out/'frozen.json'),
      'records':resultrows,'matched_controls':pairs,'seconds':time.monotonic()-start,
      'scope':'New source components/inputs after all5 RMS controls were frozen. Directed versus shuffled uses identical amplitude normalization. Family/depth/general-world-language claims remain outside this natural corpus test.'}
    save(out/'result.json',result);ledger('fresh_matched_graph_confirmation',result['seconds']);print('FRESH_MATCHED_DONE',result['seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['material','freeze','capture','confirm'],required=True);a=p.parse_args()
    globals()[a.stage]()
