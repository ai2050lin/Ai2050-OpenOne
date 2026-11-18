"""Frozen source/output factorial decomposition and held-out prediction."""
import argparse,re,time
import numpy as np
import source_output_core as S
from source_output_core import C
from source_output_runner import admitted,check_analysis
from state_formation_math import bootstrap

def ridge(x,y):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1;v=np.column_stack([np.ones(len(x)),(x-mu)/sd]);pen=np.eye(v.shape[1]);pen[0,0]=0
    return {'mu':mu.tolist(),'sd':sd.tolist(),'coef':np.linalg.solve(v.T@v+pen,v.T@y).tolist()}

def predict(x,fit):return np.column_stack([np.ones(len(x)),(x-fit['mu'])/fit['sd']])@np.asarray(fit['coef'])

def null_p(x,seed=C.SEED):
    x=np.asarray(x,float);rng=np.random.default_rng(seed);obs=abs(x.mean());values=np.abs((rng.choice([-1.,1.],size=(2048,len(x)))@x)/len(x))
    return float((1+np.sum(values>=obs))/2049)

def freeze():
    p=C.checked();assert not list(C.OUT.glob('native-calibration-[0-9]*.npz')) and not (C.OUT/'behavior-confirm.json').exists()
    names=['source_output_analysis.py','source_output_causal.py','record_selection_match.py','state_formation_math.py']
    C.save('analysis-plan.json',{'schema':'source_output_analysis_plan.v1','base_plan_sha256':C.sha(C.OUT/'plan.json'),
        'sources':{'tests/glm5/'+name:C.sha(C.ROOT/'tests/glm5'/name) for name in names},
        'primary':'fixed-candidate effects and response source/output factors; candidates never re-ranked',
        'factorial':'For each family/task/alphabet, beta_source=mean(target_sign*state), beta_output=mean(target_sign*code_sign*state), over four status_position x mapping cells.',
        'pre_codebook_negative':'beta_output is exactly zero before the codebook; h0 final token is identical',
        'prediction':'Per branch, ridge lambda1 on frozen native8 or hashed token256; compare 8-cell parsed source/map/language means and constant. No confirm refit.',
        'native_inputs':'two selected-head output-pair local writes, two physical source mass differences, one selected-unit local write, visible codebook sign, query language, input length/100',
        'unit_probes':'B26 fixed32 actual activation; independent source/output linear readouts before/after codebook; no coordinate selection',
        'causal_primary':'mean target-margin loss>=.2 and positive 99.1667% frame-bootstrap advantage over random norm control and two matched native directions; all controls must have <=2% norm error and scale<=4',
        'specificity':'Within identical families and alphabet compare status/time to explicit-number task; equivalence only if difference CI lies in [-.1,.1], otherwise no equality claim',
        'branch_gate':'all predeclared cells must pass .90 both strict-pair and full-top1; stop whole branch if it fails; no sample filtering',
        'formal_promotion':False})
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4);index=C.read(C.MODEL/'model.safetensors.index.json')['weight_map']
    def tensor(name):
        with safe_open(C.MODEL/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name)
    e=tensor('model.embed_tokens.weight');w=torch.stack([e[ids[0]].float()-e[ids[1]].float() for ids in p['answer_ids'].values()]);del e
    wo=tensor('model.layers.29.self_attn.o_proj.weight').float();down=tensor('model.layers.26.mlp.down_proj.weight').float()
    C.save_npz('weights.npz',heads=(w@wo).reshape(2,32,128).numpy(),units=(w@down).numpy(),output_reader=w.numpy())
    C.save('weights.json',{'sha256':C.sha(C.OUT/'weights.npz'),'model_snapshot':C.SNAP,'scope':'local linear writes before downstream layers and final RMSNorm'})

def token_features(tokens):
    tokens=np.asarray(tokens,np.uint64);index=((tokens*2654435761)^(tokens>>7))%256
    return np.bincount(index.astype(int),minlength=256)/np.sqrt(len(tokens))

def parsed(row):
    """A task-conditioned text parser, not a model-output or answer-label lookup."""
    lines=row['prompt'].splitlines();i=next(i for i,line in enumerate(lines) if line.startswith('Two records describe'))
    intro=lines[i];records=lines[i+1:i+3];query=lines[i+3]
    if row['task']=='status':
        valid=re.search(r'Approval (.*?) means valid;',intro).group(1)
        selected=[bool(re.search(r'(?<!\w)'+re.escape(valid)+r'(?!\w)',v)) for v in records];assert sum(selected)==1;source=1 if selected[0] else -1
    elif row['task']=='time':
        dates=[re.search(r'\d{4}-\d{2}-\d{2}',v).group() for v in records];assert dates[0]!=dates[1];source=1 if dates[0]>dates[1] else -1
    else:
        values=set(re.findall(r'(?<!\d)[12](?!\d)',query));assert len(values)==1;source=1 if values.pop()=='1' else -1
    label=re.search(r'Output codebook: record 1 -> ([12AB]);',row['prompt']).group(1);mapping=1 if label in ['1','A'] else -1
    language=-1 if re.search(r'[\u4e00-\u9fff]',query) else 1
    assert (source,mapping,language)==(row['target_sign'],row['code_sign'],row['language'])
    return source,mapping,language

def cell(row):
    source,mapping,language=parsed(row)
    return (source==1)*4+(mapping==1)*2+(language==1)

def data(split):
    p=C.checked();check_analysis();active=admitted(split);rows=C.read(C.OUT/(split+'-materials.json'))['rows'];weights=np.load(C.OUT/'weights.npz')
    assert C.sha(C.OUT/'weights.npz')==C.read(C.OUT/'weights.json')['sha256'];units=p['fixed_candidates'][1]['indices'];result={};coordinates={};start=time.monotonic()
    for key in active:
        result[key]={k:[] for k in ['x','hash','units','y','target','output','cell','family','frame','variant','head_route']}
        coordinates[key]={name:np.zeros(shape) for name,shape in [('hidden_source',(37,2,2560)),('hidden_output',(37,2,2560)),('unit_source',(2,9728)),('unit_output',(2,9728)),('head_route',(2,32))]}
    if not active:return result,coordinates
    for f in range(len(rows)//24):
        meta=C.read(C.OUT/f'native-{split}-{f:04d}.json');sc=C.read(C.OUT/f'behavior-{split}-{f:04d}.json')['scores'];byindex={r['variant']:r for r in meta['records']}
        for i,row in enumerate(rows[f*24:(f+1)*24]):
            key=S.branch(row)
            if key not in active:continue
            item=byindex[i];path=C.OUT/item['path'];assert C.sha(path)==item['sha256'];a=S.ALPHABETS.index(row['alphabet'])
            with np.load(path) as z:
                h=C.unbits(z['hidden']);u=C.unbits(z['units']);head=C.unbits(z['head_context']);mass=z['source_mass'];t=row['target_sign'];o=row['output_sign']
                hv=(head[-1,[11,15]]*weights['heads'][a,[11,15]]).sum(-1);md=mass[-1,[11,15],0]-mass[-1,[11,15],1];uv=u[-1,units]@weights['units'][a,units]
                x=np.array([*hv,*md,uv,row['code_sign'],row['language'],len(row['tokens'])/100.])
                route=t*(mass[:,:,0]-mass[:,:,1]);d=result[key]
                values={'x':x,'hash':token_features(row['tokens']),'units':u[:,units],'y':sc[i]['pair_margin'],'target':t,'output':o,'cell':cell(row),'family':f,'frame':row['frame'],'variant':i,'head_route':route[:,[11,15]].mean(1)}
                for name,v in values.items():d[name].append(v)
                for name,v in [('hidden_source',t*h),('hidden_output',o*h),('unit_source',t*u),('unit_output',o*u),('head_route',route)]:coordinates[key][name]+=v
        if (f+1)%12==0:C.log('analyze_'+split,f+1,len(rows)//24,start)
    for key in active:
        result[key]={k:np.asarray(v) for k,v in result[key].items()}
        for name in coordinates[key]:coordinates[key][name]/=len(result[key]['y'])
        assert np.abs(coordinates[key]['hidden_output'][:,0]).max()==0,'Future codebook label effect before codebook'
        assert np.abs(coordinates[key]['unit_output'][0]).max()==0
    return result,coordinates

def fit():
    C.checked();check_analysis();assert not (C.OUT/'behavior-confirm.json').exists();dataset,coordinates=data('calibration');models={}
    for key,d in dataset.items():
        model={'native':ridge(d['x'],d['y']),'token_hash':ridge(d['hash'],d['y']),'constant':float(d['y'].mean()),
            'cells':[float(d['y'][d['cell']==i].mean()) for i in range(8)],'probes':{}}
        for position in [0,1]:
            for target in ['target','output']:model['probes'][f'{position}:{target}']=ridge(d['units'][:,position],d[target])
        models[key]=model
    C.save('predictors.json',{'schema':'source_output_predictors.v1','branches':models,'fitted_on':'calibration only','no_candidate_reranking':True})
    arrays={key.replace(':','_')+'__'+name:v.astype(np.float32) for key,group in coordinates.items() for name,v in group.items()}
    C.save_npz('calibration-coordinates.npz',**arrays)
    C.save('predictor-freeze.json',{'confirm_unopened':True,'artifacts':{name:C.sha(C.OUT/name) for name in ['predictors.json','calibration-coordinates.npz','analysis-plan.json','weights.npz']},
        'active_branches':list(models),'old_candidates_sha256':C.checked()['old_selection_sha256']})

def report():
    p=C.checked();check_analysis();dataset,coordinates=data('confirm');models=C.read(C.OUT/'predictors.json')['branches'];results=[]
    for key,d in dataset.items():
        model=models[key];pred={'constant':np.full(len(d['y']),model['constant']),'parsed8':np.array(model['cells'])[d['cell']],
            'token_hash':predict(d['hash'],model['token_hash']),'native8':predict(d['x'],model['native'])}
        families=np.unique(d['family']);frames=np.array([d['frame'][d['family']==f][0] for f in families])
        errors={name:np.array([((values[d['family']==f]-d['y'][d['family']==f])**2).mean() for f in families]) for name,values in pred.items()}
        probes={}
        for pos in [0,1]:
            for target in ['target','output']:
                score=predict(d['units'][:,pos],model['probes'][f'{pos}:{target}']);probes[f'{pos}:{target}']={'accuracy':float(np.mean(np.where(score>=0,1,-1)==d[target])),
                    'family_accuracy':[float(np.mean(np.where(score[d['family']==f]>=0,1,-1)==d[target][d['family']==f])) for f in families]}
        route=d['head_route'].reshape(len(families),4,2).mean(1)
        results.append({'branch':key,'families':len(families),'mse':{name:float(v.mean()) for name,v in errors.items()},
            'native_over_parsed':float(errors['native8'].mean()/errors['parsed8'].mean()),'native_gain':bootstrap(errors['parsed8']-errors['native8'],frames,alpha=p['alpha']),
            'native_label_accuracy':float(np.mean(np.where(pred['native8']>=0,1,-1)==d['output'])),'probes':probes,
            'head_route':[{'position':i,'mean':float(route[:,i].mean()),'interval':bootstrap(route[:,i],frames,alpha=p['alpha']),'family_swap_p':null_p(route[:,i])} for i in [0,1]],
            'prefix_output_effect_zero':True,'scope':'Whole prespecified branch, no family filtering; native predictor uses actual original activations.'})
    arrays={key.replace(':','_')+'__'+name:v.astype(np.float32) for key,group in coordinates.items() for name,v in group.items()}
    C.save_npz('confirm-coordinates.npz',**arrays)
    C.save('observations.json',{'schema':'source_output_observations.v1','branches':results,'coordinates_sha256':C.sha(C.OUT/'confirm-coordinates.npz'),
        'stopped_branches':C.read(C.OUT/'admission-confirm.json')['stopped_branches'],'formal_promotion':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','fit','report']);globals()[parser.parse_args().stage]()
