"""Retrospective full-coordinate state-formation analysis. No model forward pass.

Archive values remain immutable. Binary sensitivity gates do not turn previously
exposed records or four template frames into independent confirmation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,hashlib,json,math,shutil,struct,subprocess,time
from pathlib import Path
from datetime import datetime
import numpy as np
from tools import provenance
from tools.power_analysis import check_phase
from shared.dist_protocol.natural_register_runner import snapshot

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'tests/glm5/result/mlp_puzzle_v1'
OUT=ROOT/'tests/glm5/result/hiddenstate_components_v1'
MODEL=ROOT/'models/hf/qwen3-4b'
RUN='HIDDENSTATE-COMPONENTS-20261109-v1';SEED=202611090583
TERMS=['constant','query','record','language','query_language','record_language','color','color_language']
READOUTS=['record_end','answer_prefix'];COLS4=[0,3,6,7];DEPENDENCIES=[Path(__file__),ROOT/'tools/provenance.py',ROOT/'tools/power_analysis.py',ROOT/'shared/dist_protocol/natural_register_runner.py']

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def unbits(a):return (np.asarray(a,dtype=np.uint32)<<16).view(np.float32)
def design(rows):
    return np.array([[1,r['query'],r['record'],r['language'],r['query']*r['language'],r['record']*r['language'],r['query']*r['record'],r['query']*r['record']*r['language']] for r in rows],dtype=np.float64)
def factorial(h,x):return np.einsum('vf,lrvd->lrfd',x,h,optimize=True)/len(x)
def reconstruct(beta,x,cols=None):
    cols=list(range(8)) if cols is None else cols
    return np.einsum('vf,lrfd->lrvd',x[:,cols],beta[:,:,cols,:],optimize=True)
def probe(beta,var,factor,condition=None):
    center=beta[:,:,0].copy();direction=beta[:,:,factor].copy()
    if condition is not None:
        other,value,interaction=condition;center+=value*beta[:,:,other];direction+=value*beta[:,:,interaction]
    weight=direction/np.maximum(var,1e-6)
    denom=(direction*weight).sum(-1)
    weight=np.divide(weight,denom[:,:,None],out=np.zeros_like(weight),where=denom[:,:,None]>1e-12)
    return center,weight
def score(h,center,weight):return np.einsum('lrvd,lrd->lrv',h-center[:,:,None,:],weight,optimize=True)
def correct(scores,labels):return ((scores>=0)==(np.asarray(labels)[None,None,:]>0)).mean(-1)
def cosine(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide((a*b).sum(-1),den,out=np.zeros_like(den),where=den>1e-12)
def stable_onset(values,threshold=.9):
    # A descriptive onset of this decoder, not the first existence of information.
    for layer in range(1,34):
        if all(v>=threshold for v in values[layer:layer+3]):return layer
    return None
def state_step(source,beta_delta,color_scores,language_scores):
    c=np.where(color_scores>=0,1.,-1.);l=np.where(language_scores>=0,1.,-1.)
    return source+beta_delta[:,:,0,None,:]+l[:,:,:,None]*beta_delta[:,:,3,None,:]+c[:,:,:,None]*beta_delta[:,:,6,None,:]+(c*l)[:,:,:,None]*beta_delta[:,:,7,None,:]
def setup_provenance(snap):
    provenance.RESULTS=OUT/'provenance'
    provenance._git=lambda *args:subprocess.run(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,capture_output=True,text=True).stdout.strip()
    provenance.model_revision=lambda _:snap
def save(name,body,snap):
    OUT.mkdir(parents=True,exist_ok=True);setup_provenance(snap)
    helper=provenance.write_result(Path(name).stem,body,model_path=MODEL,seed=SEED,tool='hiddenstate_components')
    prov=read(helper)['provenance'];assert not provenance.check_provenance({'provenance':prov})
    doc={**body,'provenance':prov,'helper_artifact':str(helper.relative_to(ROOT))}
    raw=json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v).encode('utf-8')
    path=OUT/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(raw);tmp.replace(path)
    return doc
def log(stage,n,total,start,**extra):
    sec=time.monotonic()-start
    doc={'time':datetime.now().astimezone().isoformat(),'stage':stage,'done':n,'total':total,'seconds':round(sec,1),
         'eta_seconds':round(sec*(total-n)/n,1) if n else None,'gpu_used':False,**extra}
    print(json.dumps(doc,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(doc)+'\n')

def freeze():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    prior=read(SOURCE/'plan.json');snap=prior['model_snapshot'];assert snapshot(MODEL)==snap
    assert shutil.disk_usage(ROOT).free>64*2**20,'Need 64MiB free for bounded CPU analysis and client output'
    binary=check_phase(.10,512,p_d=.10,alpha=.05/74);assert binary['pass']
    native=read(SOURCE/'audit.json')['native_manifest'];assert len(native)==640
    sources={str(SOURCE.relative_to(ROOT)/name):sha(SOURCE/name) for name in ['plan.json','audit.json','discovery-materials.json','holdout-materials.json','holdout-behavior.json']}
    plan={'schema':'hiddenstate_components_plan.v1','kind':'real','run_id':RUN,'created_at':datetime.now().astimezone().isoformat(),'model_snapshot':snap,
      'status':'retrospective exploratory analysis of previously exposed materials; zero new blind confirmations',
      'sources':sources,'native_files':native,'dependencies':{str(p.relative_to(ROOT)):sha(p) for p in DEPENDENCIES},
      'runtime':{'numpy':np.__version__,'python':os.sys.version,'seed':SEED},
      'design':{'shape':['family','condition8','hidden_index37','readout2','native_coordinate2560'],
        'terms':TERMS,'readouts':READOUTS,'fit':'original discovery128 families only','evaluation':'original exposed holdout512 families, all8 conditions',
        'prediction_baselines':['constant','additive q,r,L','four-state color×language','eight-cell lookup'],
        'probes':'fixed diagonal-variance centroid: q,r,L,qr; color across languages; language across colors; product of q and r decoders; vector-norm-only comparator',
        'steps':'one-step h[l]→h[l+1] conditioned on observed source state; identity,global mean increment,four-state oracle,decoded-state increment; transitions to hs[36] excluded',
        'negative_controls':['record-end prefix identical across query/language','final embedding identical','future query/color/language invisible at record end'],
        'onset':'first of3 consecutive raw residual layers with decoder accuracy>=.90; descriptive,not earliest information or causal write',
        'lens':'FP32 final-RMSNorm readout through native tied embedding rows; hs36 already normalized; endpoint max absolute score error must <=0.25 nats else suppress lens interpretation',
        'claims':'full native coordinate response statistics and candidate source-state transitions; no attention/MLP attribution or causal promotion'},
      'power':{'paired_binary_sensitivity':binary,'expected_effect':.10,'required_families':binary['n_required'],'actual_families':512,
        'scope':'paired family sensitivity illustration under independent-family assumption; four template clusters and prior exposure preclude formal confirmation; no new intervention'},
      'boundaries':{'hs0':'embedding','hs1_to35':'residual after blocks0..34','hs36':'after block35 and final RMSNorm; not raw block35 write'},
      'resources':{'gpu':False,'max_new_outputs_MiB':40,'minimum_remaining_MiB':24,'streaming':True,'native_data_rewritten':False},
      'formal_evidence_promotion':False,'limits':['all materials exposed','four heldout frames; nouns correlate with frames','output-language instruction in English, not input-language separation','no style or grammar labels','only two readouts','no QK/VO/MLP module boundaries','one physical node','original acquisition admissibility gaps retained']}
    save('plan.json',plan,snap);print(json.dumps({'frozen':True,'run_id':RUN,'binary_sensitivity':binary}),flush=True)

def checked():
    p=read(OUT/'plan.json')
    for name,h in {**p['dependencies'],**p['sources']}.items():assert sha(ROOT/name)==h,name
    return p

def native_tensor(name,rows=None):
    index=read(MODEL/'model.safetensors.index.json')['weight_map'];path=MODEL/index[name]
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size));meta=header[name];assert meta['dtype']=='BF16'
        offset=8+size+meta['data_offsets'][0]
        if rows is None:f.seek(offset);raw=f.read(meta['data_offsets'][1]-meta['data_offsets'][0]);arr=np.frombuffer(raw,'<u2').reshape(meta['shape'])
        else:
            width=meta['shape'][1];pieces=[]
            for row in rows:f.seek(offset+row*width*2);pieces.append(np.frombuffer(f.read(width*2),'<u2'))
            arr=np.stack(pieces)
    return unbits(arr)

def lens(h,gamma,weight,choices,x):
    value=h.astype(np.float32).copy()
    value[:36]=value[:36]/np.sqrt(np.mean(value[:36]**2,-1,keepdims=True)+1e-6)*gamma
    logits=np.einsum('lrvd,kd->lrvk',value,weight,optimize=True)
    grouped=np.stack([logits[:,:,:,indices].max(-1) for indices in choices],-1)
    gold=(x[:,6]<0).astype(int)+2*(x[:,3]<0).astype(int)
    correct_group=(grouped.argmax(-1)==gold[None,None,:]).mean(-1)
    # Conditional content margin is diagnostic; it receives the requested output language.
    ids=np.arange(8);other=gold^1
    margin=grouped[:,:,ids,gold]-grouped[:,:,ids,other]
    return correct_group,margin.mean(-1),grouped

def analyze():
    p=checked();snap=p['model_snapshot'];start=time.monotonic();log('discovery',0,128,start)
    manifest={r['path']:r for r in p['native_files']};materials={s:read(SOURCE/f'{s}-materials.json') for s in ['discovery','holdout']}
    x=design(materials['discovery'][:8]);assert np.array_equal(x.T@x,8*np.eye(8))
    shape=(37,2,8,2560);total=np.zeros(shape);squares=np.zeros((37,2,2560));normcoef=np.zeros((37,2,8));seen=set();embedding=None
    def family(split,f):
        nonlocal embedding
        path=SOURCE/f'{split}-{f:04d}.npz';assert sha(path)==manifest[path.name]['sha256'],path.name
        rs=materials[split][f*8:f*8+8];assert np.array_equal(design(rs),x)
        for r in rs:
            identity=hashlib.sha256(canonical_tokens(r['tokens'])).hexdigest();assert identity not in seen;seen.add(identity)
        with np.load(path) as z:bits=z['hidden']
        assert bits.shape==(8,37,2,2560)
        for ix in [[0,1,4,5],[2,3,6,7]]:
            assert np.all(bits[ix,:,0]==bits[ix[0],:,0]),'Future factors visible at record end'
            prefixes=[rs[i]['tokens'][:rs[i]['positions'][0]+1] for i in ix]
            assert all(v==prefixes[0] for v in prefixes)
        if embedding is None:embedding=bits[0,0,1].copy()
        assert np.all(bits[:,0,1]==embedding)
        return unbits(bits).transpose(1,2,0,3),rs
    for f in range(128):
        h,rs=family('discovery',f);total+=factorial(h,x);squares+=np.mean(h.astype(np.float64)**2,2)
        normcoef+=np.einsum('vf,lrv->lrf',x,np.linalg.norm(h,axis=-1))/8
        if (f+1)%16==0:log('discovery',f+1,128,start)
    beta=total/128;var=np.maximum(squares/128-beta[:,:,0]**2,0);normcoef/=128
    np.savez_compressed(OUT/'frozen-coefficients.npz',beta=beta.astype('<f4'),variance=var.astype('<f4'),norm_coefficients=normcoef,
                        provenance=json.dumps(p['provenance']))
    frozen=save('estimator.json',{'schema':'hiddenstate_estimator.v1','run_id':RUN,'model_snapshot':snap,
       'plan_sha256':sha(OUT/'plan.json'),'coefficients_sha256':sha(OUT/'frozen-coefficients.npz'),'created_at':datetime.now().astimezone().isoformat(),
       'scope':'Discovery-fitted before reading holdout hidden values in this script. Entire source corpus exposed in earlier study; not a fresh blind lockbox.'},snap)
    probes={name:probe(beta,var,factor) for name,factor in [('query',1),('record',2),('language',3),('color',6)]}
    cross={}
    for sign in [-1,1]:
        cross[f'color_train_L{sign}']=probe(beta,var,6,(3,sign,7))
        cross[f'language_train_c{sign}']=probe(beta,var,3,(6,sign,7))
    predictions={name:reconstruct(beta,x,cols) for name,cols in [('constant',[0]),('additive',[0,1,2,3]),('four_state',COLS4),('eight_cell',None)]}
    prior=read(SOURCE/'plan.json');choice_ids=prior['choice_ids'];tokens=sorted({t for v in choice_ids.values() for t in v});choices=[[tokens.index(t) for t in choice_ids[k]] for k in ['en_red','en_blue','zh_red','zh_blue']]
    gamma=native_tensor('model.norm.weight');weight=native_tensor('model.embed_tokens.weight',tokens)
    delta=beta[1:36]-beta[:35]
    prediction_errors=np.zeros((512,37,2,4));transfer_errors=np.zeros((512,35,2,4));probe_values={};lens_accuracy=[];lens_margin=[];allcoef=np.zeros(shape);coefenergy=np.zeros((37,2,8));cosines=np.zeros((512,37,2,3));max_lens_error=0
    sigma=np.sqrt(var)
    log('exposed_holdout',0,512,start)
    for f in range(512):
        h,rs=family('holdout',f);cf=factorial(h,x);allcoef+=cf;coefenergy+=np.sum(cf**2,-1)
        cosines[f]=cosine(cf[:,:,[3,6,7]],beta[:,:,[3,6,7]])
        for j,(name,pred) in enumerate(predictions.items()):prediction_errors[f,:,:,j]=np.mean((h-pred)**2,(2,3))
        scores={name:score(h,*args) for name,args in probes.items()}
        vals={name:correct(scores[name],x[:,factor]) for name,factor in [('query',1),('record',2),('language',3),('color',6)]}
        vals['color_from_query_record']=correct(np.where(scores['query']>=0,1.,-1.)*np.where(scores['record']>=0,1.,-1.),x[:,6])
        magnitude=np.linalg.norm(h,axis=-1)
        for name,factor in [('color',6),('language',3)]:
            vals[name+'_norm_only']=correct((magnitude-normcoef[:,:,0,None])*normcoef[:,:,factor,None],x[:,factor])
        for sign in [-1,1]:
            keep=x[:,3]==-sign;vals[f'color_train_L{sign}']=correct(score(h,*cross[f'color_train_L{sign}'])[:,:,keep],x[keep,6])
            keep=x[:,6]==-sign;vals[f'language_train_c{sign}']=correct(score(h,*cross[f'language_train_c{sign}'])[:,:,keep],x[keep,3])
        for name,v in vals.items():probe_values.setdefault(name,[]).append(v)
        # Only transitions between genuine, unnormalized residual boundaries.
        source=h[:35];target=h[1:36]
        trans=[source,source+delta[:,:,0,None,:],source+reconstruct(delta,x,COLS4),state_step(source,delta,scores['color'][:35],scores['language'][:35])]
        for j,pred in enumerate(trans):transfer_errors[f,:,:,j]=np.mean((target-pred)**2,(2,3))
        acc,margin,grouped=lens(h,gamma,weight,choices,x);lens_accuracy.append(acc);lens_margin.append(margin)
        old=np.array([s['scores'] for s in read((SOURCE/f'holdout-{f:04d}.json'))['scores']]);max_lens_error=max(max_lens_error,float(np.max(np.abs(grouped[36,1]-old))))
        if (f+1)%32==0:log('exposed_holdout',f+1,512,start,free_MiB=round(shutil.disk_usage(ROOT).free/2**20))
        assert shutil.disk_usage(ROOT).free>24*2**20,'Disk reserve reached; raw archive left unchanged'
    mean=allcoef/512;energy=coefenergy/512;within=energy[:,:,1:].sum(-1);probe_arrays={k:np.array(v) for k,v in probe_values.items()}
    mse=prediction_errors.mean(0);steps=transfer_errors.mean(0);lens_acc=np.mean(lens_accuracy,0);lens_marg=np.mean(lens_margin,0)
    rows=[]
    for l in range(37):
        for ro,name in enumerate(READOUTS):
            components={}
            for i,factor in enumerate([3,6,7]):
                d=mean[l,ro,factor];norm=float(np.linalg.norm(d));components[TERMS[factor]]={'contrast_norm':norm,
                   'controlled_variance_fraction':float(norm**2/within[l,ro]) if within[l,ro]>1e-20 else None,
                   'discovery_transfer_cosine':float(cosine(d,beta[l,ro,factor])) if norm*np.linalg.norm(beta[l,ro,factor])>1e-12 else None,
                   'family_cosine_mean':float(cosines[:,l,ro,i].mean()),'family_cosine_p10':float(np.quantile(cosines[:,l,ro,i],.1)),
                   'previous_direction_cosine':float(cosine(d,mean[l-1,ro,factor])) if 0<l<36 and norm*np.linalg.norm(mean[l-1,ro,factor])>1e-12 else None,
                   'contrast_increment_norm':float(np.linalg.norm(d-mean[l-1,ro,factor])) if 0<l<36 else None,
                   'dominant_native_coordinates':np.argsort(-np.abs(d))[:8].tolist(),'display_scope':'top coordinates only as lookup anchors; all2560 remain in coefficient artifact'}
            row={'hidden_index':l,'readout':name,'boundary':p['boundaries']['hs0' if l==0 else 'hs36' if l==36 else 'hs1_to35'],
              'components':components,'probes':{k:float(a[:,l,ro].mean()) for k,a in probe_arrays.items()},
              'frame_color_accuracy':{str(k):float(probe_arrays['color'][np.arange(512)%4==k,l,ro].mean()) for k in range(4)},
              'prediction':dict(zip(predictions,[float(v) for v in mse[l,ro]])),
              'eight_over_four_mse':float(mse[l,ro,3]/mse[l,ro,2]) if mse[l,ro,2]>1e-20 else None,
              'four_over_constant_mse':float(mse[l,ro,2]/mse[l,ro,0]) if mse[l,ro,0]>1e-20 else None,
              'lens':{'four_choice_accuracy':float(lens_acc[l,ro]),'conditional_color_margin':float(lens_marg[l,ro]),'endpoint_max_error':max_lens_error,'calibrated':bool(max_lens_error<=.25)},
              'step':dict(zip(['identity','mean_increment','oracle_four_state','decoded_state'],[float(v) for v in steps[l,ro]])) if l<35 else None}
            rows.append(row)
    final=[r for r in rows if r['readout']=='answer_prefix'];onsets={name:stable_onset([r['probes'][name] for r in final]) for name in probe_arrays}
    failures=[]
    for label in ['language','color']:
        layer=onsets[label] or 35;acc=probe_arrays[label][:,layer,1];worst=np.argsort(acc)[:3]
        for f in worst:failures.append({'factor':label,'hidden_index':layer,'family':int(f),'accuracy':float(acc[f]),'prompt':materials['holdout'][int(f)*8]['prompt'],
                                      'scope':'worst family at chosen descriptive onset; not an independently selected confirmation'})
    final_native=mean[:,1,[3,6,7]];transition_rank=sorted(range(1,36),key=lambda l:-np.linalg.norm(mean[l,1,6]-mean[l-1,1,6]))
    np.savez_compressed(OUT/'analysis-arrays.npz',discovery_beta=beta.astype('<f4'),evaluation_beta=mean.astype('<f4'),family_prediction_mse=prediction_errors.astype('<f4'),
       family_transfer_mse=transfer_errors.astype('<f4'),family_probe_names=np.array(list(probe_arrays)),family_probe_accuracy=np.stack(list(probe_arrays.values())).astype('<f4'),
       provenance=json.dumps(p['provenance']))
    doc={'schema':'hiddenstate_components_report.v1','kind':'real','run_id':RUN,'model_snapshot':snap,'rows':rows,'onsets':onsets,
       'plan_sha256':sha(OUT/'plan.json'),'estimator_sha256':sha(OUT/'estimator.json'),'arrays_sha256':sha(OUT/'analysis-arrays.npz'),
       'source_run':prior['run_id'],'families':{'discovery':128,'exposed_evaluation':512},'unique_inputs':len(seen),'verified_native_files':640,
       'negative_controls':{'record_end_future_factors_bitwise_invariant':True,'final_embedding_bitwise_constant':True,
         'record_end_color_accuracy':{str(l):float(probe_arrays['color'][:,l,0].mean()) for l in range(37)},
         'record_end_language_accuracy':{str(l):float(probe_arrays['language'][:,l,0].mean()) for l in range(37)}},
       'color_increment_rank_raw_only':transition_rank,'counterexamples':failures,'formal_evidence_promotion':False,
       'limits':p['limits']+['contrast fractions describe balanced task response variance,not semantic information percentages',
         'decoder onset is not information onset; product decoder may compute an operation itself',
         'one-step transition takes actual source state; does not establish a closed last-token dynamical system',
         'FP32 logit lens is a diagnostic using final RMSNorm and output weights,not the actual remaining network',
         'factor variables and color=qr come from authored design; they are not unsupervised discoveries',
         'all module-specific writes and causal effects remain unmeasured in this analysis']}
    save('report.json',doc,snap);log('complete',640,640,start,onsets=onsets,lens_error=max_lens_error)

def canonical_tokens(tokens):return json.dumps(tokens,separators=(',',':')).encode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['freeze','analyze']);args=parser.parse_args()
    globals()[args.stage]()
if __name__=='__main__':main()
