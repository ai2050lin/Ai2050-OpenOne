"""Native full-prefix observations, frozen forecasts, upstream interventions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import json
import shutil
from time import monotonic
import numpy as np
import torch
from scipy.stats import binomtest, beta
from transformers import AutoTokenizer
import mlp_trigger_design as D
import mlp_condition_run as OLD
import mlp_condition_design as OD
import mlp_condition_stream_load as LOADER
from mlp_b34_confirm_run import bits,norm_match
from shared.dist_protocol.mechanism_trace import GpuLease,unbits

class Assay(OLD.Assay):
    def vector(self,b,f):
        def hook(mod,args,out):
            if self.mode in ('base','suffix'):self.data[f'{f}{b}']=out[:,:,OLD.UNITS[b]].clone()
        return hook

    def base(self,row):
        return self.observe([{**row,'positions':[len(row['tokens'])-1]}])

    def selected(self,base,row):
        positions={'broad':len(row['tokens'])-1,'rare':64+int(base['R33'][0,64:,4].abs().argmax())}
        return positions

    def edited(self,base,b,u,pos,indices,dose,clamp=None,control=None):
        r=base[f'R{b}'].clone()
        before=r[0,pos].clone()
        if control is None:r[0,pos,indices]=(r[0,pos,indices].float()*(1-dose)).to(r.dtype)
        else:r[0,pos]=control
        aa=self.layers[b].mlp.act_fn(self.layers[b].mlp.gate_proj(self.layers[b].post_attention_layernorm(r)))*self.layers[b].mlp.up_proj(self.layers[b].post_attention_layernorm(r))
        if clamp=='unit':aa[0,pos,u]=base[f'A{b}'][0,pos,u]
        write=self.layers[b].mlp.down_proj(aa)
        if clamp=='mlp':write=self.layers[b].mlp.down_proj(base[f'A{b}'])
        state=base[f'H{b}'].clone();state[0,pos]=r[0,pos]+write[0,pos]
        return state,r[0,pos],aa[0,pos,u],float((r[0,pos].float()-before.float())[indices].sum())

    def unit_zero(self,base,b,u,pos):
        aa=base[f'A{b}'].clone();aa[0,pos,u]=0
        h=base[f'H{b}'].clone();h[0,pos]=base[f'R{b}'][0,pos]+self.layers[b].mlp.down_proj(aa)[0,pos]
        return h,-float(base[f'A{b}'][0,pos,u])

def configure(p):
    OD.OUT=D.OUT;OD.checked=lambda:p;OLD.save=lambda name,body,plan:D.save(name,body,plan)

def log(stage,n,total,start):OLD.log(stage,n,total,start)

def nuisance(row,pos):
    return [1,row['long'],row['repetition'],row['past'],row['role'],row['mapping'],
            int(row['identity']==1),int(row['identity']==2),pos/len(row['tokens']),len(row['tokens'])/1000]

def arrays(split,p):
    rows=D.materials(p,split);allr=[];ally=[];nuis=[];meta=[]
    for f in range(p['materials'][split]['families']):
        z=np.load(D.OUT/f'{split}-{f:04d}.npz');m=D.READ(D.OUT/f'{split}-{f:04d}.json')
        assert D.HASH(D.OUT/f'{split}-{f:04d}.npz')==m['native_sha256']
        rr=unbits(z['residual']);y=[];xx=[]
        for k,row in enumerate(rows[f*8:f*8+8]):
            yy=[];nn=[]
            for pair,(b,u,t,v) in D.PAIRS.items():
                pos=m['rows'][k]['positions'][pair];ids=list(OLD.UNITS)
                yy.append([float(unbits(z['prefix'])[k,ids.index(i),2,pos]) for i in (b,t)])
                nn.append(nuisance(row,pos))
            y.append(yy);xx.append(nn)
        allr.append(rr);ally.extend(y);nuis.extend(xx);meta.extend(m['rows'])
    return rows,np.concatenate(allr),np.asarray(ally),np.asarray(nuis),meta

def collect(model,p,split):
    if split=='holdout':assert (D.OUT/'frozen.json').exists()
    rows=D.materials(p,split);n=p['materials'][split]['families'];assay=Assay(model);start=monotonic();log(split,0,n,start)
    try:
        for f in range(n):
            file=D.OUT/f'{split}-{f:04d}.npz'
            if file.with_suffix('.json').exists():continue
            prefix=[];residual=[];r4fields=[];records=[]
            chunk=rows[f*8:f*8+8];maxlen=max(len(r['tokens']) for r in chunk)
            for row in chunk:
                logits,base=assay.base(row);pos=assay.selected(base,row)
                a=np.zeros((4,3,maxlen),np.uint16);r4=np.zeros((4,maxlen),np.uint16)
                for i,(b,u) in enumerate(OLD.UNITS.items()):
                    a[i,:,:len(row['tokens'])]=np.stack([bits(base[f'G{b}'][0]),bits(base[f'V{b}'][0]),bits(base[f'A{b}'][0,:,u])])
                    r4[i,:len(row['tokens'])]=bits(base[f'R{b}'][0,:,4])
                r4fields.append(r4);prefix.append(a);residual.append(np.stack([bits(base[f'R{b}'][0,pos[k]]) for k,(b,*_) in D.PAIRS.items()]))
                rr=base['R33'][0].float();aa=base['A33'][0,:,1521].float()
                event=(aa[64:].abs()>p['gates']['rare_threshold_A']).nonzero().flatten()+64
                records.append({'id':row['id'],'positions':pos,'scores':logits[0,0,p['choice_ids']].float().tolist(),
                   'top1':int(logits[0,0].argmax()),'rare_positions':event.tolist(),
                   'rare_R4_energy':(rr[event,4].square()/rr[event].square().sum(-1)).tolist(),
                   'rare_A':aa[event].tolist(),'R4_max':float(rr[64:,4].abs().max()),
                   'rare_count_first_600':int((aa[64:600].abs()>10).sum()),
                   'bos_A33':float(aa[0])})
            np.savez_compressed(file,prefix=np.stack(prefix),residual=np.stack(residual),R4=np.stack(r4fields))
            D.save(file.with_suffix('.json').name,{'rows':records,'native_sha256':D.HASH(file)},p)
            assert shutil.disk_usage(D.ROOT).free>p['resources']['free_floor_bytes'],'Disk floor reached'
            if (f+1)%8==0:log(split,f+1,n,start)
            assert monotonic()-start<60*p['resources']['stage_minutes']
    finally:assay.close()

def fit(x,y):
    mean=x.mean(0);std=x.std(0);std[std<1e-8]=1;mean[0]=0;std[0]=1
    a=(x-mean)/std;pen=np.eye(a.shape[1]);pen[0,0]=0
    return {'mean':mean.tolist(),'std':std.tolist(),'coef':np.linalg.solve(a.T@a+pen,a.T@y).tolist()}

def predict(model,x):return ((x-np.asarray(model['mean']))/np.asarray(model['std']))@np.asarray(model['coef'])

def select(p):
    rows,r,y,nuis,meta=arrays('discovery',p)
    behavior={}
    for frame in range(16):
        ii=[i for i,v in enumerate(rows) if v['frame']==frame]
        behavior[str(frame)]=float(np.mean([np.argmax(meta[i]['scores'][:2])==rows[i]['gold'] for i in ii]))
    assert min(behavior.values())>=p['gates']['behavior_frame_min'],behavior
    selected={};models={}
    for i,key in enumerate(D.PAIRS):
        if key=='rare':idx=[4]
        else:
            x=r[:,i].astype(np.float64);xc=x-x.mean(0);yc=y[:,i]-y[:,i].mean(0)
            cor=(xc.T@yc)/(np.sqrt((xc**2).sum(0))[:,None]*np.sqrt((yc**2).sum(0))[None,:]+1e-20)
            idx=np.argsort(abs(cor[:,0]*cor[:,1]))[-8:].tolist()
        selected[key]=idx
        models[key]={
          'nuisance':fit(nuis[:,i],y[:,i]),
          'source':fit(np.column_stack([nuis[:,i],y[:,i,0]]),y[:,i,1:]),
          'common':fit(np.column_stack([nuis[:,i],r[:,i,idx]]),y[:,i])}
    D.save('selection.json',{'coordinates':selected,'predictors':models,'behavior':behavior,
        'discovery_files':{str(f):D.HASH(D.OUT/f'discovery-{f:04d}.npz') for f in range(D.N_DISC)}},p)
    print({'selected':selected,'behavior_min':min(behavior.values())},flush=True)

def distribution(logits,base,pred,ids):
    q=logits[0,0].double().log_softmax(-1);b=base[0,0].double().log_softmax(-1);r=pred.double().log_softmax(-1)
    return {'scores':logits[0,0,ids].float().tolist(),'top1':int(q.argmax()),
            'kl_to_baseline':float((q.exp()*(q-b)).sum()),'kl_to_prediction':float((q.exp()*(q-r)).sum())}

def calibrate(model,p):
    sel=D.READ(D.OUT/'selection.json');rows=D.materials(p,'discovery');assay=Assay(model);start=monotonic()
    sums=np.zeros((2,2,model.config.vocab_size),np.float64);targets=np.zeros((2,2));den=np.zeros((2,2));audit=[]
    log('calibration',0,D.N_DISC,start)
    try:
        for f in range(D.N_DISC):
            row=rows[f*8+(f//16)%8];original,base=assay.base(row);pos=assay.selected(base,row)
            for i,(key,(b,u,t,v)) in enumerate(D.PAIRS.items()):
                assert torch.equal(assay.suffix(base[f'H{b}'],b+1),original),'Zero replay mismatch'
                for k,mode in enumerate(['upstream','unit']):
                    if mode=='upstream':h,_,_,delta=assay.edited(base,b,u,pos[key],sel['coordinates'][key],.5)
                    else:h,delta=assay.unit_zero(base,b,u,pos[key])
                    out=assay.suffix(h,b+1);change=(out-original).float()[0,0].cpu().numpy();change-=change.mean()
                    sums[i,k]+=delta*change;den[i,k]+=delta**2
                    targets[i,k]+=delta*float(assay.data[f'A{t}'][0,pos[key],v].float()-base[f'A{t}'][0,pos[key],v].float())
            audit.append(row['id'])
            if (f+1)%8==0:log('calibration',f+1,D.N_DISC,start)
        assert (den>0).all()
        np.savez_compressed(D.OUT/'calibration.npz',logit=sums/den[:,:,None],target=targets/den)
        D.save('frozen.json',{'selection_sha256':D.HASH(D.OUT/'selection.json'),'calibration_sha256':D.HASH(D.OUT/'calibration.npz'),
             'checks':audit,'heldout_unopened':not list(D.OUT.glob('holdout-[0-9]*.npz'))},p)
    finally:assay.close()

def intervene(model,p):
    frozen=D.READ(D.OUT/'frozen.json');sel=D.READ(D.OUT/'selection.json')
    assert frozen['selection_sha256']==D.HASH(D.OUT/'selection.json')
    assert frozen['calibration_sha256']==D.HASH(D.OUT/'calibration.npz')
    cal=np.load(D.OUT/'calibration.npz');rows=D.materials(p,'holdout');assay=Assay(model);start=monotonic();log('intervention',0,D.N_HOLD,start)
    try:
        for f in range(D.N_HOLD):
            if (D.OUT/f'causal-{f:04d}.json').exists():continue
            row=rows[f*8+(f//16)%8];original,base=assay.base(row);positions=assay.selected(base,row);records=[];commits=[]
            for i,(key,(b,u,t,v)) in enumerate(D.PAIRS.items()):
                pos=positions[key];idx=sel['coordinates'][key]
                slopes=[torch.as_tensor(cal['logit'][i,k],device='cuda') for k in range(2)]
                rng=torch.Generator(device='cuda').manual_seed(D.SEED+f*2+i)
                for dose in [.25,.5,1.]:
                    h,r,a,delta=assay.edited(base,b,u,pos,idx,dose)
                    commits.append({'pair':key,'mode':'upstream','dose':dose,'delta':delta,
                         'target_delta':float(delta*cal['target'][i,0]),'top1':int((original[0,0].double()+delta*slopes[0]).argmax())})
                uh,udelta=assay.unit_zero(base,b,u,pos)
                commits.append({'pair':key,'mode':'unit','dose':1.,'delta':udelta,
                   'target_delta':float(udelta*cal['target'][i,1]),'top1':int((original[0,0].double()+udelta*slopes[1]).argmax())})
            # Persistent all-pair predictions before the first edited suffix.
            D.save(f'prediction-{f:04d}.json',{'sample_id':row['id'],'frozen_sha256':D.HASH(D.OUT/'frozen.json'),
                 'positions':positions,'predictions':commits},p)
            for i,(key,(b,u,t,v)) in enumerate(D.PAIRS.items()):
                pos=positions[key];idx=sel['coordinates'][key];baseline=float(base[f'A{t}'][0,pos,v])
                assert torch.equal(assay.suffix(base[f'H{b}'],b+1),original)
                rng=torch.Generator(device='cuda').manual_seed(D.SEED+f*2+i)
                for dose in [.25,.5,1.]:
                    h,r,a,delta=assay.edited(base,b,u,pos,idx,dose)
                    control,norm,cnorm,valid=norm_match(base[f'R{b}'][0,pos],r,rng)
                    pred=original[0,0].double()+delta*torch.as_tensor(cal['logit'][i,0],device='cuda')
                    out=assay.suffix(h,b+1);rec={'pair':key,'mode':'upstream','dose':dose,
                        'source_A':float(a),'target_delta':float(assay.data[f'A{t}'][0,pos,v])-baseline,
                        'norm':norm,'control_norm':cnorm,'norm_valid':valid,'effect':distribution(out,original,pred,p['choice_ids'])}
                    hc,_,_,_=assay.edited(base,b,u,pos,idx,dose,control=control)
                    co=assay.suffix(hc,b+1);rec['control']=distribution(co,original,original[0,0],p['choice_ids'])
                    rec['control_target_delta']=float(assay.data[f'A{t}'][0,pos,v])-baseline
                    if dose==1.:
                        for clamp in ['unit','mlp']:
                            hh,_,_,_=assay.edited(base,b,u,pos,idx,dose,clamp=clamp)
                            oo=assay.suffix(hh,b+1);rec[clamp+'_clamp']={'target_delta':float(assay.data[f'A{t}'][0,pos,v])-baseline,
                                      **distribution(oo,original,original[0,0],p['choice_ids'])}
                    records.append(rec)
                h,delta=assay.unit_zero(base,b,u,pos);c,n,cn,ok=norm_match(base[f'H{b}'][0,pos],h[0,pos],rng)
                oo=assay.suffix(h,b+1);rec={'pair':key,'mode':'unit','dose':1.,'norm':n,'control_norm':cn,'norm_valid':ok,
                    'target_delta':float(assay.data[f'A{t}'][0,pos,v])-baseline,
                    'effect':distribution(oo,original,original[0,0].double()+delta*torch.as_tensor(cal['logit'][i,1],device='cuda'),p['choice_ids'])}
                hh=base[f'H{b}'].clone();hh[0,pos]=c;oo=assay.suffix(hh,b+1)
                rec['control_target_delta']=float(assay.data[f'A{t}'][0,pos,v])-baseline
                rec['control']=distribution(oo,original,original[0,0],p['choice_ids']);records.append(rec)
                assert torch.equal(assay.suffix(base[f'H{b}'],b+1),original)
            D.save(f'causal-{f:04d}.json',{'sample_id':row['id'],'gold':row['gold'],'positions':positions,
                'baseline':{'scores':original[0,0,p['choice_ids']].float().tolist(),'top1':int(original[0,0].argmax())},
                'prediction_sha256':D.HASH(D.OUT/f'prediction-{f:04d}.json'),'records':records,'bitwise_restored':True},p)
            if (f+1)%8==0:log('intervention',f+1,D.N_HOLD,start)
            assert monotonic()-start<60*p['resources']['stage_minutes']
    finally:assay.close()

def capability(model,p):
    assert D.HASH(D.CAP)==p['capability']['sha256']
    sel=D.READ(D.OUT/'selection.json');tok=AutoTokenizer.from_pretrained(D.MODEL,local_files_only=True);tok.padding_side='left'
    items=[]
    for row in D.READ(D.CAP):
        if not row['gate_eligible']:continue
        text=row['question']+'\n'+'\n'.join(f'{k}. {v}' for k,v in zip('ABCDE',row['choices']))+'\n请选择一个正确选项字母。'
        prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
        items.append({'id':row['id'],'source':row['source'],'gold':'ABCDE'.index(row['answer']),'n':len(row['choices']),'tokens':tok.encode(prompt,add_special_tokens=False)})
    D.save('capability-materials.json',{'rows':items},p);rows=[];start=monotonic();log('capability',0,len(items),start)
    for offset in range(0,len(items),8):
        chunk=items[offset:offset+8];batch=tok.pad({'input_ids':[x['tokens'] for x in chunk]},return_tensors='pt').to('cuda');scores={}
        for mode in ['baseline','broad-upstream','rare-upstream','broad-unit','rare-unit']:
            h=None
            if mode!='baseline':
                pair,kind=mode.split('-');b,u,_,_=D.PAIRS[pair];idx=sel['coordinates'][pair]
                if kind=='upstream':
                    def hook(mod,args,idx=idx):
                        x=args[0].clone();x[:,-1,idx]=0;return (x,)
                    # Both residual skip and MLP input must change. A norm pre-hook alone
                    # would leave skip unchanged; use the block replay through a forward hook.
                    saved={}
                    def capture(mod,args):saved['r']=args[0]
                    handle=model.model.layers[b].post_attention_layernorm.register_forward_pre_hook(capture)
                    def out_hook(mod,args,out,idx=idx,b=b):
                        r=saved['r'].clone();r[:,-1,idx]=0
                        value=r+mod.mlp(mod.post_attention_layernorm(r));y=out.clone();y[:,-1]=value[:,-1];return y
                    h=model.model.layers[b].register_forward_hook(out_hook)
                else:
                    handle=None
                    def hook(mod,args,u=u):
                        x=args[0].clone();x[:,-1,u]=0;return (x,)
                    h=model.model.layers[b].mlp.down_proj.register_forward_pre_hook(hook)
            try:scores[mode]=model(**batch,use_cache=False,logits_to_keep=1).logits[:,0,p['choice_ids']].float().cpu().numpy()
            finally:
                if h:h.remove()
                if mode!='baseline' and handle:handle.remove()
            # Only final position is edited in this battery; earlier-token rare effects
            # are not covered by a positive outcome here.
        rows.extend({'id':r['id'],'source':r['source'],'gold':r['gold'],'scores':{k:v[i,:r['n']].tolist() for k,v in scores.items()}} for i,r in enumerate(chunk))
        if len(rows)%128==0 or len(rows)==len(items):log('capability',len(rows),len(items),start)
    D.save('capability.json',{'rows':rows,'scope':'primary full removal at final token; keyed old battery only'},p)

def repeatability(model,p):
    rows=D.materials(p,'discovery');a=Assay(model);recs=[]
    try:
        for f in range(16):
            chunk=rows[f*8:f*8+4];z,b=a.base(chunk[0]);pa=a.selected(b,chunk[0])
            av=bits(b['A33'][0,:,1521]);z2,b2=a.base(chunk[0])
            z4,b4=a.observe([{**r,'positions':[len(r['tokens'])-1]} for r in chunk])
            recs.append({'family':f,'repeat_bitwise':bool(torch.equal(z,z2) and np.array_equal(av,bits(b2['A33'][0,:,1521]))),
                         'batch1_peak':pa['rare'],'batch4_peak':64+int(b4['R33'][0,64:,4].abs().argmax()),
                         'batch4_max_A':float(b4['A33'][0,64:,1521].abs().max()),
                         'batch1_max_A':float(b['A33'][0,64:,1521].abs().max()),'max_logit_diff':float((z-z4[:1]).abs().max())})
        D.save('repeatability.json',{'rows':recs},p)
    finally:a.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['discovery','select','calibrate','holdout','intervene','capability','repeatability']);stage=parser.parse_args().stage
    p=D.checked();configure(p)
    if stage=='select':select(p);return
    torch.manual_seed(D.SEED);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    with GpuLease(),torch.inference_mode():
        start=monotonic();log('load',0,1,start)
        model=LOADER.load(D.MODEL,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        log('load',1,1,start)
        if stage in ['discovery','holdout']:collect(model,p,stage)
        else:globals()[stage](model,p)
if __name__=='__main__':main()
