"""Equal realized write-norm interventions, with original downstream recomputation."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_behavior import Runtime
from source_formation_analysis import bootstrap,predict_fit,rotate_qk
from source_binding_native_v2 import bits,unbits

ARMS=['base','head_correct','head_other','forecast_qk','mlp_selected','mlp_random']

def damped_output(base,delta,target,iterations=18):
    """Binary-search the realized BF16 displacement, never amplify past full removal."""
    import torch
    low=torch.zeros_like(target);high=torch.ones_like(target)
    for _ in range(iterations):
        mid=(low+high)/2
        changed=(base.float()+mid[:,None]*delta).to(base.dtype)
        norms=(changed.float()-base.float()).norm(dim=1)
        low=torch.where(norms<target,mid,low);high=torch.where(norms>=target,mid,high)
    a=(base.float()+low[:,None]*delta).to(base.dtype);b=(base.float()+high[:,None]*delta).to(base.dtype)
    na=(a.float()-base.float()).norm(dim=1);nb=(b.float()-base.float()).norm(dim=1)
    choose=(na-target).abs()<=(nb-target).abs();out=torch.where(choose[:,None],a,b)
    return out,torch.where(choose,na,nb),torch.where(choose,low,high)

class Editor:
    def __init__(self,rt,meta):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        self.rt=rt;self.meta=meta;self.original=sdpa_attention_forward;self.arm='base';self.rows=[];self.probe=False
        self.targets={};self.proposals={};self.local=[];self.pending=None;self.forecasts={};self.batch_offset=0
    def attention(self,module,q,k,v,mask,**kwargs):
        t=self.rt.t;out,weights=self.original(module,q,k,v,mask,**kwargs)
        c=next((x for x in self.meta['heads'] if x['block']==module.layer_idx),None)
        if c is not None and self.arm=='forecast_qk':
            lo=self.batch_offset;hi=lo+q.shape[0];fq,fk=self.forecasts[module.layer_idx]
            qm=q.clone();km=k.clone();qm[:,c['head'],-1]=fq[lo:hi];km[:,c['kv_head']]=fk[lo:hi]
            changed,_=self.original(module,qm,km,v,mask,**kwargs)
            replacement=out.clone();replacement[:,-1,c['head']]=changed[:,-1,c['head']]
            return replacement,weights
        if c is None or self.arm not in ['head_correct','head_other']:return out,weights
        h=c['head'];mm=mask.expand(-1,q.shape[1],-1,-1).clone();L=q.shape[-2]
        for i,r in enumerate(self.rows):
            record=r['target_record'] if self.arm=='head_correct' else ((r['source']+1)%3)*2+r['entity']
            ix=[L-len(r['tokens'])+j for j in r['records'][str(record)]['span']];mm[i,h,-1,ix]=False
        changed,_=self.original(module,q,k,v,mm,**kwargs)
        delta=(changed[:,-1,h].float()-out[:,-1,h].float())@module.o_proj.weight[:,h*self.rt.hd:(h+1)*self.rt.hd].float().T
        self.pending=delta
        return out,weights
    def attention_hook(self,b):
        def hook(module,args,out):
            if self.arm not in ['head_correct','head_other'] or self.pending is None:return
            y,weights=out;delta=self.pending;self.pending=None
            return self.modify(y,delta,b,'head'),weights
        return hook
    def mlp_hook(self,c):
        def hook(module,args,out):
            if self.arm not in ['mlp_selected','mlp_random']:return
            ix=c['selected'] if self.arm=='mlp_selected' else c['random']
            delta=-(args[0][:,-1,ix].float()@module.weight[:,ix].float().T)
            return self.modify(out,delta,c['block'],'mlp')
        return hook
    def modify(self,y,delta,b,kind):
        base=y[:,-1];full=(base.float()+delta).to(base.dtype)
        fullnorm=(full.float()-base.float()).norm(dim=1)
        if self.probe:
            self.proposals[self.arm]=fullnorm.detach()
            return y
        changed,norm,scale=damped_output(base,delta,self.targets[self.arm])
        z=y.clone();z[:,-1]=changed
        self.local.append({'block':b,'kind':kind,'arm':self.arm,'target_norm':self.targets[self.arm].cpu().tolist(),
            'actual_norm':norm.cpu().tolist(),'full_norm':fullnorm.cpu().tolist(),'scale':scale.cpu().tolist()})
        return z

def run(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from source_formation_native import verify
    verify();proof=D.admission(model,split);root=D.OUT/model;meta=D.read(root/'formation-fit.json')
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];n=len(rows);timer=time.monotonic()
    assert not (root/f'{split}-causal-started.json').exists();D.save(f'{model}/{split}-causal-started.json',{'admission':proof,'arms':ARMS},model)
    results={a:[] for a in ARMS};local=[];checks=[];finals=[]
    params=np.load(root/'formation-fit.npz');early=np.load(root/f'{split}-native/early.npy',mmap_mode='r')
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);editor=Editor(rt,meta);rt.mod.ALL_ATTENTION_FUNCTIONS.register('source_formation',editor.attention);rt.cfg._attn_implementation='source_formation'
        for chunk in range(0,n,128):
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);states={'base':rt.embed(ids)}
            L=ids.shape[1];xearly=unbits(early[chunk:chunk+len(rs),-L:])
            fq=predict_fit(params,'q_native',xearly[:,-1]);fk=predict_fit(params,'k_native',xearly)
            for bi,c in enumerate(meta['heads']):
                qrot,krot=rotate_qk(rt,fq[:,bi*rt.hd:(bi+1)*rt.hd],fk[:,:,bi*rt.hd:(bi+1)*rt.hd],tuple(p.cpu() for p in pe))
                editor.forecasts[c['block']]=(torch.from_numpy(qrot).cuda().bfloat16(),torch.from_numpy(krot).cuda().bfloat16())
            del xearly,fq,fk
            for b in range(40):
                if b==27:
                    for a in ['head_correct','head_other','forecast_qk']:states[a]=states['base'].clone()
                if b==38:
                    for a in ['mlp_selected','mlp_random']:states[a]=states['base'].clone()
                layer=rt.block(b);handles=[]
                hc=next((x for x in meta['heads'] if x['block']==b),None);mc=next((x for x in meta['mlps'] if x['block']==b),None)
                if hc:handles.append(layer.self_attn.register_forward_hook(editor.attention_hook(b)))
                if mc:handles.append(layer.mlp.down_proj.register_forward_hook(editor.mlp_hook(mc)))
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));editor.rows=rs[lo:hi];editor.batch_offset=lo;kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],
                        'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    pair=['head_correct','head_other'] if hc else ['mlp_selected','mlp_random'] if mc else []
                    if pair:
                        editor.probe=True;editor.proposals={}
                        for arm in pair:
                            editor.arm=arm;layer(states[arm][lo:hi],**kw)
                        target=.5*torch.minimum(editor.proposals[pair[0]],editor.proposals[pair[1]])
                        editor.targets={a:target for a in pair};editor.probe=False
                    for a in list(states):
                        editor.arm=a;editor.local=[];x=states[a][lo:hi];y=layer(x,**kw)
                        if a=='base' and chunk==0 and lo==0:
                            original=y.clone();original_input=x.clone()
                        if chunk==0 and lo==0 and a!='base':
                            assert torch.equal(y[:,:-1],original[:,:-1]),'Edit escaped final position'
                        for info in editor.local:local.append({'chunk':chunk,'offset':lo,**info})
                        states[a][lo:hi]=y
                    if chunk==0 and lo==0 and pair:
                        editor.arm='base';editor.probe=False
                        restored=layer(original_input,**kw)
                        assert torch.equal(restored,original),'No-op/restoration failed'
                        checks.append({'block':b,'noop_restoration_exact':True,'earlier_positions_exact':True})
                for handle in handles:handle.remove()
                del layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(model+'/'+split+'/causal',chunk*40+(b+1)*len(rs),n*40,timer,block=b,active_arms=len(states))
            end=[]
            for a in ARMS:
                final,lg=rt.finish(states[a][:,-1]);end.append(bits(final))
                if a=='base':lp=lg.log_softmax(-1);p=lp.exp()
                kl=(p*(lp-lg.log_softmax(-1))).sum(-1).cpu().tolist();items=rt.score(lg,rs)
                for item,value in zip(items,kl):item['KL_from_base']=value
                results[a]+=items
                del final,lg
            finals.append(np.stack(end,axis=1))
            editor.forecasts.clear();del states,ids,pos,pe,mask,lp,p;gc.collect();torch.cuda.empty_cache()
        rt.cfg._attn_implementation='sdpa'
    early._mmap.close();params.close()
    base=D.read(root/f'{split}-behavior.json')['rows'];difference=max(abs(v-w) for x,y in zip(base,results['base']) for v,w in zip(x['scores'],y['scores']))
    assert difference==0 and all(x['top1']==y['top1'] for x,y in zip(base,results['base']))
    summary={};acc=np.array([x['forced_correct'] for x in base],float);margin=np.array([x['gold_margin'] for x in base])
    for a,items in results.items():
        da=np.array([x['forced_correct'] for x in items],float)-acc;dm=np.array([x['gold_margin'] for x in items])-margin
        summary[a]={'forced_accuracy':float(np.mean([x['forced_correct'] for x in items])),'top1_accuracy':float(np.mean([x['top1_correct'] for x in items])),
            'accuracy_change':float(da.mean()),'accuracy_change_ci_family':bootstrap(rows,da,f'{model}/{split}/{a}/acc'),
            'margin_change':float(dm.mean()),'margin_change_ci_family':bootstrap(rows,dm,f'{model}/{split}/{a}/margin'),
            'KL_mean':float(np.mean([x['KL_from_base'] for x in items]))}
    norm={}
    for kind in ['head','mlp']:
        errors=[]
        for info in local:
            if info['kind']==kind:
                errors.extend((np.abs(np.array(info['actual_norm'])-info['target_norm'])/np.maximum(info['target_norm'],1e-8)).tolist())
        norm[kind]={'fraction_within_5pct':float(np.mean(np.array(errors)<=.05)),'max_relative_error':max(errors),
            'admitted':bool(np.mean(np.array(errors)<=.05)>=.99)}
    path=root/f'{split}-causal-final.npy';np.save(path,np.concatenate(finals))
    D.save(f'{model}/{split}-causal.json',{'summary':summary,'rows':results,'local':local,'norm_matching':norm,'checks':checks,
        'natural_logits_replay_max_error':difference,'final_sha256':D.sha(path),'fit_sha256':meta['fit_sha256'],
        'scope':'Final-token only; paired target=.5*min(full realized effect norms at each current branch state); BF16 damping<=1. Original downstream recomputed. Matching admission requires99% within5% target, otherwise claims downgraded. Forecast arm replaces selected heads Q/K using hs20 predictions and current branch V; other heads untouched even with GQA.',
        'elapsed_seconds':time.monotonic()-timer},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--split',required=True);a=ap.parse_args();run(a.model,a.split)
