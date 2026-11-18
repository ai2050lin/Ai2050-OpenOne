"""Last-query native edge/QK/unit interventions; original downstream recomputation."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_runtime import Runtime
from source_binding_native import bits
from source_binding_analysis import verify_code,bootstrap

ARMS=['base','noop','correct_edge','wrong_edge','sparse_qk','random_qk','mlp_selected','mlp_matched']

class Editor:
    def __init__(self,rt,candidates):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        self.rt=rt;self.c=candidates;self.original=sdpa_attention_forward;self.arm='base';self.rows=[];self.deltas=[]
    def attention(self,module,q,k,v,mask,**kwargs):
        t=self.rt.t;out,weights=self.original(module,q,k,v,mask,**kwargs)
        candidate=next((c for c in self.c['heads'] if c['block']==module.layer_idx),None)
        if candidate is None or self.arm not in ['correct_edge','wrong_edge','sparse_qk','random_qk']:return out,weights
        h=candidate['head'];length=q.shape[2];qm=q;mm=mask
        if self.arm in ['correct_edge','wrong_edge']:
            mm=mask.expand(-1,q.shape[1],-1,-1).clone()
            for i,r in enumerate(self.rows):
                record=r['target_record'] if self.arm=='correct_edge' else (1-r['source'])*2+r['entity']
                idx=[length-len(r['tokens'])+j for j in r['records'][str(record)]['span']];mm[i,h,-1,idx]=False
        else:
            keep=candidate['keep'] if self.arm=='sparse_qk' else candidate['random_keep'];remove=[j for j in range(q.shape[-1]) if j not in keep]
            qm=q.clone();qm[:,h,-1,remove]=0
        changed,_=self.original(module,qm,k,v,mm,**kwargs)
        # Preserve every other native position/head bit for bit.
        replacement=out.clone();replacement[:,-1,h]=changed[:,-1,h]
        delta=(replacement[:,-1,h]-out[:,-1,h]).float()
        w=module.o_proj.weight[:,h*self.rt.hd:(h+1)*self.rt.hd].float()
        self.deltas.append({'block':module.layer_idx,'kind':'head','head':h,'write_norm':(delta@w.T).norm(dim=1).cpu().tolist()})
        return replacement,weights
    def mlp_hook(self,c):
        def hook(module,args):
            if self.arm not in ['mlp_selected','mlp_matched']:return
            t=self.rt.t;ix=c['selected'] if self.arm=='mlp_selected' else c['matched'];x=args[0];y=x.clone();y[:,-1,ix]=0
            delta=x[:,-1,ix].float()@module.weight[:,ix].float().T
            self.deltas.append({'block':c['block'],'kind':'mlp','write_norm':delta.norm(dim=1).cpu().tolist()})
            return (y,)
        return hook

def run(model,split):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    verify_code();proof=D.admission(model,split);root=D.OUT/model;c=D.read(root/'candidates.json')
    amendment=D.read(D.OUT/'causal-admission-v2.json')
    for path,sha in amendment['code_sha256'].items():assert D.sha(D.ROOT/path)==sha
    skipped=[]
    if not all(x['matched_control_valid'] for x in c['mlps']):
        skipped=['mlp_selected','mlp_matched']
        ARMS[:]=[arm for arm in ARMS if arm not in skipped]
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];n=len(rows);target=root/f'{split}-causal.json';assert not target.exists()
    D.save(f'{model}/{split}-causal-started.json',{'admission':proof,'arms':ARMS,'skipped_invalid_controls':skipped,'candidates_sha256':D.sha(root/'candidates.json')},model)
    start=time.monotonic();results={a:[] for a in ARMS};calibration=[];local=[]
    cfg=D.read(D.OUT/f'{model}-identity.json')['config'];path=root/f'{split}-causal-final.npy'
    finalarray=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint16,shape=(n,len(ARMS),cfg['hidden_size']))
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);editor=Editor(rt,c);rt.mod.ALL_ATTENTION_FUNCTIONS.register('source_binding',editor.attention)
        rt.cfg._attn_implementation='source_binding'
        for chunk in range(0,n,128):
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);base=rt.embed(ids);states={a:base.clone() for a in ARMS};del base
            for b in range(40):
                layer=rt.block(b);mc=next((x for x in c['mlps'] if x['block']==b),None);handle=layer.mlp.down_proj.register_forward_pre_hook(editor.mlp_hook(mc)) if mc else None
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));editor.rows=rs[lo:hi];kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],
                        'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    for a in ARMS:
                        editor.arm=a;editor.deltas=[];x=states[a][lo:hi];y=layer(x,**kw)
                        for d in editor.deltas:local.append({'chunk':chunk,'offset':lo,'arm':a,**d})
                        if chunk==0 and lo==0 and (mc is not None or any(x['block']==b for x in c['heads'])):
                            if a=='base':reference=y.clone();reference_input=x.clone()
                            if a=='noop':assert torch.equal(reference,y),'No-operation wrapper changed block output'
                            if a not in ['base','noop']:assert torch.equal(reference[:,:-1],y[:,:-1]),'Intervention leaked to earlier positions'
                        states[a][lo:hi]=y
                    if chunk==0 and lo==0 and (mc is not None or any(x['block']==b for x in c['heads'])):
                        editor.arm='base';rt.cfg._attn_implementation='sdpa';restored=layer(reference_input,**kw)
                        assert torch.equal(reference,restored),'Restoration failed'
                        calibration.append({'block':b,'restored_original_backend_bitwise_equal':True})
                        rt.cfg._attn_implementation='source_binding'
                if handle:handle.remove()
                del layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(model+'/'+split+'/causal',chunk*40+(b+1)*len(rs),n*40,start,block=b,arms=len(ARMS))
            for ai,a in enumerate(ARMS):
                final,lg=rt.finish(states[a][:,-1]);finalarray[chunk:chunk+len(rs),ai]=bits(final)
                if a=='base':lp0=lg.log_softmax(-1);p0=lp0.exp();baseline_logits=lg.clone()
                if a=='noop':
                    equal=bool(torch.equal(baseline_logits,lg));assert equal;calibration.append({'chunk':chunk,'noop_logits_bitwise_equal':equal})
                kl=(p0*(lp0-lg.log_softmax(-1))).sum(-1).cpu().tolist();scored=rt.score(lg,rs)
                for x,v in zip(scored,kl):x['kl_from_base']=v
                results[a]+=scored
                del final,lg
            finalarray.flush();del states,lp0,p0,baseline_logits,ids,pos,pe,mask;gc.collect();torch.cuda.empty_cache()
            D.check_write(D.OUT,8*2**30,operation='来源绑定干预批次空间检查')
        rt.cfg._attn_implementation='sdpa'
    baseline=D.read(root/f'{split}-behavior.json')['rows']
    maxdiff=max(abs(a-b) for x,y in zip(results['base'],baseline) for a,b in zip(x['scores'],y['scores']))
    assert maxdiff==0 and all(x['top1']==y['top1'] for x,y in zip(results['base'],baseline))
    summary={};baseacc=np.array([x['forced_correct'] for x in results['base']],float);basemargin=np.array([x['gold_margin'] for x in results['base']])
    for a in ARMS:
        x=results[a];change=np.array([v['forced_correct'] for v in x],float)-baseacc;dm=np.array([v['gold_margin'] for v in x])-basemargin
        summary[a]={'forced_accuracy':float(np.mean([v['forced_correct'] for v in x])),'top1_accuracy':float(np.mean([v['top1_correct'] for v in x])),
            'accuracy_change':float(change.mean()),'accuracy_change_ci_family':bootstrap(rows,change,f'{model}/{split}/{a}/acc'),
            'gold_margin_change':float(dm.mean()),'margin_change_ci_family':bootstrap(rows,dm,f'{model}/{split}/{a}/margin'),
            'kl_mean':float(np.mean([v['kl_from_base'] for v in x]))}
    D.save(f'{model}/{split}-causal.json',{'summary':summary,'rows':results,'calibration':calibration,'baseline_replay_max_choice_error':maxdiff,
        'local_removed_write_norms':local,'final_state_sha256':D.sha(path),'admission':proof,'candidates_sha256':D.sha(root/'candidates.json'),
        'scope':'all frozen heads or both MLP blocks jointly, final query only; actual downstream recomputation; wrong-source record is count/role matched, not guaranteed equal effect norm',
        'mlp_controls_valid':all(x['matched_control_valid'] for x in c['mlps']),'skipped_invalid_controls':skipped},model)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--split',required=True);a=ap.parse_args();run(a.model,a.split)
