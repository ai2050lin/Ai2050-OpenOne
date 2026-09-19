"""Posthoc position-boundary coverage repair and complete native block comparison."""
import gc
from rdc_relation_common import *
from rdc_relation_inference import CurrentRule,check_frozen
from phase2717_rdc_native_generation import BlockTrace,metric


def main():
    import torch
    out=BASE/'boundary_compilation';guard(3*1024**2)
    if (out/'result.json').exists():return
    check_frozen();save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'phase':2718,
      'question':'How much original all-source native failure was introduced by applying an anchor-trained polynomial to the first source?',
      'status':'Exploratory corrective comparison after Phase2717 all-source audit; existing fresh material reused, not an independent new confirmation.',
      'first_source_rules':'identity H23=H12; coordinatewise affine trained only on320 main-training first states, fixed covariance shrinkage1.01 and denominator1e-8. No fresh fitting or hyperparameter choice.',
      'native_candidates':['original_full_quadratic','identity_at_all_sources','first_source_identity_else_frozen_quadratic','first_source_affine_else_frozen_quadratic'],
      'information':'Only already-known full H12 source fields and known position0 boundary. Never insert actual H23, past true KV or future targets into a candidate.',
      'full_coordinate_policy':'All2560 source/target coordinates,32 heads,9728 MLP units. All-source raw fields and frozen rules retained. Save complete H24 for the fixed affine-boundary primary; other complete predictions are exactly reconstructible, with full-coordinate error profiles retained.',
      'not_assumed':'Boundary amplitude is not declared an attention sink or semantic role; a position-specific fit does not prove universal cross-model language structure.'})
    material=rows();tr=[r for r in material if r['split']=='train'];x=[];y=[]
    for r in tr:
        z=load_field(r);x.append(unbits(z['h12'][0]));y.append(unbits(z['h23'][0]))
    x=np.stack(x).astype(float);y=np.stack(y).astype(float);mx=x.mean(0);my=y.mean(0);var=np.mean((x-mx)**2,0);cov=np.mean((x-mx)*(y-my),0);a=(cov/(1.01*var+1e-8)).astype(np.float32);b=(my-a*mx).astype(np.float32)
    npz(out/'first_source_affine.npz',a=a,b=b,training_mean_h12=mx,training_mean_h23=my);trainids={r['prompt_ids'][0] for r in tr};first=[]
    for fresh in (False,True):
      for r in rows(fresh):
        if not fresh and r['split']=='train':continue
        z=load_field(r,fresh);xx=unbits(z['h12'][0]);yy=unbits(z['h23'][0]);pp=xx*a+b
        first.append({'sample_id':r['sample_id'],'language':r['language'],'split':'fresh' if fresh else r['split'],'first_token_id':r['prompt_ids'][0],'first_ID_seen_in_training':r['prompt_ids'][0] in trainids,
          'identity_MSE':float(np.mean((xx.astype(float)-yy)**2)),'affine_MSE':float(np.mean((pp.astype(float)-yy)**2)),'actual_energy':float(np.mean(yy.astype(float)**2))})
    save(out/'first_source_rows.json',first);first_summary={}
    for split in ('validation','test','fresh'):
      for seen in ('all','seen','unseen'):
        rr=[r for r in first if r['split']==split and (seen=='all' or r['first_ID_seen_in_training']==(seen=='seen'))]
        first_summary[split+'/'+seen]={'sources':len(rr),**{k:float(np.mean([r[k] for r in rr])) if rr else None for k in ('identity_MSE','affine_MSE','actual_energy')}}
    from phase2662_symmetric_mapping_contract import load_native
    rule=CurrentRule();model,tok=load_native('qwen4');trace=BlockTrace(model);device=model.get_input_embeddings().weight.device;reports=[];profiles={}
    def run(h,positions):
        n=len(h);xx=torch.as_tensor(h,dtype=torch.bfloat16,device=device)[None];pos=torch.arange(n,device=device)[None];mask=torch.full((n,n),torch.finfo(xx.dtype).min,dtype=xx.dtype,device=device).triu(1)[None,None];trace.positions=positions;trace.data={}
        z=trace.block(xx,attention_mask=mask,position_ids=pos,position_embeddings=model.model.rotary_emb(xx,pos),use_cache=False)
        return z[0,positions].float().cpu().numpy(),{k:v.copy() for k,v in trace.data.items()},bits(z[0,positions])
    try:
      with torch.inference_mode():
       for i,r in enumerate(rows(True)):
        z=load_field(r,True);h12=unbits(z['h12']);actual23=unbits(z['h23']);base=rule(h12)[:,:2560];oracle,ot,ob=run(actual23,r['anchors']);assert np.array_equal(ob,z['h24'][[0,3]])
        one=base.copy();one[0]=h12[0];aff=base.copy();aff[0]=h12[0]*a+b;candidates={'original_full_quadratic':base,'identity_at_all_sources':h12,'first_source_identity_else_frozen_quadratic':one,'first_source_affine_else_frozen_quadratic':aff};rr={'sample_id':r['sample_id'],'language':r['language'],'methods':{}}
        for name,h in candidates.items():
            p,pt,pb=run(h,r['anchors']);q=np.maximum(ot['probability'],1e-30);qq=np.maximum(pt['probability'],1e-30);kl=float(np.mean(np.sum(q*np.log(q/qq),axis=-1)));e=(p.astype(float)-oracle)**2;ue=(pt['activation'].astype(float)-ot['activation'])**2
            rr['methods'][name]={'H24_MSE':float(e.mean()),'attention_MSE':metric(ot['attention'],pt['attention'])['MSE'],'MLP_MSE':metric(ot['mlp'],pt['mlp'])['MSE'],'head_attention_KL':kl}
            if name not in profiles:profiles[name]={'H24_error':np.zeros(2560),'MLP_unit_error':np.zeros(9728)}
            profiles[name]['H24_error']+=e.sum(0);profiles[name]['MLP_unit_error']+=ue.sum(0)
            if name=='original_full_quadratic':
                with np.load(BASE/f'native/fields/{r["sample_id"]}.npz') as old:assert np.array_equal(pb,old['all_predicted_h24'])
            if name=='first_source_affine_else_frozen_quadratic':npz(out/f'fields/{r["sample_id"]}.npz',h24=pb)
        reports.append(rr)
        if i%32==31:print('BOUNDARY_NATIVE',i+1,128,flush=True)
    finally:trace.close();del model;gc.collect();torch.cuda.empty_cache()
    npz(out/'all_coordinate_and_unit_errors.npz',**{name+'_'+kind:v/256 for name,p in profiles.items() for kind,v in p.items()});save(out/'native_source_rows.json',reports)
    summary={name:{metric:float(np.mean([r['methods'][name][metric] for r in reports])) for metric in reports[0]['methods'][name]} for name in profiles};gains={}
    for name in profiles:
        values=np.array([r['methods']['original_full_quadratic']['H24_MSE']-r['methods'][name]['H24_MSE'] for r in reports]);rng=np.random.default_rng(2718);boot=[rng.choice(values,len(values),replace=True).mean() for _ in range(2000)]
        gains[name]={'original_minus_candidate_MSE':float(values.mean()),'conditional_source_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    save(out/'result.json',{'timestamp':stamp(),'first_source':first_summary,'native':summary,'paired_source_gains':gains,'original_native_replay_all_bitwise_equal':True,'queries':256,
      'limitations':'Same previously seen source set and posthoc boundary hypothesis. Purely conditional H12-to-H24 comparison through one real block, not autonomous generation or all-model boundary law. Affine boundary preserves no assumed semantic identity.'});print('BOUNDARY_COMPILATION_COMPLETE',first_summary,summary,gains,flush=True);guard(512*1024)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
