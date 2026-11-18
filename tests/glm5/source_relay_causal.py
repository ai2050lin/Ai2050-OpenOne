"""Original-model path masks and native head-context replacement, all rows retained."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
import source_relay_math as M
from source_relay_behavior import Runtime
from source_binding_native_v2 import bits

MASKS=['mask_correct','mask_records','mask_wrong','mask_question','mask_instruction','mask_assistant_tail','mask_allpast']
FORECASTS={'forecast_ridge_predV':'native_predV','forecast_aware_predV':'aware_native_predV',
           'forecast_aware_past_trueV':'aware_native_past_trueV','forecast_aware_past_predV':'aware_native_past_predV',
           'forecast_meanV':'aware_native_meanV'}
ARMS=['base']+MASKS+list(FORECASTS)

def run(split):
    import torch
    D.verify_internal();D.admission(split);root=D.OUT/D.MODEL;assert not (root/f'{split}-causal.json').exists()
    rows=D.read(D.OUT/'qwen14-materials.json')['splits'][split];base_behavior=D.read(root/f'{split}-behavior.json')['rows']
    n=len(rows);prediction=D.read(root/f'{split}-forecast.json');assert D.sha(root/f'{split}-forecast.npz')==prediction['arrays_sha256']
    data=np.load(root/f'{split}-forecast.npz');forecasts={a:data[k].copy() for a,k in FORECASTS.items()};data.close()
    timer=time.monotonic();results={a:[] for a in ARMS};checks=[];finals=[];pending={a:[] for a in ARMS};pending_rows=[]
    D.save(f'{D.MODEL}/{split}-causal-started.json',{'arms':ARMS,'start_boundary':20,'chunk':64,'batch':8})
    with D.gpu_lease(split+'/causal'),torch.inference_mode():
        rt=Runtime(D.MODEL)
        for chunk in range(0,n,64):
            rs=rows[chunk:chunk+64]
            # Preserve exactly the behavior runner's 128-row group padding width.
            parent=rows[chunk//128*128:chunk//128*128+128];longest=max(parent,key=lambda r:len(r['tokens']))
            ids,pos,pe,mask=rt.context(rs+[longest]);ids,pos,pe,mask=ids[:-1],pos[:-1],tuple(x[:-1] for x in pe),mask[:-1]
            h=rt.embed(ids);L=h.shape[1];states={'base':h};arm_masks={}
            for a in MASKS:
                ma=mask.clone()
                for i,r in enumerate(rs):
                    ix=np.array(M.block_tokens(r,a),dtype=int)+L-len(r['tokens'])
                    if len(ix):ma[i,0,-1,torch.tensor(ix,device='cuda')]=False
                assert ma[:,:,:-1].equal(mask[:,:,:-1]) and ma[:,:,-1,-1].all()
                arm_masks[a]=ma
            for b in range(40):
                if b==20:
                    for a in ARMS[1:]:states[a]=states['base'].clone()
                layer=rt.block(b);bi=[28,29].index(b) if b in [28,29] else None
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));original=None;original_input=None
                    for a in list(states):
                        kw={'attention_mask':arm_masks[a][lo:hi] if b>=20 and a in MASKS else mask[lo:hi],
                            'position_ids':pos[lo:hi],'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                        handle=None
                        if a in FORECASTS and bi is not None:
                            def replace(mod,args,arm=a,index=bi):
                                value=args[0].clone();head=D.HEADS[index]['head'];value[:,-1,head*128:(head+1)*128]=torch.tensor(forecasts[arm][chunk+lo:chunk+hi,index],device='cuda',dtype=value.dtype)
                                return (value,)
                            handle=layer.self_attn.o_proj.register_forward_pre_hook(replace)
                        x=states[a][lo:hi];y=layer(x,**kw)
                        if a=='base':original=y.clone();original_input=x.clone() if chunk==0 and lo==0 and b in [20,28,29,39] else None
                        elif chunk==0 and lo==0:assert torch.equal(y[:,:-1],original[:,:-1]),'Intervention escaped final position'
                        if handle:handle.remove()
                        states[a][lo:hi]=y
                    if original_input is not None:
                        replay=layer(original_input,attention_mask=mask[lo:hi],position_ids=pos[lo:hi],position_embeddings=tuple(p[lo:hi] for p in pe),use_cache=False)
                        assert torch.equal(replay,original),'Hook removal/noop failed';checks.append({'block':b,'noop_restoration_exact':True,'earlier_positions_equal':True})
                del layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(split+'/causal',chunk*40+(b+1)*len(rs),n*40,timer,block=b,arms=len(states))
            for a in ARMS:pending[a].append(states[a][:,-1].cpu().clone())
            pending_rows+=rs
            del states,h,ids,pos,pe,mask,arm_masks;gc.collect();torch.cuda.empty_cache()
            if len(pending_rows)==128 or chunk+len(rs)==n:
                last=[]
                for a in ARMS:
                    z=torch.cat(pending[a]).to('cuda');fn,lg=rt.finish(z);last.append(bits(fn));items=rt.score(lg,pending_rows)
                    if a=='base':lp=lg.log_softmax(-1);p=lp.exp()
                    KL=(p*(lp-lg.log_softmax(-1))).sum(-1).cpu().tolist()
                    for item,kl in zip(items,KL):item['KL_from_base']=kl
                    results[a]+=items;pending[a].clear();del z,fn,lg
                finals.append(np.stack(last,axis=1));pending_rows.clear();del lp,p;gc.collect();torch.cuda.empty_cache()
    diff=max(abs(v-w) for x,y in zip(base_behavior,results['base']) for v,w in zip(x['scores'],y['scores']))
    exact=diff==0 and all(x['top1']==y['top1'] for x,y in zip(base_behavior,results['base']))
    assert exact,('Baseline replay changed',diff)
    summary={};acc=np.array([x['forced_correct'] for x in results['base']],float);margin=np.array([x['gold_margin'] for x in results['base']])
    for a,items in results.items():
        da=np.array([x['forced_correct'] for x in items],float)-acc;dm=np.array([x['gold_margin'] for x in items])-margin
        summary[a]={'forced_accuracy':float(np.mean([x['forced_correct'] for x in items])),
            'top1_accuracy':float(np.mean([x['top1_correct'] for x in items])),
            'top1_agreement':float(np.mean([x['top1']==y['top1'] for x,y in zip(items,results['base'])])),
            'accuracy_change':float(da.mean()),'accuracy_change_ci_family':M.interval(rows,da,split+'/'+a+'/acc'),
            'margin_change':float(dm.mean()),'margin_change_ci_family':M.interval(rows,dm,split+'/'+a+'/margin'),
            'KL_mean':float(np.mean([x['KL_from_base'] for x in items]))}
    comparisons={}
    for a,b in [('mask_records','mask_correct'),('mask_records','mask_wrong'),('mask_question','mask_records'),('mask_assistant_tail','mask_records'),
                ('forecast_aware_predV','forecast_ridge_predV'),('forecast_aware_past_trueV','forecast_aware_past_predV'),('forecast_aware_predV','forecast_meanV')]:
        gain=np.array([x['forced_correct']-y['forced_correct'] for x,y in zip(results[a],results[b])],float)
        comparisons[a+' minus '+b]=M.paired(rows,gain,split+'/'+a+'/'+b)
    path=root/f'{split}-causal-final.npy';np.save(path,np.concatenate(finals))
    D.save(f'{D.MODEL}/{split}-causal.json',{'summary':summary,'rows':results,'comparisons':comparisons,'checks':checks,
        'natural_logits_replay_max_error':diff,'natural_top1_replay_exact':exact,'final_sha256':D.sha(path),
        'forecast_sha256':prediction['arrays_sha256'],'elapsed_seconds':time.monotonic()-timer,
        'scope':'Final-position interventions only; no transplanted answer vectors. Masks are total rerouting operations and are not norm matched. Forecasts observe hs20, then replace only two fixed heads; rest of original model remains. TrueV past arm excludes final token; predictedV arms use no future activation input. Intervention free-generation and unrelated-task effects not tested.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--split',required=True);a=ap.parse_args();run(a.split)
