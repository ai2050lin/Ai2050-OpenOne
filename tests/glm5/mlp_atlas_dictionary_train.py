"""Every-layer unlabelled transcoder baseline and all-native-unit feature association."""
import sys,time,json,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_natural_collect as P
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits
from shared.dist_protocol.transcoder import train,TopKTranscoder,neuron_topk_recon
from shared.dist_protocol.natural_register_runner import snapshot

def main():
    import torch
    from server.mechanism_trace_service import parameter
    bank=P.OUT/'bank';man=P.read(bank/'manifest.json');pr=P.read(P.PUBLIC/'prereg.research.json');assert man['state']=='complete';plan=pr['algorithm_plan'];prov=man['provenance'];out=P.OUT/'dictionaries';out.mkdir(exist_ok=True)
    import shared.dist_protocol.transcoder as TC
    freeze={'kind':'real','run_id':P.RUN,'model_snapshot':man['model_snapshot'],'provenance':prov,'prereg_sha256':file_sha(P.PUBLIC/'prereg.research.json'),
        'dependencies':{str(Path(p).relative_to(P.ROOT)):file_sha(p) for p in [Path(__file__).resolve(),Path(TC.__file__).resolve()]},'model_weights_unmodified':True,
        'evaluation_addendum':{'feature_contexts':'eight strongest heldout/discovery nonnegative latent activations per feature; examples, not semantic labels',
            'intervals':'paired per-document normalized SSE difference; t approximation, alpha=.05/(36 layers * 2 heldout splits * 2 baselines); no graph or causal promotion',
            'native_recompute_warning_relative_error':.05}}
    freeze_path=P.OUT/'training-execution-freeze.json'
    if freeze_path.exists():assert P.read(freeze_path)==freeze,'A resumed run must keep the frozen code and plan'
    else:P.save(freeze_path,freeze)
    summaries=[];started=time.time();torch.set_num_threads(4)
    with GpuLease():
        assert snapshot(P.MODEL)==man['model_snapshot'];P.status('dictionary_training','running',blocks_done=0,total=36)
        for b in range(36):
            completed=out/f'B{b:02d}-result.json'
            if completed.exists():
                old=P.read(completed)
                assert old['dictionary_sha256']==file_sha(out/f'B{b:02d}-transcoder.npz') and old['association_sha256']==file_sha(out/f'B{b:02d}-associations.npz')
                assert old['model_snapshot']==man['model_snapshot'];summaries.append({k:v for k,v in old.items() if k not in ['kind','run_id','model_snapshot','provenance']});continue
            block_start=time.time();X=np.load(bank/f'B{b:02d}-input.npy',mmap_mode='r');Y=np.load(bank/f'B{b:02d}-output.npy',mmap_mode='r');xt=torch.from_numpy(unbits(X[:1024]).reshape(-1,2560)).to('cuda');yt=torch.from_numpy(unbits(Y[:1024]).reshape(-1,2560)).to('cuda');rng=np.random.default_rng(P.SEED+b)
            def sample(n):
                idx=torch.from_numpy(rng.integers(len(xt),size=n)).to('cuda');return xt[idx],yt[idx]
            def log(r):
                if r['step']%200==0:P.status('dictionary_training','running',block=b,step=r['step'],total_steps=plan['steps'],blocks_done=b,total=36,train_fvu=r['fvu'])
            t,curve=train(sample,d_in=2560,d_out=2560,m=plan['m'],k=plan['k'],steps=plan['steps'],batch=plan['batch'],lr=plan['lr'],aux_k=plan['aux_k'],aux_coef=plan['aux_coef'],dead_after=plan['dead_after_tokens'],device='cuda',seed=P.SEED+b,log=log,log_every=100)
            state=out/f'B{b:02d}-transcoder.npz';np.savez(state,**t.state())
            prefix=f'model.layers.{b}.mlp.';gate=torch.from_numpy(parameter(P.MODEL,prefix+'gate_proj.weight',(slice(None),slice(None)))).to(device='cuda',dtype=torch.bfloat16)
            up=torch.from_numpy(parameter(P.MODEL,prefix+'up_proj.weight',(slice(None),slice(None)))).to(device='cuda',dtype=torch.bfloat16);down=torch.from_numpy(parameter(P.MODEL,prefix+'down_proj.weight',(slice(None),slice(None)))).to(device='cuda',dtype=torch.bfloat16);wd=down.float().T.contiguous()
            corr=None;partner=None;results={};arrays={};calibration=[]
            with torch.inference_mode():
                for split,lo,hi in [('discovery',0,1024),('confirm',1024,1536),('lockbox',1536,2048)]:
                    count=0;sy=torch.zeros(2560,dtype=torch.float64,device='cuda');s2y=torch.zeros((),dtype=torch.float64,device='cuda');sse={k:0. for k in ['dictionary','native32','discovery_mean']};perdoc={k:[] for k in sse};fired=torch.zeros(plan['m'],dtype=torch.int64,device='cuda');sa=torch.zeros(9728,dtype=torch.float64,device='cuda');saa=sa.clone();sz=torch.zeros(plan['m'],dtype=torch.float64,device='cuda');szz=sz.clone()
                    cross=torch.zeros((9728,plan['m']),dtype=torch.float64,device='cuda') if split=='discovery' else torch.zeros(9728,dtype=torch.float64,device='cuda')
                    mean=yt.mean(0);pole_values=torch.full((8,plan['m']),-float('inf'),device='cuda');pole_indices=torch.full((8,plan['m']),-1,dtype=torch.int64,device='cuda')
                    for doc in range(lo,hi,16):
                        xx=torch.from_numpy(unbits(X[doc:min(doc+16,hi)]).reshape(-1,2560)).to('cuda');yy=torch.from_numpy(unbits(Y[doc:min(doc+16,hi)]).reshape(-1,2560)).to('cuda')
                        zz=t.encode(xx);pred=t.decode(zz);aa=torch.nn.functional.silu(torch.nn.functional.linear(xx.to(torch.bfloat16),gate))*torch.nn.functional.linear(xx.to(torch.bfloat16),up)
                        full=torch.nn.functional.linear(aa,down).float();neuron=neuron_topk_recon(aa.float(),wd,plan['k']);n=len(xx);count+=n;sy+=yy.sum(0,dtype=torch.float64);s2y+=(yy.double()**2).sum();fired+=(zz>0).sum(0)
                        values,indices=torch.cat([pole_values,zz],0).topk(8,dim=0)
                        new_indices=torch.arange(doc*16,doc*16+n,device='cuda')[:,None].expand(-1,plan['m']);pole_indices=torch.cat([pole_indices,new_indices],0).gather(0,indices);pole_values=values
                        for key,yhat in [('dictionary',pred),('native32',neuron),('discovery_mean',mean[None])]:
                            loss=((yy-yhat).double()**2).sum(-1);sse[key]+=float(loss.sum());perdoc[key].extend(loss.reshape(-1,16).sum(1).cpu().tolist())
                        calibration.append(float((full-yy).norm()/yy.norm().clamp_min(1e-8)))
                        af=aa.float();sa+=af.sum(0,dtype=torch.float64);saa+=(af.double()**2).sum(0);sz+=zz.sum(0,dtype=torch.float64);szz+=(zz.double()**2).sum(0)
                        if split=='discovery':cross+=(af.T@zz).double()
                        else:cross+=(af.double()*zz[:,partner].double()).sum(0)
                    den=float(s2y-(sy*sy).sum()/count);assert den>0
                    va=(saa-sa*sa/count).clamp_min(0);vz=(szz-sz*sz/count).clamp_min(0)
                    if split=='discovery':
                        cov=cross-sa[:,None]*sz[None]/count;scale=torch.sqrt(va[:,None]*vz[None]);corr=torch.where(scale>1e-16,cov/scale,torch.zeros_like(cov)).clamp(-1,1).float();partner=corr.abs().argmax(1);arrays['all_discovery_correlations']=corr.cpu().numpy();arrays['discovery_partner']=partner.cpu().numpy().astype(np.int16);rho=corr[torch.arange(9728,device='cuda'),partner]
                    else:
                        scale=torch.sqrt(va*vz[partner]);cov=cross-sa*sz[partner]/count;rho=torch.where(scale>1e-16,cov/scale,torch.zeros_like(cov)).clamp(-1,1).float()
                    arrays[split+'_partner_correlation']=rho.cpu().numpy();arrays[split+'_feature_firings']=fired.cpu().numpy()
                    arrays[split+'_feature_pole_values']=pole_values.cpu().numpy();arrays[split+'_feature_pole_indices']=pole_indices.cpu().numpy().astype(np.int32)
                    for k,v in perdoc.items():arrays[split+'_doc_sse_'+k]=np.asarray(v)
                    results[split]={'n_documents':hi-lo,'n_native_positions':count,'FVU':{k:v/den for k,v in sse.items()},'active_dictionary_features':int((fired>0).sum()),'zero_dictionary_features':int((fired==0).sum()),'denominator_variance_sum':den}
                    if split!='discovery':
                        from scipy.stats import t as student_t
                        intervals={};alpha=.05/(36*2*2)
                        for baseline in ['native32','discovery_mean']:
                            advantage=(np.asarray(perdoc[baseline])-np.asarray(perdoc['dictionary']))*(hi-lo)/den
                            mean_advantage=float(advantage.mean());half=float(student_t.ppf(1-alpha/2,len(advantage)-1)*advantage.std(ddof=1)/np.sqrt(len(advantage)))
                            intervals[baseline]={'mean_FVU_advantage':mean_advantage,'lower':mean_advantage-half,'upper':mean_advantage+half,'alpha':alpha,'method':'paired-document t approximation; no causal claim'}
                        results[split]['paired_intervals']=intervals
            associations=out/f'B{b:02d}-associations.npz';np.savez_compressed(associations,**arrays)
            summary={'block':b,'dictionary_sha256':file_sha(state),'association_sha256':file_sha(associations),'m':plan['m'],'k':plan['k'],'curve':curve,'results':results,'native_full_reconstruction_max_relative_error':max(calibration),
                'native_recompute_warning':max(calibration)>.05,'association_repeated_units':int(((np.abs(arrays['discovery_partner_correlation'])>=.2)&(arrays['confirm_partner_correlation']*arrays['discovery_partner_correlation']>0)&(arrays['lockbox_partner_correlation']*arrays['discovery_partner_correlation']>0)&(np.abs(arrays['confirm_partner_correlation'])>=.2)&(np.abs(arrays['lockbox_partner_correlation'])>=.2)).sum()),
                'scope':'correlation nomination andFVU only; no feature semanticlabels, no causalnecessaryunit orfullmodel losssplicerecovery','seconds':time.time()-block_start}
            P.save(out/f'B{b:02d}-result.json',{'kind':'real','run_id':P.RUN,'model_snapshot':man['model_snapshot'],'provenance':prov,**summary});summaries.append(summary)
            P.status('dictionary_training','running',blocks_done=b+1,total=36,last_FVU=results['lockbox']['FVU']);del t,xt,yt,gate,up,down,wd,corr,arrays;gc.collect();torch.cuda.empty_cache()
        assert snapshot(P.MODEL)==man['model_snapshot']
    result={'schema':'mlp_dictionary_atlas_report.v1','kind':'real','run_id':P.RUN,'model_snapshot':man['model_snapshot'],'provenance':prov,'layers':summaries,'n_layers':36,'dictionary_features':36*plan['m'],'native_units_with_association_attempts':350208,'seconds':time.time()-started,
        'limitations':['1024dictionaryfeatures perblock isacompressionbudget, notacompletebasisoflanguage.','FVU reconstruction andlatentcorrelation do notestablish a semanticmeaning orcausaluse.','Nativebaseline usesallnativeactivations torank32contributions, soequalactivecount isnot equalFLOP/parameterbudget.','Trainingdraws resample16,384naturalpositions perblock; repeatdraws arenotindependentnewdata.','No modelweights wereedited. Fullnetworkspliceloss andcapabilityneedseparate tests beforea controlclaim.']}
    P.save(P.OUT/'dictionary-result.json',result);P.status('dictionary_training','done',layers=36,features=result['dictionary_features'],seconds=result['seconds']);print(json.dumps({'layers':36,'dictionary_features':result['dictionary_features'],'seconds':result['seconds']}),flush=True)
if __name__=='__main__':main()
