"""Discovery-only sparse-feature nomination, then scoped heldout tests and full correlations."""
import sys,time,gc,sqlite3,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits
from shared.dist_protocol.transcoder import TopKTranscoder

def contrasts(x):
    # background,R,T,Q,readout,feature. Do not count positions/variants as independent backgrounds.
    loc=x[:,:,1,0]-x[:,:,0,0]
    industry=(x[:,:,1,1]-x[:,:,0,1])*np.array([1.,-1.])[None,:,None,None]
    query=(x[:,:,:,1]-x[:,:,:,0]).reshape(len(x),4,2,x.shape[-1])
    return {'location':loc,'industry':industry,'query':query}

def main():
    import torch
    from scipy.stats import binom
    plan=P.read(P.OUT/'plan.json');pr=P.read(P.PUBLIC/'controlled-v4-prereg.research.json');bank=P.OUT/'controlled-bank';man=P.read(bank/'manifest.json');assert man['state']=='complete' and man['design_version']==4;out=P.OUT/'semantics';out.mkdir(exist_ok=True)
    db=sqlite3.connect(bank/'samples.sqlite3');rows=[json.loads(r[0]) for r in db.execute('SELECT body FROM samples ORDER BY idx')];db.close()
    assert len(rows)==4096
    for b in range(512):
        for r in [0,1]:
            for t in [0,1]:assert rows[b*8+r*4+t*2]['selected_record_hidden_sha256']==rows[b*8+r*4+t*2+1]['selected_record_hidden_sha256']
            for q in [0,1]:assert rows[b*8+r*4+q]['before_selector_hidden_sha256']==rows[b*8+r*4+2+q]['before_selector_hidden_sha256']
    cells=[]
    for key in sorted({tuple(r[k] for k in ['split','mapping','fact','question','frame']) for r in rows if r['split']!='discovery'}):
        rr=[r for r in rows if tuple(r[k] for k in ['split','mapping','fact','question','frame'])==key];assert len(rr)==96;acc=sum(r['correct'] for r in rr)/96;cells.append(dict(zip(['split','mapping','fact','question','frame'],key))|{'n':96,'accuracy':acc,'strict_format_accuracy':sum(r['strict_format'] for r in rr)/96,'passed':acc>=.9})
    behavior=all(c['passed'] for c in cells);fidelity=P.read(P.OUT/'fresh-evaluation-result.json');elig={r['block']:r['both_fidelity_gates'] for r in fidelity['layers']};alpha=pr['semantic_criteria']['alpha'];records=[];start=time.time();torch.set_num_threads(4)
    P.save(P.OUT/'semantics-execution-freeze.json',{'kind':'real','run_id':P.RUN,'source_sha256':file_sha(__file__),'prereg_sha256':file_sha(P.PUBLIC/'controlled-v4-prereg.research.json'),'bank_sha256':file_sha(bank/'manifest.json'),'selection_sha256':file_sha(P.OUT/'selection-extension.json'),'criteria':pr['semantic_criteria']})
    with GpuLease():
        for block in [8,24,35]:
            td=P.read(P.OUT/f'trained/B{block:02d}-result.json');state=P.ROOT/td['state_path'];assert file_sha(state)==td['state_sha256']
            with np.load(state) as st:t=TopKTranscoder.from_state(st,device='cuda')
            y=np.load(bank/f'B{block:02d}-output.npy',mmap_mode='r');disc=unbits(y[:1024]).reshape(-1,2560);zs=[]
            with torch.inference_mode():
                for lo in range(0,len(disc),256):zs.append(t.encode(torch.from_numpy(disc[lo:lo+256]).to('cuda')).cpu().numpy())
            zd=np.concatenate(zs).reshape(128,2,2,2,2,t.m);dd=contrasts(zd);nominated=[];rms=np.sqrt((zd.astype(np.float64)**2).mean((0,1,2,3,4)))
            for factor,limit in [('location',6),('industry',6),('query',4)]:
                delta=dd[factor];sign=np.sign(delta.mean((0,1)));acc=(delta*sign[None,None]>0).mean(0).min(0)[1];order=sorted(range(t.m),key=lambda f:(-acc[f],-abs(float(delta[:,:,:,f].mean()))/max(1e-8,rms[f]),f));chosen=[f for f in order if f not in [v['feature'] for v in nominated]][:limit]
                nominated +=[{'feature':f,'nomination_factor':factor,'discovery_min_order_accuracy':float(acc[f])} for f in chosen]
            # Freeze nomination before encoding any confirm/lockbox native values.
            P.save(out/f'B{block:02d}-nomination.json',{'kind':'real','run_id':P.RUN,'block':block,'nominated':nominated,'discovery_documents':128,'fidelity_eligible':elig[block],'behavior_gate':behavior,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance']})
            allz=np.lib.format.open_memmap(out/f'B{block:02d}-latent.npy',mode='w+',dtype=np.float32,shape=(4096,2,t.m));allz[:1024]=zd.reshape(1024,2,t.m)
            with torch.inference_mode():
                for lo in range(1024,4096,128):allz[lo:lo+128]=t.encode(torch.from_numpy(unbits(y[lo:lo+128]).reshape(-1,2560)).to('cuda')).cpu().numpy().reshape(-1,2,t.m)
            allz.flush();stats={};feature_records=[];controls={}
            for factor,delta in dd.items():
                sign=np.sign(delta.mean((0,1))).astype(np.int8);stats[factor+'_sign']=sign
                for split,lo,hi in [('confirm',128,320),('lockbox',320,512)]:
                    z=np.asarray(allz[lo*8:hi*8]).reshape(hi-lo,2,2,2,2,t.m);diff=contrasts(z)[factor];success=(diff*sign[None,None]>0).sum(0);nonzero=(diff!=0).sum(0);p=binom.sf(success-1,len(z),.5);passed=(success/len(z)>=.8)&(nonzero/len(z)>=.8)&(p<alpha)&(sign[None]!=0)
                    for k,v in [('successes',success),('nonzero',nonzero),('p',p),('passed',passed)]:stats[f'{factor}_{split}_{k}']=v
                    flips=np.ones(len(z),np.float32);flips[:len(z)//2]=-1;np.random.default_rng(P.SEED+block+(0 if split=='confirm' else 10)).shuffle(flips);ns=(diff*flips[:,None,None,None]*sign[None,None]>0).sum(0);npv=binom.sf(ns-1,len(z),.5);controls[f'{factor}_{split}']=int(((ns/len(z)>=.8)&(npv<alpha)).all(0).sum())
            for item in nominated:
                f=item['feature'];factor=item['nomination_factor'];passes=[]
                for stage in [0,1]:passes.append(all(stats[f'{factor}_{sp}_passed'][:,stage,f].all() for sp in ['confirm','lockbox']))
                zz=np.asarray(allz[320*8:]).reshape(192,2,2,2,2,t.m);cc=contrasts(zz)[factor][:,:,1,f];orientation=stats[factor+'_sign'][1,f];worst=np.unravel_index(int(np.argmin(cc*orientation)),cc.shape);bb=320+int(worst[0]);component=int(worst[1])
                if factor=='query':rr,tt=divmod(component,2);examples=[bb*8+rr*4+tt*2+q for q in [0,1]]
                else:rr=component;qq=0 if factor=='location' else 1;examples=[bb*8+rr*4+tt*2+qq for tt in [0,1]]
                feature_records.append({**item,'replicated_by_readout':list(map(bool,passes)),'functional_eligible':bool(passes[1] and behavior and elig[block]),'scope':'conditionalgeneratedattributeanswer response; prequestion role separatelytested; notunique semanticunit','counterexample_indices':[i for i,r in enumerate(rows) if r['split']=='lockbox' and not r['correct']][:8],'response_counterexample':{'indices':examples,'oriented_delta':float(cc[worst]*orientation),'is_sign_failure':bool(cc[worst]*orientation<=0),'reason':'leastconsistent lockboxcomponent; apositivevalue isnotan actualsigncounterexample'}})
            # Complete native×latent correlation on discovery at both readouts; no TopK matrix.
            with torch.inference_mode():
                aa=torch.from_numpy(unbits(np.load(bank/f'B{block:02d}-A.npy',mmap_mode='r')[:1024]).reshape(-1,9728)).to('cuda');zz=torch.from_numpy(zd.reshape(-1,t.m)).to('cuda');a0=aa-aa.mean(0);z0=zz-zz.mean(0);cov=a0.T@z0;scale=torch.sqrt(a0.square().sum(0)[:,None]*z0.square().sum(0)[None]);corr=torch.where(scale>1e-12,cov/scale,0).clamp(-1,1).cpu().numpy();np.savez_compressed(out/f'B{block:02d}-native-feature-correlation.npz',correlation=corr)
            selected=[r['feature'] for r in nominated];random=[];matched=True;norms=t.W_dec.norm(dim=1).cpu().numpy()
            for f in selected:
                pool=[j for j in range(t.m) if j not in selected+random and .8<=rms[j]/max(rms[f],1e-12)<=1.25 and .8<=norms[j]/max(norms[f],1e-12)<=1.25]
                if not pool:matched=False;break
                random.append(min(pool,key=lambda j:digest({'seed':P.SEED,'block':block,'feature':f,'candidate':j})))
            stats['discovery_feature_RMS']=rms;stats['decoder_norm']=norms;np.savez_compressed(out/f'B{block:02d}-statistics.npz',**stats);record={'block':block,'features':feature_records,'selected_group':selected,'random_group':random,'random_matched':matched,'behavior_gate':behavior,'fidelity_gate':elig[block],'negative_controls':controls,'latent_sha256':file_sha(out/f'B{block:02d}-latent.npy'),'statistics_sha256':file_sha(out/f'B{block:02d}-statistics.npz'),'native_correlation_sha256':file_sha(out/f'B{block:02d}-native-feature-correlation.npz')};P.save(out/f'B{block:02d}-result.json',{'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],**record});records.append(record);P.status('semantics','running',block=block,eligible_features=sum(r['functional_eligible'] for r in feature_records));del t,zd,dd,allz,aa,zz,a0,z0,cov,scale,corr,stats;gc.collect();torch.cuda.empty_cache()
    P.save(P.OUT/'semantics-result.json',{'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'behavior_gate':behavior,'behavior_cells':cells,'behavior_errors':[{k:r[k] for k in ['id','bank_index','split','expected','generated_text','frame','queried_attribute']} for r in rows if not r['correct']],'prefix_controls':{'Q_invisible_at_selected_record_all37hidden_exact':True,'T_before_selector_all37hidden_exact':True},'layers':records,'seconds':time.time()-start});P.status('semantics','done',behavior_gate=behavior,layers=3)
if __name__=='__main__':main()
