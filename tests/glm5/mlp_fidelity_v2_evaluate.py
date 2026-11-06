"""First fresh metrics, after fixed selection; balanced language splits and all layers."""
import sys,time,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
import mlp_fidelity_v2_engine as E
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits
from shared.dist_protocol.transcoder import TopKTranscoder,neuron_topk_recon
from shared.dist_protocol.natural_register_runner import snapshot

def main():
    import torch
    from scipy.stats import t as student_t
    from server.mechanism_trace_service import parameter
    plan=P.read(P.OUT/'plan.json');selection=P.read(P.OUT/'selection-extension.json');trained=P.read(P.OUT/'trained-result.json');man=P.read(P.OUT/'natural-bank/manifest.json');assert man['state']=='complete';docs=P.read(P.OUT/'fresh-documents.json')['natural'];ti,_=E.indices(P.read(P.LONG/'documents.json')['docs'],P.SEED);out=P.OUT/'fresh-evaluation';out.mkdir(exist_ok=True)
    P.save(P.OUT/'fresh-evaluation-freeze.json',{'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'source_sha256':file_sha(__file__),'selection_sha256':file_sha(P.OUT/'selection-extension.json'),'trained_sha256':file_sha(P.OUT/'trained-result.json'),'natural_bank_sha256':file_sha(P.OUT/'natural-bank/manifest.json'),'criteria':plan['fresh_evaluation']['fidelity_gate']})
    start=time.time();summaries=[];torch.set_num_threads(4)
    with GpuLease():
        assert snapshot(P.MODEL)==plan['model_snapshot']
        for b in range(36):
            if (out/f'B{b:02d}-result.json').exists():summaries.append(P.read(out/f'B{b:02d}-result.json'));continue
            row=trained['layers'][b];state=P.ROOT/row['state_path'];assert file_sha(state)==row['state_sha256']
            with np.load(state) as st:t=TopKTranscoder.from_state(st,device='cuda')
            with np.load(P.OLD/f'dictionaries/B{b:02d}-transcoder.npz') as st:old=TopKTranscoder.from_state(st,device='cuda')
            proto=P.OUT/f'development/B{b:02d}-tc4096.npz';tc=None
            if proto.exists():
                with np.load(proto) as st:tc=TopKTranscoder.from_state(st,device='cuda')
            X=np.load(P.OUT/f'natural-bank/B{b:02d}-input.npy',mmap_mode='r');Y=np.load(P.OUT/f'natural-bank/B{b:02d}-output.npy',mmap_mode='r');baseY=np.load(P.OLD/f'bank/B{b:02d}-output.npy',mmap_mode='r');mean=torch.from_numpy(unbits(baseY[ti]).reshape(-1,2560).mean(0)).to('cuda');prefix=f'model.layers.{b}.mlp.';gate=torch.from_numpy(parameter(P.MODEL,prefix+'gate_proj.weight',(slice(None),slice(None)))).to('cuda',dtype=torch.bfloat16);up=torch.from_numpy(parameter(P.MODEL,prefix+'up_proj.weight',(slice(None),slice(None)))).to('cuda',dtype=torch.bfloat16);down=torch.from_numpy(parameter(P.MODEL,prefix+'down_proj.weight',(slice(None),slice(None)))).to('cuda',dtype=torch.bfloat16);wd=down.float().T.contiguous();arrays={};results={};maxerror=0
            for split in ['confirm','lockbox']:
                indices=np.asarray([i for i,r in enumerate(docs) if r['split']==split]);assert len(indices)==256 and sum(docs[i]['language']=='en' for i in indices)==128
                keys=['dictionary','archived_tc1024','native32','native_matched_k','development_mean']+(['diagnostic_tc4096'] if tc else []);sse={k:[] for k in keys};sy=torch.zeros(2560,dtype=torch.float64,device='cuda');s2=0.;fired=torch.zeros(t.m,dtype=torch.long,device='cuda');poles=torch.full((8,t.m),-float('inf'),device='cuda');poleidx=torch.full((8,t.m),-1,device='cuda',dtype=torch.long);bycoordinate=torch.zeros(2560,dtype=torch.float64,device='cuda');n=0
                with torch.inference_mode():
                    for lo in range(0,256,32):
                        ix=indices[lo:lo+32];x=torch.from_numpy(unbits(X[ix]).reshape(-1,2560)).to('cuda');y=torch.from_numpy(unbits(Y[ix]).reshape(-1,2560)).to('cuda');z=t.encode(y);pred=t.decode(z);a=torch.nn.functional.silu(torch.nn.functional.linear(x.to(torch.bfloat16),gate))*torch.nn.functional.linear(x.to(torch.bfloat16),up);full=torch.nn.functional.linear(a,down).float();maxerror=max(maxerror,float((full-y).norm()/y.norm().clamp_min(1e-8)));outputs={'dictionary':pred,'archived_tc1024':old(x),'native32':neuron_topk_recon(a.float(),wd,32),'native_matched_k':neuron_topk_recon(a.float(),wd,t.k),'development_mean':mean[None]}
                        if tc:outputs['diagnostic_tc4096']=tc(x)
                        for key,yy in outputs.items():sse[key].extend(((yy.double()-y.double())**2).sum(-1).reshape(-1,8).sum(1).cpu().tolist())
                        bycoordinate+=((pred.double()-y.double())**2).sum(0);sy+=y.sum(0,dtype=torch.float64);s2+=float((y.double()**2).sum());n+=len(y);fired+=(z>0).sum(0);v,i=torch.cat([poles,z],0).topk(8,dim=0);newidx=torch.from_numpy((ix[:,None]*8+np.arange(8)).ravel()).to('cuda')[:,None].expand(-1,t.m);poleidx=torch.cat([poleidx,newidx],0).gather(0,i);poles=v
                den=s2-float(sy.square().sum()/n);fvu={k:sum(v)/den for k,v in sse.items()};alpha=.05/(36*2);diff=(np.asarray(sse['archived_tc1024'])-np.asarray(sse['dictionary']))*256/den;half=float(student_t.ppf(1-alpha/2,255)*diff.std(ddof=1)/16);ci=[float(diff.mean()-half),float(diff.mean()+half)];passed=fvu['dictionary']<=.2 and ci[0]>0
                results[split]={'n_documents':256,'language_counts':{'en':128,'zh':128},'positions':n,'FVU':fvu,'paired_advantage_vs_archived_tc_CI':ci,'alpha':alpha,'fidelity_gate':passed,'active_features':int((fired>0).sum()),'denominator':den,'source_note':'SAE seesactualMLPoutput; lowerFVU thanTC isrepresentationcompression, notinputpredictionimprovement'}
                arrays[split+'_doc_indices']=indices;arrays[split+'_feature_firings']=fired.cpu().numpy();arrays[split+'_feature_pole_values']=poles.cpu().numpy();arrays[split+'_feature_pole_indices']=poleidx.cpu().numpy().astype(np.int32);arrays[split+'_coordinate_sse']=bycoordinate.cpu().numpy()
                for k,v in sse.items():arrays[split+'_doc_sse_'+k]=np.asarray(v)
            path=out/f'B{b:02d}-statistics.npz';np.savez_compressed(path,**arrays);summary={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'block':b,'candidate':row['candidate'],'results':results,'both_fidelity_gates':all(r['fidelity_gate'] for r in results.values()),'native_reconstruction_max_relative_error':maxerror,'dictionary_sha256':row['state_sha256'],'statistics_sha256':file_sha(path)};P.save(out/f'B{b:02d}-result.json',summary);summaries.append(summary);P.status('fresh_evaluation','running',blocks_done=b+1,total=36,last_FVU=results['lockbox']['FVU']['dictionary'],last_gate=summary['both_fidelity_gates']);del t,old,tc,gate,up,down,wd,mean,arrays;gc.collect();torch.cuda.empty_cache()
        assert snapshot(P.MODEL)==plan['model_snapshot']
    result={'schema':'mlp_fidelity_evaluation.v2','kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'layers':summaries,'seconds':time.time()-start,'passed_layers':[r['block'] for r in summaries if r['both_fidelity_gates']],'selection':selection};P.save(P.OUT/'fresh-evaluation-result.json',result);P.status('fresh_evaluation','done',passed_layers=result['passed_layers'],seconds=result['seconds'])
if __name__=='__main__':main()
