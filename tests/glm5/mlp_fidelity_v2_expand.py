"""Development-only sparsity-budget extension, frozen before fresh metric inspection."""
import sys,json,time,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
import mlp_fidelity_v2_engine as E
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits

def main():
    import torch
    plan=P.read(P.OUT/'plan.json');cfg=plan['training'];out=P.OUT/'development-expansion';out.mkdir(exist_ok=True)
    contract={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'reason':'k32 developmentvalidation stillpoor atB8/B24; test representation densitybefore any freshmetrics. Existing5000step result andselection retained.',
        'candidates':[{'id':'sae4096k128','kind':'sae_output','m':4096,'k':128},{'id':'sae4096k256','kind':'sae_output','m':4096,'k':256}],
        'selection':'among allSAE candidates at5000steps, minimize mean developmentvalidationFVU + .05*k/256; save explicit extensionselection, no newheldout used; matchednativek baseline required. This changes activebudget andisnot a samek improvement claim.',
        'source_sha256':file_sha(__file__),'engine_sha256':file_sha(E.__file__),'fresh_metrics_inspected':0}
    P.save(P.PUBLIC/'development-extension.research.json',contract);P.save(P.OUT/'extension-freeze.json',contract);docs=P.read(P.LONG/'documents.json')['docs'];ti,vi=E.indices(docs,P.SEED);records=[];torch.set_num_threads(4)
    with GpuLease():
        for b in plan['prototype_layers']:
            yy=torch.from_numpy(unbits(np.load(P.OLD/f'bank/B{b:02d}-output.npy',mmap_mode='r')).reshape(-1,2560)).to('cuda')
            it=torch.tensor((ti[:,None]*16+np.arange(16)).ravel(),device='cuda');iv=torch.tensor((vi[:,None]*16+np.arange(16)).ravel(),device='cuda')
            for ci,c in enumerate(contract['candidates']):
                path=out/f'B{b:02d}-{c["id"]}-result.json'
                if path.exists():records.append(P.read(path));continue
                def log(r):P.status('development_expansion','running',block=b,candidate=c['id'],**r)
                t,curve=E.train(yy[it],yy[it],yy[iv],yy[iv],m=c['m'],k=c['k'],steps=5000,batch=cfg['batch'],lr=cfg['lr'],lr_end=cfg['lr_end'],seed=P.SEED+b+100*(ci+3),checkpoints=cfg['checkpoints'],log=log)
                state=out/f'B{b:02d}-{c["id"]}.npz';np.savez(state,**t.state());r={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'block':b,'candidate':c,'curve':curve,'state_path':str(state.relative_to(P.ROOT)),'state_sha256':file_sha(state)};P.save(path,r);records.append(r);del t;gc.collect();torch.cuda.empty_cache()
            del yy,it,iv;gc.collect();torch.cuda.empty_cache()
    records+= [r for r in P.read(P.OUT/'development-result.json')['layers'] if r['candidate']['kind']=='sae_output'];table=[]
    for c in [next(r['candidate'] for r in records if r['candidate']['id']=='sae4096'),*contract['candidates']]:
        mean=float(np.mean([r['curve'][-1]['development_validation_FVU'] for r in records if r['candidate']['id']==c['id']]));table.append({'candidate':c,'step':5000,'mean_validation_FVU':mean,'selection_score':mean+.05*c['k']/256})
    best=min(table,key=lambda r:r['selection_score']);P.save(P.OUT/'selection-extension.json',{**best,'table':table,'contract_sha256':file_sha(P.OUT/'extension-freeze.json'),'fresh_metrics_inspected':0});P.status('development_expansion','done',selection=best)
if __name__=='__main__':main()
