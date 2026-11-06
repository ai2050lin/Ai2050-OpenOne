"""Fixed expanded recipe atall36layers; no fresh metric based selection."""
import sys,time,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
import mlp_fidelity_v2_engine as E
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits

def main():
    import torch
    plan=P.read(P.OUT/'plan.json');choice=P.read(P.OUT/'selection-extension.json');c=choice['candidate'];cfg=plan['training'];out=P.OUT/'trained';out.mkdir(exist_ok=True);docs=P.read(P.LONG/'documents.json')['docs'];ti,vi=E.indices(docs,P.SEED);rows=[]
    P.save(P.OUT/'trained-extension-freeze.json',{'kind':'real','run_id':P.RUN,'source_sha256':file_sha(__file__),'engine_sha256':file_sha(E.__file__),'choice_sha256':file_sha(P.OUT/'selection-extension.json'),'fresh_metrics_read':0,'train_documents':ti.tolist(),'validation_documents':vi.tolist()});start=time.time();torch.set_num_threads(4)
    with GpuLease():
        for b in range(36):
            path=out/f'B{b:02d}-result.json'
            if path.exists():rows.append(P.read(path));continue
            y=torch.from_numpy(unbits(np.load(P.OLD/f'bank/B{b:02d}-output.npy',mmap_mode='r')).reshape(-1,2560)).to('cuda');it=torch.tensor((ti[:,None]*16+np.arange(16)).ravel(),device='cuda');iv=torch.tensor((vi[:,None]*16+np.arange(16)).ravel(),device='cuda');ci={'sae4096':2,'sae4096k128':3,'sae4096k256':4}[c['id']]
            def log(r):P.status('all_layer_training','running',block=b,candidate=c['id'],**r)
            t,curve=E.train(y[it],y[it],y[iv],y[iv],m=c['m'],k=c['k'],steps=5000,batch=cfg['batch'],lr=cfg['lr'],lr_end=cfg['lr_end'],seed=P.SEED+b+100*ci,checkpoints=cfg['checkpoints'],log=log)
            state=out/f'B{b:02d}-dictionary.npz';np.savez(state,**t.state());row={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'block':b,'candidate':c,'curve':curve,'state_sha256':file_sha(state),'state_path':str(state.relative_to(P.ROOT)),'source':'observedMLPoutputSAE, notx-to-y prediction'};P.save(path,row);rows.append(row);del y,it,iv,t;gc.collect();torch.cuda.empty_cache()
    P.save(P.OUT/'trained-result.json',{'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'layers':rows,'seconds':time.time()-start,'choice':choice});P.status('all_layer_training','done',layers=36,seconds=time.time()-start)
if __name__=='__main__':main()
