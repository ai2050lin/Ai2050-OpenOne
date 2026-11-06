"""Development-only selection, then fixed-recipe all-layer training, no fresh outputs read."""
import sys,json,time,gc,argparse
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
import mlp_fidelity_v2_engine as E
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,unbits

def main(stage):
    import torch
    plan=P.read(P.OUT/'plan.json');cfg=plan['training'];out=P.OUT/stage;out.mkdir(exist_ok=True);docs=P.read(P.LONG/'documents.json')['docs'];ti,vi=E.indices(docs,P.SEED)
    P.save(P.OUT/(stage+'-freeze.json'),{'kind':'real','run_id':P.RUN,'plan_sha256':file_sha(P.OUT/'plan.json'),'source_sha256':file_sha(__file__),'engine_sha256':file_sha(E.__file__),'source_bank_sha256':file_sha(P.OLD/'bank/manifest.json'),'train_documents':ti.tolist(),'validation_documents':vi.tolist(),'fresh_outputs_read':0})
    candidates=plan['candidates'];blocks=plan['prototype_layers'] if stage=='development' else list(range(36));choice=P.read(P.OUT/'selection.json') if stage=='trained' else None
    if choice:candidates=[choice['candidate']]
    records=[];torch.set_num_threads(4);started=time.time()
    with GpuLease():
        for b in blocks:
            xx=torch.from_numpy(unbits(np.load(P.OLD/f'bank/B{b:02d}-input.npy',mmap_mode='r')).reshape(-1,2560)).to('cuda');yy=torch.from_numpy(unbits(np.load(P.OLD/f'bank/B{b:02d}-output.npy',mmap_mode='r')).reshape(-1,2560)).to('cuda')
            it=torch.tensor((ti[:,None]*16+np.arange(16)).ravel(),device='cuda');iv=torch.tensor((vi[:,None]*16+np.arange(16)).ravel(),device='cuda')
            for c in candidates:
                tag=f'B{b:02d}-{c["id"]}';path=out/(tag+'-result.json')
                if path.exists():records.append(P.read(path));continue
                x=yy if c['kind']=='sae_output' else xx;seed=P.SEED+b+100*next(i for i,v in enumerate(plan['candidates']) if v['id']==c['id']);steps=choice['step'] if choice else cfg['steps'];begin=time.time()
                def log(row):P.status(stage,'running',block=b,candidate=c['id'],**row)
                t,curve=E.train(x[it],yy[it],x[iv],yy[iv],m=c['m'],k=c['k'],steps=steps,batch=cfg['batch'],lr=cfg['lr'],lr_end=cfg['lr_end'],seed=seed,checkpoints=[s for s in cfg['checkpoints'] if s<=steps],log=log)
                state=out/(tag+'.npz');np.savez(state,**t.state());result={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'block':b,'candidate':c,'curve':curve,'state_sha256':file_sha(state),'state_path':str(state.relative_to(P.ROOT)),'seed':seed,'seconds':time.time()-begin,'train_documents':len(ti),'validation_documents':len(vi),'scope':'old exposed development only; freshheldout results not inspected'}
                P.save(path,result);records.append(result);del t,x;gc.collect();torch.cuda.empty_cache()
            del xx,yy,it,iv;gc.collect();torch.cuda.empty_cache()
    if stage=='development':
        table=[]
        for c in candidates:
            rr=[r for r in records if r['candidate']['id']==c['id']]
            for step in cfg['checkpoints']:table.append({'candidate':c,'step':step,'mean_validation_FVU':float(np.mean([next(q['development_validation_FVU'] for q in r['curve'] if q['step']==step) for r in rr]))})
        best=min(table,key=lambda r:r['mean_validation_FVU']);P.save(P.OUT/'selection.json',{**best,'selection_table':table,'selection_rule':plan['selection'],'fresh_outputs_read':0,'plan_sha256':file_sha(P.OUT/'plan.json')});print(json.dumps({'selected':best}),flush=True)
    P.save(P.OUT/(stage+'-result.json'),{'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'layers':records,'seconds':time.time()-started,'stage':stage});P.status(stage,'done',trained=len(records),seconds=time.time()-started)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['development','trained']);main(p.parse_args().stage)
