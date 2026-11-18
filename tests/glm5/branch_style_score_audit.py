"""Independent cache-free full-prefix candidate rescoring; old scores retained."""
import sys
from pathlib import Path
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C

def score_full(engine,rows):
    import torch
    device=engine.model.device
    seq=[r['tokens']+c for r in rows for c in r['candidate_tokens']]
    cand=[c for r in rows for c in r['candidate_tokens']]
    width=max(map(len,seq));keep=max(map(len,cand))+1
    ids=torch.full((len(seq),width),engine.tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for i,tokens in enumerate(seq):ids[i,-len(tokens):]=torch.tensor(tokens,device=device);mask[i,-len(tokens):]=1
    # Only the final candidate prediction positions are unembedded; the entire
    # actual prefix is processed, without cached states or answer-derived inputs.
    result=engine.model(input_ids=ids,attention_mask=mask,position_ids=mask.cumsum(-1)-1,use_cache=False,logits_to_keep=keep).logits.float().log_softmax(-1)
    values=[]
    for i,tokens in enumerate(cand):
        start=keep-len(tokens)-1
        lp=result[i,start:start+len(tokens)]
        values.append(lp.gather(1,torch.tensor(tokens,device=device)[:,None]).sum().item())
    return np.array(values,dtype=np.float32).reshape(len(rows),4)

def main():
    import torch
    start=time.time();report={'scope':'Numerical audit after cache/full mismatch on exposed pilot. Full-prefix, cache-free scoring is authoritative for final behavioral gate; no method selection or refitting. Original scores and frozen native inference retained.','sets':{}}
    sets=[(p,C.OUT/p,C.selected_rows(p)) for p in ['pilot','discovery','confirm']]
    folder=C.OUT/'fresh_fusion_probe';sets.append(('fresh_2x2',folder,C.read(folder/'inputs.json')['rows']))
    with C.gpu_lease(),torch.inference_mode():
        engine=C.Engine(False)
        for name,folder,rows in sets:
            if (folder/'full_score_summary.json').exists():report['sets'][name]=C.read(folder/'full_score_summary.json');continue
            path=folder/'scores_full.npy';state=folder/'full_score_state.json';done=C.read(state)['done'] if state.exists() else 0
            array=np.lib.format.open_memmap(path,mode='r+' if done else 'w+',dtype=np.float32,shape=(len(rows),4))
            C.log('full_score_'+name,done,len(rows),start)
            for off in range(done,len(rows),4):
                batch=rows[off:off+4];array[off:off+len(batch)]=score_full(engine,batch);array.flush()
                C.save(state,{'done':off+len(batch)})
                if (off+len(batch))%64==0 or off+len(batch)==len(rows):C.log('full_score_'+name,off+len(batch),len(rows),start)
            old=np.load(folder/'scores.npy',mmap_mode='r');before=np.max(old[:,:2],1)>np.max(old[:,2:],1);after=np.max(array[:,:2],1)>np.max(array[:,2:],1)
            if name!='fresh_2x2':
                behavior=C.behavior(rows,array)
                gates=C.read(C.OUT/'prereg.json')['behavior_gate'];passed=behavior['content_accuracy']>=gates['content_pooled'] and min(v['accuracy'] for v in behavior['cells'].values())>=gates['min_cell'] and min(v['positive_shift_fraction'] for v in behavior['style'].values())>=gates['paired_style_per_task']
            else:
                groups={};shifts={str(t):[] for t in [0,1]}
                for off in range(0,len(rows),4):
                    s=array[off:off+4];shifts[str(rows[off]['task'])].append(float((s[1,0]-s[1,1])-(s[2,0]-s[2,1])))
                for i,r in enumerate(rows):groups.setdefault((r['branch'],r['task'],r['mode']),[]).append(bool(after[i]))
                behavior={'content_accuracy':float(after.mean()),'min_cell_accuracy':min(float(np.mean(v)) for v in groups.values()),'style_shift_positive_fraction':{k:float(np.mean(np.array(v)>0)) for k,v in shifts.items()}}
                passed=behavior['content_accuracy']>=.85 and behavior['min_cell_accuracy']>=.75 and min(behavior['style_shift_positive_fraction'].values())>=.65
            result={'kind':'real','name':name,'rows':len(rows),'behavior':behavior,'gate_passed':bool(passed),'max_abs_cache_difference_nat':float(np.max(abs(array-old))),'mean_abs_cache_difference_nat':float(np.mean(abs(array-old))),'content_decision_disagreements':int(np.sum(before!=after)),'model_snapshot':C.read(C.OUT/'frozen.json')['model_snapshot'],'source_sha256':C.file_sha(__file__)}
            C.save(folder/'full_score_summary.json',result);report['sets'][name]=result
    C.save(C.OUT/'full_score_audit.json',report)
    print({k:{'accuracy':v['behavior']['content_accuracy'],'gate':v['gate_passed'],'disagreements':v['content_decision_disagreements']} for k,v in report['sets'].items()},flush=True)

if __name__=='__main__':main()
