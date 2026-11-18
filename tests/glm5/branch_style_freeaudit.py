"""Fixed independent-family greedy audit; no candidate/feature selection."""
import re
import sys
from pathlib import Path
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C

def main():
    import torch
    frozen=C.read(C.OUT/'frozen.json')
    rows=C.selected_rows('confirm');families=sorted({r['family'] for r in rows})
    selected=[]
    for i,f in enumerate(families):
        selected.append(next(r for r in rows if r['family']==f and (r['branch'],r['task'],r['style'])==(i%3,(i//3)%2,(i//6)%2)))
    out=[];start=time.time();C.log('confirm_free_audit',0,len(selected),start)
    with C.gpu_lease(),torch.inference_mode():
        engine=C.Engine(False)
        # Independent full-concatenation scorer audit on already-exposed pilot.
        audit_rows=C.selected_rows('pilot')[:2]
        _,cached,_=engine.run(audit_rows,score=True)
        brute=np.zeros((2,4))
        for i,r in enumerate(audit_rows):
            for j,candidate in enumerate(r['candidate_tokens']):
                ids=torch.tensor([r['tokens']+candidate],device='cuda')
                logits=engine.model(input_ids=ids,use_cache=False).logits[0].float()
                start_at=len(r['tokens'])-1
                pred=logits[start_at:start_at+len(candidate)].log_softmax(-1)
                brute[i,j]=pred.gather(1,torch.tensor(candidate,device='cuda')[:,None]).sum().item()
        scorer_error=float(np.max(abs(brute-cached)))
        C.save(C.OUT/'scorer_audit.json',{'cached':cached.tolist(),'full':brute.tolist(),'max_abs_error_nat':scorer_error,'pilot_ids':[r['id'] for r in audit_rows],'scope':'Engineering audit; no thresholds relaxed.'})
        # Keep the failed cached-scoring audit visible. Free generation is
        # independent of that scorer; full-prefix rescoring is a separate step.
        for off in range(0,len(selected),4):
            batch=selected[off:off+4];width=max(len(r['tokens']) for r in batch)
            ids=torch.full((len(batch),width),engine.tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
            for i,r in enumerate(batch):
                n=len(r['tokens']);ids[i,-n:]=torch.tensor(r['tokens'],device='cuda');mask[i,-n:]=1
            value=engine.model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=64,do_sample=False,pad_token_id=engine.tok.pad_token_id)
            for r,seq in zip(batch,value[:,width:]):
                text=engine.tok.decode(seq,skip_special_tokens=True)
                colors=[v[r['task']] for v in C.COLORS]
                found=[(m.start(),color) for color in colors for m in re.finditer(re.escape(color),text,re.I)]
                first=min(found)[1] if found else None
                expected=r['slots']['EC'] if r['task'] else r['slots']['C']
                out.append({'id':r['id'],'family':r['family'],'branch':r['branch'],'task':r['task'],'style':r['style'],'expected':expected,'first_color':first,'first_color_correct':first==expected,'tokens':seq.tolist(),'text':text,'reached_64_tokens':len(seq)==64 and int(seq[-1]) not in {engine.tok.eos_token_id,engine.tok.pad_token_id}})
            if len(out)%16==0 or len(out)==len(selected):C.log('confirm_free_audit',len(out),len(selected),start)
    report={'kind':'real','model_snapshot':frozen['model_snapshot'],'frozen_sha256':C.file_sha(C.OUT/'frozen.json'),'rows':out,'families':len(selected),'first_color_accuracy':float(np.mean([r['first_color_correct'] for r in out])),'independent_cached_vs_full_score_max_error_nat':scorer_error,'cached_scorer_audit_passed':scorer_error<1.,'scope':'One balanced condition per independent confirmation family; greedy 64 tokens. First color is an operational lexical screen, not complete semantic/style correctness. All mismatches retained for review.','source_sha256':C.file_sha(__file__)}
    C.save(C.OUT/'free_audit.json',report)
    print({'families':len(out),'first_color_accuracy':report['first_color_accuracy']},flush=True)

if __name__=='__main__':main()
