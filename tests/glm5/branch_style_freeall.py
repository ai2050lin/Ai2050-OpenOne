"""Complete confirmation-cohort free generation and task-compliance audit.

Secondary audit after discovering forced-choice/translation-compliance mismatch;
all cells retained, no tuning or feature refitting.
"""
import re
import sys
from pathlib import Path
import time
import json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C

def inspect(row,text):
    palette=[v[row['task']] for v in C.COLORS]
    expected=row['slots']['EC'] if row['task'] else row['slots']['C']
    if row['task']:
        pattern=r'\b(?:is|are|was|were|has)\s+(?:(?:a|an)\s+)?('+ '|'.join(palette)+r')\b'
        pred=re.search(pattern,text,re.I)
        found=pred.group(1).lower() if pred else None
        english=not bool(re.search(r'[\u4e00-\u9fff]',text)) and bool(re.search(r'[A-Za-z]',text))
        if found is None:
            colors=[(m.start(),p) for p in palette for m in re.finditer(r'\b'+p+r'\b',text,re.I)]
            if len(colors)==1:found=colors[0][1]
    else:
        name=re.escape(row['slots']['X'])
        pred=re.search(name+r'[^。；\n]{0,25}?('+ '|'.join(palette)+')',text)
        if pred:found=pred.group(1)
        else:
            colors=[(m.start(),p) for p in palette for m in re.finditer(p,text)]
            found=min(colors)[1] if colors else None
        english=True
    return {'predicted_color':found,'color_correct':found==expected,'output_language_compliant':english,'joint_color_language':found==expected and english}

def summarize(rows):
    cells={}
    for b in range(3):
        for s in range(2):
            for t in range(2):
                a=[r for r in rows if (r['branch'],r['style'],r['task'])==(b,s,t)]
                cells[f'b{b}-s{s}-t{t}']={'n':len(a),**{k:float(np.mean([r[k] for r in a])) for k in ['color_correct','output_language_compliant','joint_color_language']}}
    return cells

def main():
    import torch
    path=C.OUT/'free_all_rows.jsonl';rows=C.selected_rows('confirm');done=[]
    if path.exists():done=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    assert all(x['id']==r['id'] for x,r in zip(done,rows))
    start=time.time();C.log('confirm_free_all',len(done),len(rows),start)
    with C.gpu_lease(),torch.inference_mode():
        engine=C.Engine(False)
        for off in range(len(done),len(rows),4):
            batch=rows[off:off+4];width=max(len(r['tokens']) for r in batch)
            ids=torch.full((len(batch),width),engine.tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
            for i,r in enumerate(batch):
                n=len(r['tokens']);ids[i,-n:]=torch.tensor(r['tokens'],device='cuda');mask[i,-n:]=1
            value=engine.model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=64,do_sample=False,pad_token_id=engine.tok.pad_token_id)
            for r,seq in zip(batch,value[:,width:]):
                text=engine.tok.decode(seq,skip_special_tokens=True)
                rec={'id':r['id'],'family':r['family'],'branch':r['branch'],'style':r['style'],'task':r['task'],'expected':r['slots']['EC'] if r['task'] else r['slots']['C'],'text':text,'tokens':seq.tolist(),**inspect(r,text)}
                done.append(rec)
                with path.open('a',encoding='utf-8') as out:out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush()
            if len(done)%32==0 or len(done)==len(rows):C.log('confirm_free_all',len(done),len(rows),start)
    C.save(C.OUT/'free_all_report.json',{'kind':'real','families':128,'rows':len(done),'cells':summarize(done),'joint_color_language_accuracy':float(np.mean([r['joint_color_language'] for r in done])),'language_failure_count':sum(not r['output_language_compliant'] for r in done),'scope':'All 1536 confirmation prompts, greedy max 64 tokens. Prespecified transparent predicate/lexical screen, imperfect for unusual paraphrases; source texts and all outputs retained. Secondary audit, not retrospective feature selection.','source_sha256':C.file_sha(__file__),'model_snapshot':C.read(C.OUT/'frozen.json')['model_snapshot']})
    print({'rows':len(done),'joint_accuracy':np.mean([r['joint_color_language'] for r in done]),'language_failures':sum(not r['output_language_compliant'] for r in done)},flush=True)

if __name__=='__main__':main()
