"""Additional pre-run registered no-conflict control, not a new primary test."""
from pathlib import Path
import sys,json,re
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from phase2794_run import install
from phase2794_analyze import ci
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    c=read(P.OUT/'formal-corpus.json');selection=read(P.OUT/'selection.json');items=[]
    for r in c['items']:
        if r['split']!='unseen_combination' or r['frame']!=2:continue
        # Remove the false fact without changing the independent verified card.
        source=r['source_text'].replace('原子序数'+r['document'],'原子序数'+r['prior']).replace('为'+r['document'],'为'+r['prior'])
        assert source!=r['source_text']
        items.append({**r,'id':r['id']+':NO-CONFLICT','prompt':r['prompt'].replace(r['source_text'],source),
            'source_text':source,'document':r['prior'],'expected':r['prior']})
    assert len(items)==128
    P.save(P.OUT/'numeric-control-contract.json',{'kind':'real','run_id':P.RUN,'script_sha256':file_sha(__file__),
        'selection_sha256':file_sha(P.OUT/'selection.json'),'inputs':items,'conditions':['base','mlp32','random32'],
        'status':'additional control registered before own outputs; not H2 reselection; conflicting fact replaced by matching real number; token length may differ by1',
        'criterion':'descriptive paired loss difference: similar damage suggests numerical-answer support rather than arbitration-specific gate'},True)
    rows=[]
    with GpuLease():
        torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        for condition,group in [('base',[]),('mlp32',selection['units']),('random32',selection['random_units'])]:
            hs=install(m,group,1.)
            try:
                for r in items:
                    ids=tok.encode(r['prompt'],add_special_tokens=False)
                    with torch.inference_mode():out=m.generate(input_ids=torch.tensor([ids],device='cuda'),max_new_tokens=4,do_sample=False,pad_token_id=tok.eos_token_id)
                    actual=tok.decode(out[0,len(ids):],skip_special_tokens=True).strip()
                    rows.append({'sample_id':r['id'],'family_id':r['family_id'],'scope':r['scope'],'condition':condition,'input_tokens':len(ids),
                        'expected':r['expected'],'generated':actual,'correct':actual==r['expected']})
            finally:
                for h in hs:h.remove()
    base={r['sample_id']:r for r in rows if r['condition']=='base'};effects={}
    for condition in ['mlp32','random32']:
        rr=[r for r in rows if r['condition']==condition]
        effects[condition]=ci([np.mean([int(base[r['sample_id']]['correct'])-int(r['correct']) for r in rr if r['family_id']==e['id']]) for e in c['entities'][:64]])
    result={'kind':'real','run_id':P.RUN,'rows':rows,'effects':effects,'accuracy':{s:float(np.mean([r['correct'] for r in rows if r['condition']==s])) for s in ['base','mlp32','random32']},
        'provenance':selection['provenance'],'limit':'additional descriptive control, no primary promotion; length may differ by1token'}
    P.save(P.OUT/'numeric-control.json',result,True);print(json.dumps({'effects':effects,'accuracy':result['accuracy']},ensure_ascii=False))
if __name__=='__main__':main()
