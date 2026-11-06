"""Predeclared repeated-pool monitor for a discovery-selected native MLP group."""
from pathlib import Path
import json
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from phase2794_run import install
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.natural_register_runner import snapshot


def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    out=P.OUT;selection=json.loads((out/'selection.json').read_text(encoding='utf-8'))
    inputs=P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json';qs=json.loads(inputs.read_text(encoding='utf-8'))
    prov=selection['provenance'];units=selection['units']
    contract={'kind':'real','run_id':P.RUN,'status':'prospectively declared repeated-pool capability monitor, selected32 MLP alpha1',
        'input_sha256':file_sha(inputs),'selection_sha256':file_sha(out/'selection.json'),'units':units,
        'script_sha256':file_sha(__file__),'margin':.02,'confidence':.95,'method':'paired simultaneous Clopper-Pearson loss/gain bounds',
        'gate_eligible':sum(q['gate_eligible'] for q in qs),'reference_only':sum(not q['gate_eligible'] for q in qs),'provenance':prov}
    P.save(out/'capability-contract.json',contract,True)
    with GpuLease():
        assert snapshot(P.MODEL)==prov['model_revision'];torch.set_num_threads(4)
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);tok.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
        letterids=[tok.encode(l,add_special_tokens=False) for l in 'ABCDE'];assert all(len(x)==1 for x in letterids)
        ids=[x[0] for x in letterids];rows={};started=time.time()
        for condition in ['base','selected_units']:
            handles=[]
            if condition=='selected_units':
                handles=install(model,units,1.,'mlp')
            records=[]
            try:
                for start in range(0,len(qs),4):
                    chunk=qs[start:start+4];tokens=[]
                    for q in chunk:
                        text=q['question']+'\n'+'\n'.join(f'{s}. {choice}' for s,choice in zip('ABCDE',q['choices']))+'\n请只回答正确选项字母。'
                        prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        tokens.append(tok.encode(prompt,add_special_tokens=False))
                    batch={k:v.cuda() for k,v in tok.pad({'input_ids':tokens},padding=True,return_tensors='pt').items()}
                    with torch.inference_mode():o=model(**batch,use_cache=False,logits_to_keep=1)
                    for j,q in enumerate(chunk):
                        scores=o.logits[j,-1,ids[:len(q['choices'])]].float().cpu();pred='ABCDE'[int(scores.argmax())]
                        records.append({'id':q['id'],'prediction':pred,'answer':q['answer'],'correct':pred==q['answer'],
                            'source':q['source'],'gate_eligible':q['gate_eligible'],'choice_logits':scores.tolist(),'input_tokens':len(tokens[j])})
                    del o,batch
                    if start%400==0:print(json.dumps({'condition':condition,'done':start+len(chunk),'total':len(qs)}),flush=True)
            finally:
                for h in handles:h.remove()
            rows[condition]=records;P.save(out/f'capability-{condition}.json',records,True)
        result=P.OLD.W.paired_interval([x['correct'] for x in rows['base']],[x['correct'] for x in rows['selected_units']],np.asarray([q['gate_eligible'] for q in qs]))
        result.update(kind='real',run_id=P.RUN,scope=contract['status'],seconds=time.time()-started,provenance=prov)
        P.save(out/'capability-result.json',result,True);print(json.dumps(result),flush=True)
        assert snapshot(P.MODEL)==prov['model_revision']


if __name__=='__main__':main()
