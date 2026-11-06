"""Calibration accesses only discovery entities and frames, not heldout outputs."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha
def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    c=json.loads((P.OUT/'pilot-corpus.json').read_text(encoding='utf-8'));rows=[]
    P.save(P.OUT/'pilot-contract.json',{'kind':'real','run_id':P.RUN,'corpus_sha256':file_sha(P.OUT/'pilot-corpus.json'),
        'script_sha256':file_sha(__file__),'criterion':'each frame/scope accuracy>=.8; no exclusions; only16 discovery entities, F0/F1,1024tokens'},True)
    with GpuLease():
        assert torch.cuda.is_available();torch.set_num_threads(4)
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        for item in c['items']:
            with torch.inference_mode():generated=m.generate(input_ids=torch.tensor([item['token_ids']],device='cuda'),max_new_tokens=4,do_sample=False,pad_token_id=tok.eos_token_id)
            actual=tok.decode(generated[0,len(item['token_ids']):],skip_special_tokens=True).strip()
            rows.append({'sample_id':item['id'],'scope':item['scope'],'frame':item['frame'],'expected':item['expected'],'predicted':actual,'correct':actual==item['expected']});del generated
        cells=[{'frame':f,'scope':s,'n':16,'acc':sum(x['correct'] for x in rows if x['frame']==f and x['scope']==s)/16} for f in [0,1] for s in ['文档','现实']]
        d={'kind':'real','run_id':P.RUN,'cells':cells,'rows':rows,'passed':all(x['acc']>=.8 for x in cells),'model_snapshot':c['model_snapshot']}
        P.save(P.OUT/'pilot-result.json',d,True);print(json.dumps({'cells':cells,'passed':d['passed']},ensure_ascii=False))
if __name__=='__main__':main()
