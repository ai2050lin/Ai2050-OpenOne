"""Post-hoc numerical sham audit, separate from frozen research verdicts."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
from native_mlp_long_validate import install
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha
from shared.dist_protocol.natural_register_runner import snapshot
def main():
    import torch
    from transformers import AutoModelForCausalLM
    s=P.read(P.OUT/'selection.json');d=P.read(P.OUT/'documents.json');prov=s['provenance'];raw=np.load(P.OUT/'input-tokens.npy',mmap_mode='r')
    P.save(P.OUT/'numeric-contract.json',{'kind':'real','run_id':P.RUN,'script_sha256':file_sha(__file__),'n':32,'selection':'first32 frozen lockbox A-documents, not chosen by effect',
        'scope':'post-hoc exact-copy alpha0 numerical sham; no new mechanism claim','provenance':prov})
    rows=[]
    with GpuLease():
        assert snapshot(P.MODEL)==prov['model_revision'];torch.set_num_threads(4)
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        for pair in d['pairs']['lockbox'][:32]:
            ids=torch.tensor(np.asarray(raw[pair['a'],:1024])[None],device='cuda');q=pair['position']
            with torch.inference_mode():base=m(input_ids=ids,use_cache=False,logits_to_keep=torch.tensor([q],device='cuda')).logits[0,-1].clone()
            hs=install(m,s['units'],0,q)
            try:
                with torch.inference_mode():sham=m(input_ids=ids,use_cache=False,logits_to_keep=torch.tensor([q],device='cuda')).logits[0,-1]
            finally:
                for h in hs:h.remove()
            rows.append({'doc_index':pair['a'],'position':q,'bitwise_equal':bool(torch.equal(base,sham)),'max_abs_error':float((base.float()-sham.float()).abs().max())})
        assert snapshot(P.MODEL)==prov['model_revision']
    P.save(P.OUT/'numeric-result.json',{'kind':'real','run_id':P.RUN,'rows':rows,'all_equal':all(r['bitwise_equal'] for r in rows),'provenance':prov});print(json.dumps({'all_equal':all(r['bitwise_equal'] for r in rows)}))
if __name__=='__main__':main()
