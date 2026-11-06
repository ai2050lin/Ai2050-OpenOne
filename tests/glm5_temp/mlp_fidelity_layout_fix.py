from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[2];base=root/'tests/glm5';old=base/'mlp_fidelity_v2_causal.py';text=old.read_text(encoding='utf-8')
changes={
 'import mlp_fidelity_v2_tail as T':'import mlp_fidelity_v2_tail_v2 as T',
 "P.OUT/('tail-'+part)":"P.OUT/('tail-'+part+'-v2')",
 'T.tail_logits(m,r,y)':"T.tail_logits(m,r,y,ctx['native_norm_input'],positions,keep)",
 "T.tail_logits(m,r,ops['sae_full'])":"T.tail_logits(m,r,ops['sae_full'],ctx['native_norm_input'],positions,keep)",
 'T.tail_logits(m,r,yy)':"T.tail_logits(m,r,yy,ctx['native_norm_input'],positions,keep)",
 "('final-hidden',m.model.norm((r+y)[:,None])[:,0])":"('final-hidden',ctx['native_final_hidden'])"}
for a,b in changes.items():assert a in text,a;text=text.replace(a,b)
needle="        if part=='natural':\n            items="
addition="""        def norm_in(_m,args):
            if ctx['capture']:ctx['native_norm_input']=args[0].detach().clone()
        def norm_out(_m,args,value):
            if ctx['capture']:ctx['native_final_hidden']=take(value)
        hooks.extend([m.model.norm.register_forward_pre_hook(norm_in),m.model.norm.register_forward_hook(norm_out)])
"""
assert needle in text;text=text.replace(needle,addition+needle);new=base/'mlp_fidelity_v2_causal_v2.py';new.write_text(text,encoding='utf-8')
d={'schema':'research_execution_correction.v1','kind':'real','run_id':'MLP-FIDELITY-20261104-v2','reason':'native norm andresidualsum wereexact; only BF16lm_head inputstride changed. CUDAfull-layout replay restoresbitwise equality. No tolerance relaxed; oldnatural resultvalidbecauseits tensor-indexedheadwascontiguous andpassedtruehookverification. Failedgeneratedv1 retained withoutresults.',
 'old_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_source_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'math_sha256':hashlib.sha256((base/'mlp_fidelity_v2_tail_v2.py').read_bytes()).hexdigest(),'diagnostic_sha256':hashlib.sha256((root/'tests/glm5/result/mlp_fidelity_v2/tail-layout-diagnostic.json').read_bytes()).hexdigest(),'strict_bitwise_requirement_preserved':True}
(root/'research/mlp_fidelity/v2/tail-layout-correction.research.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');print(d)
