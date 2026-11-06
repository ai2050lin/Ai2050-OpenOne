from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2];base=root/'tests/glm5';old=base/'mlp_fidelity_v2_causal_v2.py';s=old.read_text(encoding='utf-8')
a="prompts.append(tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False))"
b="prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))"
assert a in s;s=s.replace(a,b).replace("'tail-'+part+'-v2'","'tail-'+part+'-v3'").replace("choices=['natural','generated','capability']","choices=['capability']")
p=base/'mlp_fidelity_v2_causal_v3.py';p.write_text(s,encoding='utf-8')
d={'schema':'research_execution_correction.v1','kind':'real','run_id':'MLP-FIDELITY-20261104-v2','reason':'Installed tokenizer apply_chat_template(tokenize=True) returned BatchEncoding, not a token list. Render text then encode exactly as archived capability monitor. Failed firstbatch before forward, no capability results produced; thresholds and source questions unchanged.','old_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'output':'tail-capability-v3'}
(root/'research/mlp_fidelity/v2/capability-tokenizer-correction.research.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');print(d)
