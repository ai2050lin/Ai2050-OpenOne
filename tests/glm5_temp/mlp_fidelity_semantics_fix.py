from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2];base=root/'tests/glm5';old=base/'mlp_fidelity_v2_semantics.py';text=old.read_text(encoding='utf-8')
for a,b in [('GpuLease,file_sha,unbits','GpuLease,file_sha,unbits,digest'),("P.OUT/'semantics'","P.OUT/'semantics-v2'"),('semantics-execution-freeze.json','semantics-execution-freeze-v2.json'),('semantics-result.json','semantics-result-v2.json')]:
 assert a in text;text=text.replace(a,b)
new=base/'mlp_fidelity_v2_semantics_v2.py';new.write_text(text,encoding='utf-8')
for p in [base/'mlp_fidelity_v2_causal.py',root/'server/mlp_fidelity_service.py']:
 t=p.read_text(encoding='utf-8').replace('semantics-result.json','semantics-result-v2.json').replace("f'semantics/B","f'semantics-v2/B");p.write_text(t,encoding='utf-8')
doc={'schema':'research_execution_correction.v1','kind':'real','run_id':'MLP-FIDELITY-20261104-v2','reason':'semantics stopped by missing digest import during random-control nomination; no criterion or numerical algorithm changed','old_source_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'new_source_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'old_partial_directory_preserved':'tests/glm5/result/mlp_fidelity_v2/semantics','corrected_directory':'tests/glm5/result/mlp_fidelity_v2/semantics-v2'}
(root/'research/mlp_fidelity/v2/semantics-import-correction.research.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8');print(doc)
