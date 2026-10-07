"""Recover exact executed revisions by matching the hashes in execution receipts."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[2]
out=root/'tests/glm5/result/rdc_trusted_rebuild_20260923'
dest=out/'code_snapshots';dest.mkdir(exist_ok=True)
file=root/'tests/glm5/phase2751_trusted_rebuild.py'
saved_v2=dest/'f27442596554e79d85d9b27e77ce779d03225359261cf68dcd227556d9b823c0.py'
v2=saved_v2.read_text(encoding='utf-8') if saved_v2.exists() else file.read_text(encoding='utf-8')
start=v2.index("    elif side=='14B':\n        # Cost-only")
end=v2.index('    if limit:rows=rows[:limit]',start)
v1=v2[:start]+v2[end:]
start=v1.index("        write(dest/'anchors.json',anchors)\n        # Accelerate")
end=v1.index("        np.savez(dest/'lastblock_weights.npz'",start)
original="""        weights={k:v.detach().float().cpu().numpy() for k,v in dict(Wg=last.mlp.gate_proj.weight,Wu=last.mlp.up_proj.weight,
            Wd=last.mlp.down_proj.weight,gamma=last.post_attention_layernorm.weight,
            final_gamma=model.model.norm.weight).items()}
"""
v0=v1[:start]+original+v1[end:]
v0=v0.replace("                validation=measure(j,fd)\n                assert validation['relative_error']<1e-5, validation\n                jvp_validation.append(dict(id=rows[c]['id'],finite_difference=validation))",
              "                jvp_validation.append(dict(id=rows[c]['id'],finite_difference=measure(j,fd)))")
versions={hashlib.sha256(v.encode()).hexdigest():v for v in [v0,v1,v2,file.read_text(encoding='utf-8')]}
receipts=[]
for p in out.glob('*/execution.json'):
    d=json.loads(p.read_text(encoding='utf-8'));key=d['script_sha256']
    assert key in versions,(str(p),key,list(versions))
    target=dest/(key+'.py');target.write_bytes(versions[key].encode())
    assert hashlib.sha256(target.read_bytes()).hexdigest()==key
    receipts.append(dict(execution=str(p.relative_to(out)),source='code_snapshots/'+target.name,
                         match=True,note='Exact content reconstruction checked against execution-time SHA, not a retroactive timestamp assertion.'))
for name in ['rdc_trusted_measurements.py','phase2751_evidence_repair.py','phase2751_rebuild_summary.py','phase2751_length_control.py']:
    source=root/'tests/glm5'/name;data=source.read_bytes();key=hashlib.sha256(data).hexdigest()
    (dest/(key+'.py')).write_bytes(data)
(dest/'recovery_receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print('Exact executed source versions recovered:',len(versions),'receipts:',len(receipts))
