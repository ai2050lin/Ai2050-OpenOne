"""Narrow the closeout inventory to this run's explicitly owned scripts.

The first inventory glob also matched other ongoing work in this shared folder.
Keep that inventory as history; never alter the other task's files.
"""
import sys,json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
owned={
    'conditional_lookup_native.py','conditional_lookup_prereg.md',
    'conditional_lookup_audit.py','conditional_lookup_closeout.py',
    'conditional_lookup_collect.py','conditional_lookup_analyze.py',
    'conditional_lookup_natural_design.md','conditional_lookup_natural_audit.py',
    'conditional_lookup_plot.py','conditional_lookup_examples.py',
    'conditional_lookup_natural_closeout.py','conditional_lookup_manifest_repair.py',
}
p=OUT/'natural-manifest.json';old=D.read(p);kept={};excluded=[]
for n,h in old['files'].items():
    path=ROOT/n
    if path.parent==ROOT/'tests/glm5' and path.name not in owned:excluded.append(n)
    else:kept[n]=h
for n,h in kept.items():assert D.sha(ROOT/n)==h,(n,'scientific artifact changed')
historical=OUT/'natural-manifest-v1-broad-glob.json';assert not historical.exists();p.rename(historical)
timestamp=datetime.now().strftime('%Y-%m-%d %H:%M')
note=f'\n\n**Phase 2967 清单校正 [{timestamp}]。** 最终核对发现初版清单用conditional_lookup*通配符，误收录共享目录中另一项正在编写的方案/审计脚本，其中机制方案随后变化。本轮冻结的采集/分析代码、封条和所有科学产物哈希均未变化。初版清单保留为natural-manifest-v1-broad-glob.json；现版采用本轮明确脚本白名单，排除其他工作，不修改其文件。修正不改变结果或任何验收门槛。\n'
memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
with memo.open('ab') as f:f.write(note.encode('utf-8'))
report=OUT/'REPORT.zh-CN.md'
with report.open('ab') as f:f.write(note.encode('utf-8'))
kept[str(report.relative_to(ROOT))]=D.sha(report)
kept[str(historical.relative_to(ROOT))]=D.sha(historical)
kept[str(Path(__file__).relative_to(ROOT))]=D.sha(Path(__file__))
new=dict(provenance=old['provenance'],files=kept,status=old['status'],inventory_revision=2,reason='explicit ownership; exclude unrelated concurrently maintained files',excluded_from_this_run=excluded)
p.write_text(json.dumps(new,indent=2),encoding='utf-8')
for n,h in kept.items():assert D.sha(ROOT/n)==h,n
print(json.dumps({'verified_files':len(kept),'excluded_unrelated_files':excluded}),flush=True)
