"""Append the final identity-cluster sensitivity qualification and refresh manifest."""
import json,hashlib,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tests/glm5'))
from phase2751_trusted_rebuild import OUT,sha,write
now=datetime.now().astimezone()
marker='C004：统计单位的进一步收紧'
note=f'''

### {marker} [{now:%Y-%m-%d %H:%M}]

最终复核补充：不同任务族复用了相同的实体编号，旧材料各族也复用了相同连接词编号。因此语义源组数量不等于身份或模板完全独立的统计重复数。冻结`material.json`中的`independent_source_groups=48`字段只按语义源组计数，其名称过强，**不得据此声称48个统计独立实体/模板**；主4B完整设计只有16个主实体编号、14B为8个，每个留出拆分分别只有8/4个编号。字段保留以维持原材料哈希，解释在此追加纠正。

已执行第二套分组敏感性分析：把同一实体/连接词编号跨任务族、条件和表述一起重采样4000次，结果保存在`bootstrap_sensitivity.json`。这是更粗分组的依赖敏感性检查，区间不保证一律更宽，不能挑选更有利口径。

等长且实体隔离的4B诊断中，固定交互校正相对加性的配对误差差值：实体留出−0.0477，编号块95%区间[−0.0800,−0.0403]；表述留出+0.0096，区间[−0.0031,+0.0372]；双留出+0.0060，区间[−0.0218,+0.0454]。因此“相同表述下有收益、跨表述收益未稳定复现”的范围限制仍应保留。

14B表述留出区间是否跨零随分组口径变化，且只有4个编号块，不能升级为稳健推广；双留出两种口径均跨零。主要材料中的参与者重叠、固定的两种表述和三个任务族仍限制总体推断。后续扩规模应增加独立词汇集合、表述来源和关系结构，而不仅增加同一身份的改写与前向次数。
'''
report=OUT/'phase_report.md';glm=root/'research/glm5/docs/AGI_GLM5_MEMO.md'
for p in [report,glm]:
    before=p.read_bytes();assert marker not in before.decode('utf-8-sig')
    with p.open('ab') as f:f.write(note.encode('utf-8'))
    assert p.read_bytes()[:len(before)]==before
ledger_path=OUT/'evidence/claims.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
for r in ledger['claims']:
    if r['id']=='RDC-N01':
        r['sampling_qualification']='Semantic-source and identity-block sensitivity both reported; only8/4 identity blocks per holdout split. Primary entity split has partner overlap.'
        r['sensitivity']='bootstrap_sensitivity.json'
write(ledger_path,ledger)
index_path=OUT/'index.json';index=json.loads(index_path.read_text(encoding='utf-8'))
index.update(report='phase_report.md',trusted_kernel='trusted_kernel.json',sampling_sensitivity='bootstrap_sensitivity.json',
             material_audit='material_audit.json',length_control='length_control/4B/summary.json')
write(index_path,index)
source=root/'tests/glm5/phase2751_cluster_sensitivity.py'
(OUT/'code_snapshots'/(sha(source)+'.py')).write_bytes(source.read_bytes())
receipt_path=OUT/'completion_receipt.json';receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
receipt['sampling_note_added_local']=now.isoformat()
receipt['results']={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=receipt_path}
write(receipt_path,receipt)
print('Sampling qualification appended; manifest refreshed.')
