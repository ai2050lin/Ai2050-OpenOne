"""Finalize this audit and append its report to the mandated GLM memo only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

root = Path(__file__).resolve().parents[2]
out = root / 'tests/glm5/result/gpt5_memo_audit_20260923'
target = root / 'research/glm5/docs/AGI_GLM5_MEMO.md'
sha = lambda b: hashlib.sha256(b).hexdigest()
inv = json.loads((out / 'inventory.json').read_text(encoding='utf-8'))
art = json.loads((out / 'artifact_checks.json').read_text(encoding='utf-8'))
stats = json.loads((out / 'continuum_reanalysis.json').read_text(encoding='utf-8'))
checks = json.loads((out / 'mathematical_checks.json').read_text(encoding='utf-8'))
assert inv['sections'] == 344 and inv['lines'] == 13134
assert inv['memo_sha256'] == sha((out / 'memo_snapshot.md').read_bytes())
assert len(art['selected_phases']) == 46 and len(art['result_files']) == 49
assert len(art['script_sha256']) == 47
assert sum(r.get('seal_result_matches') is True for r in art['result_files']) == 21
assert not any(r.get('seal_result_matches') is False for r in art['result_files'])
assert abs(stats['analyses']['n12']['p_two_sided'] - 2/24) < 1e-12
assert abs(stats['analyses']['n15_14B']['p_two_sided'] - .15) < 1e-12
assert checks['audit_self_checks'] == 'passed'
assert checks['phase3075_empirical_submodularity']['positive_over_002'] == 467
assert checks['phase3075_empirical_submodularity']['conditional_second_difference_count'] == 1792

before = target.read_bytes()
before_text = before.decode('utf-8-sig')
marker = 'GPT5 Phase 2750–3100 研究可信度全面审查'
assert marker not in before_text, 'Audit already appended; do not duplicate.'
phases = [int(x) for x in re.findall(r'^## Phase (\d+)', before_text, re.M)]
phase = max(phases) + 1
now = datetime.now().astimezone()
header = f'## Phase {phase}: {marker} [{now:%Y-%m-%d %H:%M}]'
report = (out / 'review.md').read_text(encoding='utf-8')
body = report.split('\n', 1)[1].lstrip()
addition = ('\n\n' + header + '\n\n'
    '**状态：已完成文档、代码与冻结数据审查；未执行新的模型实验。** '
    '本节接续 GLM5 研究记录的 Phase 编号；所审对象为 GPT5 分支的 Phase 2750–3100，二者勿混淆。'
    '遵照当前用户要求，以追加方式保存审查；不覆盖历史结论或原始产物。\n\n' + body + '\n').encode('utf-8')
assert target.read_bytes() == before, 'Memo changed before append; inspect and retry.'
with target.open('ab') as f:
    f.write(addition)
after = target.read_bytes()
assert after[:len(before)] == before
assert after[len(before):] == addition

receipt = dict(
    completed_local=now.isoformat(), completed_utc=datetime.now(timezone.utc).isoformat(),
    timezone_note='Environment timezone America/Chicago; local datetime includes actual UTC offset.',
    scope='GPT5 memo frozen through Phase3100; targeted code/data audit, not all GPU experiments replicated.',
    gpu_experiments_run=0, original_experiment_files_modified=False,
    snapshot_sha256=inv['memo_sha256'], source_current_sha256=sha((root/'research/gpt5/docs/AGI_GPT5_MEMO.md').read_bytes()),
    append=dict(path=str(target), phase=phase, heading=header, start_line=before.count(b'\n')+3,
                bytes_before=len(before),bytes_appended=len(addition),before_sha256=sha(before),
                after_sha256=sha(after),original_prefix_preserved=True),
    artifacts={str(p.relative_to(root)):sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file() and p.name!='audit_receipt.json'},
    audit_script_sha256=sha((root/'tests/glm5/audit_gpt5_memo_20260923.py').read_bytes()),
    finalizer_script_sha256=sha(Path(__file__).read_bytes()),
    verification='Artifact counts, snapshot identity, block-permutation outputs, mathematical checks and append-only preservation passed.'
)
(out/'audit_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(phase=phase,heading=header,start_line=receipt['append']['start_line'],
                     bytes_appended=len(addition),original_prefix_preserved=True,report_characters=len(report)),ensure_ascii=False))
