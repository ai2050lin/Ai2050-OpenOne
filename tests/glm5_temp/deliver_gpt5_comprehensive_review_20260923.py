"""Create the requested review tables and append a complete record safely."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/gpt5_comprehensive_review_20260923'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
REPORT = OUT / 'review.md'

def sha(b):
    return hashlib.sha256(b).hexdigest()

def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def expand_numbers(s):
    ids=set()
    for m in re.finditer(r'(\d{4})(?:[–-](\d{4}))?', s):
        ids.update(range(int(m[1]), int(m[2] or m[1])+1))
    return ids

def main():
    report=REPORT.read_text(encoding='utf-8')
    inv=json.loads((OUT/'source_inventory.json').read_text(encoding='utf-8'))
    checks=json.loads((OUT/'artifact_checks.json').read_text(encoding='utf-8'))
    claims=[]
    for line in report.splitlines():
        if re.match(r'^\| R\d\d \|', line):
            parts=[p.strip() for p in line.strip('|').split('|')]
            assert len(parts)==7, (len(parts),line)
            keys=['id','gpt_phases','claim','judgment','retained_basis','problem','next_step']
            claim=dict(zip(keys,parts))
            wanted=expand_numbers(claim['gpt_phases'])
            refs=[]
            for p in inv['phases']:
                if expand_numbers(p['phase']) & wanted:
                    refs.append(dict(phase=p['phase'],line=p['line']))
            assert refs, claim['id']
            claim['source_sections']=refs
            claim['evidence_note']='Route-level judgment; specific numerical rechecks are separately enumerated in artifact_checks.json and mathematical_checks.json.'
            claims.append(claim)
    assert len(claims)==57
    assert len({c['id'] for c in claims})==57
    write_json('claim_ledger.json',dict(source_sha256=inv['sha256'],claims=claims))
    src=(ROOT/'research/gpt5/docs/AGI_GPT5_MEMO.md').as_posix()
    snap=(OUT/'source_snapshot.md').as_posix()
    rows=['# 全部Phase标题段覆盖索引','',
          '共344段；合并编号保持原样。此表表示进入审查范围和关联主张，不表示每段的全部数值都独立重算。',
          '“结果身份核对”仅指选定result/seal；是否做CPU数值重算另见本轮结果。所有判断须连同review.md的边界阅读。','',
          '| 序号 | 原Phase标题 | 核心主张ID | 结果身份核对 | 冻结来源 |',
          '| --- | --- | --- | --- | --- |']
    selected=set(checks['selected_phases'])
    coverage=[]
    for i,p in enumerate(inv['phases'],1):
        ids=expand_numbers(p['phase'])
        related=[c['id'] for c in claims if c['id'] not in ('R56','R57') and any(expand_numbers(s['phase'])&ids for s in c['source_sections'])]
        if not related: related=['R57（整体与证据边界）']
        checked=bool(ids&selected)
        title=p['title'].strip().removeprefix('## ').replace('|','\\|').replace('[','\\[').replace(']','\\]')
        rows.append(f"| {i} | [{title}]({src}:{p['line']}) | {', '.join(related)} | {'是（抽查）' if checked else '未逐段核对'} | [快照]({snap}:{p['line']}) |")
        coverage.append(dict(**p,claim_ids=related,result_identity_spotcheck=checked))
    assert len(coverage)==344
    assert all('\r' not in row and '\n' not in row for row in rows)
    (OUT/'phase_coverage.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    write_json('phase_coverage.json',coverage)
    # Capture cited subsequent corrections before appending this review.
    current=MEMO.read_bytes()
    gt=current.decode('utf-8-sig')
    gh=list(re.finditer(r'^## Phase (\d+)[:：].*$',gt,re.M))
    selected_sections=[]
    follow_refs=[]
    for i,m in enumerate(gh):
        if 2751<=int(m[1])<=2756:
            section=gt[m.start():gh[i+1].start() if i+1<len(gh) else len(gt)]
            selected_sections.append(section)
            follow_refs.append(dict(phase=int(m[1]),source_line=gt.count('\n',0,m.start())+1,
                                    section_sha256=sha(section.encode('utf-8'))))
    (OUT/'followup_context_snapshot.md').write_text('\n'.join(selected_sections),encoding='utf-8')
    write_json('followup_context_sources.json',dict(source=str(MEMO),source_sha256=sha(current),sections=follow_refs,
               note='Existing later GLM results, not experiments run in this review.'))
    # Validate key reported rechecks against machine outputs.
    mat=json.loads((OUT/'mathematical_checks.json').read_text(encoding='utf-8'))
    cont=json.loads((OUT/'continuum_reanalysis.json').read_text(encoding='utf-8'))
    assert mat['phase3075_empirical_submodularity']['positive_over_002']==467
    assert mat['phase3075_empirical_submodularity']['conditional_second_difference_count']==1792
    assert abs(cont['analyses']['n12']['rho']-.9296005160692782)<1e-12
    assert cont['analyses']['n15_14B']['p_two_sided']==.15
    assert not any(r.get('seal_result_matches') is False for r in checks['result_files'])
    validation=dict(claims=57,phase_sections=344,selected_result_files=len(checks['result_files']),
                    all_claims_have_source_sections=True,reported_numerical_rechecks_match=True,
                    model_forwards_this_run=0,method='CPU frozen-array reanalysis, targeted source inspection, mathematical checks and route-level review.')
    write_json('validation.json',validation)
    if '--append' not in sys.argv:
        print(json.dumps(validation,ensure_ascii=False))
        return
    receipt=OUT/'append_receipt.json'
    if receipt.exists():
        print(receipt.read_text(encoding='utf-8'))
        return
    # Refuse a stale append. Keep the prefix byte-for-byte and never rewrite old phases.
    before=MEMO.read_bytes()
    assert before==current, 'Memo changed during preparation; rerun to choose the latest Phase.'
    numbers=[int(x) for x in re.findall(r'^## Phase (\d+)[:：]',before.decode('utf-8-sig'),re.M)]
    phase=max(numbers)+1
    local=datetime.now().astimezone()
    heading=f"## Phase {phase}: GPT5研究历史综合审查、57项主张分级与完整覆盖表 [{local:%Y-%m-%d %H:%M}]"
    body='\n'.join(report.splitlines()[1:]).strip()
    body=re.sub(r'^(#{2,5}) ',r'\1# ',body,flags=re.M)
    lead=f'\n\n{heading}\n\n状态：已完成文档与CPU审查。本轮未加载模型。结果目录：tests/glm5/result/gpt5_comprehensive_review_20260923/。审查脚本：tests/glm5/comprehensive_gpt5_review_20260923.py；交付脚本：tests/glm5_temp/deliver_gpt5_comprehensive_review_20260923.py。\n\n'
    addition=(lead+body+'\n').encode('utf-8')
    with MEMO.open('ab') as f:
        f.write(addition)
        f.flush()
    after=MEMO.read_bytes()
    assert after[:len(before)]==before, 'Append-only prefix check failed'
    assert after[len(before):len(before)+len(addition)]==addition
    artifacts={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in OUT.iterdir() if p.is_file() and p.name!='append_receipt.json'}
    write_json('append_receipt.json',dict(phase=phase,heading=heading,source_sha256=inv['sha256'],
               completed_local=local.isoformat(),completed_utc=datetime.now(timezone.utc).isoformat(),
               memo=str(MEMO),start_line=before.count(b'\n')+3,
               bytes_before=len(before),appended_bytes=len(addition),original_prefix_preserved=True,
               before_sha256=sha(before),after_sha256=sha(after),artifacts=artifacts,
               validation=validation))
    print(json.dumps(dict(phase=phase,heading=heading,start_line=before.count(b'\n')+3,
                         original_prefix_preserved=True,claims=57,phase_sections=344),ensure_ascii=False))

if __name__=='__main__':
    main()
