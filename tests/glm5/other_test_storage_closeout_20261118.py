"""Close the non-model storage continuation and append its result to GLM5 memo."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import check_write
import other_test_storage_cleanup_20261118 as C

OUT=C.OUT
MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def write(name,text):
    path=OUT/name;payload=text.encode('utf-8');check_write(path,len(payload),operation='其他测试清理交付')
    with path.open('xb') as f:f.write(payload)


def main():
    plan=read('plan.json');verification=read('verification.json');extra=read('extra-protected-verification.json')
    assert verification['passed'] and extra['passed']
    assert verification['completed_scopes']==['native','workspace']
    assert plan['script_sha256']==C.sha(ROOT/'tests/glm5/other_test_storage_cleanup_20261118.py')
    receipts=[read(f'receipt-{s}.json') for s in ['workspace','native']]
    assert all(r['plan_sha256']==C.sha(OUT/'plan.json') for r in receipts)
    assert not (OUT/'memo-append-verification.json').exists()
    grouped=collections.defaultdict(lambda:dict(files=0,logical_bytes=0,retained_review_bytes=0))
    marker_paths=set()
    for receipt in receipts:
        for r in receipt['deleted']:
            g=grouped[r['group']];g['files']+=1;g['logical_bytes']+=r['bytes'];g['retained_review_bytes']+=r.get('review_bytes',0)
            marker=Path(r['path']).parent/'RAW_STORAGE_STATUS_20261118_CONTINUATION.json'
            assert marker.is_file();m=json.loads(marker.read_text(encoding='utf-8'));assert m['plan_sha256']==receipt['plan_sha256']
            marker_paths.add(str(marker))
    reasons=collections.Counter(x['reason'] for x in plan['kept_decisions'])
    frozen_shards=len({r['proof'] for r in plan['records'] if r['group'].startswith('family:')})
    inventory_progress=[json.loads(x) for x in (OUT/'inventory-progress.jsonl').read_text(encoding='utf-8').splitlines()]
    initial_free=inventory_progress[0]['free_bytes'];current_free=shutil.disk_usage(ROOT).free
    inventory=inventory_progress[-1]
    group_rows=json.loads((OUT/'inventory-groups.json').read_text(encoding='utf-8'))
    before=MEMO.read_bytes();phase=max(map(int,re.findall(rb'^## Phase (\d+):',before,re.M)))+1
    timestamp=C.stamp()[:16].replace('T',' ')
    raw_total=sum(x['logical_bytes'] for x in grouped.values());review_total=sum(x['retained_review_bytes'] for x in grouped.values())
    observed=sum(r['observed_free_increase_bytes'] for r in receipts)
    table='| 类别 | 删除文件数 | 删除逻辑GiB | 另存复核GiB |\n| --- | ---: | ---: | ---: |\n'
    csvbuf=io.StringIO(newline='');writer=csv.writer(csvbuf);writer.writerow(['group','deleted_files','deleted_logical_bytes','retained_review_bytes'])
    for name,g in grouped.items():
        table+=f'| `{name}` | {g["files"]} | {g["logical_bytes"]/C.GIB:.6f} | {g["retained_review_bytes"]/C.GIB:.6f} |\n'
        writer.writerow([name,g['files'],g['logical_bytes'],g['retained_review_bytes']])
    write('group-summary.csv',csvbuf.getvalue())
    body=fr'''
**目标、授权与完成状态。** 用户在上一轮A1/A2清理之后，明确要求继续检查其他测试的大体积中间数据，按“只保留脚本、题目、结果和必要复核记录”的相同标准清理，并禁止运行模型。本轮完成只读盘点、用途与身份核查、原场退役、混合NPZ复核成员另存、重复副本清理、保留文件校验和本条尾部追加。没有导入/调用科研模型、CUDA前向、重采、训练、提交、上传、发布或修改正式协议、客户端源代码；没有新科学样本或证据晋级。原工作区其他任务的修改保留。

**盘点范围和准入原理。** 元数据盘点覆盖项目`tests/`、`results/`、`research/`及`D:/ai2050_native`，共{inventory['files']:,}个文件、{inventory['groups']:,}个分组；没有遍历模型权重、Git、依赖环境，也没有删除这些目录。16个目录联接/重解析点未跟随，避免跨根误清理；实际删除目标另外逐级解析并拒绝链接。盘点的Windows目录项可能不提供有效inode/nlink，所以盘点身份不能用作删除依据：清理计划全部用`Path.stat()`刷新非零卷/文件ID、硬链接数、大小、mtime和SHA-256，再检查文件能否独占只读打开。所有进入删除的文件硬链接数均为1，不以多个路径重复计算可回收空间。

语言模式族只准入能够与保留的`distributed_data/bundles/<tm_id>/v<version>/corpus.json`按原协议材料指纹相等、且冻结`runner.py`存在的原场；共{frozen_shards}个原场分片通过。其原始L层、MLP G/V与部件Z/M属于可重采状态，不是题目或主要输出评分。材料、契约、运行器、原生manifest、模型/精度与条目ID绑定留存并纳入保护哈希。输入指纹沿用原协议canonical JSON的SHA-256前16位，完整题库文件的本轮SHA-256也保存，不将短指纹单独当成新的科学确认。

对混合NPZ，不按扩展名或体积整包删除：先按已核实的生产器逐成员区分原场与结果。MF的Hg/Hs、MV/XM的H/B/bare/codes、两套近期试验的全HiddenState和MLP G/U/A列为原场；保留题目ID、token/位置/划分、实际logits/top、来源质量、写入归因、gate评分等非原场成员。25份新`.review.npz`保存于本审计目录`retained_npz/`，每个保留成员在删除原包前后均校验SHA-256；归档路径、成员表、精度/形状、原包哈希与新包哈希可回查。大原场仅留头信息和少量字节作身份线索，这些线索不能恢复原场。Transcoder只清理采集激活x_train/x_eval，训练词元、窗口索引和训练好的字典保留。测试研究库对象仅在正式canonical对象存在、大小及完整SHA-256完全相同且目标不被使用时清理重复副本；正式对象、库数据库、题库、脚本、结果和必要反例不删除。

形式上，文件退役的输入身份检查为

$$
H_f=\operatorname{{SHA256}}(\operatorname{{bytes}}(f)),\qquad
\mathrm{{identity}}_f=(\mathrm{{volume}},\mathrm{{fileID}},\mathrm{{size}},\mathrm{{mtime}},\mathrm{{nlink}}).
$$

只有冻结计划中的路径、用途、身份复核一致且nlink=1才执行逐文件unlink。保留成员检验为

$$
\operatorname{{SHA256}}(\mathrm{{oldNPZ}}[k])=\operatorname{{SHA256}}(\mathrm{{reviewNPZ}}[k]),\quad k\in K_{{keep}}.
$$

这些是存储与字节一致性验收，不是新的语言机制公式。混合包里的原始数值字节不重算、不重新量化，不用拟合近似替代结果。

**实际清理结果。** 共删除{verification['deleted_files']:,}个原场或重复中间文件，逻辑大小{raw_total:,}字节（{raw_total/C.GIB:.6f}GiB）；保留成员的新复核包共{review_total:,}字节（{review_total/C.GIB:.6f}GiB）。逐类如下：

{table}

工作区清理时段实测净增加{receipts[0]['observed_free_increase_bytes']/C.GIB:.6f}GiB，原生目录清理时段增加{receipts[1]['observed_free_increase_bytes']/C.GIB:.6f}GiB，两时段合计{observed/C.GIB:.6f}GiB（{observed/1e9:.3f}GB）。这不含时段之外部分审计材料写入。以本轮盘点开始时D盘可用{initial_free:,}字节、报告生成前{current_free:,}字节计算，包含审计与复核归档开销的全轮净增约{(current_free-initial_free)/C.GIB:.6f}GiB，当前可用约{current_free/C.GIB:.6f}GiB；报告追加等小文件会再略占空间。逻辑大小、删后复核包体积、分阶段空间变化及全轮空间变化分别报告，不能互相冒充。

本轮没有递归删除目录，{len(marker_paths)}个实际受影响父目录新增`RAW_STORAGE_STATUS_20261118_CONTINUATION.json`。旧manifest、报告与审计保持字节不变，它们描述历史执行，不再自动意味着全部原场现存。状态文件指向本轮清单和复核包；可从`plan.json`及两份receipt定位每个原文件、保存成员与恢复依据。

**明确保留及停止的范围。** 12份清单找不到对应冻结题库，10份存在材料指纹或运行器匹配问题，均保留原场，不用当前题库静默替换历史输入；2份PS最终状态缺原场manifest、4份融合激活缺原场manifest，也保留；另1个混合文件没有可准入原场成员，保留。安装和展示实际使用的`visualization-research`、客户端本机原生数据、现有发布包、模型/权重缓存、Git、依赖环境、节点账号配置与任务正式输出、普查结果、训练好的字典及其他未核实结果均不清理。大目录不等于可删除缓存，旧磁盘表的clean字段没有作为删除命令。

首次计划核查在PS缓存找不到manifest时停止，尚未删除数据；随后收窄为保留这些缺件文件，只继续不受影响、身份足够的范围，旧停止痕迹保留于进度日志。这是工程准入修正，不是模型实验失败或科学结果变化。`_cross_model`里的历史int8与BF16仍按原精度记录；本轮不运行旧精度模型、也不把不同桶合并。

**验收、来源与复算边界。** 最终5项边界单测通过，覆盖混合NPZ的结果/输入保护、已知原场成员准入、模型/服务/客户端路径拒绝、目录联接拒绝以及Windows真实文件ID检查。{verification['protected_files_rehashed']:,}个保护文件清理后重新哈希全部一致，25份复核归档及其所有保留成员校验通过，{verification['deleted_files']:,}个计划路径均不存在。另行核验训练好的Transcoder字典及三个客户端manifest共4项，哈希不变。此次没有重跑模型、协议全套或浏览器功能测试，保留数组与身份校验不能替代这些验收，也不能将原科学主张自动升级。

实际命令为项目venv下`python -B -u tests/glm5/other_test_storage_inventory_20261118.py`；清理器`other_test_storage_cleanup_20261118.py plan`、`execute-workspace`、`execute-native`、`verify`；单测`python -B -m unittest discover -s tests/glm5 -p test_other_test_storage_cleanup_20261118.py -v`。工作区外D盘原生目录通过工具权限审查执行已授权清理，没有请求额外模型任务。盘点约81.6秒，冻结哈希计划约180.3秒；日志逐批且最长约25秒写一行进度，含已完成/总数、已用时间和ETA。没有GPU任务，因此不伪填显存测量。

审计根目录为`tests/glm5/result/other_test_storage_cleanup_20261118_v1/`：`inventory-files.jsonl.gz`为完整盘点；`inventory-groups.json`和`inventory-skipped.json`记录分组与边界；`plan.json`记录冻结删除表和保护表；`receipt-workspace.json`、`receipt-native.json`、`verification.json`及`extra-protected-verification.json`记录执行与复核；`retained_npz/`存必要结果成员。109,507,059字节本轮盘点文本经gzip无损保存为5,088,609字节，解压SHA-256与原文相同后才删除本轮生成的未压缩副本，没有删除其他任务的临时脚本或日志。

没有另外保存完整原场备份。题目/结果仍可复核，但新的全坐标、单元联合或跨层方法分析可能需要按保留冻结题库、运行器、模型/tokenizer身份和精度重新采集；旧int8、运行库及硬件差异尤其不能保证逐位复现。未来重采先登记预算和新目录，不覆盖旧结果，不把已暴露确认材料重新当盲测。重复测试对象可按清单从未改动的canonical路径恢复，这一类不需要模型。完整原场退役、可复核结果保留、客户端数据保留和科学结论等级是不同状态，本轮分别登记。

**理论增量与下一阶段。** 本轮只改数据可用性，没有新语言编码规律或RDC公式进展，沿用Phase2984的研究结论与接口。关键认识是科研可追溯性应依靠输入身份、真实结果、反例、可执行做法和清晰的数据可用性，而不是默认永久保留所有大场；但删除原场确实牺牲免重采换方法分析的能力，不能把它描述成没有代价。后续应作为一个统一存储合同任务，在采集/训练/打包流水线按角色登记“输入、结果、学习到的模型、必要复核、可重采原场、在用客户端依赖”，结束时自动生成轻量交付和退役清单，并检验缺件与恢复路径；本轮没有擅自修改这些生产流程或启动新模型研究。

**通俗说明。** 又清掉了约106GiB的逐层录像和已确认有正式原件的测试副本，留下题目、答案、失败记录、训练成果和必要查证数据。需要的新角度全场分析仍可按原做法重采；当前任务没有为清理再运行模型。
'''
    title=f'## Phase {phase}: 其他测试原场、混合复核包与重复副本的同标准清理 [{timestamp}]'
    write('REPORT.md','# 其他测试中间数据清理报告\n\n'+f'{timestamp} America/Chicago；Phase {phase}。\n\n'+body.strip()+'\n')
    structured={'schema':'other_test_cleanup_closeout.v1','created_at':C.stamp(),'phase':phase,'groups':dict(grouped),
                'inventory_files':inventory['files'],'inventory_groups':inventory['groups'],'frozen_material_shards':frozen_shards,
                'deleted_files':verification['deleted_files'],'deleted_logical_bytes':raw_total,'review_bytes':review_total,
                'measured_deletion_interval_gain_bytes':observed,'whole_turn_initial_free_bytes':initial_free,
                'whole_turn_observed_free_before_report_bytes':current_free,'kept_decision_reasons':dict(reasons),
                'status_markers':len(marker_paths),'new_model_runs':0,'new_scientific_claims':0}
    write('report.json',json.dumps(structured,ensure_ascii=False,indent=2)+'\n')
    entry=('\n\n'+title+'\n\n'+body.strip()+'\n').encode('utf-8')
    check_write(MEMO,len(entry),operation='尾部追加其他测试清理记录')
    assert MEMO.read_bytes()==before,'Concurrent memo writer; stop before append'
    with MEMO.open('ab') as f:f.write(entry);f.flush();os.fsync(f.fileno())
    after=MEMO.read_bytes();assert after[:len(before)]==before and after[len(before):]==entry
    append={'phase':phase,'title':title,'prefix_bytes':len(before),'prefix_sha256':hashlib.sha256(before).hexdigest(),
            'prefix_preserved':True,'append_sha256':hashlib.sha256(entry).hexdigest(),'memo_after_sha256':hashlib.sha256(after).hexdigest(),
            'memo_line':before.count(b'\n')+3,'report_sha256':C.sha(OUT/'REPORT.md')}
    write('memo-append-verification.json',json.dumps(append,ensure_ascii=False,indent=2)+'\n')
    sources=[ROOT/'tests/glm5'/name for name in ['other_test_storage_inventory_20261118.py','other_test_storage_cleanup_20261118.py',
             'test_other_test_storage_cleanup_20261118.py','other_test_storage_closeout_20261118.py']]
    sources += [ROOT/'deploy/storage_policy.py',ROOT/'deploy/storage_policy.json']
    with zipfile.ZipFile(OUT/'cleanup-sources.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(OUT/'cleanup-sources.zip') as z:assert z.testzip() is None
    files={p.relative_to(OUT).as_posix():{'bytes':p.stat().st_size,'sha256':C.sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()}
    write('delivery-manifest.json',json.dumps({'files':files,'self_excluded':True,'phase':phase},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'phase':phase,'memo_prefix_preserved':True,'report':str(OUT/'REPORT.md'),
                      'deletion_interval_gain_GiB':observed/C.GIB,'whole_turn_gain_GiB':(shutil.disk_usage(ROOT).free-initial_free)/C.GIB,
                      'free_GiB':shutil.disk_usage(ROOT).free/C.GIB},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
