"""Immutable evidence ledger and append-only closeout of this stage review."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import re
import shutil
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from conditional_lookup_stage_audit import SOURCE, OUTPUT, read, sha
from deploy.storage_policy import check_write


def main():
    report_path = OUTPUT / 'stage-evidence-ledger.json'
    assert not report_path.exists(), 'review already closed'
    pre = read(SOURCE / 'prereg.json')
    discover = read(SOURCE / 'discovery-analysis.json')
    confirm = read(SOURCE / 'confirm-analysis.json')
    audit = {s: read(OUTPUT / f'{s}-audit.json') for s in ('discovery', 'confirm')}
    arithmetic = read(OUTPUT / 'frozen-prediction-arithmetic-audit.json')
    scalar = read(OUTPUT / 'native-parameter-scalar-audit.json')
    assert all(a['artifact_and_arithmetic_pass'] for a in audit.values())
    assert arithmetic['all_arithmetic_checks_pass'] and scalar['model_snapshot_verified']
    assert not confirm['mechanism_accepted'] and not confirm['causal_tested']
    base = max(confirm['comparisons'], key=lambda k: confirm['comparisons'][k]['joint'])
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    memo = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
    index = ROOT / 'research/claude/docs/AGI_CLAUDE_MEMO.md'
    before, index_before = memo.read_bytes(), index.read_bytes()
    phase = max(map(int, re.findall(rb'^## Phase (\d+):', before, re.M))) + 1
    index_phase = max(map(int, re.findall(rb'^## Phase (\d+):', index_before, re.M))) + 1
    scripts = ['conditional_lookup_mechanism_plan.md', 'conditional_lookup_stage_audit.py',
               'test_conditional_lookup_stage_audit.py', 'conditional_lookup_mechanism_cards.py',
               'conditional_lookup_parameter_slice_audit.py', 'conditional_lookup_prediction_audit.py',
               'conditional_lookup_stage_review_closeout.py']
    files = {str((ROOT / 'tests/glm5' / n).relative_to(ROOT)): sha(ROOT / 'tests/glm5' / n) for n in scripts}
    files.update({str(p.relative_to(ROOT)): sha(p) for p in OUTPUT.iterdir() if p.is_file()})
    source_names = ['prereg.json', 'natural-plan.json', 'calibration.json', 'candidate.json',
                    'discovery-collection.json', 'confirm-collection.json', 'discovery-analysis.json',
                    'confirm-analysis.json', 'readout-models.npz', 'native-parameters.npz', 'natural-audit.json']
    files.update({str((SOURCE / n).relative_to(ROOT)): sha(SOURCE / n) for n in source_names})
    log = [json.loads(line) for line in (SOURCE / 'natural-progress.jsonl').read_text().splitlines()]
    summaries = [read(p) for s in ('discovery', 'confirm') for p in SOURCE.glob(s + '-??.json')]
    telemetry = dict(cuda_allocator_peak_bytes=max(p['cuda_peak'] for p in summaries),
                     sampled_dedicated_peak_bytes=max(p.get('dedicated_gpu_bytes', 0) for p in log),
                     sampled_shared_peak_bytes=max(p.get('shared_gpu_bytes', 0) for p in log),
                     source_artifact_bytes=sum(p.stat().st_size for p in SOURCE.iterdir() if p.is_file()),
                     review_artifact_bytes=sum(p.stat().st_size for p in OUTPUT.iterdir() if p.is_file()),
                     disk_free_bytes=shutil.disk_usage(ROOT).free,
                     cleanup=dict(owned_plot_cache='tests/glm5_temp/conditional_lookup_review_mpl', removed_bytes=85845))
    ledger = dict(schema='clt_stage_evidence_ledger.v1', kind='real', provenance=pre['provenance'],
                  created_at=stamp, phase=phase, source_run='CLT-20261117-v1 natural extension',
                  reuse='same discovery/confirm forward run as GLM5 Phase2967; no new samples or independent model node added by review',
                  stages=dict(behavior={'discovery_correct': audit['discovery']['strict_correct'], 'discovery_n':1024,
                                        'confirm_correct':audit['confirm']['strict_correct'], 'confirm_n':2048},
                              readable={'boundary':'hs18 (output of block17)', 'native_coordinate_budget':32,
                                        'confirm':confirm['encoding'], 'strongest_random_joint':confirm['comparisons'][base]['joint'],
                                        'gain':confirm['comparisons'][base]['advantage'], 'specificity_gate':confirm['encoding_gate']},
                              local_predictive={'same_position_upstream_input':True, 'parameter_model':confirm['parameter'],
                                                'closed_selector_implemented':False},
                              observed_routing={'block':26,'head':36,'kv_head':7,**arithmetic['measured_routing'],
                                                'causal_use_certified':False},
                              original_parameters={'scalars_verified':scalar['total_scalars'],'bitwise_identity':True},
                              causal={'tested':False,'reason':'No frozen upstream-only compressed selector; original mechanism admission not met'}),
                  engineering_tests=dict(command='.venv/Scripts/python.exe -B tests/glm5/test_conditional_lookup_stage_audit.py',
                                         passed=6,exit_code=0, scope='isolated engineering positives/negatives; not scientific sample counts'),
                  telemetry=telemetry, files=files,
                  limitations=['Only new-family confirmation within the same exposed16-cell task',
                               'No free multi-token answer generation', 'Not a two-source semantic fusion test',
                               'Readability does not identify the actual causal route',
                               'Local parameter model uses true same-position upstream states',
                               'No full-vocabulary distribution predictor implemented in this extension',
                               'No causal intervention or independent model-node rerun',
                               'No server publication or client registration'])
    check_write(report_path, 1 << 20, operation='CLT stage evidence ledger')
    report_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')
    note = f'''

## Phase {phase}: 条件选择大阶段的原生坐标、神经元和标量参数审计 [ {stamp} ]

**目标、授权与状态。** 用户授权以“条件如何选择、融合和复用知识”为下一大阶段，制定方案并开始真实测试，严格区分可读出、可预测和因果作用。本轮完成大阶段方案、自然发现/确认结果的独立算术复核、全原生坐标观察图和原参数追溯。复用Phase2967的同一CLT-20261117-v1自然扩展运行：64发现家族1024输入、128确认家族2048输入；这里不重复增加样本、实验次数或独立节点数。发现/确认CUDA前向已完成并释放GPU，因果干预未进入。完整方案 tests/glm5/conditional_lookup_mechanism_plan.md；审计、图和卡片 tests/glm5/result/conditional_lookup_stage_review_v1；原生场和冻结对象仍在conditional_lookup_native_v1。

**执行协调和冻结。** 首次接入同名discovery-console.log遇到持有文件锁，随后确认原采集PID45260已逐层运行，复用冻结采集而未另开并发模型。自然候选于2026-11-17 11:36:29（Chicago）冻结，确认在之后采集；原参数、层位、坐标和分析代码哈希均保持原冻结版本。发现编码门失败的候选按原先允许的负结果复现进入确认，不修改门槛。当前确认已经暴露，后继新候选不得再把它作为盲测。

**方法、数学对象与实际范围。** 外部金答案行位 r=(1-b)r0+b*r1 只定义任务。真实MLP为 G_j=sum_k Wg[j,k]z_k、U_j=sum_k Wu[j,k]z_k、A_j=SiLU(G_j)U_j、残差写入d为Wd[d,j]A_j；这些是原生计算定义。少输入模型用发现均值mu补全省略坐标：Ghat_j=Wg_j mu+sum_(k in K) Wg[j,k](z_k-mu_k)，Uhat同理，再计算Ahat。它使用同位置真实上游z，不能视为从文本出发的闭合机制。

自然场实际范围：实体末token全部41残差边界；B20–22实体G/U/A/z及最终位置A/z；B20–30分组来源与局部OV记账，B20–22另含实体位置。未采的层/位置不是零效应。派生图保存每个家族完整2×2行位设计的beta0=mean((2r0-1)H)、beta1和beta01=mean((2r0-1)(2r1-1)H)，对应平衡因子模型系数，不是均值差本身（均值差为两倍主效应系数）。原始单位、逐坐标背景标准差及标准化视图分别保留；保持全部5120残差坐标、三块各17408个MLP单元顺序，无PCA/Top-K定义主干。颜色为对称对数，图只作描述，不进行逐坐标显著性或语义功能判定。

**独立核验与实际结果。** 发现1024/1024、确认2048/2048，全部16格严格四候选和全词表top1均100%。独立脚本核对材料/分片/候选哈希、ID与顺序、金标签和逐条margin，发现256对、确认512对后置条件实体状态在全部41边界逐位一致。两集合B20–22全部17408单元的BF16 SiLU(G)*U与真实A逐位一致；六来源质量归一化检查通过。这是技术与行为证据。

可读出：冻结hs18（B17输出）的32原生坐标，对确认512个独立实体前缀的两位和联合准确率均100%。最强随机32坐标联合93.75%，优势6.25个百分点，97.5%家族bootstrap区间[3.71,9.18]个百分点；未达到冻结的10个百分点优势，因此原编码特异性准入False。不能把该失败写成“信息不存在”：两个行位确实可由冻结线性读出器在新家族读出。随机组也很强，说明该读出不证明所选坐标独占功能。嵌入和干扰基线联合25%，与两个平衡二值变量的该口径机会水平一致。随机可读性也不否定稀疏因果电路存在。

局部参数预测：选择B20的冻结32单元，32原生z输入相对均值的MSE改善10.13%、相对随机32输入改善12.56%；原MSE分别0.0514190346、0.0572140478、0.0588082075，均未达到20%门。独立FP64算术复核得到0.0514190360、0.0572140465、0.0588082062，在事先写入独立复核脚本的数值容差内一致。精确GUA复算成功与降维预测不足分别记账，不能互相替代。

来源观察：冻结B26/H36（KV头7）的实际注意力，确认后置1023/1024（99.90%）、前置1019/1024（99.51%）关注正确正文记录。它读取的是已计算attention，不是操作前紧凑预测器。query_entity分组只有查询名字末token，其余名字token在other，不能以该分组质量小排除名字缓存。正文路由准确也可能是重复读取已经选中的答案，不能由此确定首次选择位置、必要性或取值算法。

**原生索引与标量参数。** h18冻结坐标示例1200、335、1541、4992、3988等，完整32索引在卡片。B20单元示例10256、7153、3036，完整32单元附真实gate/up行与down列。单元10256的原Wg[10256,3852]=-0.07470703125，Wu[10256,3988]=-0.302734375；这些是原生连接与实际输入乘积的定位实例，不赋予单参数语义。B26/H36的Q原矩阵行4608..4735、K/V行896..1023、O列4608..4735，以及Q/K归一化和输入归一化参数均保存为BF16位模式。另一次CPU原文件核验重新确认模型/tokenizer快照，并对这些头参数与32单元参数合计3,118,336标量逐位核对成功。摘取约束原参数是结构来源证据，不是已经压缩出语言机制。

**证据分级与硬伤。** stage-evidence-ledger.json将技术/行为、可读出、局部预测、实际来源和因果五类独立登记。本轮未实现操作前紧凑选择器，未做原模型因果干预、自由多token回答、完整词表分布预测、独立物理节点模型复跑、跨句式/跨深度推广或跨模型验证。确认128家族的旧二值规划（预期15点，p_d=.25，alpha=.025）由tools/power_analysis得到所需106、power=.875，只约束原冻结二值主张，不给MSE、坐标筛选或因果试验补功效。注意力数值门为pilot校准的绝对容差0.065418，不能冒充逐头相对误差保证。原生坐标地图可受词序、身份、位置和范数影响；完整16格已在试跑暴露，只能称新家族复现。

**第一性洞察与下一大阶段。** 当前拼图是“同一实体位置可读到两个行位，同时一个下游头的实际来源随条件指向正确正文”，两点间的运算链尚未闭合。应区分信息保存、条件选择、内容取值与输出读出，而不预设名字向量容纳完整知识。后继协议在新家族上记录全层原生来源，判断B26/H36是首个选择器还是接续读取；仅用其操作前归一化query/source状态、实际位置和原Q/K参数，保留RMSNorm/RoPE/GQA，以32/128输入坐标和同预算随机/均值/干扰基线预测选择及QK间隔。再用原V/O预测写入和下游GUA。新材料、具体索引、预算、指标和功效须在运行前另冻结，不能用本轮确认调参。原选择/写入共同门通过后，才冻结单坐标、单元或标量参数的剂量、每格效应、等强度随机/sham、自身恢复与副作用检验；BF16参数编辑按实际舍入后delta登记。真正的两源语义融合另设计确需两个知识源的任务，不能把选择性交互当作已经验证融合。

外部图谱保留条件/内容/行位/时序的任务关系，内部图谱新增原生信息卡与参数来源，两图谱关联目前仅达到新家族信息复现和来源观察；未新增因果边。无新RDC统一公式、数学定理或AGI结论。现有失败限定32输入近似与坐标优势主张，不说明现有数学体系不足。

**命令、复核与资源。** 本轮新增并实际执行：python -B tests/glm5/test_conditional_lookup_stage_audit.py（6项工程正/负对照通过，非科学样本）；python -u -B tests/glm5/conditional_lookup_stage_audit.py discovery/confirm；conditional_lookup_mechanism_cards.py；conditional_lookup_parameter_slice_audit.py；conditional_lookup_prediction_audit.py。全部使用.venv/Scripts/python.exe，结果在tests/glm5/result下。冻结预测器独立复算与保存row_scores最大差0，原参数身份、全坐标原生算术和分片准入通过。真实CUDA采集复用collect.py discovery/confirm，原采集耗时209.3/419.1秒，不能将本轮CPU审计当额外模型重复。分片summary CUDA allocator峰值{telemetry['cuda_allocator_peak_bytes']}字节，逐层DXGI共享显存采样峰值{telemetry['sampled_shared_peak_bytes']}字节，采样不代表所有瞬时高水位。研究原场、冻结文件和复算依据保留；仅清除本轮独占MPL缓存85,845字节。清理前核对绝对与解析路径。D盘始终执行storage_policy保留20GiB，当前余量约{telemetry['disk_free_bytes']/2**30:.2f}GiB。后继新增采集须重新核对峰值预算。未上传正式中心或发布客户端，本轮不动其他任务文件。备忘录与统一索引仅append，旧内容完整前缀核验通过。
'''
    # Required title format has no extra spaces inside timestamp brackets.
    note = note.replace(f'[ {stamp} ]', f'[{stamp}]')
    idx_note = f'\n\n## Phase {index_phase}: 条件选择大阶段的原生坐标与参数独立审计 [{stamp}]\n\n见GLM5 Phase {phase}，承接同一自然运行Phase2967，不重复计模型样本或独立节点。发现/确认分片及全MLP算术通过，3,118,336原参数标量身份核验通过；两位可读出100%，但坐标优势与局部预测门未过，因果未测试。大阶段方案及QK操作前预测后继设计已保存。\n'
    for path, old, text in ((memo, before, note), (index, index_before, idx_note)):
        check_write(path, len(text.encode('utf-8')), operation='CLT stage append')
        assert path.read_bytes() == old, 'concurrent memo append; retry without overwriting'
        with path.open('ab') as stream:
            stream.write(text.encode('utf-8'))
        assert path.read_bytes() == old + text.encode('utf-8')
    print(json.dumps(dict(ledger=str(report_path), phase=phase, index_phase=index_phase,
                          memo_line=before.count(b'\n')+3, old_prefix_preserved=True,
                          ledger_sha256=sha(report_path), telemetry=telemetry)), flush=True)


if __name__ == '__main__':
    main()
