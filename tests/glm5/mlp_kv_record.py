"""Append the KV experiment and its reproducibility record to the project memos."""
from datetime import datetime
from statistics import median
from zoneinfo import ZoneInfo

import mlp_kv_common as C
from mlp_atlas_publish_final import Archive
from mlp_functional_record import append, phase


def main():
    output=C.OUT/'phase-registration.json'
    if output.exists():return
    archive=Archive(C.PUBLIC,C.RUN,C.identity());reg=C.read(C.OUT/'registration-final.json');report=archive.lib.artifact(reg['report_sha256'])
    med=report['mediation'];earlier=report['earlier'];target=report['targeted'];cap=report['capability'];formal=report['formal'];pilot=report['pilot']
    full=med['summary']['all'];low=target['summary']['all'];screen=report['screen']
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';claude=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    assert C.RUN not in memo.read_text(encoding='utf-8')[-20000:]
    n=phase(memo);cn=phase(claude);stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    api=C.read(C.OUT/'live-acceptance.json');proof=archive.put('delivery-proof',{'schema':'mlp_kv_delivery_acceptance.v1','api':api,
        'screenshots':{p.name:C.file_sha(p) for p in C.OUT.glob('client-*.png')},
        'logs':{name:C.file_sha(C.OUT/name) for name in ['frontend-build.log','frontend-tests.log','storage-audit.jsonl']}})
    actual=C.read(C.OUT/'targeted/result-layout-v2.json')['rows']
    old=C.read(C.PREVIOUS/'revision/generated-causal.json')['rows'];old_median=median(r['conditions']['clean']['answer_gap'] for r in old if r['q']==1)
    good=target['screen_to_replay_exact'];nlow=low['n'];npower=target['power_plan']['n_required'];gap=median(r['conditions']['clean']['gap'] for r in actual)
    def table(groups):
        return '\n'.join(f'- {name}：首词正确率 {v["accuracy"]:.2%}，翻错／转对 {v["losses"]}/{v["gains"]}，KL {v["mean_KL"]:.8g}，间隔配对差 {v["gap_delta"]}。' for name,v in groups.items())
    ability='。'.join(f'{name} {v["n"]} 题，翻错／转对 {v["losses"]}/{v["gains"]}，净损界 [{100*v["loss_interval"][0]:.3f}, {100*v["loss_interval"][1]:.3f}] pp，{v["status"]}'
                     for name,v in cap['summary'].items())
    par=pilot['layout_parity'];mx=max(r['max_abs_logit_difference'] for r in par);top=sum(sum(r['argmax_equal']) for r in par)
    storage=sum(1 for _ in (C.OUT/'storage-audit.jsonl').open(encoding='utf-8'))
    body=f'''## Phase {n}: B34 写入的 KV 时序路径与低余量留出复核 [{stamp}]

运行 `{C.RUN}` 使用本地 CUDA Qwen3-4B BF16，模型快照 `{C.identity()['model_snapshot']}`。承接 Phase 2797 的 B34/H2、B35/H25 观察与时间边界。程序、原始结果和哈希索引位于 `tests/glm5/mlp_kv_*.py`、`tests/glm5/result/mlp_kv_mediation_v1/` 和研究库。用单 GPU 顺序执行。

**材料。** 新背景取此前未用的固定 Wikipedia 行组，正式样本均为 1034 token，真实文章背景最少 831 token。试跑在 32 个新背景上比较四种文法，只用试跑确定错误草稿提示档。用签署记录给出唯一答案，错误草稿是明确的竞争线索。正式确认、锁箱各 128 个背景、每个两种选择，512 条真实生成，正确率 {formal['accuracy']:.2%}，行为门 {'通过' if formal['passed'] else '未通过'}。首词候选间隔中位 {formal['median_gap']:.2f}，上一轮暴露的生计样本中位 {old_median:.2f}。后者是跨材料参照。主干预按哈希在每背景预选一个选择，256 条中低余量 {med['summary']['low_margin']['n']} 条。分词、模板、来源、逐题输出和失败材料均有内容索引。

**计算路径。** 设 P 为到源 token #991 为止的固定前缀，C(P) 为原生缓存。对 B34/H2 的原生 Head 向量 z 仅在这个位置施加 `(1−α)z`，固定前缀其余输入。B35 输入残差在真实 RMSNorm 前读取，K 在真实 RoPE 后读取，V 在原生缓存读取。真实 edit 与干净 C 的差只落在 B35、#991 的八组 K/V。令 L(q,C) 为缓存上的后续查询词表 logits，D 为编辑的 B35 槽。回放检验 `L(q,C[slot←D]) = L(q,C_edit)`，恢复检验 `L(q,C_edit[slot←C_slot]) = L(q,C)`。256/256 的全词表结果逐位成立，实际生成也逐项匹配，且所有 256 条的概率分布出现非零变化。K、V 单独回放和等范数符号排列对照另做。原序列一体前向和切分前向在 16 条试跑的最高 logit 最大绝对差 {mx:.3f}，argmax {top}/16 一致。正式操作均使用同一切分布局，逐位等价限定在该布局。

**主生成干预。** 以真实生成首词、配对精确 McNemar 和等范数对照分别判定。预注册门为净损 10pp、p<.05，并较匹配对照多损 5pp。主门 {'通过' if full['passed'] else '未通过'}。以下 KL 为相对干净分布的全词表平均，间隔是第一候选 token 的 logit 差。

{table(full['operations'])}

**独立低余量筛查。** 另冻结 {screen['n_screened']} 个新文档，每文档一条，先完成全部干净筛查再运行干预。按干净首词正确且 `0<gap≤8` 入组 {screen['eligible']} 条，实际 {nlow} 条，间隔中位 {gap:.2f}。按翻转率 0.10 与 10pp 目标，最低 {npower} 条，预期功效 {target['power_plan']['achieved_power']:.3f}。初次最后一条因单样本布局，筛查间隔 7.50 在重算为 7.25，保留原结果并依据原筛查的四条批次复算其全部操作。其他三条仅作计算布局补齐，最终筛查首词与间隔 {good}/{nlow} 一致，仍沿用冻结门。主门 {'通过' if low['passed'] else '未通过'}，原生 KV 回放和恢复分别 {target['checks']['replay_logits_bitwise_equal']}/{nlow}、{target['checks']['rescue_logits_bitwise_equal']}/{nlow} 逐位相符。

{table(low['operations'])}

**早层与末层控制。** B33 MLP 写入移除的 32 条原生前缀，只在 B34、B35 的同一源槽产生缓存差。两层一起回放与恢复分别 {earlier['checks']['replay_logits_bitwise_equal']}/32 和 {earlier['checks']['rescue_logits_bitwise_equal']}/32 逐位相符。只回放 B34、只回放 B35、联合回放的 KL 分别为 {earlier['summary']['operations']['only_B34']['mean_KL']:.8g}、{earlier['summary']['operations']['only_B35']['mean_KL']:.8g}、{earlier['summary']['operations']['replay_both']['mean_KL']:.8g}。32 条只是计算边界及行为描述。B35/H25 在 8 条负对照改变了局部残差，缓存、后续 logits 和生成仍相同。这与末块写入发生在本层 K/V 形成之后的时间顺序一致。

**能力门。** 使用已暴露的固定池 2200 正式题，另 140 仅参考。两种实际操作在末位置前 42 token 施加，短提示限制在有效前缀内。采用两操作校正的配对精确损失／增益界，容许净损 2pp。{ability}。结论只覆盖测试操作与监测题。

**研究账本。** 归档 {reg['cards']} 张定位卡、{reg['edges']} 条类型边，分开保存真实路径、生成负结果、筛查排除项和逐题错误。原生 K/V 的 BF16 位型与逐样本归档逐项比对。正式 API {len(api['checks'])} 项、前端 73 项和客户端生产构建通过。工作盘对 {storage} 个已有归档执行透明压缩及前后哈希核验，历史结果保留。报告 `{reg['report_sha256']}`，图谱 `{reg['graph_sha256']}`。

**理论增量。** 固定前缀中，历史写入只有进入未来计算可读取的状态才有跨 token 因果路径。这里可把这种路径写成“局部写入 → 持续缓存 → 当前查询读出”的组合，并用反向恢复核对。方向可读、分布可达和首词决定是三个需各自检验的命题。低余量筛查缩小了先前生成天花板的影响，真实作用量仍以分布小扰动为主。下一阶段应在发现集选择真实被回答查询读取的更早源 token 与多写入组，冻结未见组合、同位置等范数控制和能力门，再比较 K/V 方向及联合干预能否复现可观行为差。
'''
    sha=archive.put('phase-summary',{'schema':'research_phase_summary.v1','phase':n,'markdown':body,'timestamp':stamp,'report_sha256':reg['report_sha256'],
                                      'graph_sha256':reg['graph_sha256'],'acceptance_sha256':proof})
    append(memo,body)
    append(claude,f'''## Phase {cn}: 问题前原生 KV 传递和低余量留出（GLM5 Phase {n} 详记） [{stamp}]

`{C.RUN}` 的 B34/H2 在 256 个正式新背景只改变源 token 的 B35 K/V，回放、恢复均逐位验证。B33 MLP 需要 B34/B35 双层源槽。独立低余量 {nlow} 条与两项能力监测已归档，报告 `{reg['report_sha256']}`。''')
    C.save(output,{**C.identity(),'phase':n,'claude_phase':cn,'summary_sha256':sha,'acceptance_sha256':proof,'timestamp':stamp})
    print('Phase',n,'Claude',cn,'summary',sha,flush=True)


if __name__=='__main__':main()
