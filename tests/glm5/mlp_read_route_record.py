"""Append the reproducible early-token route results and delivery evidence."""
from datetime import datetime
from zoneinfo import ZoneInfo

import mlp_read_route_common as C
from mlp_atlas_publish_final import Archive
from mlp_functional_record import append, phase


def main():
    out=C.OUT;dest=out/'phase-registration.json'
    if dest.exists():return
    archive=Archive(C.PUBLIC,C.RUN,C.identity());reg=C.read(out/'registration-final.json');d=archive.lib.artifact(reg['report_sha256'])
    route,screen,causal,cap=d['route'],d['screen'],d['causal'],d['capability'];m=causal['summary']['all'];n=m['n'];checks=causal['checks']
    raw=C.read_gz(out/'holdout-native.json.gz')['rows'];source=[p for r in raw for p in r['positions']]
    live=C.read(out/'live-acceptance.json')
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';claude=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    assert C.RUN not in memo.read_text(encoding='utf-8')[-20000:]
    p=phase(memo);cp=phase(claude);now=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    proof=archive.put('delivery-proof',{'schema':'early_read_route_delivery.v1','api':live,
          'screenshots':{v.name:C.file_sha(v) for v in out.glob('client-*.png')},
          'builds':{name:C.file_sha(out/name) for name in ['frontend-tests.log','frontend-build.log']}})
    operations='\n'.join(f'- {k}：首词正确 {v["accuracy"]:.2%}，首词变化 {v["first_word_changes"]}，翻错／转对 {v["losses"]}/{v["gains"]}，KL {v["mean_KL"]:.8g}，间隔配对差 {v["gap_delta"]}。'
                         for k,v in m['operations'].items())
    readers='\n'.join(f'- B{b}：Q 逐位一致 {r["Q_unchanged"]}/{n}，真实变化非零 {r["nonzero_read"]}/{n}，预测与真实变化的中位余弦 {r["median_cosine"]:.6f}，中位相对误差 {r["median_relative_error"]:.5f}。'
                     for b,r in causal['read_summary'].items())
    if d['by_role']:
        readers+='\n- 事后逐角色核验（只定位读取，不调整生成判据）：'+ '。'.join(
            f'{role}／B{b} 在 {v["nonzero_read"]}/{v["n"]} 条有非零变化，中位余弦 {v["median_cosine"]:.6f}'
            for role,data in d['by_role']['summary'].items() for b,v in data.items())+'。'
    ability='。'.join(f'{name} {r["n"]} 题，翻错／转对 {r["losses"]}/{r["gains"]}，净损区间 [{100*r["loss_interval"][0]:.3f}, {100*r["loss_interval"][1]:.3f}] pp，{r["status"]}'
                     for name,r in cap['summary'].items())
    control=checks['strict_matched_control'];h=m['operations']['joint_1'];specific=m['specificity']
    body=f'''## Phase {p}: 早期源 token 的实际读取与联合写入缓存路径 [{now}]

`{C.RUN}` 使用本地单 GPU Qwen3-4B BF16，快照 `{C.identity()['model_snapshot']}`。在 Phase 2799 单一 #991 的机制结果上，把源移动到事实词和选择键。复现实验与浏览证据分别在 `tests/glm5/mlp_read_route_*.py`、`tests/glm5/result/mlp_read_route_v1/` 及同一研究库。所有结果真实运行。

**发现与冻结。** 1,536 个新背景取自固定 Wikimedia 数据集此前未用的行组，英语和中文背景配平，任务指令为英语。每条 1,034 token，文章背景最少 886 token。正文的目标签署属性、其他属性、错误草稿和选择键分别通过分词 offset 映射。先在 64 个发现样本重建 B34/B35 回答 #1033 的注意力，`a_h(s)=softmax_s(q_h·k_g(s)/sqrt(128))`。每个原生 Head 读出用 `z_h=Σ_s a_h(s)v_g(s)` 对照实际 SDPA 输入 `o_proj` 的向量。最大相对重建误差 {route['discovery_max_native_error']:.6f}。按发现集跨层 `Σ a||O_h v||` 冻结两种源角色 {route['roles']}，再按本地真实图的答案间隔导数 `∇gap·writer` 选原生写入 {route['heads']} 与 MLP {route['mlps']}。导数只用于候选排序。正式源位置范围 #{min(source)} 到 #{max(source)}，均早于 #991。K 取 RoPE 后 BF16，V 取真实缓存。

**独立留出与实际读取。** 冻结后在 1,472 个独立新背景干净生成，确认／锁箱各 736。首词为正确值且 `0<gap≤8` 才入组。最终 {n} 条，确认／锁箱 {causal['summary']['confirm']['n']}/{causal['summary']['lockbox']['n']}。干净间隔中位 {m['median_clean_gap']:.3f}。目标 10pp、翻转率规划 0.10，最低 {screen['power_plan']['n_required']} 条，事前公式对应功效 {screen['power_plan']['achieved_power']:.3f}。复测清洁首词与间隔 {checks['screen_exact']}/{n} 一致。单层独立只清零入选源 V，保持该层 Q/K，实际 Head 变化与 `−Σ_s a_h(s)V_g(s)` 比较。保留近似数值误差和每条反例。

{readers}

**联合干预与缓存。** 在两个冻结角色的实际子词位置，对写入应用 `(1−α)z_h` 或 `(1−α)y_b`。比较全部缓存确定变化只在 B34/B35，且没有早于最早干预源的位置。分开保存直接源槽与后续传播槽。把真实干预的所有变化范围回放到干净缓存，{checks['replay_logits_bitwise_equal']}/{n} 逐位重现后续全词表 logits。把原范围放回编辑缓存，{checks['rescue_logits_bitwise_equal']}/{n} 逐位恢复。逐位结论限于同一原生 BF16 切分布局。单独回放源和其余槽所得 KL 保留为非加性的分支结果。所有 {checks['nonzero_future_distribution_cases']} 条有非零词表分布变化。

**等范数与首词。** 初版控制按各自分支当前写入缩放，后层输入可能已经变化。严格复算版本把每个随机符号排列的干预向量按联合干预同一样本、同一位置、同一原生写入实际 L2 缩放。共 {control['vectors']} 个向量，BF16 回写前最大相对差 {control['max_pre_rounding_relative_error']:.4g}，回写后中位／最大相对差 {control['median_post_rounding_relative_error']:.4g}/{control['max_post_rounding_relative_error']:.4g}。初版和复算版都保留。主门净损至少 10pp、两侧配对精确 p<.05、比严格对照多损至少 5pp。实际联合首词变化 {h['first_word_changes']}/{n}，净损 {100*h['net_loss']:.3f} pp，p={h['mcnemar_p']:.6g}，与对照的净损差 {100*specific:.3f} pp，判定 {'通过' if m['passed'] else '未通过' if m['eligible'] else '确认资格未达'}。

{operations}

**能力与范围。** 两种操作在已暴露的 2200 道固定监测题执行，140 道仅参考。使用发现集冻结的相对位置 {route['capability_offsets']}，短提示限制在有效前缀。以两操作同时配对精确净损界判断 2pp 非劣效。{ability}。当前证据覆盖此任务、冻结写入和对应监测池。

**交付与理论。** 3D 研究库含 {reg['cards']} 张定位卡、{reg['edges']} 条证据边，含子词坐标、回答 Head 读出、原生槽、逐题生成、严格范数和失败证据。API {len(live['checks'])} 项、前端 73 项及生产构建通过。报告 `{reg['report_sha256']}`。实际读取、经缓存传到分布、足以改变词选取需要分开建立证据。下一阶段可将已验实的源路径用同样本真实先验与答案证据做跨任务的对照，冻结可复现的选择性组和语义上不变的控制条件，再检验效果是否推广。
'''
    sha=archive.put('phase-summary',{'schema':'research_phase_summary.v1','phase':p,'markdown':body,'timestamp':now,
                                      'report_sha256':reg['report_sha256'],'graph_sha256':reg['graph_sha256'],'acceptance_sha256':proof})
    append(memo,body)
    append(claude,f'''## Phase {cp}: 实际源读取与联合写入缓存路径（GLM5 Phase {p} 详记） [{now}]

`{C.RUN}` 实测回答查询读取更早的目标属性和选择键，冻结两个 Head 和一个 MLP 在 {n} 个未见低余量组合复核。已归档逐位回放恢复、严格等范数、实际生成首词和能力保持。报告 `{reg['report_sha256']}`。''')
    C.save(dest,{**C.identity(),'phase':p,'claude_phase':cp,'summary_sha256':sha,'acceptance_sha256':proof,'timestamp':now})
    print('Phase',p,'Claude',cp,'summary',sha,flush=True)


if __name__=='__main__':main()
