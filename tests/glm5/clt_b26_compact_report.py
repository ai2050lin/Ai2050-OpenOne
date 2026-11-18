"""Render native-coordinate evidence and append the audited experiment to both memos."""
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import numpy as np

from clt_b26_compact import OUT, ROOT, TMP, candidate, checked, commit
from deploy.storage_policy import check_write, checked_dir
from source_binding_design import read, sha


def draw(out, report, data):
    cfg = checked_dir(TMP / 'report-mpl', 8 << 20)
    os.environ['MPLCONFIGDIR'] = str(cfg)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for key in ('q', 'k', 'v'):
        score = data['score_' + key]
        axes[0, 0].plot(score / max(score.max(), 1e-20), lw=.6, label=key.upper())
    axes[0, 0].set(title='Discovery | native Q/K/V input ranking', xlabel='Input coordinate (original order)', ylabel='Score / branch maximum')
    axes[0, 0].legend()
    score = data['unit_sse'] / (16 * 64)
    axes[0, 1].plot(score, lw=.6, color='#99426b')
    axes[0, 1].set(title='Discovery | native MLP activation contrast', xlabel='MLP unit (original order)', ylabel='Mean squared change in A')
    arms = ('native32', 'random32', 'native128', 'random128', 'mean')
    route = [report['arms'][arm]['route_agreement'] for arm in arms]
    axes[1, 0].bar(arms, route, color=['#285f87', '#b0b8c7', '#285f87', '#b0b8c7', '#b0b8c7'])
    axes[1, 0].axhline(.9, color='#995100', ls='--', lw=1, label='Absolute gate')
    axes[1, 0].set(title='Holdout | four-record routing agreement', ylabel='Fraction', ylim=(0, 1))
    axes[1, 0].legend()
    bases = ('random128', 'mean', 'zero')
    for index, (metric, color, label) in enumerate((('write', '#285f87', 'H36 write'), ('downstream', '#99426b', 'MLP activation'))):
        points = [i for i, base in enumerate(bases) if base in report['gates'][metric]]
        gains = [report['gates'][metric][bases[i]]['fraction_gain'] for i in points]
        axes[1, 1].bar(np.asarray(points)+(index-.5)*.32, gains, width=.32, color=color, label=label)
    axes[1, 1].set_xticks(range(len(bases)), bases)
    axes[1, 1].axhline(.2, color='#995100', ls='--', lw=1, label='Relative gain gate')
    axes[1, 1].set(title='Holdout | native128 MSE gain against baselines', ylabel='Fractional MSE reduction')
    axes[1, 1].legend()
    figure.suptitle('Qwen3-14B B26/H36 | prediction from native upstream state')
    check_write(out, 8 << 20, operation='CLT figure')
    assert not out.exists()
    figure.savefig(out, dpi=155)
    plt.close(figure)
    trace = OUT / 'causal-causal-traces.npz'
    if trace.exists():
        with np.load(trace) as x:
            gold = x['gold_record']
            rr = np.arange(len(gold))

            def margin(key):
                mass = x[key]
                return np.log(np.maximum(mass[rr, :, gold], 1e-20))-np.log(np.maximum(mass[rr, :, gold^1], 1e-20))

            observed, predicted = margin('actual_mass'), margin('compact_mass')
            observed_delta, predicted_delta = observed[:, 1]-observed[:, 0], predicted[:, 1]-predicted[:, 0]
            advantage = observed[:, 2]-observed[:, 1]
            ratio = x['q_dose'][:, 2]/x['q_dose'][:, 1]
        causal, panels = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
        panels[0].scatter(predicted_delta, observed_delta, s=13, alpha=.65, color='#285f87')
        bound = max(np.abs(predicted_delta).max(), np.abs(observed_delta).max(), .1)
        panels[0].plot([-bound, bound], [-bound, bound], color='#717781', ls='--')
        panels[0].set(title='Counterfactual prediction', xlabel='Predicted margin change', ylabel='Measured margin change')
        panels[1].hist(advantage, bins=16, color='#99426b')
        panels[1].axvline(0, color='#717781', ls='--')
        panels[1].set(title='Matched random minus target', xlabel='Same-archive log mass ratio', ylabel='Families')
        panels[2].hist(ratio, bins=18, color='#347c70')
        panels[2].axvspan(.9, 1.1, color='#347c70', alpha=.10)
        panels[2].set(title='Actual Q perturbation dose', xlabel='Random / target (post-RoPE L2)', ylabel='Families')
        causal.suptitle('B26/H36 native Q contribution intervention | 128 fresh families')
        causal.savefig(check_write(OUT/'b26-q-matched-intervention.png', 8 << 20, operation='CLT causal figure'), dpi=155)
        plt.close(causal)
    # This directory is owned by this invocation. Delete only verified direct cache children.
    assert cfg.resolve() == (TMP / 'report-mpl').resolve() and cfg.resolve().is_relative_to(ROOT.resolve())
    removed = 0
    for p in cfg.iterdir():
        if p.is_file() and not p.is_symlink() and p.name.startswith('fontlist-') and p.suffix == '.json':
            removed += p.stat().st_size
            p.unlink()
    if not any(cfg.iterdir()):
        cfg.rmdir()
    if not any(TMP.iterdir()):
        TMP.rmdir()
    return removed


def close():
    prereg, seal = checked(), candidate()
    report, audit = [read(OUT / name) for name in ('evaluation.json', 'audit.json')]
    assert sha(OUT / 'evaluation.json') == audit['eval_sha']
    calibration_path = OUT / 'calibration-causal-summary.json'
    intervention_path = OUT / 'causal-causal-summary.json'
    calibration = read(calibration_path) if calibration_path.exists() else None
    intervention = read(intervention_path) if intervention_path.exists() else None
    if report['causal_admitted']:
        assert calibration is not None, 'Complete the admitted calibration before closeout'
        if calibration['technical_pass']:
            assert intervention is not None, 'Complete the admitted intervention before closeout'
            causal_audit = audit['causal'] or read(OUT/'causal-audit.json')
            assert causal_audit['analysis_sha'] == sha(intervention_path)
    assert not (OUT / 'delivery.json').exists()
    discovery, confirm = [read(OUT / f'{s}-collection.json') for s in ('discovery', 'confirm')]
    with np.load(OUT / 'predictor.npz') as data:
        idx = {kind: data[f'native128_idx_{kind}'].tolist() for kind in ('q', 'k', 'v')}
        q_coord = idx['q'][0]
        q_bits = data['param_q'][:, q_coord].astype(np.uint32)
        qvals = (q_bits << 16).view(np.float32)
        scalar_idx = int(np.argmax(np.abs(qvals)))
        scalar_value = float(qvals[scalar_idx])
        removed = draw(OUT / 'b26-compact-results.png', report, data)
    summary = report['arms']['native128']
    assert len(seal['units']) == 32 and all(len(v) == 128 for v in idx.values())
    by_cell = summary['by_cell_route_agreement']
    worst = int(np.argmin(by_cell))
    ts = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    memo = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
    index = ROOT / 'research/claude/docs/AGI_CLAUDE_MEMO.md'
    old, ix = memo.read_bytes(), index.read_bytes()

    def phase_number(blob):
        return max(map(int, re.findall(rb'^## Phase (\d+):', blob, re.M))) + 1

    p, ip = phase_number(old), phase_number(ix)
    logs = [json.loads(line) for f in OUT.glob('*progress.jsonl') for line in f.read_text().splitlines()]
    cuda = max(v['cuda_peak'] for v in logs)
    shared = max(v.get('shared_gpu_bytes') or 0 for v in logs)
    disk = shutil.disk_usage(OUT).free / 2**30
    scores = []
    for arm in ('native32', 'random32', 'native128', 'random128', 'mean', 'zero', 'full'):
        s = report['arms'][arm]
        agreement = f"{s['route_agreement']:.2%}" if 'route_agreement' in s else '未定义'
        mse = f"{s['write_mse']:.6g}" if 'write_mse' in s else '校准或零头背景'
        scores.append(f"| {arm} | {agreement} | {mse} | {s['mlp_mse_by_GUA'][2]:.6g} |")
    table = '\n'.join(scores)
    comparisons = []
    for block, comparisons_by_base in report['gates'].items():
        for base, s in comparisons_by_base.items():
            effect = s.get('fraction_gain', s['mean'])
            comparisons.append(f"| {block} | {base} | {effect:.2%} | {s['ci975'][0]:.6g} 至 {s['ci975'][1]:.6g} | {s['passes']} |")
    comparisons = '\n'.join(comparisons)
    scalar_count = audit['scalars']
    replay_max = max(z['mass_abs'] for r in audit['cpu_replay'] for z in r['differences'].values())
    replay_gua = np.max([z['GUA_abs'] for r in audit['cpu_replay'] for z in r['differences'].values()], axis=0).tolist()
    offline = {key: float(np.mean([r['offline_restore_full_input'][key]['write_mse'] for r in audit['cpu_replay']]))
               for key in ('none', 'Q', 'K', 'V', 'QKV')}
    failure = report['example_failures'][0] if report['example_failures'] else '无路由不匹配例'
    if intervention:
        execution = ('达到了联合预测门，随后冻结剂量与主要因果指标。独立校准过门，'
                     '完成128个额外新家族的真实Q贡献阻断、匹配随机、sham和同输入恢复。')
        effects = f'''

**原模型因果试验。** 128个新家族每族固定代表题，16格每格8题，档案物理顺序与字面标题配平。介入仅对最终位置H36的原生Q投影删减冻结128坐标中相对发现均值的权重乘积之和的50%。随机组匹配RoPE后实际Q变化L2，匹配率{intervention['dose_match_fraction']:.2%}、比值范围{intervention['dose_ratio_range']}，目标剂量中位数{intervention['median_target_dose']:.6g}。用同输入原始Q在Q RMSNorm出口恢复，sham等值复制。逐位检查{intervention['bitwise']}。干预与恢复目标的归一化前编辑一致{intervention['touched_and_restored']}。技术和行为准入{intervention['technical_pass']}。

匹配组减目标组的同档竞争行对数质量差为{intervention['matched_route_effect']:.5f}，家族97.5%区间{intervention['matched_route_interval']}，范围内路由作用门{intervention['route_effect_admitted']}。原样/目标/匹配/sham/恢复的自然金路由为{intervention['gold_source_top1_by_arm']}。紧凑模型预测目标相对原样的对数质量变化均值{intervention['margin_compact_change']:.5f}，真实{intervention['margin_actual_change']:.5f}，RMSE {intervention['margin_change_rmse']:.5f}，方向一致率{intervention['margin_change_direction_agreement']:.2%}。写入变化预测MSE {intervention['write_delta_mse']:.6g}（零变化基线{intervention['write_delta_zero_mse']:.6g}），MLP A变化预测MSE {intervention['A_delta_mse']:.6g}（零变化基线{intervention['A_delta_zero_mse']:.6g}）。重复说明完整自然预测门与因果变化预测是独立终点。

匹配组减目标组最终金答案分差{intervention['answer_margin_matched_effect']:.5f}，区间{intervention['answer_margin_matched_interval']}，全词表首选{intervention['top1_by_arm']}。各分支对原样的全词表KL {intervention['kl_by_arm']}。试验为对冻结坐标组成的乘积组的干预，原生标量参与计算，单个标量、单个坐标或单个单元的必要性仍未检验。
'''
    elif calibration:
        execution = '联合预测门通过。独立干预校准未达到技术门，正式干预家族保持封存。'
        effects = f"\n\n**干预校准。** 历史独立32家族的实际剂量匹配率{calibration['dose_match_fraction']:.2%}，逐位检查{calibration['bitwise']}，技术门{calibration['technical_pass']}。校准仅作技术证据。\n"
    else:
        execution = '未达到联合预测门，冻结干预家族仍封存。'
        effects = ''
    content = f'''

## Phase {p}: B26/H36 紧凑原生 QKV、残差写入与下游 MLP 留出预测 [{ts}]

**任务和范围。** 用户要求先检验操作前 Q/K 至来源选择、V/O 写入和下游 MLP 的紧凑预测，过门后进入匹配对照、恢复干预。本轮 `clt_b26_compact_v1` 完成发现、候选冻结、128新家族确认和CPU复核。{execution} Phase 2969已有的局部原生干预另行保留。

**材料与功效。** 本地 Qwen3-14B 原 BF16 快照 `{prereg['provenance']['model_revision']}`。发现64家族1024输入，确认128家族2048输入。每族两档案的行序、目标身份、档案提示完全交叉16格，档案提示晚于实体。不同运行的既有暴露名字与本轮三个划分的名字隔离，同结构模板和24值词表复用。可推广口径限新家族。配对二值事前参考效应15个百分点、discordance=.25、alpha=.025，所需{prereg['power']['n_required']}家族，128家族的功效{prereg['power']['achieved_power']:.3f}。连续MSE的改进与区间单独报告。确认首词严格四候选正确{audit['splits']['confirm']['strict']}/2048、全词表首选正确{audit['splits']['confirm']['top1']}/2048，逐格行为门{confirm['behavior_pass']}。B26前实体位置同前缀逐位一致，发现{audit['splits']['discovery']['pairs']}对、确认{audit['splits']['confirm']['pairs']}对。

**预测算法。** 对Q最终位置和K/V所有有效前缀，从发现均值 μ、方差与原参数列平方和选32/128个原生输入坐标。给定分支的坐标集合 S，预测投影为 `p_hat = W μ + sum_(j in S) W[:,j](z[j]-μ[j])`，加法后依原流程舍入BF16。Q/K各自RMSNorm、RoPE与GQA组7原样参与。`α_hat = softmax(q_hat·k_hat/√128 + causal_mask)`。按原token跨度聚合四条记录，写入 `w_hat = W_O[H36] Σ_i α_hat[i] v_hat[i]`。自然QKV重算比对实测SDPA的最大绝对差为发现{discovery['max_head_rebuild_abs']:.6g}、确认{confirm['max_head_rebuild_abs']:.6g}，冻结门0.065419。

输出端使用真实B26输入残差和另外39个头独立重算的O写入。把预测的H36写入加入该背景，经过post-attention RMSNorm和原gate/up参数预测32个冻结单元的G/U/A。单元由发现集实测A与无H36输入时重算A的均方变化筛选。对照包含同预算随机坐标、发现均值及无头背景。来自其他头的背景是已声明的同时边界输入，不能把下游成绩解释成只用128坐标的完整块预测。128方案的Q/K/V各保留16384个矩阵标量及128截距，O仍使用655360个标量，MLP gate/up保留327680个标量及归一化参数。真实G/U/A只用于测量或发现阶段筛选。full行反映分流后重加的BF16舍入误差。

**确认结果。** 自然的四记录质量argmax对任务金记录命中{report['natural_gold_route']:.2%}。下表路由列比较预测与自然头的四记录argmax，写入MSE使用全部5120坐标，下游A使用同一组32冻结单元。

| 方法 | 路由一致率 | H36写入MSE | MLP A MSE |
| --- | ---: | ---: | ---: |
{table}

| 预测项 | 对照 | 准确率百分点优势或相对MSE增益 | 家族bootstrap 97.5%区间（配对原尺度差） | 过门 |
| --- | --- | ---: | --- | --- |
{comparisons}

操作前native128预测的同档竞争行对数质量差RMSE为{summary['same_archive_margin_rmse']:.4f}。最弱的cell {worst}只有{by_cell[worst]:.2%}，它对应两档案行序位均为1、身份0、档案1。预注册90%门用于总体，不能解释为各条件都达到90%。失败示例 ` {failure} `。保留真实质量与预测质量、H36上下文向量、原写入全坐标、实际与预测G/U/A及逐题行为，支持复核和分层分析。`evaluation.json`保存每格路由结果。

**原生索引和参数。** Q坐标起始{idx['q'][:6]}，K起始{idx['k'][:6]}，V起始{idx['v'][:6]}。下游B26单元示例{seal['units'][:6]}。按方差加权排名首位Q输入为{q_coord}，其绝对值最大的原生Q标量示例 `model.layers.26.self_attn.q_proj.weight[{36*128+scalar_idx},{q_coord}] = {scalar_value:.9g}`。排名和参数只定位运算中一项，不证明其语言含义。`predictor.npz`保留完整索引、Q/K/V/O、gate/up/down及归一参数。独立读取原权重核对{scalar_count:,}个BF16标量一致，复算各分片5120维写入MSE。CPU逐确认分片重放一例，路线质量与CUDA最大绝对差{replay_max:.6g}，G/U/A最大绝对差{replay_gua}，为跨设备舍入诊断。没有第二物理节点模型复跑。

16个分片首题均为cell 0。对此样本在CPU代理中把各分支逐一换回完整输入，写入MSE的描述性结果为{offline}。该诊断使用已开封样本，需用新的层位、条件和家族验证后才能概括分支瓶颈。

{effects}

**证据等级、局限与下一阶段。** Phase 2968的行位可读出和Phase 2969的局部K/V干预与此处的留出预测分别登记。128列按各投影的对角方差选项，忽略输入协方差和Q/K相互作用。早层是否已选定来源、从文本开始的全模型输出保真及自由生成、跨模型/句式和真正两源融合仍待检验。下游预测依赖他头背景。下一大阶段使用新冻结材料，落实身份与行位线索指向不同记录的来源冲突设计，分开评估Q、K、V预算，比较以QK竞争间隔和原生写入为目标的稀疏选列与协方差校正。候选通过后，按范围内预测误差冻结更小贡献组或单标量，并验证原模型联系和恢复。图谱保留真实原生计算索引、留出误差及实际通过的范围内干预关系。本轮未建立新的RDC统一公式或AGI结论。

**复算与资源。** `.venv/Scripts/python.exe -u -B tests/glm5/clt_b26_compact.py freeze/discovery/select/confirm/evaluate`。过联合门时，串行入口为 `clt_b26_compact_causal.py freeze/calibration/causal`，之后运行 `clt_b26_compact_audit.py` 和 `clt_b26_compact_report.py`。分片日志记录真实进度、ETA、CUDA分配和DXGI共享显存。发现{discovery['elapsed']:.1f}秒、确认{confirm['elapsed']:.1f}秒，采样CUDA分配峰值{cuda}字节，共享显存峰值{shared}字节。D盘余量{disk:.2f}GiB，保留20GiB。清理当前Matplotlib缓存{removed}字节，原生证据和日志留在 `tests/glm5/result/clt_b26_compact_v1`。本地证据已核验，未发布正式服务器。
'''
    if (OUT / 'calibration-causal-console.log').exists() and 'GPU is leased' in (OUT / 'calibration-causal-console.log').read_text(encoding='utf-8'):
        content += '\n首次校准遇到异任务持有GPU租约，未进入前向。失败日志保留，设备释放后使用同一冻结方案重试。\n'
    index_note = (f'\n\n## Phase {ip}: B26/H36 紧凑预测留出验证 [{ts}]\n\n'
                  f'见GLM5 Phase {p}。新128家族确认，操作前native128路由一致率{summary["route_agreement"]:.2%}。'
                  f'{execution} 原生参数及全坐标写入误差已独立复算。\n')
    files = {f.name: sha(f) for f in OUT.iterdir() if f.is_file()}
    commit('delivery.json', dict(provenance=prereg['provenance'], phase=p, timestamp=ts, files=files,
                                 causal_admitted=report['causal_admitted'], causal_tested=intervention is not None, cleanup_bytes=removed,
                                 report_code_sha=sha(Path(__file__))))
    for path, original, extra in ((memo, old, content), (index, ix, index_note)):
        payload = extra.encode('utf-8')
        check_write(path, len(payload), operation='CLT compact memo appendix')
        assert path.read_bytes() == original
        with path.open('ab') as stream:
            stream.write(payload)
        assert path.read_bytes() == original + payload
    print(f'Closed Phase {p}, index {ip}, original memo bytes unchanged.', flush=True)


if __name__ == '__main__':
    close()
