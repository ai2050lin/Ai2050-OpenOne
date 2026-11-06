"""Append the phase from frozen outputs and the isolated API acceptance proof."""
from datetime import datetime
from zoneinfo import ZoneInfo

import mlp_factor_steer_common as C
from mlp_atlas_publish_final import Archive
from mlp_functional_record import append, phase


def main():
    memo = C.ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
    claude = C.ROOT / 'research/claude/docs/AGI_CLAUDE_MEMO.md'
    assert all(C.RUN not in p.read_text(encoding='utf-8')[-20000:] for p in (memo, claude))
    archive = Archive(C.PUBLIC, C.RUN, C.identity())
    reg = C.read(C.OUT / 'registration-final.json')
    proof = C.read(C.OUT / 'api-upload-proof.json')
    package = C.read(C.OUT / 'upload-package.json')
    doc = archive.lib.artifact(reg['report_sha256'])
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    title = '属性证据、选择键与错误草稿的最小对照及原生写入跨任务检验'
    phase_n = phase(memo)
    outcomes = []
    for task, result in doc['tasks'].items():
        v, audit = result['summary'], result['checks']
        s = doc['screen'][task]
        ops = v['operations']
        hits = '、'.join(f'{key} {ops[key]["targeted_alt"]}/{v["n"]}' for key in ops)
        gap = ops['targeted']['target_gap_gain']
        rows = C.read_gz(C.OUT / f'{task}-result.json.gz')['rows']
        flips = {op: {r['index'] for r in rows if r['conditions'][op]['word'] == r['contrast']}
                 for op in ('targeted', 'mlp_only', 'head_only', 'source_only')}
        outcomes.append(f'- {task} 筛查 {s["screened"]}/{s["available"]}，干净首词正确率 {s["clean_accuracy"]:.2%}，纳入 {v["n"]}，'
                        f'规划功效 {s["power"]["achieved_power"]:.3f}。反答案首词计数为 {hits}。首词门 {v["passed"]}，'
                        f'联合相对等范数优势 {v["specificity"]:.2%}，相对干净精确 p={ops["targeted"]["targeted_mcnemar_p"]:.8g}。'
                        f'联合的平均定向间隔变化 {gap["mean"]:.4f}，按文档自助区间 {gap["ci95"]}，全词表 KL {ops["targeted"]["mean_KL"]:.6f}。'
                        f'联合与仅 MLP 的成功交集 {len(flips["targeted"] & flips["mlp_only"])} 条，和仅源槽回放的交集 {len(flips["targeted"] & flips["source_only"])} 条。'
                        f'干净布局一致 {audit["clean_exact"]}/{v["n"]}。回放与恢复首步 logits 逐位一致分别 {audit["replay_logits_bitwise_equal"]} 和 {audit["rescue_logits_bitwise_equal"]}，'
                        f'四 token 生成一致分别 {audit["replay_generation_equal"]} 和 {audit["rescue_generation_equal"]}。'
                        f'范数比较 {audit["norm_vectors"]} 个向量，原生舍入前最大相对误差 {audit["max_pre_rounding_norm_error"]:.3e}，'
                        f'舍入后 {audit["max_post_rounding_norm_error"]:.3%}，起点前无改变 {audit["no_cache_changes_before_origin"]}。')
    contrasts = [f'- {k} 均值 {v["mean"]:.4f}，95% 区间 {v["ci95"]}。'
                 for k, v in doc['factorial']['factorial_effects'].items()]
    capability = []
    for k, v in doc['capability']['summary'].items():
        lo, hi = v['loss_interval']
        capability.append(f'- {k}，{v["n"]} 道门内题，干净 {v["base_accuracy"]:.2%}，编辑 {v["edited_accuracy"]:.2%}，'
                          f'丢失 {v["losses"]}、新增正确 {v["gains"]}，净损失 {v["net_loss"]:.2%}，'
                          f'同步精确区间 {lo*100:.2f} 至 {hi*100:.2f} 个百分点，2 点保持门 {v["status"]}。')
    sel = doc['selection']
    lines = '\n'.join
    body = f"""## Phase {phase_n}: {title} [{stamp}]

### 原理与预注册

本地 Qwen3-4B BF16，固定模型和 tokenizer 快照 `{reg['model_snapshot']}`。3200 篇新背景按文档隔离，英文、中文各 1600 篇。每条实际模型输入为 1034 token，背景至少 841 token。64 篇用于 E/S/D 2×2×2 析因，共 512 条；768 条成对边逐 token 核对，唯一变化分别位于签署属性、选择键、草稿。另用 64 篇挑选写入，两个未见任务各冻结 1536 篇。属性查询更换实体、键和背景；派遣任务额外要求呼号→工单→许可→队伍。两任务共享 fishing/mining 答案词。

设 \\(g=\\ell_{{fishing}}-\\ell_{{mining}}\\)，估计固定其他因子时的成对 \\(\\Delta g\\)，选择键改变的效应按目标 A 的属性定向。对原生写入通道 \\(w\\)，用发现集签署属性变化均值 \\(e_w\\) 和草稿、同答案键切换均值组成的干扰子空间 \\(U_w\\)，方向为 \\(v_w=(I-P_{{U_w}})e_w\\)。拟合位置为属性 token、键 token 和共同前缀末两 token。干预在真实 Head 的输出投影前以及 MLP 残差写出位置施加 \\(x_w\\leftarrow\\operatorname{{BF16}}(x_w+\\alpha s v_w)\\)，\\(s\\) 按预先给定反目标定向。对照使用同通道的符号置换方向，逐样本、逐位置和逐通道匹配实际施加向量的范数，再舍入 BF16。

### 结果

{lines(contrasts)}

六个发现集配方在干净正确的样本中均无定向首词切换。按冻结的次级排序取 `{sel['recipe']['id']}`，Head {sel['recipe']['heads']}，MLP {sel['recipe']['mlps']}，α={sel['recipe']['alpha']}。首词确认门为相对干净至少 10 个百分点、相对等范数至少 5 个百分点、双侧配对 p<.025，两个任务分别判定。保持门按既定 2200 道题评估，另有 140 道参考题。

{lines(outcomes)}

{lines(capability)}

### 结论与瓶颈

最小对照把签署属性、选择键和草稿对输出的效应分开。未选记录仍可影响答案，错误草稿也有独立影响。回放与恢复约束的是冻结干预通过改变后的前缀缓存对生成的传递。在 B35 展示四个源位的 K/V，全部改变槽的回放跨层执行，不能据此赋予某层或某个 Head 独立因果归因。仅 MLP、仅 Head 和源槽回放均单独存档，效应交集由逐题结果确定。等范数控制定义在舍入前，舍入差异单列；它是一种方向对照，不能穷尽所有等能量干预。低余量样本和显式二词首词指令限定了外推范围；本试验没有验证无提示自由文本、未见答案词、真实世界知识或单参数机制。保持门仅覆盖冻结的选项题条件和给定偏移，需与首词门独立解释。

从编码结构看，相同输入 token 可以在选择关系变化时产生不同效应。下一阶段应按记录结构冻结第三类任务和未见答案词，区分绝对词方向、角色与属性组合；以仅 MLP 的结果作为基线，做发现集内的组缩减和新留出复现。在原生轨迹上继续分解缓存路径，检查哪个原生写入携带、读取或放大条件信息。最小对照与真实可达输出的联合预测是后续图谱的可证伪判据。

### 图谱和发布

运行 `{C.RUN}`，报告 `{reg['report_sha256']}`，图谱 `{reg['graph_sha256']}`。中心索引保存原生写入候选、观察边、两个首词门和逐题反例。能力评估发布题号、来源和预测结果，题干保留在节点。发布包 {package['objects']} 个内容寻址对象。{proof['health']['version']} 程序包独立启动后，两次执行真实数据上传，匿名客户端索引及 BF16 数组读取验证均通过。隔离验收端点 `{proof['endpoint']}`，正式中心由用户手动升级后再导入。
"""
    append(memo, body)
    hits = '、'.join(f'{task} {v["summary"]["operations"]["targeted"]["targeted_alt"]}/{v["summary"]["n"]}'
                    for task, v in doc['tasks'].items())
    append(claude, f"""## Phase {phase(claude)}: {title} [{stamp}]

详见 GLM5 Phase {phase_n}。本地 Qwen3-4B 通过 768 条单 token 析因边分离事实、选择键和草稿。冻结写入在独立低余量样本上得到定向首词 {hits}。能力保持、等范数、回放恢复及失败样本逐题保留。研究库报告 `{reg['report_sha256']}`，图谱 `{reg['graph_sha256']}`。服务器 {proof['health']['version']} 独立发布包与研究上传包已在隔离中心通过 API 联调。
""")
    C.save(C.OUT / 'phase-registration.json', {'phase': phase_n, 'timestamp': stamp, **reg})


if __name__ == '__main__':
    main()
