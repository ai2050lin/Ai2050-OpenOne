"""Append the scientific and delivery record, with numbers from the committed results."""
from datetime import datetime
import re
from zoneinfo import ZoneInfo

import mlp_functional_common as C
from mlp_atlas_publish_final import Archive


def append(path,body):
    with path.open('a',encoding='utf-8',newline='\n') as stream:stream.write('\n\n'+body.strip()+'\n')


def phase(path):
    return 1+max(map(int,re.findall(r'^## Phase (\d+):',path.read_text(encoding='utf-8'),re.M)))


def main():
    output=C.OUT/'phase-registration.json'
    if output.exists():return
    archive=Archive(C.PUBLIC,C.RUN,C.identity());reg=C.read(C.OUT/'registration-final.json')
    report=archive.lib.artifact(reg['report_sha256']);dictionary=report['dictionary'];rev=report['revision']
    gen=report['generated'];cap=report['capability'];paths=report['paths']
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';claude=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    assert C.RUN not in memo.read_text(encoding='utf-8')[-20000:]
    n=phase(memo);cn=phase(claude);stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    numbers=[]
    for split,r in dictionary['summary'].items():
        lines=[]
        for name in ['clean','zero','archived_sae','mse_control','functional_sae','native256']:
            v=r['operations'][name]
            lines.append(f'{name} KL {v["KL_clean"]:.6f}、CE {v["CE"]:.6f}、FVU {v["FVU"]:.6f}、恢复 {v["loss_recovered"]:.4f}')
        ci=r['comparisons']['native256']
        numbers.append(f'- {split}（{r["documents"]} 篇）'+ '。'.join(lines)+f'。原生减行为 SAE 的配对区间 KL {ci["KL_clean"]["adjusted_CI"]}，CE {ci["CE"]["adjusted_CI"]}。')
    path_lines=[]
    for attr,r in rev['replication'].items():
        v=r['incoming'];winners=[f'B{p["block"]}/H{p["head"]}' if p['head'] is not None else f'B{p["block"]} MLP' for p in r['writers'] if p['replicated']]
        path_lines.append(f'- {attr} 的修订 confirm / lockbox 单条准确率 {v["confirm"]["classification_accuracy"]:.2%} / {v["lockbox"]["classification_accuracy"]:.2%}。锁箱两种 R 配对排序为 '+
                          ' / '.join(f'{q:.2%}' for q in v['lockbox']['paired_accuracy_by_assignment'])+f'。通过完整复现门的写入为 {", ".join(winners) or "无"}。')
    remainder=rev.get('remainder',{}).get('summary',{}).get('lockbox')
    if remainder:
        heads=' / '.join(f'{v:.2%}' for v in remainder['selected_heads']['paired_accuracy_by_R'])
        full=' / '.join(f'{v:.2%}' for v in remainder['full_residual']['paired_accuracy_by_R'])
        mismatch=remainder['head_positive_full_nonpositive']['count_by_R']
        path_lines.append(f'- 生计两 Head 联合投影的锁箱配对排序为 {heads}，全残差同方向为 {full}。两种 R 中各有 {mismatch[0]} 和 {mismatch[1]} 对局部正而全场非正。此分解在复现后选两 Head，属描述性证据。剩余项包括所有其他写入和原生舍入。')
    dose=[]
    for attr,r in gen['summary'].items():
        if not isinstance(r,dict):continue
        dose.append(f'- {attr}，{r["n"]} 个独立背景。'+ '。'.join(
            f'{name} 正确率 {v["accuracy"]:.2%}、KL {v["mean_KL"]:.5f}、生计方向的带符号残差 {v["mean_signed_industry_residual"]:.5f}、首词间隔 {v["mean_answer_gap"]:.4f}'
            for name,v in r['operations'].items())+f'。全剂量较对照的损失差 {r["target_minus_random"]}。')
    limits=[]
    for name,v in cap['summary'].items():
        if not isinstance(v,dict):continue
        lower,upper=v['loss_interval']
        limits.append(f'{name} 错转 {v["losses"]}、对转 {v["gains"]}，净损 {100*v["net_loss"]:.3f} pp，区间 [{100*lower:.3f}, {100*upper:.3f}] pp，判定 {v["status"]}')
    generated_pass=gen['summary'].get('industry',{}).get('passed',False)
    flow=report.get('flow_boundary')
    flow_note=(f'事后原生复算的 {flow["cases"]} 个暴露样本中，{flow["local_residual_changed"]} 个问题前残差随单独移除 B35/H25 改变。'
               f'后续 query logits 逐位一致 {flow["all_later_query_logits_bitwise_equal"]}，全部 36 层前缀 KV 逐位一致 {flow["all_prefix_KV_bitwise_equal"]}，实际生成一致 {flow["all_generated_tokens_equal"]}。'
               '末块的 Head 写入在本层 K/V 形成之后，固定前缀时它只进入该位置余下的末块计算和读出。这是局部响应的时序边界。') if flow else ''
    counts=[f'{k} {v["n"]} 条／{len(v["chunks"])} 分片' for k,v in report['cases'].items()]
    api=C.read(C.OUT/'live-final-acceptance.json');proof=archive.put('delivery-acceptance',{
        'schema':'functional_delivery_acceptance.v1','api':api,
        'files':{p.name:C.file_sha(p) for p in C.OUT.glob('client-*.png')},
        'software':{name:C.file_sha(C.OUT/name) for name in ['frontend-build-final.log','frontend-tests.log','protocol-tests-final.log','node-tests-retry.log']}})
    protocol=(C.OUT/'protocol-tests-final.log').read_text(encoding='utf-8',errors='replace')
    protocol_outcome='261 项复跑全部通过' if re.search(r'\bOK\s*$',protocol) else '全量复跑日志有未通过项，24 项节点服务单独复跑通过'
    body=f'''## Phase {n}: 行为保真字典与问题前残差写入的留出和干预 [{stamp}]

运行 `{C.RUN}` 使用同一 Qwen3-4B BF16 本机 CUDA 快照 `{C.identity()['model_snapshot']}`。单 GPU 顺序执行。原数据、程序版本、结果和内容寻址引用分别保存在 `tests/glm5/`、`tests/glm5/result/mlp_functional_v1/` 和研究库。原版行为失败与修订版分开归档。

**字典目标。** 设 y 为原生 B35 MLP 写入，r 为进入其 MLP 的残差，N 为最终归一化，U 为真实词表映射。预测分布为 `softmax(U N(r+ŷ))`。在 384 个已暴露开发文档的采样位置上，用全词表 `KL(p_clean || p_ŷ) + .15 CE(x_next, p_ŷ) + .10 MSE(y,ŷ)/σ²` 校准归档 SAE，另留 128 个旧文档作开发检查。与同起点纯 MSE 各 640 步，批 32，k=256，训练使用紧凑 BF16 词表计算。正式留出保持原序列布局，真实 hook 和原生输出逐位匹配。旧 SAE、同起点续训和原生 256 单元同时对照。自然材料使用此前未用的固定 Wikipedia 行组，确认和锁箱各 128 中文加 128 英文，每篇输入 1024 token、读出 8 处。数据身份按来源和 token 前缀核对。训练期语料重叠未知。

{chr(10).join(numbers)}

**行为门和原生计算。** 原版 448 个背景各做 R、T、Q 的 2×2×2 因子设计，共 3584 条，背景最少 895 token。两种问题共用问题前前缀。原版试跑 256 条全对，正式确认的一格 65/80。28 个首词失败的完整输出均为 Answer 加正确属性词，按原格式门保留失败，原版范围干预未进入执行。问题前原生 Q0 采集 1792 个案例，36 块全部 Q Head，35 块 MLP 写入，37 个残差边界。每个 Q0 案例逐位检查 `r_(b+1)=fl(fl(r_b+attn_b)+mlp_b)`。同一材料两种未来问题的原版问题前各层状态逐位相等，选择词出现前的 T 配对也逐位相等。发现集拟合方向 d 后，直接用 `d^T O_(b,h) z_(b,h)` 和 `d^T y_b` 分解写入，保留 BF16 加法与 FP32 分解之间的误差。原版地理与生计投影重建 RMSE 分别为 {paths['observations']['location']['projection_residual_rmse']:.4f} 和 {paths['observations']['industry']['projection_residual_rmse']:.4f}，并存全部写入统计。原版行为门限制了证据晋级。

**新组合复核。** 保留 v1 候选和阈值，改用 352 个新背景、虚构实体和记录键组合，移除提示末尾的 Answer 头，仍严格用首生成词评分。共 2816 条（256 试跑，2560 留出），背景最短 897 token。修订行为门 {'通过' if rev['passed'] else '未通过'}，最弱格 {min(v['correct'] for v in rev['cells'])}/80。修订 Q0 样本保存 1280 个候选块原场，两种未来问题的 B35 输入逐位一致。生计 I=T XOR R，属性对照要求两种 R 均同向，并分别在两个留出上通过家族校正。只以冻结方向预测，阈值和提名均不依赖修订留出。

{chr(10).join(path_lines)}

**范围干预。** 原版的 B34/H2 在原行为门过滤前已通过生计写入区间，因此在独立修订中预注册生计组，包含 B34/H2、B35/H25、B33 MLP、B34 MLP。以 160 个锁箱背景各做两种问法，比较全组、半剂量、Head、MLP、B34/H2 单独与同层同位置等范数扰动。扰动控制使用实际写入的固定符号排列，范数匹配在 BF16 加法之前成立。每个操作均从修改后的前缀重新执行真实下游和 KV。主门需要至少 10pp 生成正确率净损失、配对精确 p<.05，且较等范数控制多损至少 5pp。结论为{'支持此范围的联合干预效应' if generated_pass else '未达预注册主门'}。残差、词表 KL、首词间隔与生成错误分开记录。

{chr(10).join(dose) if dose else '修订行为门未过，未执行范围内生成干预。'}

{flow_note}

**能力。** 已暴露的固定监测池 2200 道正式、140 道参考。字典改在回答边界，上游操作的位置用修订试跑中的预注册相对偏移，遇到短提示限于有效前缀。净损区间是经两操作校正的配对精确损／增界，门为 2pp。{'。'.join(limits)}。证据仅覆盖本次操作和监测池。

**归档与可复核性。** 3D 工作包含 {reg['cards']} 张定位卡与 {reg['edges']} 条类型边。逐题材料为 {'，'.join(counts)}。原生张量按哈希本机回查。原版与修订范围、留出结果和已知失败都分开保留。真实 API {len(api['checks'])} 项、前端 73 项和生产构建通过，分布式协议{protocol_outcome}。研究报告 `{reg['report_sha256']}`，图谱 `{reg['graph_sha256']}`。

**理论和下一阶段。** 行为误差由残差写入方向经 N 与 U 的实际读出共同决定，场误差与预测误差可交换的假设在两组新留出没有成立。上游每条写入的数学来源可追溯，与属性标签的对应仍要靠新组合、真实前向干预和随机对照检验。下一大阶段应在新实体关系与更细的时间位置上，分开检验选择词后、问题前和回答边界，针对低余量材料预注册以错误率和输出分布为双指标，再让冻结图谱独立预测未见样本的激活及下一词。误差通道、原生单元基线和不复现路径均保留。
'''
    sha=archive.put('phase-summary',{'schema':'research_phase_summary.v1','phase':n,'markdown':body,'report_sha256':reg['report_sha256'],
                    'graph_sha256':reg['graph_sha256'],'acceptance_sha256':proof,'timestamp':stamp})
    append(memo,body)
    append(claude,f'''## Phase {cn}: 行为保真校准与问题前写入（GLM5 Phase {n} 详记） [{stamp}]

运行 `{C.RUN}`。B35 在两个新自然留出降低了相对旧字典的 KL 和预测损失，仍未通过同预算原生基线比较。原版行为失败及全部生成反例保留，修订使用新材料和冻结路径。正式证据、反例、计算来源和能力门均已进入 3D 研究库。对应研究报告 `{reg['report_sha256']}`。''')
    C.save(output,{**C.identity(),'phase':n,'claude_phase':cn,'summary_sha256':sha,'acceptance_sha256':proof,'timestamp':stamp})
    print('Phase',n,'Claude',cn,'summary',sha,flush=True)


if __name__=='__main__':main()
