"""Verify both frozen studies, append one research phase, and bind delivery supplements."""

from pathlib import Path
import json

import numpy as np

import paradigm_routes as X
import paradigm_routes_report as V
import paradigm_source_causal as Q


R,C,OUT=X.R,X.C,X.OUT


def main():
    R.OUT=OUT
    plan=X.verify()
    source=Q.verify()
    audit=R.read('native-audit.json')
    assert audit['plan']==C.A.digest(OUT/'plan.json') and audit['parameters']==C.A.digest(OUT/'native-parameters.npz')
    assert C.A.digest(R.ROOT/audit['script'])==audit['script_sha']
    assert sum(r['n'] for r in audit['records'])==1024
    assert all(r['GUA_native_formula_max_abs']==0 for r in audit['records'])
    assert (OUT/'attention-write-map.png').is_file() and (OUT/'mlp-all-unit-map.png').is_file()
    predictions=X.read_np('confirm-predictions.npz')
    random_gain=V.family_summary((predictions['native'][:,0]-predictions['random'][:,0]).reshape(-1,8).mean(1))
    counts=0
    for receipt in sorted([*OUT.glob('pilot-???.json'),*OUT.glob('discovery-???.json'),*OUT.glob('confirm-???.json')]):
        record=R.read(receipt.name)
        with np.load(OUT/record['file'],allow_pickle=False) as saved:
            embed=saved['field'][:,0].reshape(-1,8,5120)
            assert np.array_equal(embed,np.repeat(embed[:,:1],8,axis=1))
            counts+=len(embed)
    assert counts==128
    deviations=dict(at=C.stamp(),initial_source_attempt=dict(status='native_process_0xc0000005',last_saved_progress='110/160',
                    log='source-causal/campaign-first-attempt.log',log_sha=C.A.digest(Q.OUT/'campaign-first-attempt.log')),
                    initial_cpu_attempt=dict(status='process_0xc0000005',result='no audit result written'),
                    recovery='Same frozen causal plan rerun with unchanged code and material, then CPU re-read executed serially. Both complete. Original native study intact.',
                    causal_plan=C.A.digest(Q.OUT/'plan.json'),causal_audit=C.A.digest(Q.OUT/'audit.json'),native_audit=C.A.digest(OUT/'native-audit.json'),
                    unresolved='Initial process exit cause not established.')
    R.write('execution-deviations.json',deviations)
    extras=f'''### 两条主线的具体推进与独立复算补充

本轮采集 128 个家族的 1,024 个前缀，在同一共享 query token 保存 40 层 × 17,408 个 MLP 单元，共 696,320 个原生单元的 G/U/A；每层40头与全部41个原生残差边界。16 家族另作128次指定 print 引号前缀的条件语言读取。所有128家族各八种范式组合在 query 的 embedding 逐位相同。新来源因果面板64家族512视图。它们与父研究的发现、确认和锁箱内容无重合。

五个原生分片共1,024视图在CPU复算 G/U/A，保存 BF16 门乘法与原始激活最大差均为0。根据 Q/K 原生来源 logsumexp 重建源组注意力，最大误差 {max(r['attention_reconstruction_max_abs'] for r in audit['records']):.8f}。把全部头和全部MLP原生写入投影求和，与 raw 残差实际增量核对，各分片平均绝对差约 {np.mean([r['raw_write_sum_vs_residual_mean_abs'] for r in audit['records']]):.5f}，最大 {max(r['raw_write_sum_vs_residual_max_abs'] for r in audit['records']):.5f}，包括原生 BF16 投影与残差相加舍入。参数导出含32所选和32同层随机单元的 gate/up 行与 down 列，以及所选/对照头的 Q/K/V/O、归一化和实际读出行，均可按原模型标量索引回查。

预先选定的32单元增强全词表预测相对文本基线改善11.66%。随机同层32单元与同一个头摘要的KL更低，native−random的家族均值差为 {random_gain['mean']:.5f}，99% 区间 {random_gain['ci']}。因此以局部输出投影强弱选出的集合，其预测特异性未获支持。候选限定层12–31，只有该读出方向全部单位平方响应的2.51%。全头热力图显示更深部还有显著写出。后续选择须联合自然输入来源、G与U的条件依赖、真实读者连接、路径因果及未见预测，并保留同预算随机集合。

源因果试验的主效果接近零，剂量P95约29.8%且大于冻结3%门槛。部分原生下游单元对两个来源的编辑有响应，全词表少数首选变化和完全自恢复保留。单头、两个短源组的结果约束这一候选的来源解释。尚需追踪分散在指令、格式说明及上下文的真实读者与并行路径。Python请求在部分条件原生首词为引号，校准的功能限制也完整保留。

两条路线的共同记录对象设为“条件来源 → Q/K 选取 → V/O 写入 → MLP 单元 G/U/A → 下一层读取或输出”。对源组 R 的 m_R=Σ_i∈R α_i 与 w_R=W_O Σ_i∈R α_i v_i 分别保存选取量与实际写入。多个范式贡献到同一残差和同一门控，比较单条件与组合条件的主效应与交互。来源竞争可在同头 softmax 的归一化中检验，跨头写入的支持/抑制及下游门控另行检验。范式节点可与多个头、单元和语境关联，其物理定位和跨层重用按原生索引登记。

下一大阶段由现有全场找主干并局部细化到单坐标和单标量参数。材料正交交叉输出语言、论文/代码文体、因果/转折逻辑及语法槽位，加入内容知识关系与角色变换。先比较相同 token 和相同读出位置上的全坐标响应；按 G、U 及注意力真实来源得到候选计算角色，再追踪连接与融合/竞争位置。正确、错误和竞争输出都进入结构发现，行为评分界定功能命题的适用域。冻结后用新内容和新范式组合预测响应、全词表分布及原生干预方向。按原模型匹配阻断、独立恢复、并行替代路径及跨物理节点复算分别验收。当前阶段中的最强局部写出与预测最有用组件是两个可比较的选择目标，应由同一留出证据决定是否一致。

注意力图 attention-write-map.png 显示全部40×40个头在请求格式、示例格式及二者交互上的原生写出。MLP图 mlp-all-unit-map.png 保留40×17,408原生坐标顺序，以对称对数色标显示原始贡献量。原数值保存在同目录NPZ中。

来源与原理对照见 [Goodfire 神经元研究](https://www.goodfire.com/research/neurons-as-an-interpretable-basis)、[QK/OV 计算分解](https://transformer-circuits.pub/2021/framework/index.html)、[任务信息由注意力头传递的因果个例](https://arxiv.org/abs/2310.15213)。本轮研究靶标从原生单元和语言范式两端对接，进展按真实计算联系、未见预测及范围内原生作用计量。

来源试验首次到110/160后进程返回0xC0000005，同期CPU审计也返回相同退出码，未形成该次因果结果。随后同一冻结计划串行重跑，原生来源检验和CPU审计均完成，原因未确定。原始失败日志、恢复入口及实际产物绑定在execution-deviations.json。暂存清理只涉及本轮图表字体缓存与独占目录。
'''
    original=V.render
    V.render=lambda p,r:original(p,r)+'\n\n'+Q.paragraph()+'\n\n'+extras
    V.close(plan)
    Q.write('manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=C.A.digest(p)) for p in Q.OUT.iterdir()
                                      if p.is_file() and p.name!='manifest.json'},sources=source['sources']))
    manifest=R.read('manifest.json')
    for script in ('paradigm_routes_audit.py','paradigm_routes_visualize.py','paradigm_routes_finish.py','paradigm_source_causal.py'):
        key='tests/glm5/'+script
        manifest['sources'][key]=C.A.digest(R.ROOT/key)
    manifest['substudies']={'source-causal/manifest.json':C.A.digest(Q.OUT/'manifest.json')}
    R.write('manifest.json',manifest)
    print('Verified, appended and bound supplementary evidence',flush=True)


if __name__=='__main__':
    main()
