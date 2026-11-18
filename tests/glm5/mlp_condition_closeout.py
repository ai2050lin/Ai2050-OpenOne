"""Append evidence-scoped closeout only after server and visual acceptance."""
from datetime import datetime
import hashlib
import json
import re
import mlp_condition_v2 as V
import mlp_condition_design as D

V.configure()
r=D.READ(D.OUT/'report.json');reg=D.READ(D.OUT/'registration.json')
accept=D.READ(D.OUT/'server-acceptance.json');visual=D.READ(D.OUT/'visual-acceptance.json')
client=D.READ(D.OUT/'client-release.json');cleanup=D.READ(D.OUT/'cleanup.json')
assert accept['bundle_checks']['remote_verified']==reg['objects']
assert visual['passed']
now=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
files=[D.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md',D.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md']
first=files[0].read_bytes()
phase=max(map(int,re.findall(r'^## Phase (\d+):',first.decode('utf-8-sig'),re.M)))+1
rows=[]
for label,title in [('broad','B29/U17 → B30/U2'),('rare','B33/U1521 → B34/U6097')]:
    o=r['observations'][label]
    rows.append(f"| {title} | {o['pearson']:.6f} | {o['spearman']:.6f} | {o['winsorized']:.6f} | {o['conditional']['relative_mse']:.6f} | {o['conditional']['family_wins']}/512 | 未通过 |")
cap='；'.join(f"{k}: 原对改错{v['lost']}/{v['n']}、损失上界{v['loss_upper']:.4%}、{'通过' if v['passed'] else '未通过'}" for k,v in r['capability'].items())
doses='\n'.join(f"| {c['pair']} | {c['dose']} | {c['n']}/512 | {c['full_vocab_changes']} | {c['target_prediction_mse']:.7g} / {c['target_nochange_mse']:.7g} | {c['distribution_prediction_KL']:.7g} / {c['distribution_nochange_KL']:.7g} |" for c in r['dose_curve'])
body=f'''

## Phase {phase}: 条件响应卡、长文本冻结验证与正式中心闭环 [{now}]

**授权与范围。** 延续用户要求，以本地 Qwen3-4B CUDA BF16 验证广泛关联与稀有强响应，控制词身份、角色、位置和答案映射，先预测再干预，保留严格等范数、恢复与能力保持。用户恢复正式中心后继续协议上传。本轮没有换模型、更新权重或远程部署服务器程序。网页更新仍由用户手动发布。

**材料、身份与失败记录。** 模型快照 `{D.SNAPSHOT}`，运行 `{D.RUN}`，预注册 SHA256 `{r['plan_sha256']}`。v1 在发现阶段的四句式准确率为 seal 72.85%、board 80.27%、sample 93.55%、screen 100%，未过行为门；锁箱未打开，失败材料保留。v2 改用明确 APPROVER/OBSERVER 角色分配，另建发现与留出材料并重新冻结。128 个发现家族 × 16 变体、512 个新留出家族 × 16 变体，发现长643–683 token，留出641–673 token。每家族四因子完全交叉、token长度相同、角色最小对词袋相同，六个读出位置均在完整证据之后；答案映射只在末位置作为预测输入。发现四句式全对；新留出 logbook、directory、assignment 全对，allocation 99.8047%。这是四类人工角色任务和重复背景段落，不是512篇独立自然文档，也不是广泛语义任务的确认。

**原理和算法。** 原生单元 `A_j=SiLU(G_j)·V_j`，其写出为 `A_j W_down[:,j]`。旧统计的同token方差比为 `Σ(A−mean(A|token))² / Σ(A−mean(A))²`，仅表示同词条件变化；负门控的 A² 份额没有乘写出权重，不等于知识量或因果贡献。条件预测为 `ŷ=βᵀz+αA_source+Σγ_k A_source z_k`，岭惩罚 λ=1；z含词身份、角色、位置、答案映射、末位置标记及发现token指示。三方法同材料比较：仅z、z+源A、z+源A及条件交互。以文本家族聚合误差，冻结门为胜出家族≥65%、单侧二项p≤.005且总MSE至少降低10%。源A在预测时已知，不是图谱独立预测模型。

**协议审计。** 冻结文字另外提到了句式指示变量，但冻结代码未实现；本轮保留差异并按实际算法报告，没有在开封后改代码补过门。所有产物 `formal_evidence_promotion=false`，数值门不触发正式语义/因果晋级。稀有层按旧发现 q99 冻结阈值，至少128个留出家族才做确认；本轮稀有候选仅14位置/3家族，分层功效不足。预注册的家族功效与能力非劣检验补充均保留；能力补充在留出及能力运行前冻结，n=2200、α=.0125、2%损失门，针对预期0.5%损失的精确功效>99.99%。这不保证任意微小效应有同等功效。

| 候选 | Pearson | 排序关联 | 削弱极值后 | 条件预测MSE/基线 | 家族胜出 | 数值门 |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

广泛候选仍有负相关，但MSE只降低8.89%、胜出61.13%，不能把显著p值当作过门。稀有候选的Pearson接近1，排序关联和削弱极值后的关联却为负，且总MSE反而增加；这给出了“极少强点制造近完美相关”的真实反例。即使348个家族相对基线有微小改善，也不能抵消总误差恶化。

**先预测、原生写入与恢复。** 发现64家族校准完整151,936词表logit有限编辑斜率，按角色×映射四层冻结；预测承诺在每家族每候选编辑后缀前写盘。源单元末token施加±发现标准差的0.5/1/2倍，方向由冻结斜率选择以反对原始A/B胜者。保留原模型完整后缀重算、1%容差BF16实际残差等范数对照、目标单元恢复和等范数假恢复。3,072组对照全部合格；512家族取消全部编辑后logits逐位一致。三个剂量两对候选均无A/B翻转、无全词表首词变化，不支持行为开关或中介命题。有效目标恢复对照 broad 459/512、rare 189/512，余者仍保留，不能用源端全合格掩盖目标端缺口。这里重算的是末token写入后的后缀，未新增早期token KV历史中介结论。

| 对 | 剂量 | 源对照有效 | 首词变化 | 目标预测MSE/不变基线 | 输出预测KL/不变基线 |
| --- | --- | --- | --- | --- | --- |
{doses}

仍有一个可保留的正面拼图：在1倍有限写入下，冻结预测器的目标单元变化MSE分别是不变基线的0.3358和0.2550，即能预测部分微小的内部响应。不过全词表分布预测 broad 仅小幅改善，rare 的KL反而为不变基线的1.378倍，2倍剂量更达到4.353倍。局部响应预测、整体输出保真和行为控制因此不能混成一个通过标志；这里没有预注册将这些描述性MSE改善晋级为语义机制。

能力保持：{cap}。只针对1倍剂量、两个源单元的正负编辑及既有2200题，不外推到2倍或自由长生成。全部原始分数、预测误差、{len(r['counterexamples'])}条未产生目标行为/控制问题记录、16条最大相对预测误差家族示例与完整输入保留。无翻转是干预命题的负结果，不等于冻结预测器曾预言一定翻转。

数值边界：全词表KL用FP32计算，逐样本约10⁻⁸及更小量级可出现舍入导致的微小负值，原值保留，不赋予负KL物理含义，也不据此宣称保真改进。报告首次输出因NumPy int64计数不可直接JSON序列化而中断；单独输出包装器只把NumPy标量转换为同值Python标量，冻结分析代码、阈值与原始数值未改，附serialization-audit.json。发布日志不作为其自身的固定哈希输入；所有真正列入清单的原始产物已逐一复核。

**工程故障与修复。** 常规 Transformers safetensors映射加载曾在Windows原生存储切片处退出(-1073741819)，关闭异步后仍复现。新增顺序读原始张量加载器，沿用原模型类与BF16计算，并核对全部398张量CPU原始字节与CUDA往返SHA；加载约20秒。该修复随v2冻结，不修改模型值。留出耗994秒、干预316秒，留出峰值显存9121MiB（约8.91GiB），共享显存约82MiB。逐批日志、退出信息、哈希、原始BF16采集与失败v1全部保留。

**服务器和客户端闭环。** 正式中心1.0.29恢复。研究包 `{reg['bundle']}`，SHA `{reg['bundle_sha256']}`。通过正式客户端管理员上传界面逐对象登记；回读并验证全部{reg['objects']}个CAS对象。首次/重复上传的界面回执见 `visual-acceptance.json`。8个服务器条件层对象覆盖77,824个原生编号；其中只有4个单元接受本轮受控检验，另携带4×3×2560完整gate/up/down参数、全部干预记录、能力分数、文本和所选原值、无损全词表斜率及复算清单。完整本地采集原场未重复上传。

新版客户端保持菜单与配色，默认条件卡、关键词折叠；支持八层3D、全部单元编号、条件指标、三种关联口径、剂量、能力保持、反例和完整参数。未测关系不填零；节点着色与关联线度量分开。专用新版页面在本机5035读取正式中心API并验收，不把本机文件伪装为服务器数据。正式网站程序尚待用户部署网页包 `{client['path']}`（SHA `{client['sha256']}`）；旧网页能读取通用对象，不能声称新专用画面已远程部署。前端56项测试通过，两个Vite生产构建通过，研究算法4项植入/负控检查此前通过；完整协议回归未跑，本轮未修改共享协议或中心代码。大JS分块警告保留。

**理论增量与下一阶段。** 当前更支持“条件响应与计算用途必须分开”的组织方式，而不是单元贴一个永久词义。相关结构可跨长文本保留，却没有成为本任务答案杠杆；稀有强响应应单独建触发条件与频度模型，不能让极值支配整张语义图谱。下一大阶段应冻结完整基线规范与新的文档家族：① 从发现材料提取稀有触发的前缀、位置及G/V符号，明确普通/强响应两分支，独立留出至少128个触发家族；② 对广泛候选比较共同残差输入、词身份/位置、范数与条件非线性基线，再在真正触发的位置做写入和下游读取检验；③ 先预言在哪个新语境失效，再与原模型对照；④ 第二物理节点同快照独立复算。未达到分层功效或未胜过基线时缩小/替换命题，不能不断给同一高相关换名称继续升级。当前没有新增已破解语义单元的计数。

测试结束清理仅涉及本轮可重建网页打包目录，释放{cleanup['released_bytes']:,}字节；保留正在展示的本机客户端、正式包、关键原场、日志与复算输入。核验记录位于 `tests/glm5/result/mlp_condition_v2/`。
'''
with files[0].open('ab') as f:f.write(body.encode('utf-8'))
assert files[0].read_bytes().startswith(first)
old=files[1].read_bytes();nextid=max(map(int,re.findall(r'^## Phase (\d+):',old.decode('utf-8-sig'),re.M)))+1
short=f'''\n\n## Phase {nextid}: 条件响应长文本验证与正式中心归档 [{now}]

本地Qwen3-4B BF16完成128发现、512留出家族×16条件，冻结后执行三剂量、严格等范数、恢复和2200题能力保持。广泛候选MSE降低8.89%但未过门；稀有候选Pearson .9976但排序-.128、强响应仅3家族，不能视为稳定语义。3072组源对照全部合格，512家族取消编辑逐位恢复，实际首词零翻转。预注册文字/代码的句式变量差异、目标恢复对照不足、既有能力题库和单物理节点限制均保留，无正式机制晋级。

正式中心1.0.29通过客户端协议接收{reg['objects']}对象，公开回读哈希核验；新条件3D界面在本机客户端读取正式服务器数据验收。八层77824条件观察与4个新受控单元分开。网页客户端更新包待用户手动部署。完整算法、公式、反例、能力结果、上传/去重、边界与下一大阶段见 GLM5 Phase {phase} 和 `tests/glm5/result/mlp_condition_v2/`。本轮不覆盖他人并行的客户端改动，未改共享协议或中心程序。
'''
with files[1].open('ab') as f:f.write(short.encode('utf-8'))
assert files[1].read_bytes().startswith(old)
D.SAVE(D.OUT/'closeout.json',{'phase':phase,'claude_phase':nextid,'time':now,
    'prefix_sha256':{str(p.relative_to(D.ROOT)):hashlib.sha256(b).hexdigest() for p,b in zip(files,[first,old])},
    'provenance':D.checked()['provenance']})
print({'phase':phase,'claude_phase':nextid,'appended':True})
