"""Append-only closeout for the actual completed natural CLT experiment."""
import os,sys,json,re,hashlib,shutil
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
import source_binding_design as D
from deploy.storage_policy import check_write
OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
pre=D.read(OUT/'prereg.json');seal=D.read(OUT/'candidate.json');d=D.read(OUT/'discovery-analysis.json');c=D.read(OUT/'confirm-analysis.json');audit=D.read(OUT/'natural-audit.json')
assert c['candidate_sha']==D.sha(OUT/'candidate.json')
for n,h in seal['files'].items():assert D.sha(OUT/n)==h
assert not (OUT/'natural-closeout.json').exists()
now=datetime.now().strftime('%Y-%m-%d %H:%M')
memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';index=ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
old=memo.read_bytes();oldidx=index.read_bytes()
phase=max(map(int,re.findall(rb'^## Phase (\d+):',old,re.M)))+1
phaseidx=max(map(int,re.findall(rb'^## Phase (\d+):',oldidx,re.M)))+1
enc=c['encoding'];par=c['parameter'];head=c['routing']['chosen'];cache=c['cache']
strong=max(c['comparisons'],key=lambda k:c['comparisons'][k]['joint']);base=c['comparisons'][strong]
pct=lambda x:f'{100*x:.2f}%'
table='\n'.join(f"| {name} | {pct(v['bit0'])} | {pct(v['bit1'])} | {pct(v['joint'])} |" for name,v in [('筛选32坐标',enc)]+list(c['comparisons'].items()))
title='Qwen14B 独立行号的64家族发现与128家族冻结确认：广泛可读信息、正文路由与局部参数缺口'
text=f'''

## Phase {phase}: {title} [{now}]

**完成范围。** 用户授权继续 CLT-20261117-v1，自然采集64个发现家族1024输入，冻结后采集128个确认家族2048输入，合计3072输入。每家族4种独立行序×2个档案条件×前置/后置条件=16格。只用本地 Qwen3-14B 原始BF16，CPU读取权重、CUDA RTX5080逐层计算，固定真实chat template、关闭thinking、Answer换行；没有GLM复现、自由文本生成或干预。本轮“完成”指自然发现、冻结与确认，不表示选择机制验收完成。

**冻结、可见性和执行修订。** 继承 conditional_lookup_prereg.md；扩展说明 conditional_lookup_natural_design.md，实际执行代码绑定 natural-plan.json。最初启动在磁盘检查失败：0.5GiB表达式为float，接口要求int；尚未前向计算、未分析发现集。保留 natural-plan-v1-storage-error.json 与 calibrate-storage-error.log，改为整数2**29后重新冻结。试跑首家族16题先校准原生attention输出对账，最大误差0.0327091，冻结容差0.0654182。候选封条时间 {seal['utc']}，SHA256 {D.sha(OUT/'candidate.json')}；封条包含执行/分析代码、发现结果、读出模型和参数文件哈希。开封前未读确认材料；确认采集代码在读取材料前验证封条，审计核对其早于首个确认产物。没有依据确认改层、索引、alpha或阈值。初始工程自检曾给随机连续信号的线性分类器指定任意80%门槛而得79.6875%，没有进入科研指标；改用可精确线性解码的植入信号验证实现得到100%，未改分析代码或科研门槛。

**原理与对象。** 两档案中被查询名字的行号为 r0,r1∈{{0,1}}，独立变化；条件 b 决定外部任务目标槽位 2b+(1-b)r0+b r1。这是任务定义，不是已经找到的神经网络算法。旧材料 r1=1-r0 的固定一位+XOR无法统一覆盖本设计的00/11。后置条件时，名字末token不能知道未来b。行号读出只取每家族后置b=0的4个不同前缀，发现256、确认512个实体状态；统计单位仍是64/128个家族，不能把重复b或16格当成独立家族。

**自然采集。** 保存名字位置h00–h40全部5120原生坐标（h18指18个块后，即0编号B17输出），B20–22全部17408单元的G/U/A、名字和最终位置的MLP归一化输入z及A。保存B20–30各头的实际QK注意力来源与原参数OV局部写入，六个来源为四条记录、查询名字末token、其他位置。B20–22记录名字及最终位置，B23–30记录最终位置。不是全部token全场，也没有全部层的MLP采集，覆盖范围如上。每块注意力保留q/k额外RMSNorm、RoPE、GQA和实际因果mask；原生BF16以uint16位模式无损保存。

**计算方法。** 原生坐标或MLP单元组上拟合 r_hat=μy+((xK−μK)/sK)W，K为1或32个真实索引。按两个行号的平方相关和选索引，逐家族五折内部重新选择/标准化；alpha∈{{.01,.1,1,10,100}}，用发现OOF联合准确率选择池与alpha。h01–h40及A20–22共43池；所有排名记录，不以投影坐标替换原生索引。发现结果有选模型乐观偏差，正式数值来自冻结后的确认。注意力来源计算 alpha(t,s)=softmax_s(q_t·k_s/√d)，来源写入=Σ_s∈来源 alpha(t,s) W_O v_s。再以原始代码unembedding作局部对比，仅为未归一化局部代数贡献，不是最终logit或因果贡献。

原生参数局部近似在B{seal['parameter']['layer']}冻结32个MLP单元，保存原始gate/up的权重行及down权重列。单元A_j=SiLU(G_j)U_j，G_j=Σ_k Wg[j,k]z_k，U同理。冻结共同32输入坐标K，令 G_hat_j=Wg[j,:]μ+Σ_k∈K Wg[j,k](z_k−μ_k)，U同理，A_hat=SiLU(G_hat)U_hat；省略输入固定为发现均值，权重没有再训练。K按发现输入方差×gate/up权重平方和选择；另冻结随机32输入和均值A对照。这仍依赖该层真实z，只是局部近似，不是从文本出发的闭合机制。

**行为与技术结果。** 发现 {audit['splits']['discovery']['strict_correct']}/1024、确认 {audit['splits']['confirm']['strict_correct']}/2048 的四候选严格margin>0；完整词表首选分别 {audit['splits']['discovery']['top1_correct']}/1024 与 {audit['splits']['confirm']['top1_correct']}/2048。16格分别全通过95%门。发现256对、确认512对后置实体状态在全部41边界逐位相同。原生attention对账最大误差发现 {audit['splits']['discovery']['source_max_abs']:.8f}、确认 {audit['splits']['confirm']['source_max_abs']:.8f}，均小于预先容差。选定32单元的CPU BF16 SiLU(G)U与实际A逐位一致，最大差 {par['gua_max_abs']}。独立产物审计校验材料、原场和模型文件绑定、答案/间隔算术、名字隔离和确认读出指标；不是第二节点独立模型复跑。

**行号信息结果。** 发现选中h18的32坐标，alpha=10，OOF两位联合100%。同池随机32的最佳OOF为96.484375%，仅领先3.515625点，预注册10点优势门未过；按冻结扩展保留该候选作负结果复现。确认如下（同一512前缀，128家族）：

| 冻结读出 | r0准确率 | r1准确率 | 两位同时正确 |
|---|---:|---:|---:|
{table}

与最强基线{strong}的联合增益 {100*base['advantage']['mean']:.4f}个百分点，97.5%家族bootstrap区间 [{100*base['advantage']['ci975'][0]:.4f}, {100*base['advantage']['ci975'][1]:.4f}]点。编码验收={c['encoding_gate']}。准确率门与特别集中/超过随机的门分开记账。名字位置确有两个独立变化行号的可读信息，不能据此证明两套独立物理寄存器或专属32坐标。发现嵌入/长度位置代码基线联合25%；最佳单坐标线性读出联合50%，不排除单实数的非线性多值编码。发现h13开始有较强两位读出，h18–h25达到100% OOF；这条层曲线是探索性覆盖，确认只验证冻结h18，不逐层声称确认。

**答案缓存诊断。** 在同一冻结32坐标上固定alpha=1读出两档案代码词，确认代码0={pct(cache['code0'])}、代码1={pct(cache['code1'])}、同时正确={pct(cache['joint'])}（词表{cache['vocab']}项）。这里只检测为行号筛出的坐标上的线性代码读出，不是与最佳代码缓存模型的公平竞争；不能因低分排除别的坐标、非线性或多位置缓存。完整表型已用于pilot，结果只涉及新名字家族，不声称未见表型、中文、翻译或递归深度泛化。

**自然选择来源。** 发现冻结B{head['layer']}H{head['head']}（块/头均0编号），最终位置对四条正文记录的注意力质量argmax；确认后置命中目标记录 {pct(head['late_slot_acc'])}，前置 {pct(head['early_slot_acc'])}，同前缀两种b都命中 {pct(head['late_both_requests_correct'])}。全部确认输入平均正文质量 {pct(head['record_mass'])}、查询名字末token {pct(head['entity_mass'])}、金记录 {pct(head['gold_slot_mass'])}。表明存在准确访问正文的实际路由，不是只有名字处的可读信息。该指标使用操作后的真实注意力，不满足“仅操作前紧凑载体预测选择”的正式验收；一个头的来源质量不代表整个网络，也不证明必要性。先用行号地址去正文取值与收到条件重新匹配名字，都可产生相同注意力现象；目前不能区分。两代码缓存也可与正文通路并存。

自然路由反例已逐条列入 routing-counterexamples.json：确认2048输入有6次该头的记录质量argmax不等于金记录（5次前置、1次后置），原模型这6题仍全对。例如 confirm-023-11-0-0 的金答案为dog，该头对错误第一行的注意力质量0.2442、正确第二行0.1697，但原模型正确答案间隔14.875。说明“最大质量记录”只是局部诊断，不是整个模型的决策规则；不能直接据此证明补偿或该头没有作用。

**参数结果。** 32个输入坐标的原参数近似，相比均值A降低MSE {100*par['relative_mse_gain_vs_mean']:.2f}%，相比随机32输入降低 {100*par['relative_mse_gain_vs_random']:.2f}%，预注册双20%门通过={par['pass_20percent']}。原始G/U/A可精确对账，并不表示32输入可以解释这些神经元；尚缺的输入项/交互与门控范围保留为未解释残差。保存native-parameters.npz的真实参数行/列及索引，不能把这些切片统称为最小因果电路。

**科学判断与硬伤。** 得到可复现的信息候选和实际正文路由，未获得通过门槛的紧凑独立表编码/完整选择机制，未进行消融和恢复。两行可读≠原模型使用这份读出；随机坐标强≠模型必然冗余容错；路由正确≠必要；原参数对账≠机制提取。本轮未冻结从上游紧凑状态预测选择、写入并通过匹配干预的可执行小机制，因此原预注册的选择共同终点和因果验收均未满足，不能用观察指标冒充。词汇/模板范围窄，读出只在线性1/32坐标预算内，MLP只覆盖3块，attention只覆盖11块；后续扩展需新冻结和新确认材料，本批确认已开封，不能用其调参后再当盲测。

**理论增量与下一大阶段。** 外部图谱明确两行独立变化和晚到条件；内部图谱新增h18原生信息候选卡、B20局部参数切片与B26H36实际来源记录；关联边为新家族信息复现及原生计算来源，未升级因果边。重要线索是“地址信息可在名字处读到，同时实际读取仍落在正文”，适合检验混合地址—取值算法，而不是预设单向量闭合。下一阶段围绕一个完整问题推进：B26H36的Q/K究竟从哪些上游原生坐标取得档案条件、行号或名字身份？冻结Q/K分量与来源位置的小模型，预测自然选择及可区分的冲突题；用同输入原生贡献阻断、等强度随机/sham及自身恢复检验即时QK、OV、答案和补偿路径，避免供体差分搬运。并把32输入参数残差作条件化分解，比较更大预算或非线性交互的预测收益，不能在旧确认中筛阈值。只有改变行号地址与保留名字匹配产生不同预测的对照，才有机会区分查表取值与正文重匹配。无新数学定理、递归泛化或AGI结论。

**资源、产物与复核。** scripts: conditional_lookup_collect.py、conditional_lookup_analyze.py、conditional_lookup_natural_audit.py、conditional_lookup_plot.py、conditional_lookup_natural_closeout.py，均在tests/glm5；材料和结果在tests/glm5/result/conditional_lookup_native_v1。运行命令为 .venv/Scripts/python.exe -u -B tests/glm5/conditional_lookup_collect.py calibrate/discovery/confirm（逐次执行）；分析器 discovery 在confirm采集之前，confirm 在之后；最后audit/plot/closeout。逐层换行日志包含完成层、分片进度、耗时、ETA及DXGI共享显存；CPU/GPU串行分层，统一磁盘检查，每片0.5GiB峰值预算，额外总预算3GiB并保留D盘20GiB。原场按冻结例外保留用于复核，不清理别的研究数据。未上传、未正式服务器登记、未客户端读回；当前是本地已审计的自然观察包，非正式因果机制发布。标准provenance回执与natural-manifest记录实际哈希，客户端/服务器交付状态单列，不声称已发布。
'''
for p in (memo,index):check_write(p,100000,operation='CLT natural append')
with memo.open('ab') as f:f.write(text.encode('utf-8'))
assert memo.read_bytes().startswith(old)
indextext=f'\n\n## Phase {phaseidx}: 独立行号自然发现与冻结确认 [{now}]\n\n见GLM5 Phase {phase}。Qwen14B完成64发现家族1024题和128确认家族2048题；h18冻结32坐标联合读出{pct(enc["joint"])}，最强随机基线{pct(base["joint"])}，编码优势门={c["encoding_gate"]}。B26H36确认后置正文路由{pct(head["late_slot_acc"])}，不证明查表或重查的独占因果机制；原参数32输入近似20%门={par["pass_20percent"]}。确认已开封，不得调参后复用为盲测。\n'
with index.open('ab') as f:f.write(indextext.encode('utf-8'))
assert index.read_bytes().startswith(oldidx)
(OUT/'REPORT.zh-CN.md').write_text(text.lstrip(),encoding='utf-8')
from tools.provenance import write_result
os.environ['GIT_CONFIG_COUNT']='1';os.environ['GIT_CONFIG_KEY_0']='safe.directory';os.environ['GIT_CONFIG_VALUE_0']=str(ROOT)
receipt=write_result('conditional_lookup_natural',dict(provenance=pre['provenance'],phase=phase,confirmation=c,audit=audit,candidate_sha=D.sha(OUT/'candidate.json')),model_path=D.MODELS['qwen14'],seed=pre['seed'],tool='conditional_lookup_natural_closeout.py')
metadata=dict(provenance=pre['provenance'],phase=phase,index_phase=phaseidx,receipt=str(receipt.relative_to(ROOT)),created=now,mechanism_accepted=False,publication={'uploaded':False,'registered':False,'client_readback':False},retention='retain original materials, BF16 observations, frozen models, native parameter slices and necessary audit logs',cleanup='only this-run matplotlib/temp cache; no original evidence removal')
(OUT/'natural-closeout.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
files={str(p.relative_to(ROOT)):D.sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='natural-manifest.json'}
files.update({str(p.relative_to(ROOT)):D.sha(p) for p in (ROOT/'tests/glm5').glob('conditional_lookup*') if p.is_file()})
files[str(receipt.relative_to(ROOT))]=D.sha(receipt)
(OUT/'natural-manifest.json').write_text(json.dumps(dict(provenance=pre['provenance'],files=files,status='natural discovery and confirmation complete; mechanism unaccepted'),indent=2),encoding='utf-8')
print(json.dumps(metadata,ensure_ascii=False),flush=True)
