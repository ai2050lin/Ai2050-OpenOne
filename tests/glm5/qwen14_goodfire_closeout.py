"""Append-only research memo, cross-index and reproducible delivery record."""
from datetime import datetime
import hashlib
import json
import re
import shutil
import qwen14_goodfire_extension as E
C=E.C


def append_checked(path,text):
    before=path.read_bytes()
    with path.open('ab') as stream:stream.write(text.encode('utf-8'))
    after=path.read_bytes();assert after[:len(before)]==before
    sha=lambda b:hashlib.sha256(b).hexdigest()
    return {'path':str(path),'before_bytes':len(before),'after_bytes':len(after),'prefix_sha256':sha(before),'new_sha256':sha(after)}


def main():
    E.configure();r=C.read(E.OUT/'report.json');audit=C.read(E.OUT/'audit.json');query=C.read(E.OUT/'query-validation.json')
    assert audit['passed'] and query['passed']
    natural=r['new_interventions']['natural'];cap=r['new_interventions']['capability']
    original={f:sum(x['prediction_candidates'][f] for x in r['layers']) for f in ['G','V','A']}
    scaled=r['secondary_standardized_prediction']['layers']
    counts={f:sum(x['candidates'][f] for x in scaled) for f in ['G','V','A']}
    reg={x['block']:x for x in r['regimes']};pred={x['block']:x for x in scaled}
    table='\n'.join(f"| B{b} | {reg[b]['holdout_DN_mass_mean']:.4f} | {reg[b]['holdout_negative_silu_derivative_mean']:.4f} | {pred[b]['candidates']['G']} / {pred[b]['candidates']['V']} / {pred[b]['candidates']['A']} |" for b in E.ALL)
    primary=[x for x in natural['rows'] if x['condition']=='native' and x['dose']==0]
    effect_table='\n'.join(f"| B{x['block']}/U{x['unit']} | {x['argmax_changes_all']}/{x['n_total']} | {x['mean_nll_change_all']:.6g} | {x['mean_kl_all']:.6g} | {x['n_matched']} |" for x in primary)
    cap_native=[x for x in cap['rows'] if x['condition']=='native']
    telemetry=[]
    for p in E.OUT.glob('*.log'):
        for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
            if line.startswith('{'):
                try:telemetry.append(json.loads(line))
                except ValueError:pass
    resources={'cuda_peak_bytes':max((x.get('cuda_peak',0) for x in telemetry),default=0),
        'max_measured_shared_gpu_bytes':max((x.get('shared_gpu_bytes',0) or 0 for x in telemetry),default=0),
        'min_logged_available_RAM_bytes':min(x['ram_available_bytes'] for x in telemetry if 'ram_available_bytes' in x),
        'min_logged_available_commit_bytes':min(x['commit_available_bytes'] for x in telemetry if 'commit_available_bytes' in x),
        'new_directory_bytes':sum(p.stat().st_size for p in E.OUT.iterdir() if p.is_file())}
    E.save(E.OUT/'resource-summary.json',resources)
    memo=E.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';old=memo.read_text(encoding='utf-8')
    title='Qwen3-14B B28–B39全量Goodfire式原生分析与双负分支复核'
    assert title not in old,'Do not append this completed phase twice'
    phase=max(map(int,re.findall(r'^## Phase (\d+)',old,re.M)))+1
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=fr'''

## Phase {phase}: {title} [{stamp}]

**目标与完成范围。** 用户要求按Goodfire思路完成Qwen3-14B中B28之后所有层的神经元分析，并确认“islu”是“思路”的拼写。按已说明的零基B28–B39（包含B28）执行：12×17,408={r['native_units']:,}个原生MLP单元；旧B28–B35只读复用，新增B36–B39的69,632个。完成全量观测档案、读写参数、双极语境、符号区间、读出近似、预测筛查及8个新单元的有限干预。未逐个完成208,896个单元的语义解释或完整因果论证。RUN {E.RUN}，BF16快照{E.SNAP}。方法来源：[Goodfire原文](https://already-clean.goodfire.com/)。没有发布或改动正式服务；旧客户端范围未自动更新，新全量索引与本地查询入口单独交付。

**材料与采集。** 1,024篇既有合法来源的中英文长文本，每篇1,024 token，512发现、256确认分片、256留出分片，继承文档家族划分。语料以前用于4B与14B研究，不是全项目新盲箱。每篇8位置{C.POSITIONS}保存新增四层原生BF16的G/V/A、归一化MLP输入和输出；另在512–1023全部位置扫描每个G/V/A单元的正负极值。各层完整gate/up/down参数保留。模型、tokenizer及旧依赖哈希采集前复核。CPU读原safetensors、原生HF块逐层送CUDA、256篇状态驻留、前向batch=2、SDPA与显式因果mask、无量化。4篇发现侧试跑通过后扩大，采集用时{r['collection_seconds']:.2f}秒。B35全部文档×8位置的G/V/A与旧原场逐位一致；40块后已保存的最终归一化状态也逐位一致。新增16组原生MLP算式回放通过。

**数学对象。** z为实际MLP归一化输入，g_j、u_j为gate/up第j行，d_j为down第j列：

$$G_j=g_j^Tz,\quad V_j=u_j^Tz,\quad A_j=\operatorname{{SiLU}}(G_j)V_j,\quad \Delta h_j=A_jd_j.$$

对各单元和划分，记录四个G/V符号区间。双负绝对激活份额定义为

$$m_j^{{--}}=\frac{{\sum_i |A_{{ij}}|\mathbf{{1}}[G_{{ij}}<0,V_{{ij}}<0]}}{{\sum_i|A_{{ij}}|}}.$$

下表先求每个单元份额，再在层内取均值；不是MLP输出能量或语义贡献比例。能量份额与零值约定另存。SiLU导数为s'(G)=sigmoid(G)[1+G(1−sigmoid(G))]，存在负导数区间。连续实数SwiGLU的局部导数恒等式是

$$J_j(z)=d_j\left[V_js'(G_j)g_j+s(G_j)u_j\right]^T.$$

它表达固定写方向与条件化读方向；是已知计算式，不是新语言定律，也不等于BF16有限扰动后的下游因果效应。本轮没有用它完成全模型输出预测。

**预测方案。** 原口径使用262维文本特征及固定ridge，比较当前token哈希、前128 token频率、背景变量与文档移位负对照。发现侧拟合；两留出分片均比词身份MSE降低至少10%，并胜过均值和移位对照才记候选。历史失败已知后，另行冻结一个次级分析：仅用发现侧均值/标准差标准化特征，正则仍0.01×训练行数，增加相同128桶当前token的匹配对照。没有覆盖或追认原结果。哈希基线仍弱于完整词身份、embedding或强n-gram，候选不是多重检验校正后的语义确认。

相邻层为每个目标单元搜索全部17,408个发现侧源单元，冻结最大绝对相关父单元及线性系数，与均值、同编号、固定随机父对照比较。新补四边界，合计11个；自由递推从实测B28 A开始。查询可返回全部相邻单元的参数余弦和各划分相关行，未用Top-K替代完整关系；静态对齐、响应相关和已检验因果边分别记账。

**全量结果。** 原未标准化方法G/V/A候选总数{original['G']}/{original['V']}/{original['A']}；次级标准化方法{counts['G']:,}/{counts['V']:,}/{counts['A']:,}。这些是三个响应字段的候选计数，不能相加称为独立神经元或已破解数量。

| 原生块 | 留出双负绝对激活份额均值 | 留出SiLU负导数区间比例 | 标准化G / V / A预测候选 |
| --- | --- | --- | --- |
{table}

双负份额约21.6%–37.2%，不能把负门值一律视为关闭。B39负导数区间约75.3%，仅限本语料8个读出位置。G/V的一部分可从粗文本上下文预测，乘积A仍难；这提示联合条件与非线性是后续问题，不支持直接给单元贴概念标签。

四个新增边界的逐单元MSE比均值为1.0684、1.0805、1.0765、1.3272，中位数也均大于1，未建立普遍有效的单父接力。另保留事后描述性的原幅值汇总：总MSE比1.0242、1.0828、0.8749、0.8399，后两边界的部分大幅值响应具有预测信息。不能混淆两种加权口径。B39/U9191约占该层均值基线总误差23.5%，冻结父为B38/U10628，留出相关约0.244、MSE比约0.944，说明少数大幅值单元可明显影响总体指标。B28起始的自由递推到B39误差比约1，基本退回均值；未取得文本到全场或完整下一token分布的图谱保真闭合。

**有限原模型作用检验。** 发现侧预定每个新块选A标准差最大、绝对A均值最大的两个不同单元：B36/U88、U11576；B37/U17020、U5800；B38/U9395、U3139；B39/U9191、U13992。最后token的原生down列写入倍数设0、0.5、2，设实际BF16扰动范数≤2%误差的随机方向对照与重复恢复。保持完整query形状的前缀KV回放，4篇校准KL为0后绑定代码，再跑256篇自然文本、512道历史有答案选择题。它是有限列写入操作，不是重算权重矩阵的精确权重消融或语义steering。

自然回放通过{natural['replay_passed']}/{natural['samples']}，最大KL={natural['max_replay_kl']:.8g}；能力回放通过{cap['replay_passed']}/{cap['samples']}，最大KL={cap['max_replay_kl']:.8g}。恢复逐位一致分别{natural['restore_all']}、{cap['restore_all']}。有效范数对照自然{natural['norm_controls_valid']}/{natural['norm_controls_total']}，能力{cap['norm_controls_valid']}/{cap['norm_controls_total']}；失败行保留，匹配比较用同一有效样本。功效沿用更保守的16重比较规划：自然n=256针对≥8个百分点配对变化，能力n=512针对4个百分点变化，均通过tools/power_analysis；小效应不具有同等功效。能力保持另以原对改错比例单侧95%上界≤2%判断，新24项原生剂量通过{sum(x['capability_gate'] for x in cap_native)}/{len(cap_native)}。旧16个加新8个共24个被测单元，各运行保留自己的数值基线与校准。

| 新单元，0倍写入 | 下一token argmax改变 | 下一词损失平均变化 | 平均KL（编辑对基线） | 有效匹配n |
| --- | --- | --- | --- | --- |
{effect_table}

逐条分数、真实下一词、恢复、无变化与受损反例均保存。这些数据刻画操作的影响，未回答单元的语言功能，也未证明独有性、必要性或自然生成语义保持。

预选使用8位置的A变异，干预只测位置1006，不能据此否定其他触发位置的作用。原始A幅值也不是参数化不变的功能强度：连续实数下把up行乘c、down列除c可保留写出而改变A。额外描述性诊断保留std(A)×||d||，它仍不包含其他单元抵消、归一化或下游读取，不能当作因果重要性。

**审计与硬伤。** 12层全数组维度、有限性、符号分区及质量守恒检查通过；10项查询验收覆盖越界拒绝、两端层真实字段、完整相邻行与冻结父的一致性，以及原checkpoint的gate/up/down向量逐位核对。热图保持原生编号，概览显示相邻16单元均值；原分辨率图每列一个单元，全坐标NPZ保留。执行源码与模型在采集前冻结，第三方包版本在采集后分析期间补录，不追认更早冻结。实际随机流沿用源码的20261109派生式，20261113是本轮provenance种子标签；映射见random-seed-audit.json。

旧B28–B35初始provenance缺陷保留，不因补齐四层而晋级。已暴露语料、有限位置采样、关键词弱解释、哈希碰撞、单父模型、没有字典法同预算比较、没有新的语义最小对与自由生成评估、没有独立节点，是当前硬伤。正式语义确认数仍为0；它描述本套流程的确认状态，不表示全部单元无功能。

**三图谱与理论。** 内部图谱增量是完整B28–B39对象覆盖、双负/非单调状态说明、可复核的预测失败和局部成功。外部图谱仅关联文档、来源、检索语境和有限能力题，没有新增已确认知识/语法/推理族。跨图谱增加次级响应预测候选，没有升级语义或因果机制边。RDC名称和主体公式不变。

第一性原理上的工作对象是“固定读写参数＋随上下文改变的G/V组合”。相同A可来自不同G/V组合，局部敏感性不同；只看正端或给单元取一个名词都会丢失可检验条件。这是数学结构提示，不能推出人脑机制或AGI结论。

**下一阶段的大任务。** (1) 建立跨知识、语法、推理族的条件说明书：从全坐标档案按预定规则分层选正负端、双负/双正、高低幅值单元，冻结新词、新句式、新组合家族；先过行为和因果可见性门，在同批材料比较完整词身份/embedding、位置、n-gram及背景白化基线。(2) 建立可预测的原生读写链：同时预测G、V、A和完整词表输出，检验联合条件、归一化和attention路径；与单父、单元组及适用SAE/transcoder同预算比较，保留残差。(3) 对冻结候选做坐标/单元/参数的成组干预、匹配负对照和恢复，检验语义选择性、自由生成和副作用；在新家族及独立节点成立后，再按模型串行原则安排GLM等复现。以上均是计划，未冒充本轮已完成。

**资源与交付。** 新数据真实路径D盘，运行前登记12GiB峰值并保留20GiB；旧C盘联接只读。实测CUDA峰值{resources['cuda_peak_bytes']/2**30:.3f}GiB，共享显存峰值{resources['max_measured_shared_gpu_bytes']/2**20:.1f}MiB。逐批日志包含进度、耗时、ETA及内存，模型任务依次执行。脚本tests/glm5/qwen14_goodfire_*.py；报告、索引、图和清单tests/glm5/result/qwen14_mlp_b28_b39_v2/。原生场、参数、预测系数登记为后续查询与复算的复用数据，复查期限2026-12-13。独占缓存及重复进度副本按cleanup.json核对后清理；历史证据保留。

通俗地说：B28到末层的全部原生MLP单元已有可查档案，“两个负输入也能形成正写入”广泛存在。部分上下文信号可以预测，但多数单元的组合响应、跨层接续与语言含义仍需验证。档案数量不是破解比例。
'''
    m=append_checked(memo,text)
    claude=E.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md';ct=claude.read_text(encoding='utf-8')
    cp=max(map(int,re.findall(r'^## Phase (\d+)',ct,re.M)))+1
    cross=f'\n\n## Phase {cp}: Qwen3-14B B28–B39原生神经元全量分析索引 [{stamp}]\n\n详见GLM5 Phase {phase}，RUN {E.RUN}，结果tests/glm5/result/qwen14_mlp_b28_b39_v2/report.json。覆盖208,896个原生MLP单元，新增69,632个；旧B28–B35只读复用。双负绝对激活份额均值21.6%–37.2%；次级标准化G/V/A预测候选7,976/3,548/92，未作语义确认。新增8单元的有限写入和512题能力检查，反例与不合格范数对照保留。本地新索引和查询已交付，未部署正式服务；旧准入缺陷、语料暴露和无独立复算的边界不变。\n'
    cm=append_checked(claude,cross)
    snapshots=[]
    scripts=sorted((E.ROOT/'tests/glm5').glob('qwen14_goodfire_*.py'))
    scripts += [E.ROOT/'tests/glm5'/name for name in ['qwen14_mlp_atlas.py','qwen14_mlp_analysis.py','qwen14_mlp_intervene.py','qwen14_mlp_publish.py','qwen14_mlp_provenance.py','mlp_tail_finalize.py','mlp_tail_goodfire.py','source_binding_resources.py']]
    scripts += [E.ROOT/p for p in ['shared/dist_protocol/mechanism_trace.py','deploy/storage_policy.py','deploy/storage_policy.json','tools/provenance.py','tools/power_analysis.py']]
    for p in scripts:
        dest=E.OUT/('source-'+p.name);shutil.copyfile(p,dest)
        snapshots.append({'source':str(p),'snapshot':dest.name,'sha256':C.file_sha(dest)})
    E.save(E.OUT/'delivery.json',{'phase':phase,'claude_index_phase':cp,'timestamp':stamp,'memo':m,'claude_memo':cm,
        'source_snapshots':snapshots,'native_units':r['native_units'],'new_native_units':r['new_native_units'],
        'query_checks':len(query['checks']),'full_layer_checks':len(audit['checks']),'semantic_confirmed':0,
        'remote_deployed':False,'formal_evidence_promotion':False,'resources':resources})
    print('GLM5 Phase',phase,'Claude index',cp,flush=True)


if __name__=='__main__':main()
