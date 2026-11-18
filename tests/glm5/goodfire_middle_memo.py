"""Append the completed, independently checkable campaign to the research memo."""
from datetime import datetime
import hashlib
import re
import goodfire_middle as M
from deploy.storage_policy import check_write
C=M.C


def main():
    delivery=C.read(M.BASE/'delivery.json');cleanup=C.read(M.BASE/'cleanup.json')
    reports={k:C.read(M.BASE/k/'report.json') for k in M.SPECS}
    digest={k:C.read(M.BASE/k/'descriptive-digest.json') for k in M.SPECS}
    for k in M.SPECS:
        assert reports[k]['semantic_confirmed_units']==0
        assert C.read(M.BASE/k/'audit.json')['passed']
    path=M.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';before=path.read_bytes()
    marker='Qwen3-4B与14B的B18–B28全量原生单元研究和客户端交付'
    assert marker.encode('utf-8') not in before,'Campaign already appended'
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',before,re.M)]
    phase=max(phases)+1;stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    lines=[f'\n\n## Phase {phase}: {marker} [{stamp}]\n']
    def add(s):lines.append(s.strip()+'\n')
    add('''**范围与完成状态。** 按用户要求，研究零基B18–B28（含两端），逐模型串行使用本地CUDA。Qwen3-4B每层9,728个单元，共107,008个；Qwen3-14B每层17,408个，共191,488个；合计298,496个原生MLP单元。每个都有G/V/A统计、两极长语境、完整读写参数、符号分区、近似输出读出及冻结预测分数；相邻层完整关联可查。另对每模型6个预选单元做有限写入干预。完成的是全量观察、预测筛查与局部作用测试，未逐个完成语言功能解释或全电路因果闭合。正式语义确认数仍为0，不能把档案数当成破解比例。

方法入口是Goodfire的[原生神经元研究工作假说](https://already-clean.goodfire.com/)：联合看原生单元、正负端、长上下文和上下游线索。本轮没有检验其整体“可解释比例”，也没有把线性预测失败当作对该假说的否证。[Interpreting Language Model Parameters](https://www.goodfire.com/research/interpreting-lm-parameters)是另一个参数分解/VPD研究方向；本轮未训练VPD，不能称为其复现。''')
    add('**冻结身份。** '+ '；'.join(f"{M.SPECS[k]['name']}：RUN `{reports[k]['run_id']}`，BF16快照 `{reports[k]['model_snapshot']}`" for k in M.SPECS)+'。模型、tokenizer、代码、材料、阈值和运行库版本在v2采集/拟合前绑定，随机种子20261113及各派生式见源码。具体阶段有各自的干预计划与执行绑定，不跨版本合并。')
    add('''**材料和可见性。** 使用同一批1,024篇既有合法来源中英文文档，每篇1,024 token，继承家族划分512发现、256确认分片、256留出分片。材料过去用于其他研究，不是项目级全新盲箱。每篇保存位置533、598、687、787、813、840、945、1006的全部目标MLP单元、归一化输入与输出，以及embedding和每块残差状态、最终归一化状态；第二半段512–1023全部位置扫描G/V/A两端极值。8,192个位置不是8,192个独立样本。预测特征只用当前位置及其前缀；下一词1007只作干预结果评分。完整语境用于解释线索，不能把后文当作该状态的因果输入。

模型以CPU读原生safetensors、逐块送CUDA、256篇状态分块、前向batch=2运行，无量化，无同时加载两个模型。基础档案保留完整原生BF16位模式，热图保持原编号；概览对log(1+std(A))或双负份额按相邻16单元平均后自适应定色标，另有每列一个单元的原分辨率图和完整NPZ。原分辨率图按各自全值范围定色标，两模型未共享色标，不能直接用相同颜色比较幅值；精确范围见delivery.json的plot_audit。未通过PCA或Top-K定义主干，低幅值保留。''')
    add(f"v2正式采集耗时：4B {reports['qwen4']['collection_seconds']:.2f}秒，14B {reports['qwen14']['collection_seconds']:.2f}秒；不含后续分析和干预。")
    add('''**数值门及失败留档。** 4B v1首次完整采集后，与原始HF前向对照的A相对L2最大0.02048479，超过预注册0.02，因此未进入响应拟合。诊断发现RoPE逆频率在CUDA初始化与HF默认CPU float32初始化再迁移并不逐位一致，最大频率差约1.86×10⁻⁹，少数三角值经BF16和多层计算放大。仅把HF参考也改成CUDA初始化即可逐位重现v1，权重未变。修正执行口径、升为v2、重新冻结并重采，没有放宽阈值。两模型各用首4篇发现文档验证原始HF完整前向，68项G/V/A和最终状态检查均逐位一致。这个参考检查只覆盖4篇，不冒充全部1,024篇的独立重跑。

旧4B B28原场已不在现有路径，使用上述原始HF参考，保留缺失事实。14B旧版B28与新规范执行的G/V/A全场相对L2分别0.010256/0.015911/0.020231，最大32篇批次A误差0.037424；旧结果保留为另一执行版本，不悄悄拼成一条新旧混合轨迹。v1首4篇精确失败证据、原始输入、诊断、源码和日志保存；重复大场在最终核对后清理。''')
    add(r'''**数学对象与可执行算法。** 令z为实际归一化MLP输入，g_j、u_j为gate/up第j行，d_j为down第j列，s=SiLU：

$$G_j=g_j^Tz,\quad V_j=u_j^Tz,\quad A_j=s(G_j)V_j,\quad \Delta h_j=A_jd_j.$$

每单元双负绝对激活份额为

$$m_j^{--}=\frac{\sum_i|A_{ij}|\mathbf1[G_{ij}<0,V_{ij}<0]}{\sum_i|A_{ij}|}.$$

表中先按单元计算再层内平均；它不是整体MLP能量或语义贡献。四分区覆盖全部状态，零属于非负侧，另存能量口径。连续实数下

$$s'(G)=\sigma(G)[1+G(1-\sigma(G))],\qquad J_j(z)=d_j[V_js'(G_j)g_j+s(G_j)u_j]^T.$$

这是已知SwiGLU算式及局部导数恒等式，说明固定写方向、条件化读方向；不是新语言定律，不能直接代替BF16有限干预。G<0不能一概等同于“关门”。

文本预测采用发现侧标准化、固定岭回归正则0.01×训练行数，262维特征。当前token哈希、同桶数当前token、前128 token频率加背景、完整token身份的收缩均值加背景（收缩强度10）及文档移位负对照分别保留。G/V/A直接预测候选必须在两留出分片中均比词哈希MSE至少低10%，且胜过均值、匹配当前token、完整token加背景和移位对照。该筛查不是经过全体多重检验校正的语义确认。

另外冻结组合预测 $\widehat A=s(\widehat G)\widehat V$：两分片均比直接A预测MSE低至少10%且胜过均值才记组合候选。它没有直接A筛查的全部强对照，计数不能混加。一般有

$$E[s(G)V\mid X]=E[s(G)\mid X]E[V\mid X]+\operatorname{Cov}(s(G),V\mid X),$$

且 $s(E[G\mid X])\ne E[s(G)\mid X]$。因此分别预测G/V后相乘不是条件期望恒等式；联合分布、非线性和拟合偏差都可能造成残差。

每个相邻目标单元在全部源单元中按发现侧最大绝对相关选择单父，冻结线性系数，与均值、同编号和固定随机父比较。自由递推只从实测B18 A出发，之后用自身预测继续；不能把每层都重新读真实状态当成自由预测。静态参数余弦、响应相关与干预效应分别记账。直接unembedding投影只作读出近似，未包含后续归一化、attention和其他单元作用。''')
    add('**逐层结果。** 下表DN为留出双负绝对激活份额均值；负导数为留出SiLU负导数区间比例；G/V/A为通过直接预测筛查的单元数。')
    add('| 模型/块 | DN | 负导数比例 | G候选 | V候选 | A候选 | 组合候选 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for k in M.SPECS:
        for r in digest[k]['layers']:
            add(f"| {M.SPECS[k]['name']}/B{r['block']} | {r['DN_mass']:.4f} | {r['negative_derivative']:.4f} | {r['G']} | {r['V']} | {r['A']} | {r['composed']} |")
    for k in M.SPECS:
        d=digest[k];p=d['pooled_A'];s=d['counts']
        add(f"{M.SPECS[k]['name']}合计G/V/A候选{s['G']}/{s['V']}/{s['A']}；组合候选{d['composed_count']}，与直接A筛查交集{d['composed_and_direct_screen']}。按原幅值汇总的留出A总MSE/均值基线：直接{p['direct_over_mean']:.6f}、组合{p['composed_over_mean']:.6f}、完整token加背景{p['exact_token_over_mean']:.6f}。B18自由递推至B28的MSE比：逐单元均值{d['rollout_final']['mean_mse_ratio']:.6f}，总MSE比{d['rollout_final']['total_MSE_ratio']:.6f}。")
    add('直接和组合预测都有局部候选，但总体A拟合未胜过均值；自由递推基本退回均值，未形成可推广的简单单父链。完整token基线提示词汇身份不可忽略。不能由一个弱预测器的失败推出单元无语言功能，也不能由少量候选推出已解释语义。不同模型同编号层处于不同相对深度，材料也相同，不作规模因果结论或独立样本合并。')
    add('''**有限原模型作用检验。** 在B18/B23/B28各选发现侧std(A)×||d||最高者，及不同的双负|A|均值×||d||最高者，每模型6单元，写入倍数0、0.5、2。这个选择对正的up/down互补缩放更稳健，但仍不是因果重要性。实际BF16扰动范数≤2%误差的随机方向为负对照；不匹配行保留，对照比较使用相同有效样本。操作只改变最后token的列写入，不是重算修改权重后的精确权重消融。

每模型256篇自然文本、512道历史有答案选择题，分别建立自己的基线。保留原查询形状的前缀KV回放经4篇校准后运行，每块独立恢复。自然检验按单模型6单元校正，预设配对不一致概率0.1，n=256对8个百分点效应的规划功效0.921，所需n=190；不是对所有小效应、全部剂量和跨模型比较都有同等功效。能力规划n=512针对4个百分点变化；保持门另要求每项原对改错占全部题目的比例单侧95%上界≤2%，没有给出全部剂量联合95%的保证。通过仅覆盖这些题目和操作，不能推出通用能力或自然生成保持。''')
    add('| 模型/单元（0倍写入） | argmax改变/256 | 全样本平均ΔNLL | 全样本平均KL | 范数匹配n | 匹配ΔNLL：原生 / 随机 |\n| --- | ---: | ---: | ---: | ---: | ---: |')
    for k in M.SPECS:
        for r in digest[k]['natural_zero']:
            rows=reports[k]['intervention']['natural']['rows']
            pair={x['condition']:x for x in rows if x['block']==r['block'] and x['unit']==r['unit'] and x['dose']==0}
            matched=' / '.join(f"{pair[s]['mean_nll_change_matched']:.8f}" for s in ['native','norm_random'])
            add(f"| {M.SPECS[k]['name']}/B{r['block']}/U{r['unit']} | {r['argmax_changes_all']}/256 | {r['mean_nll_change_all']:.8f} | {r['mean_kl_all']:.8f} | {r['n_matched']} | {matched} |")
    for k in M.SPECS:
        d=digest[k];r=d['replay'];n=r['natural'];c=r['capability']
        add(f"{M.SPECS[k]['name']}：自然回放{n['replay_passed']}/256，能力回放{c['replay_passed']}/512，最大KL分别{n['max_replay_kl']}/{c['max_replay_kl']}，独立恢复逐位一致分别{n['restore_all']}/{c['restore_all']}；有效范数对照自然{n['norm_controls_valid']}/{n['norm_controls_total']}、能力{c['norm_controls_valid']}/{c['norm_controls_total']}。能力基线{d['capability_baseline_correct'][0]}/512，原生剂量保持门通过{d['capability_native_passed']}/{d['capability_native_total']}。")
    add('匹配列只在相同有效样本上描述原生与随机方向的损失变化，没有由这些小效应建立语义选择性，也没有把随机对照差异追认为新的确认检验。逐样本真实下一词、完整结果、反例、无变化及匹配失败均可回查。极小负KL是浮点求和误差，保留原值，不解释成负散度。未针对每单元最强触发位置做语义干预；预选覆盖8个位置而干预仅1006，不能据此否定其他位置的作用。高极值语境或输出词列表只作检索线索，尚无新语义最小对、自由生成或参数组干预。')
    checks=delivery['checks'];b=checks['browser'];e=checks['desktop']
    add(f"**客户端与交付验收。** 客户端{delivery['desktop_version']}，中心{delivery['center_version']}。两模型的11层全阵列、每个单元的G/V/A、参数末坐标、完整相邻关系、直接/组合预测和逐样本干预已接入成果库、完整报告、主空间对象详情。同一对象携带run与snapshot；新运行范围外不混用旧档案。API核对22层、44个端点单元、20个相邻边界、12个被测单元及错误关卡；浏览器{len(b['checks'])}项检查通过，含刷新、手机、旧版隔离，页面异常和GPU请求均为0。真实打包Electron通过{len(e['checks'])}项检查，确认isPackaged、D盘隔离配置及两模型最后单元末参数，{e['payloadFiles']}个内置文件与源文件一致，额外节点资源也一致。未运行用户系统安装向导。")
    add(f"原生服务9项、科研算法7项、前端63项通过。完整协议315项中314通过，唯一失败为既有`SOURCE-BINDING-20261110.source`语言项映射缺失，未修改无关冻结文法来掩盖失败。专用组件lint通过，共享组件既有2错误3警告签名与基线一致；最初版本隔离条件引入的3个编译规则错误已修复并复验。中心包{delivery['center_members']}个源文件逐一核对，两个研究登记包对象哈希、两个原生传输包与安装包一致性核验通过，完整六包路径和SHA256见`delivery.json`。本轮只交付本机和手动发布包，未更新公网。")
    add('桌面首轮验收的Playwright HTTP辅助请求继承了含中文产品名的User-Agent，Node请求头校验拒绝发送；应用页面本身已正常读取档案。将测试辅助请求显式设为ASCII User-Agent后，重新完成两模型实际打包程序验收。产品文件未因此修改，初次失败JSON和日志保留。14B概览另检查了原图左侧裁剪，层标签18–28与轴标题齐全；缩小预览不能代替原图核对。')
    add('''实际执行入口：`goodfire_middle.py prepare/admit/pilot/collect`、`goodfire_middle_analysis.py profiles/regimes/predict/neighbors/readout`、`goodfire_middle_audit.py`、`goodfire_middle_intervene.py freeze/calibrate/natural/capability`，由`goodfire_middle_pipeline.py`串行调度；`goodfire_middle_report.py build/plots/manifest`、`goodfire_middle_digest.py`、`goodfire_middle_client.py`及delivery流水线构建档案；API、browser、desktop、closeout脚本核验；unittest/vitest/lint和中心、桌面构建日志保存在结果根目录。模型任务日志逐行显示阶段、完成/总数、耗时、ETA及独占/共享显存。所有测试脚本在tests/glm5，临时脚本在tests/glm5_temp，结果在tests/glm5/result/goodfire_middle_20261113_v2。原始测试源码和集成源码均有哈希及归档，发布操作详见deploy/README_NATIVE_MIDDLE.md。''')
    add('**资源和清理。** 初始空间门未过，用户选择先释放D盘使空余至少35GiB；外部释放后空间达到约149GiB，正式按34GiB新增峰值加20GiB保留过门。传输打包另按实际增量检查。所有新增项目数据在D盘，无默认C盘回落。')
    for k in M.SPECS:
        r=delivery['resources'][k]
        add(f"{M.SPECS[k]['name']}日志内进程CUDA峰值{r['cuda_peak']/2**30:.3f}GiB，DXGI共享显存峰值{r['shared_gpu_bytes']/2**20:.1f}MiB。系统可用RAM低点{r['minimum_machine_available_RAM']/2**30:.3f}GiB，属于全机计数，不能归因全部给本实验；运行速度与共享显存没有显示超过预设1GiB的GPU溢出。")
    add(f"最终清理仅本轮已无用途的v1大场、逐位相同参数副本、完整结果的重复进度副本、独占浏览器/构建/协议缓存与未发布中间中心包，共释放{cleanup['freed_bytes']:,}字节；保留失败证明、日志、两套v2全量原生数据、可用客户端和传输包。明细见cleanup.json。原生场、系数、完整参数与查询档案作为后续复用数据，复查期限2026-12-13；不清理其他任务文件。")
    add('''**严格审视。** 当前主要硬伤是：历史材料已经暴露；每篇只保留8个完整读出位置，极值扫描不能替代全位置原场；完整token基线仍弱于embedding、强n-gram和有语序的上下文模型；没有同预算SAE/transcoder/VPD对照；大量单元筛查未获得语义层面的多重检验控制；单父算法过于简单；输出投影不是下游完整计算；只有12个单元的有限列写入测试；没有全词表图谱预测保真、自由生成或独立节点复现。两种模型共用材料不算两套独立样本。数值执行口径的改变揭示：模型权重快照相同还不足以保障响应字段逐位一致，图谱必须绑定算子、初始化、批形状与校准。这些限制都在报告和客户端可查，不因工程审计通过而升级结论。

**三图谱、理论与下一阶段大任务。** 内部图谱增加22个模型层的完整原生对象、双负/非单调状态和失败预测；外部图谱仍只关联自然文档、语言来源与历史能力题，没有新增已确认知识、语法、推理族。跨图谱增加冻结响应筛查候选和有限操作效应，相关边不升级为已证实的计算链，RDC主体理论不变。

第一性原理的研究对象应是“固定参数读入的两个条件量如何联合形成写出，并被后续过程使用”。同一个A可能来自不同G/V组合，局部导数与下游条件可以不同；单个名词标签、正极关键词或一个线性父单元都不足以描述这类条件化计算。以上是从已知算式和本次负结果得到的研究方向，不是人脑机制或AGI结论。

下一阶段应作为一个共同问题的大Phase推进：(1) 按预定分层规则从正负端、双负/双正、高低幅值选单元，建立知识、语法、推理的条件说明书，冻结新词、新句式、新组合家族，先过行为与因果可见性门；保留不激活和近邻反例。(2) 在同一材料、预算和数值执行口径上比较完整词身份/embedding、强上下文基线、原生单元组与适用字典/参数方法，预测G、V的联合条件、A和完整词表输出，同时保留残差和不确定性；不能仅扩大普查数量。(3) 对冻结候选做坐标、单元组及真实参数干预，区分QK选取、V/O写入、归一化和MLP作用，加入匹配负控制、恢复、自由生成与副作用，检验未见家族和独立节点；再串行安排其他本地模型复现。统一数值口径后才考虑与历史尾层拼接。

通俗说明：这次为两个模型的第18至28号块建立了完整“神经元档案”，能看到每个单元读了什么数、怎样组合、往哪里写，也做了少量改变写入的实验。两个负输入产生正输出的情况很常见，但把这些单元串成可以自己推算答案的规则还没有成功。客户端解决了“全部可查”，接下来要验证“为什么响应、什么时候失效、是否真参与某种语言能力”。''')
    # Table rows must remain contiguous in Markdown; prose blocks have blank lines.
    body='\n'.join(lines)
    body=re.sub(r'(?m)(\|[^\n]*\|)\n\n(?=\|)',r'\1\n',body)
    raw=body.encode('utf-8');check_write(path,len(raw))
    with path.open('ab') as stream:stream.write(raw)
    after=path.read_bytes();assert after[:len(before)]==before and after[len(before):]==raw
    M.configure('qwen4');M.OUT=M.BASE;M.RUN='NATIVE-MIDDLE-DELIVERY-20261113-v2'
    M.write(M.BASE/'memo-append.json',dict(phase=phase,timestamp=stamp,path=str(path),before_bytes=len(before),
        before_sha256=hashlib.sha256(before).hexdigest(),appended_bytes=len(raw),appended_sha256=hashlib.sha256(raw).hexdigest(),
        preserved_prefix=True,memo_after_sha256=hashlib.sha256(after).hexdigest()))
    print(f'Appended Phase {phase}, preserved {len(before)} prior bytes',flush=True)


if __name__=='__main__':main()
