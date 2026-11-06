"""Append scientific closeout and client acceptance; never rewrite prior memo."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import collections
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_confirm as C
import phase2773_w6_loop as W


def main():
    root=W.ROOT;r=W.read(C.OUT/'result.json');audit=W.read(C.OUT/'verification.json')
    offline=W.read(C.OUT/'offline-acceptance.json');reg=W.read(C.OUT/'client-registration-current.json')
    plan=W.read(C.OUT/'dataset-atlas-plan-registration-current.json');h=r['hypotheses']['legacy512_a6'];cap=h['capability']
    assert offline['report_sha256']==reg['report_sha256'] and offline['dataset_plan_sha256']==plan['sha256']
    acceptance={'kind':'software_browser_acceptance','run_id':C.RUN,'passed':True,
        'report_sha256':reg['report_sha256'],'graph_sha256':reg['graph_sha256'],'plan_sha256':plan['sha256'],
        'checks':['formal report displays failed heldout measurement gate separately from successful NI',
            '256 exact-length pairs and two heldout frames visible','337 mixed cases and qualitative scoring/content counterexamples visible',
            'five calibration runs grouped under main workpackage','proposal status and dataset usage states visible',
            'full model identity verified; B18/U4427 locator works; exact down_proj column inspected',
            'all9728 native BF16 response values loaded for actual384-token sample'],
        'screenshots':{p.name:W.sha(p) for p in C.OUT.glob('visual-*.jpg')},
        'frontend_tests':23,'cpu_measurement_tests':15,'frontend_build':'passed; existing chunk-size warnings',
        'static_html_validation':'generated; embedded browser file protocol blocked, no workaround attempted; actual client plan inspected instead',
        'limits':['no browser network-offline emulation; actual generated offline pack verified separately',
            'no remote deployment, new installer or second-node replication in this phase']}
    assert len(acceptance['screenshots'])==2
    W.save(C.OUT/'browser-acceptance.json',acceptance,True)
    C.status('client_acceptance',report_sha256=reg['report_sha256'],browser_checks=True,offline_pack_verified=True)
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    glm=root/'research/glm5/docs/AGI_GLM5_MEMO.md';claude=root/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    marker='完整运行标识：`'+C.RUN+'`'
    text=glm.read_text(encoding='utf-8');assert marker not in text,'Closeout already appended'
    phase=max(map(int,re.findall(r'^## Phase (\d+):',text,re.M)))+1
    pilot_lines=[]
    for suffix in ['', '_v2','_v3','_v4','_v5']:
        p=W.read(root/f'tests/glm5/result/phase2777_natural{suffix}/pilot-result.json')
        pilot_lines.append(f'| {p["run_id"]} | {p["style_accuracy"]["acad"]:.2%} | {p["style_accuracy"]["colloq"]:.2%} | {p["content_accuracy"]:.2%} | 未通过 |')
    pct=lambda x:f'{100*x:.4f}'
    entry=f'''

## Phase {phase}: W6 等长度留出句式、完整自然生成与数据—图谱组织闭环 [{stamp}]

完整运行标识：`{C.RUN}`。本条按用户明确要求追加 GLM5 备忘录，并在 Claude 统一叙述中追加交叉索引。运行编号 P2777 与其他研究脚本可能共享数字前缀，以完整 run_id、design_sha、模型快照和产物哈希区分研究，不能按文件名前缀合并。

### 1. 任务、结论与真实完成范围

用户要求排除句式及长度混杂，以本地 Qwen3-4B 的自然完整生成验证旧 W6 参数编辑，保留能力保持，将结果、反例和图谱碎片持续存入客户端；随后整理数据集和图谱的整体组织及界面方案。

已完成单机真实闭环，**自然语域控制仍未验收**。新输入总长度严格匹配且有留出句式，但冻结联合评分器在留出句式上失效；不得把零合格翻转写成“不存在门控”或“容量只能分布式承载”。能力保持通过仅限重复的固定监测题池。本轮不是多节点复算，不是新安装器发布。

### 2. 五版发现集校准与前瞻性收窄

每版8记录家族、2发现句式、2语域要求，共32输入；分别保存原模型与继承旧编辑的自由输出。逐版冻结文法、评分器与脚本再试跑，所有失败和全文保留。版本依次检查：温和风格源文本、相同事实的明确要求、末尾面对面交流要求、系统角色要求、异领域示例迁移。既有短文本能力检查能生成口语、回读指定字串；长输入的单条与批处理检查未发现混入另一条件。短文本只作实现控制，不能替代用户要求的长文本研究。

| 运行 | 书面符合率 | 口语符合率 | 内容字面代理 | 双向行为门 |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(pilot_lines)}

v3在“书面≥.8、内容≥.8”的发现筛选内具有最高平衡语域符合率，选择规则在新留出输出产生前登记。但五版均未建立可用的双向自然语域对照，**没有从这些失败口语条件提取新的均值差方向**。正式问题因此预先收窄为：继承的具体旧编辑是否足以将原本可靠书面输出变为合格口语。正式方向、每层列数512、α=6及两负对照均在新记录3000–3127输出前冻结；没有看到留出结果后更改词表、剂量、阈值或另选方向。

### 3. 材料、采集与算子

- 模型路径 `{W.MODEL}`；完整权重及tokenizer快照 `{r['model_snapshot']}`，前后相同。CUDA bf16、RTX5080、transformers {r['env']['transformers']}、推理种子{W.SEED}。一次只加载这一模型；没有运行GLM或14B。
- 128新记录家族，留出事实句式/要求句式F2、F3；384与512有效token各64家族。每个家族有两句式、两种语域要求，共512条；共256对，长度不相等对数0。两语域的事实、中性整句背景以及其余要求完全相同，只改变等token长度的语域要求；无截断、空格补齐或masked padding充当长度控制。
- 长度在家族之间分配，不能据此估计同一家族的纯长度作用；词汇仍可能与发现材料重叠，仅声称新记录及新句式。长输入包含共同中性背景，是受控中文记录，不是自然几千token篇章。
- 保留所有37层HiddenState、B18–27全部9728 MLP中间单元，在末尾两个可见位置的原BF16位模式；512样本、128批、{audit['native_bytes']:,}字节。不是全token场，也不含生成阶段轨迹。另归档两条输入×10层共20份完整阵列供客户端查看。
- 四种权重条件：原模型、旧定向编辑、同列且Frobenius范数匹配随机方向seed0、按激活能量分层随机列seed0。每种512次自然生成，共2048完整输出；greedy、max_new_tokens160、无thinking、无固定续写候选池。四条件均无截断。编辑后以精确张量相等核对内存恢复，不修改模型文件。

定义输入到MLP输出：`a_l = silu(G_l h_l) ⊙ (V_l h_l)`，`y_l = W_l a_l`。旧单位方向为`u_l`，所选列的对角掩码为`M_l`，操作为：

`W'_l = W_l − α u_l (u_l^T W_l) M_l`。

此处W是down_proj，l为B18–27，M每层选512列，α=6；它在每个块上是秩至多一的参数更新，而α>1是过度移除并改变所选投影的符号/强度，**不是简单把门关闭**。十层联合、13,107,200个寻址标量；不能认定某个展示单元独立具有因果作用。没有为自然生成找到可行的最小编辑量，应记录`min_edit_params=null / not_established_for_natural_generation`；旧偏好任务的可行上界不迁移为自然生成紧致度。

### 4. 判据与统计对象

完整输出先由冻结语言特征组规则、再由未编辑原Qwen在两种A/B标签顺序下盲评；盲评仅见输出，不见输入要求和编辑条件，置信阈值.6。两者一致且达到45字、两句才接受标签，否则unknown。它们并非独立人类评审。内容代理只核查允许物品名称/别名、已给定最终数量及明显多余数量，不证明完整保义。

对家族i、句式f的书面输入，合格事件：

`Q_if = 1[base书面 ∧ edit口语 ∧ edit内容代理通过 ∧ 未截断]`。

主效应为全部256书面条件输入的Q均值；先对家族内两句式平均，128家族重采样6000次，95%百分位区间。主判要求均值≥.20、下界>.10、超过两个自身负对照、内容损失及原有口语响应损失上界≤.05，并通过能力NI。新材料任何一步失败均不支持预注册主张。家族bootstrap只在可交换近似下解释，不是从全部自然语言随机抽样。

能力门以同题配对：损失`d=(原对改错数−原错改对数)/n`；分别为损失、收益的二项概率建立同时Clopper–Pearson界，再作差。Bonferroni保证联合覆盖至少95%，损失上界≤2个百分点才通过。不能用“差异不显著”代替非劣效。

### 5. 真实结果及测量硬伤

- 留出原模型书面联合符合率77/256={r['baseline_by_register']['acad']:.4%}，口语5/256={r['baseline_by_register']['colloq']:.4%}，正式行为门失败。书面逐格F2/384=.328125、F2/512=.578125、F3/384=.140625、F3/512=.15625。内容代理原465/512，编辑461/512。
- **评分器失效证据：** 原Qwen盲评把256/256书面条件输出都判书面，规则却只接受77；这些输出常用核验报告语气但没有两个指定学术词。基线512输出的unknown为410，且均未截断。不能把规则假阴性误判成模型不会书面语。
- 合格书面→口语0/256，家族bootstrap退化为[0,0]，两负对照比较下界也不为正。不能说真实效应恰为零。128家族零事件的单侧95%二项上界约2.3%只描述**当前合格事件管线**，由于行为门失效，不能界定真实语域机制效应。
- 内容代理净损失{h['content_loss']['mean']:.4%}，95%区间[{h['content_loss']['ci'][0]:.4%}, {h['content_loss']['ci'][1]:.4%}]；原有口语合格响应损失区间[{h['colloquial_loss']['ci'][0]:.4%}, {h['colloquial_loss']['ci'][1]:.4%}]。这些门通过不能补救失败的语域行为门。
- 事后描述：只看同模型盲评，口语要求下原36/256、旧编辑11/256、随机方向28/256、随机列15/256。存在“编辑可能压低口语”的新候选，而不是书面→口语成功；两个对照也影响它。它没有独立评分与预注册推广资格，不能事后换主判宣布成功。
- 保义反例：物品是记录册、地点是图书室，编辑输出却新增“记录中所涉物品均为图书”。同时命中记录册和151仍不能证明保义。该全文、规则假阴性、温和口语漏检都已在客户端逐例保存，并标注为事后AI审阅。
- 能力正式2200题：{pct(cap['base_accuracy'])}%→{pct(cap['edited_accuracy'])}%；23原对改错、20原错改对；净损失{pct(cap['net_loss'])}个百分点，95%保守区间[{pct(cap['loss_interval'][0])},{pct(cap['loss_interval'][1])}]个百分点，NI通过。分层题池为C-Eval100、MMLU100、CMExam1900、本地lang_bench100；140 M3Exam镜像均仅参考。医学占比高，不能推广为全域能力保持，题池重复使用也不是新盲验收。
- 337混合记录=305生成/测量未达标+23正式能力回退+9参考答案变化。原runner把9参考变化沿用keyed标签；追加更正版及errata，不修改冻结原始结果，正式NI自始排除它们。
- 正式运行{r['seconds']:.2f}秒（约{r['seconds']/60:.2f}分钟）；CUDA峰值allocated {r['peak_cuda_bytes']:,}字节。五版试跑、实现诊断另计。

### 6. 持续积累与客户端验收

最新报告CAS `{reg['report_sha256']}`；图谱CAS `{reg['graph_sha256']}`；参考标签更正CAS `{reg['reference_label_errata_sha256']}`。旧版报告、旧W6固定候选偏好正结果及此前负版本均保留。10张B18–27卡明确属于1个联合干预，逐层因果未测，主张状态为测量门失败、未验收；不算10个已破解机制。

归档包括冻结512输入和token ID、4×512自由生成全文、逐题能力结果与题目、源脚本文本及哈希、预注册、原场manifest、20份完整9728维响应、五版试跑全文和337混合记录。约393MB原生NPZ仍在本机，离线包含元数据/预览与重算来源，不声称大张量已全部上传。

正式客户端实际检查了：长度/句式页、全部反例与事后审阅、五版子运行折叠、待实施方案及数据集使用状态；完整快照核验后定位B18/U4427，实读down_proj列4427，并加载384-token原输入的全部9728维响应，A[4427]=4.03125。该值来自未编辑原模型保存场，不是即时推理、因果重要性或编辑后激活。报告及参数空间截图保存在`tests/glm5/result/phase2777_confirm/visual-*.jpg`。

15项材料/统计测试、23项前端检查及生产构建通过；构建保留既有chunk体积告警。真实服务器生成的主包与研究库附加包通过核验，包含全部输入、2048输出、2340题监测、5版失败、20阵列、旧偏好证据、最新报告/图谱和方案。不是浏览器断网模拟，也没有远端部署或新安装器。静态HTML已生成；内置浏览器安全策略禁止file协议，未尝试绕过，已改为验收正式客户端中的结构化方案。

### 7. 数据集—图谱总体方案与界面

方案CAS `{plan['sha256']}`；可读界面文件`research/phase2777/dataset-atlas-plan-v2.html`，合同/路线`research/phase2777/dataset-atlas-plan-v2.research.json`，正式工作包内可展开查看。它是提案：规范化注册表、新job_kind、多节点复算和导航重构未冒充已经实现。

**选择：图谱为主、数据集为证据与管理入口。** 不按每个数据集各建孤立本体。三图层为外部模式族、模型特定的内部特征/路径、有范围的两图谱关联主张；三能力为知识网络、分析推理（含翻译和编程）、语法。语言、领域、语域、长度、组合深度为交叉条件。主张可挂接多个来源；同一数据集亦可服务多个图谱问题。

数据分为自然发现库（A引擎字典与协方差）、受控机制库（B引擎与行为门）、外部行为库（能力保持）、反例/复算库（C引擎与新设计）。CLiMP/BLiMP、GSM8K、HumanEval、FLORES+、XNLI、LongBench v1为按用途建议的材料，本轮未新增下载或运行；C-Eval/MMLU/CMExam已有监测子集的实际使用量独立标注。短最小对不能靠补中性背景就成为真实长篇；LongBench v2的超长选择题不替代自由生成语域验证。候选上游资料已查阅官方仓库/论文/数据卡，链接写入方案。平台不能用自身统一许可证覆盖上游数据许可。

身份层须保存dataset/version/source/license、原文档与样本血缘、同源组、研究切分、原split、分词器及模型快照；翻译、改写和重复来源按组切分、去重，不把不同文件名当独立证据。原生位置身份与方向/字典特征身份分开，run_id放到证据层；不能因同一单元索引就合并语义，也不能因换一次运行就凭空新增一个机制。迁移只追加canonical alias与claim映射，不改旧CAS。

界面建议：

| 一级 | 二级分组 | 渐进显示 |
| --- | --- | --- |
| 进展 | 总览→三能力卡→来源矩阵 | 首先显示有冻结分母的执行率，再展开有效行为格、留出证据、反例及预测保真度；无分母“—”，未测“未测”。 |
| 空间 | 语言图谱 / 参数结构 / 样本响应 | 模型/快照与来源过滤；节点右栏先一句主张和实例，展开才给原文、对照及参数；同一候选的不同范围正负结果并列。 |
| 研发 | 计划与执行 / 数据与模板 / 证据审核 | 数据集视角用于材料质量、版本、切分与队列，提供反向进入关联图谱；统一实时分析入口，保留模型/样本上下文。 |
| 配置 | 模型与节点 / 算力贡献 / 账号 | 仅资源、托管与权限；不再复制研究分析页面。 |

图谱布局距离与动画不是模型几何/计算边。数量、数据集数、分片完成率不是破解比例。最终要测留出前缀上`mean KL(p_model || p_atlas)`，以及`(L_simple−L_atlas)/(L_simple−L_model)`；分母为正才定义，负恢复不截断。图谱预测不能读取目标状态、未来答案，或把完整模型重跑当提取算法。

### 8. 理论进展、第一性原则与下一大阶段

本轮没有发现新的普遍语言数学结构；推进在于把“可改变固定候选偏好”和“能控制完整生成”分成不同、可追溯的命题，并找到**评价算子也会受句式混杂**。长度配平与留出句式不能自动保证测量本身有效。能力保持是选择编辑的约束，不是控制机制已经成立的证明。自然生成还包括自回归状态接续；单个读出边界的变化未必在后续历史中持续。

下一阶段围绕“可推广的条件化语言状态与连续生成机制”组织大任务：

1. **测量与材料大任务：** 多领域真实连贯书面/口语重述，先由独立双人盲评校准语域、保义、自然度及分歧处理，冻结标注规则；把词表规则降为辅助诊断。在同一内容家族内交叉512/1024/2048有效token及多构式，试跑各格通过后再按功效规划正式家族量，保留词汇/句式/领域留出。不能只继续增样或调到能过门。
2. **三引擎与动态场大任务：** A自然字典发现、B受控映射、C因果边在同一材料比较；采提示及生成步全层原生场，检验旧编辑究竟改变哪种响应、是否被后续恢复、是否只是评分器漏检。拟合可解释的条件转移关系，保留hs[0]、长度/位置、随机标签及真实参数来源；不要只寻找必需头或只反复作差搬运。
3. **可复算图谱与分布式大任务：** 实施注册表与claim映射，新增`natural_register_intervention.v1`冻结任务适配，以整家族分片、显存互斥和断点产物执行；第二节点复算与争议隔离。先Qwen3-4B，随后GLM4及14B逐个运行，按实际显存安排合适节点或另桶offload/量化，不把16GB卡当作可直接装入14B bf16。
4. **图谱解释增量大任务：** 外部族结构、内部响应/字典特征、真实计算边共同构成一个可执行的图谱预测器，在未见组合上超越词汇/位置/n-gram基线；达到这个条件才能把累积碎片逐步升级为机制解释。

通俗说明：这轮把相同内容和相同长度的材料真正跑了一遍，也保存了每条生成。最需要修正的是尺子：许多明显书面的回答没出现预定关键词，就被规则判成“不确定”。所以现在能说能力监测保持住了，不能说自然语域开关已经找到。系统应围绕共同图谱积累这些有范围的证据，用数据集追溯来源，而不是按数据集画许多互不相通的图。
'''
    with glm.open('a',encoding='utf-8',newline='') as f:f.write(entry)
    prior=claude.read_text(encoding='utf-8');cp=max(map(int,re.findall(r'^## Phase (\d+):',prior,re.M)))+1
    cross=f'''

## Phase {cp}: W6 等长留出句式的测量失败与数据—图谱方案（运行P2777，GLM5 Phase{phase}详记） [{stamp}]

按用户强制追加要求，完整原理、公式、五版校准、正式数据、保义反例、评分器硬伤、能力NI、客户端/离线验收与阶段路线已追加`research/glm5/docs/AGI_GLM5_MEMO.md` Phase{phase}。完整run_id `{C.RUN}`，design `{r['design_sha256']}`，模型快照 `{r['model_snapshot']}`。

真实Qwen3-4B CUDA bf16：128新记录家族，2留出句式，384/512有效token，256严格等长对、512输入×4权重条件=2048自由生成。五版发现双向语域校准均失败，正式预先收窄为旧编辑的书面→口语充分性，不从失败口语条件提取新方向。留出原模型盲评256书面皆书面，但冻结规则只接受77，联合行为门失败；合格翻转0不能解释为没有门控。字面保义代理还遗漏“记录册/图书室→物品均为图书”的语义新增。全部生成均未截断。

能力重复监测2200题：68.5455%→68.4091%，23损失/20收益，95%配对CP损失界[−0.7367,1.0084]个百分点，低于2点NI门；医学1900题占比高，非全域证明。140M3仅参考；9条参考变化的标签误用已追加更正，原始文件不改，NI从始排除。337混合记录、5版试跑、约393MB本机原生场和20完整阵列保存；10张参数定位卡仍是1个联合编辑。

最新报告 `{reg['report_sha256']}`，图谱 `{reg['graph_sha256']}`，数据图谱方案 `{plan['sha256']}`。旧偏好正结果仍保留，不能被自然生成测量失败覆盖。15CPU、23前端检查与构建通过；正式浏览器核验报告、反例、待实施方案、完整快照B18/U4427及9728维响应；真实离线主/附加包核验通过。没有远端部署、新安装器或第二节点复算。

方案采用图谱主入口+数据证据维度，保留研发数据管理的反向入口；外部模式/内部对象/有范围主张分开，run与数据集作为证据血缘。未来依次实施数据身份契约、独立多领域语域评价、提示—生成连续原场的三引擎、分布式复算及图谱留出预测。当前增量是发现测量本身也须跨句式推广，不是建立了普遍容量×门控定理。
'''
    with claude.open('a',encoding='utf-8',newline='') as f:f.write(cross)
    W.save(C.OUT/'memo-index.json',{'glm_phase':phase,'claude_phase':cp,'time':stamp,'run_id':C.RUN},True)
    print(json.dumps({'glm_phase':phase,'claude_phase':cp,'time':stamp,'browser_acceptance':True,'offline_acceptance':True}))


if __name__=='__main__':main()
