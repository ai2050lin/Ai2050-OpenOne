"""Append-only research closeout; requires completed local delivery audit."""
import hashlib,re
from datetime import datetime
import state_formation_core as C

def main():
    C.checked();r=C.read(C.OUT/'client-report.json');a=C.read(C.OUT/'audit.json');d=C.read(C.OUT/'delivery-audit.json')
    reg=C.read(C.OUT/'registration.json');errors=C.read(C.OUT/'behavior-error-diagnosis.json')
    assert d['passed'] and r['diagnostic_only'] and not r['behavior']['passed']
    assert all(not e['scope_use_supported'] and not e['effect_prediction_supported'] for e in r['effects'])
    center=next(x for x in r['center_rows'] if x['primary']);acc=center['accuracies'];ci=center['gain']['frame_interval']
    table='\n'.join(f"| {e['candidate']} | {e['method']} / {e['dose']} | {e['mean_target_margin_loss']:.6f} | {e['mean_control_margin_loss']:.6f} | [{e['matched_excess']['frame_interval'][0]:.6f}, {e['matched_excess']['frame_interval'][1]:.6f}] | {e['prediction_to_best_baseline']:.6f} | {e['first_correct_to_wrong']} / {e['first_wrong_to_correct']} |" for e in r['effects'])
    formations='\n'.join(f"- {c['id']}：发现期同一后边界读出器的准确率增益 {c['accuracy_gain']:.6f}，发现门{'通过' if c['selection_support'] else '未过、弱候选'}；新样本各边界各自读出准确率 {c['confirmation_probe_accuracy_before']:.6f} → {c['confirmation_probe_accuracy_after']:.6f}；同一后边界方向的有符号对照 {c['same_post_decoder_signed_contrast_before']:.6f} → {c['same_post_decoder_signed_contrast_after']:.6f}。" for c in r['formation'])
    baselines='\n'.join(f"- {r['curves'][i]['label']}：常数、q/r/L加法、四状态、八条件 MSE 依次为 "+', '.join(f"{v:.7g}" for v in r['curves'][i]['state_prediction_mse'].values())+f"；八条件/四状态={r['curves'][i]['eight_over_four']:.6f}。" for i in [48,66,68,70,72,73])
    cap='\n'.join(f"- {key}：基线 {v['base_correct']}/{v['n']}，编辑后 {v['edited_correct']}/{v['n']}；失去 {v['lost']}、新增答对 {v['gained']}。" for key,v in r['capability'].items())
    gatefails='；'.join(f"frame/history={key} 为 {v['four_choice_accuracy']:.6%}（n={v['n']}）" for key,v in r['behavior']['cells'].items() if v['four_choice_accuracy']<.9)
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';index=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    old=memo.read_bytes();oldindex=index.read_bytes();assert C.RUN.encode() not in old,'Run already recorded'
    phase=max(map(int,re.findall(rb'^## Phase (\d+):',old,re.M)))+1
    iphase=max(map(int,re.findall(rb'^## Phase (\d+):',oldindex,re.M)))+1
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    entry=rf'''

## Phase {phase}: 新样本条件中心、注意力与MLP状态形成，以及行为门失败后的全样本作用诊断 [{stamp}]

**任务、交付与结论边界。** 承接Phase 2887，先解决D盘容量，再以本地Qwen3-4B的新样本检验条件中心偏移、注意力/MLP写入及简单基线，保留失败并同步客户端。运行 `{C.RUN}`。本轮真实完成8192个唯一输入的全层/多位置采集、预冻结候选与作用预测、新确认集行为检查，以及768个预定样本的原模型干预诊断；但**原确认流程未通过行为门，不能宣布语义机制确认成功**。原causal阶段在任何确认干预前主动报错停止，失败日志保留；随后另行冻结diagnostic-amendment，按原样本/候选/剂量/种子对全部768例做故障诊断，不剔除失败格、不改阈值、不把该诊断恢复成盲测。原冻结分析仍以行为门为必要条件，所有scope_use_supported、effect_prediction_supported及center_primary_supported保持false。0项正式机制晋级，未运行14B/GLM4，无跨模型结论。

**先保全证据再采集。** 起始D盘余量约0.14GiB。把旧mlp_puzzle_v1的1863文件、2701122204字节搬到获准可写的C盘研究存储，逐文件SHA256核对后才删除D盘重复副本，原D路径保留为Junction。首次创建链接受沙盒权限限制时已恢复原目录，再以有界目录操作完成；没有丢弃旧结果腾空间。收尾再次通过原逻辑路径核对全部1863文件。新结果逻辑路径tests/glm5/result/state_formation_v1同样链接到C盘；原子写入使用解析后的真实C路径，避免跨卷os.replace错误。首次未冻结写入错误、迁移回退日志均保留。新成果实际保留 {d['result_retained_bytes']/2**30:.3f}GiB，冻结预算32GiB、C盘最低预留30GiB；收尾C盘 {d['C_free_bytes']/2**30:.2f}GiB、D盘 {d['D_free_bytes']/2**30:.2f}GiB可用。只清理本轮可重建构建目录及已经归档并校验的临时日志重复件，保留全部原生场、输出、失败日志、坐标和复算依据。目录链接依赖C盘目标，不应把D路径当作第二份备份。

**冻结、数据与计算身份。** 模型/tokenizer完整快照 `{C.SNAP}`，本地models/hf/qwen3-4b，原生BF16，无量化，batch=1、CUDA、SDPA显式布尔因果mask、use_cache=False、TF32关闭。每次加载逐个核对磁盘及GPU参数字节SHA。2026-11-09 06:41冻结原plan、12项依赖源码及材料；发现选择在读取确认场前冻结，64个发现家族的作用估计也在确认采集前冻结。24个作者生成并经shared template_grammar.v1校验的文法：8发现记录表述×32家族=256；16新记录表述×48家族=768。每家族8个q/r/L条件，c=qr，q为查询X/Y、r为当前红蓝映射、L为输出英文/中文。新确认名词8种与表述、历史条件、重复交叉，实体家族不跨划分；8192分词输入与旧运行无完全重合。pilot256输入只是发现集子集，不能另算新增样本。

新确认变化的是**记录表述、名词及实体**，查询和语言指令框架沿用发现集；主体输入均为英文，不能称为输入中英文普遍机制，也没有检验诗歌、讨论、论文风格或泛化递归。历史分为无可用旧映射、与当前r独立随机的已废旧映射；旧映射在同家族八条件中固定，因此不会与当前颜色恒定混淆。功效预算Δ=.05、n=768、p_d=.10、α=.05/8，工具给所需512、预算功效约.95；这是二值配对敏感度近似，不能保证16个人工表述代表语言总体，不能直接套到连续logit效应。统计同时列家族与句式簇重采样，主要解释采用16表述均值的4096次bootstrap区间；这也不是完整的句式×家族双层随机采样模型。

**测量与可复算数学对象。** 每家族原始BF16位模式保存：hidden[8,37,4,2560]（embedding+36个纯block后残差），post_attention/attn_write/mlp_write各[8,36,4,2560]，final_norm[8,4,2560]。四位置是record_end、query_end、instruction_end、answer_prefix；最后位置的条件均因果可见。分析链74边界：embedding、每块注意力后/MLP后、最终RMSNorm。不同于旧数据，本轮B35原始输出与最终归一化已分开。运行中逐例、逐块、逐位置核验真实加法；事后独立CPU BF16舍入又检查32家族全36块四位置。未来q/L在相同record_end前缀的最大相对差 {a['prefix_negative_control_max_relative_l2']:.8g}，全部逐位一致={a['prefix_negative_controls_all_bitwise_equal']}。不能把同一坐标跨层自动命名为同一功能。

$$
R_b=\operatorname{{BF16}}(H_b+A_b),\qquad H_{{b+1}}=\operatorname{{BF16}}(R_b+M_b),\qquad
X=(1,q,r,L,qL,rL,qr,qrL),\quad X^TX=8I,\quad \beta=X^TH/8.
$$

前两式是采集到的原模型计算关系；后两式是正交实验编码和条件均值估计，均不自动成为新的语言定理。候选四状态拟合为

$$
\hat h(c,L)=\mu+c\,a+L\,b+cL\,d,\qquad
\text{{语言方向}}=b+c\,d,\quad \text{{条件中心}}=\mu+c\,a.
$$

读取方向按发现集逐坐标方差缩放并固定；不做Top-K截断，不输入待预测层之外的未来信息。固定/修正方法都先从当前h解码颜色ĉ，再用相反颜色s=−ĉ条件上拟合的语言方向；修正只把中心从μ+s a改成μ+ĉ a。真实确认颜色只给显式oracle基线。两个颜色都参与发现拟合/尺度估计，不能声称训练时从未见过另一颜色。该实验预测的是当前状态能否被读出，不是从token独立算出模型状态。八条件多项式对二元设计本身可饱和表达，写出公式不证明模型真的按该多项式计算。

**行为门失败是本轮的首要实证结果。** 全6144个确认条件首词top1为 {r['behavior']['first_top1']:.6%}，四候选错误50条；预注册逐记录表述×历史条件准确率至少90%，失败格为 {gatefails}。其他还有frame2历史格4条错误但未低于门。更细256格的最小n={a['fine_condition_cells']['minimum_n']}、最小准确率={a['fine_condition_cells']['minimum_accuracy']:.6%}；这是收尾描述性审计，不能冒称为预注册细格门。50条错误全发生在旧映射与新映射冲突时，50/50颜色与已作废旧映射一致；中文30、英文20。无旧映射或旧映射相同的条件全对。这个吻合定位到“新旧来源竞争”的问题，不是旧映射已被证明经某个头传播。旧记录文字更直接、记录表述显式颜色词等仍可能是替代解释。

**条件中心：必须同时看强基线与准入。** hs[34]（B33后）的固定中心准确率 {acc['fixed']:.6%}，由当前h读出颜色后修正 {acc['decoded_center']:.6%}，提供真实颜色的oracle {acc['oracle_center']:.6%}，整体线性读出 {acc['pooled']:.6%}，仅范数 {acc['norm']:.6%}，反向中心 {acc['wrong_center']:.6%}，打乱发现标签 {acc['permuted_labels']:.6%}。修正相对固定中心平均增益 {center['gain']['mean']:.6%}，句式簇区间[{ci[0]:.6%},{ci[1]:.6%}]，α=.00625。即使数值门看似满足，原行为门未过，整体冻结命题不晋级；也不能把修复一个条件阈值有偏的读出器解释成优于整体线性基线。次要hs33/35只作描述，不能替换主要边界。

**BF16并列评分的细节。** 共3例四候选最高分并列；其中1例四候选按预定(en_red,en_blue,zh_red,zh_blue)顺序判错，但完整词表按token ID的argmax判对。所以四候选判错50、实际首词判错49；50/50吻合旧记录应严格理解为“按冻结四候选规则选出的颜色”，不能把并列的这1例描述成唯一确定的错误输出。两组行为门仍失败，不更换并列规则补过门。四候选argmax的输出语言准确率仍100%。

**单次置乱负对照的硬伤。** 打乱标签比较器的固定符号准确率只有10.904948%，但事后将其预测标签取反得到89.095052%。这不作为新选出的预测器或确认结果，却足以说明“低于50%”不是“没有编码语言”。本轮只有一个固定种子的家族内标签打乱均值方向，未形成多次家族/句式级置乱的零分布；不能用这个弱对照证明所提方向的特异性。仅范数也达85.481771%，需警惕幅度、共享条件结构和读出器适配。后续须在发现阶段冻结多次置乱、随机方向及范数/位置控制，保留双向可读性口径；当前不补调阈值来消除这项硬伤。

**注意力与MLP形成：观察和作用分账。** 只用发现前128家族拟合、后128选择，随后256重拟合；语言在B12–18、颜色在B18–26窗口内，分别冻结一个attention与MLP候选。选择用同一后边界解码器的前/后准确率差，仍可能受全局中心变化影响；确认曲线的前/后准确率各用各自冻结解码器，所以另列同一后边界方向的有符号条件对照，不把曲线跳跃直接称为模块写入因果效应。

{formations}

早层曲线是另一项重要限制：答案前embedding语言/颜色均50%，但B0注意力后语言已99.088542%，B0 MLP后81.852214%，B1注意力后100%。因此不能把B13或B16叫作语言信息第一次形成的地方；所选后边界读出器在前边界表现差，可能只是阈值/坐标不适配，而该前边界自己的冻结读出器已100%。这也与旧材料hs15才达到所用持续读出判据不同，说明可读出“起点”强烈依赖材料和读出器，不能固化成跨任务的语言层位。注意力/MLP写入对已存在信息的变换、放大、抑制和下游利用，应与信息首次出现分别检验。

**全状态的简单基线。** 在相同新材料上比较常数、q/r/L加法、四状态(c,L,cL)、八条件，全部由发现拟合、评估原始全向量MSE，不减去测试家族真实中心。四/八状态模型接收实验的正确c/L标签，属于条件摘要预测，尚未从原始文本闭环计算这些标签；也不是完整词表分布预测。几个代表边界如下，所有74边界与残差保留：

{baselines}

这里出现了对旧局部规律的重要限制：Phase 2887旧材料在hs34为八条件/四状态≈1.024，四状态更好；新材料hs34变成≈0.876，八条件更好，尾部多个边界也如此。因此不能保留“尾层总会充分压成四状态”作为一般规律。新旧材料同时改变了表述、实体、背景和历史设计，当前差异不能单独归因于旧记录冲突；需要下一轮正交设计。八条件多出的q/r及其语言交互能够降低当前误差，只是保留了额外条件响应的证据，不等于找到了模型实际执行的八格查表器。

**作用诊断的方法与结果。** 原确认流程停止后，diagnostic-amendment另行冻结并标明材料已暴露。沿用原先确定的每家族一个条件，共768例；四候选分别在最后位置缩放模块写入，剂量.5/1。对照匹配实际BF16残差改变量的范数（误差≤2%），不是只匹配FP32理想值；注意力后状态替换重放原块MLP及残差，前16家族/候选恒等和零剂量与原输出逐位一致，每例全部编辑后恢复逐位一致。注意力另屏蔽最后查询对语言指令区或当前记录区的全部头连边，与等token数量背景屏蔽比较；这会重新归一化注意力，不是等注意力质量、单头或QK参数干预。

源屏蔽只涉及该块最后位置到源区的直接边，其他位置或更早层已传播的信息仍可绕行；因此弱效应不能否定整个来源区的作用，不能把这等同于删除全部当前记录。

目标作用量是正确候选相对同另一因子条件下竞争词的logit优势下降，KL为KL(编辑分布||原分布)。四候选首token评分并非整段生成质量。作用预测在确认采集前由64发现家族拟合四(c,L)层的分数变化；输入仍包括原模型四候选分数和正确实验c/L，不是输入到输出的机制替代。比较同批案例的无变化与全局平均效应，以下比值为预测MSE/二者中较小者；≤.8只是冻结数值门，行为失败时不接受科学确认。区间基于16句式簇；所有768例均计入，没有删除无效对照或失败行为。

| 候选 | 操作/剂量 | 目标优势下降 | 对照下降 | 超过对照的句式区间 | 预测/最佳基线MSE | 首词正确→错 / 错→对 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
{table}

统计失去/新增答对与连续分数效应分别报告；“另一因子正确”是条件化二选一口径，不等于整段能力保持。源屏蔽不能证明唯一路径，完整模块删除也不能证明它独有地存储语言/颜色；无明显效应只否定此次位置/剂量的命题，可能存在冗余和别处计算。effect-error-diagnosis保留新旧记录冲突、基线对错分组的描述数据，分组重叠且未另设过门标准。

全量删除下，语言MLP B13目标优势平均下降约0.291、颜色attention B23约3.215，均值作用超过范数对照，分层作用预测误差比约0.460/0.494；这些是**未通过整体行为门时的数值诊断**，不是已晋级的语言/颜色模块。语言attention B16整块移除的目标下降为负，颜色MLP B19也为负，故发现期读出增益不保证方向性的预测作用。B23仅屏蔽当前记录区直接边时优势下降约−0.881，与整块删除方向相反：不能把整块作用直接归到当前记录的这组直接边，信息可能已在其他位置/层传播，源区状态本身也已上下文化，注意力重归一化还是替代解释。四候选分组中，B23删除使原本正确761例中的9例变错，也使7例原本错中的2例变对；首词口径对应8错/2恢复，两种评分不可混同。错误组仅7例，不能把它升级为通用纠错法。

**副作用。** 512道已暴露历史带答案选择题的首字母强制选择结果如下，仅作宽编辑副作用，不是新的综合能力盲测：

{cap}

**审计与客户端。** 全1024家族原生文件SHA及元数据/冻结身份通过；3840份基线/全量模块删除完整词表BF16数组的四候选分数和argmax独立重算，FP64与保存FP32 KL最大误差 {a['maximum_float64_vs_float32_KL_error']:.8g}；768次干预基线与原采集分数逐项相同，所有恢复通过。技术复算在同一物理节点上，不能叫独立节点复现。报告、5张全向量/模块候选卡、4条观察关联边同步两个本机库；146个全2560坐标数组、每库{d['coordinate_values_per_library']}个浮点数与原件逐位核对，{reg['objects']}种内容寻址对象及包SHA回读。卡片单坐标只是导航锚点，不当作单神经元或唯一概念位置。无正式因果边晋级。

客户端复用已有实验记录，四页是条件中心检验、注意力与MLP形成、作用与副作用、原始证据与失败；并列强基线、74状态边界、真实注意力/MLP写入全坐标、50条行为错误、全部作用失败理由和原始产物。行为门失败与诊断身份默认可见，缺数据/模型身份不符明确停止显示，保留恢复入口。科学6项、前端56项、浏览器{d['browser_checks']}项通过，页面错误/模型启动请求0；新增组件lint及生产构建通过，既有模块类型/大分块提示保留。真实桌面曲线/原生坐标/作用表和390px手机视口截图已目视核对。没有改共享协议/runner，未重跑整套305项；安装版、离线安装包、正式远端未验收或部署。

报告SHA `{reg['report_sha256']}`，图SHA `{reg['graph_sha256']}`，研究包SHA `{reg['bundle_sha256']}`。本机入口为 /workbench?lens=process&station=execution&run={C.RUN}&snapshot={C.SNAP}。重复包登记不增加样本，远端发布仍由用户手动安排。核心脚本state_formation_design/math/runner/analyze保持原冻结哈希；门失败后的state_formation_diagnostic另有来源哈希。publish、failure_analysis、delivery、memo脚本、全部命令日志和5字段溯源均在指定目录，详见delivery-audit、audit、diagnostic-amendment及memo-append。

工程审计首次在保存统计字段cells_below_point90时遇到numpy.int64不能被溯源JSON编码器序列化，改成内置int后全量重审通过；未改变冻结测量或分析。另有一个日志保留缺口：PowerShell经跨卷Junction移动首轮audit.log失败，随后重跑覆盖了原完整traceback。audit-recovery.json如实保留工具已观察到的错误尾部、exit=1与恢复过程，不能称为保存了完整首轮traceback。原模型行为门失败causal.log和诊断协议完整保留；这一工程缺口没有被补写成实验通过记录。

交付对账首轮因通用产物并非全部含schema字段而触发KeyError，改为仅对带坐标schema的文档做数组校验后重跑通过；完整首次日志保存在delivery-initial-schema-check-failure.log。首次本机登记早于工程恢复说明链接补入，随后补登记最终204对象版本，原登记哈希保留在registration-before-recovery-link.json；没有新增样本、改写科学结果或宣称重复复制为独立证据。

**资源和硬伤。** 依次加载同一4B，没有并行加载其他模型；本轮设计执行量为29568次真实GPU前向（含重复校准/干预/复原，不能当独立样本数）。阶段耗时详见delivery-audit。进度每8家族、能力题每32题写换行并刷新；各完成阶段最大日志间隔均<5分钟。WDDM采样专用显存峰值 {d['wddm_peak_dedicated_bytes']/2**30:.3f}GiB，共享显存 {d['wddm_peak_shared_bytes']/2**20:.2f}MiB；这是分配计数，不等于传输量，不能据此宣称绝对无CPU溢出。没有新增自然语料、无监督SAE/transcoder同预算比较、完整背景协方差白化、自由生成、参数级分解或跨模型检验。当前场只保留四位置；attention源token如何形成各自K/V尚未细分。本轮获得的是带失败范围的编码观察与宽模块作用诊断，不是整体语言编码机制的闭合。

**第一性原理、理论增量与下一大阶段。** 有限状态的几何可读性、输出规则和产生状态的算法是三个不同对象。相同最后token的embedding不能独自决定答案，前缀中的实体绑定、有效性/时序及语言要求必须通过实际读写过程进入状态。条件中心可以解释某个读者的阈值为什么失准，但不自动解释网络如何挑选正确记录；在冲突输入上，仅有干净颜色/语言分离甚至高总体准确率仍不足够。RDC本轮不新增统一定理，最有价值的拼图是将“条件状态几何”与“来源有效性选择”分开登记，并用门失败精确界定其连接尚不可靠。

下一阶段应围绕一个共同问题组织：**有效记录如何战胜过时记录，并生成可复用的颜色/语言状态**。先设计新的未暴露查询/指令框架和独立记录文法，正交控制颜色词显式性、对象位置、记录新旧、冲突强度及语序；先在发现集校准行为，不再复用此次确认集调门。采一次保存全层和所有相关源位置；用QK选择权重、V/O真实写入及MLP条件响应提出可执行来源规则，比较最近记录、关键词/位置、词袋、简单绑定器、现有四状态模型和适用字典/电路方法。冻结可比预算后，先预测未见来源选择/状态/输出及干预效应，再检验具体头、原生坐标/单元组与匹配反例，分开必要性、充分性和副作用。只有算子相对简单基线带来冻结留出预测增益，才升级为更细齿轮；若只能改阈值、或宽模块扰动有作用但效应不能预测，回到来源与替代路径，不继续给热区命名或重复升级同一失败规则。随后再引入独立风格/句法因子，检验同一算子的组合与竞争；不能直接把颜色查表推广到翻译、推理、递归或AGI。

通俗说明：这轮不仅检查“颜色和语言能不能从脑内读出来”，还检查“哪些计算步骤会影响答案”。最重要的反例是：部分新说法下，模型会沿用已经作废的颜色记录。中心偏移和模块作用的数值都保留了，但这道行为门没过，因此不会把它们包装成已经破解的通用规则。
'''
    cross=f'\n\n## Phase {iphase}: 新样本状态形成与行为门失败诊断 [{stamp}]\n\n详见GLM5 Phase {phase}，运行{C.RUN}。保全1863旧文件并迁移2.516GiB至C盘，新采集256发现/768确认家族共8192输入、74状态边界及4位置。确认首词99.2025%但两记录表述历史格未过90%行为门；50条错误均与已废旧记录颜色吻合。原确认causal停止，另行冻结全768预定样本诊断，全部机制确认标记false；保留条件中心、四候选attention/MLP作用、强基线及512旧能力题副作用。两个本机库{reg["objects"]}产物、146全坐标数组、5卡4观察边、客户端行为失败与恢复验证完成。详细数值、硬伤、存储/显存和下一阶段来源有效性竞争方案见主备忘录。未远端发布、无正式机制晋级。\n'
    assert memo.read_bytes()==old and index.read_bytes()==oldindex,'Memo changed; retry using latest tails'
    with memo.open('ab') as f:f.write(entry.encode('utf-8'))
    with index.open('ab') as f:f.write(cross.encode('utf-8'))
    assert memo.read_bytes().startswith(old) and index.read_bytes().startswith(oldindex)
    C.save('memo-append.json',{'phase':phase,'index_phase':iphase,'timestamp':stamp,'memo':str(memo),
        'previous_bytes':len(old),'previous_sha256':hashlib.sha256(old).hexdigest(),'entry_sha256':hashlib.sha256(entry.encode()).hexdigest(),
        'append_only_verified':True,'memo_source_sha256':C.sha(__file__)})
    print({'phase':phase,'index_phase':iphase,'time':stamp},flush=True)

if __name__=='__main__':main()
