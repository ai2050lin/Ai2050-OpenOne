"""Publish the local November review and append its audited cleanup to the memo."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import check_write
import november_storage_cleanup_20261118 as C

OUT = C.OUT


def write_new(name: str, content: str):
    path = OUT/name
    data = content.encode('utf-8')
    check_write(path, len(data), operation='十一月审计交付')
    with path.open('xb') as stream:
        stream.write(data)
    return path


def main():
    plan = json.loads((OUT/'plan.json').read_text(encoding='utf-8'))
    receipt = json.loads((OUT/'deletion-receipt.json').read_text(encoding='utf-8'))
    verification = json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    historical = json.loads((OUT/'historical-manifest-crosscheck.json').read_text(encoding='utf-8'))
    assert verification['passed'] and historical['passed']
    assert receipt['deleted_files'] == verification['deleted_paths_absent'] == plan['delete_files']
    assert receipt['plan_sha256'] == C.sha(OUT/'plan.json')
    assert not (OUT/'memo-append-verification.json').exists()
    before = C.MEMO.read_bytes()
    assert hashlib.sha256(before[:plan['memo_before_bytes']]).hexdigest() == plan['memo_before_sha256'], 'Earlier memo bytes changed; stop before append'
    existing = before.decode('utf-8')
    phase = max(map(int, re.findall(r'^## Phase (\d+):', existing, re.M))) + 1
    timestamp = C.stamp()[:16].replace('T', ' ')
    grouped = collections.defaultdict(lambda: dict(paths=0, logical=0, unique=0, kept=0))
    seen = set()
    for r in plan['records']:
        g = grouped[r['tier']]
        if r['action'] == 'delete_raw':
            g['paths'] += 1
            g['logical'] += r['bytes']
            key = (r['device'], r['inode'])
            if key not in seen:
                g['unique'] += r['bytes']
                seen.add(key)
        else:
            g['kept'] += r['bytes']
    source_lines = {int(re.match(r'## Phase (\d+):', e['header'])[1]): e['line'] for e in plan['monthly_index']}
    index = io.StringIO(newline='')
    writer = csv.writer(index)
    writer.writerow(['phase', 'timestamp', 'source_line', 'title', 'characters', 'entry_sha256'])
    for entry in plan['monthly_index']:
        m = re.match(r'## Phase (\d+): (.*?) \[(.*?)\]', entry['header'])
        writer.writerow([m[1], m[3], entry['line'], m[2], entry['characters'], entry['sha256']])
    write_new('november-index.csv', index.getvalue())

    tests = [
        ('2775–2778', 'W6 权重编辑与自然语域',
         '修改指定 MLP 写出方向，分别看候选偏好、自然生成、保义和能力保持。',
         '固定候选分差可以变化，但自然语域联合门未通过；160条旧确认没有跨候选决策边界，不能把评分移动当作自然能力控制。',
         '保留局部评分结果；自然控制未支持，不能推出容量必然分布式。'),
        ('2781–2792', '属性仲裁、原生单元与字典保真',
         '先检查模型是否能按资料/现实范围作答，再比较原生组、字典的激活误差与输出损失。',
         '2781资料192/192，现实61/192，行为门失败。2792自然文本SAE损失恢复0.589192、KL 0.105219；同预算原生256恢复1.054527、KL 0.041184。',
         '激活FVU更低不保证行为更好；恢复大于1限于该批预测损失，非一般能力增强。'),
        ('2797–2809', '早期写入、来源读取与缓存时序',
         '对来源位置、Head/MLP写入及K/V做分离、回放和恢复，检验新任务与新答案词。',
         '2801派遣/查找各104个低余量输入，联合操作分别诱发20/18个反答案；2809三方法×三留出的九格均未达0.35 logit最小效应。',
         '局部计算传递成立的范围保留，跨任务推广未过；回放一致不是完整机制成立。'),
        ('2851–2885、2896、2916、2919', '全量原生档案、相对编码与跨层预测',
         '保留原生G/V/A、参数及完整坐标，比较词汇基线、相邻层预测和实际干预。',
         '2860的38912单元中31949个在各自留出八个正极未命中发现关键词。2866四剂量目标与对照均0/768翻转。2885共同输入可绕过所选源单元影响下层；单独关闭源单元还可能增强目标。',
         '普查覆盖不等于语义解释率；相关边、实际参数来源、因果边必须分开。'),
        ('2880、2887–2895、2901', '条件状态与强基线挑战',
         '用析因状态中心预测响应，比较四状态、八条件和简单来源规则，并检查旧/新记录竞争。',
         '2880四状态激活MSE在16组中的11组优于八条件；2888整体首词99.20%仍有两个正式条件格失败，50条错误全部符合过时颜色映射。',
         '四状态包含实验者提供的正确语义变量，是条件基线；极小KL不能作为从文本独立破解的比例。'),
        ('2905', 'B1修订、自然写入与自主递推',
         '增加原生供体、多个随机对照、实际BF16范数匹配、输出权重方向和完整行为门。',
         '原生响应或输出改写部分成立，逐题路由预测相关仅0.210/0.235；中心化后的MSE增益约−11.27%/+5.01%。',
         '相当部分平均增益来自整体置信度下降，未完成逐句来源选择算法。'),
        ('2906', 'source_binding_v1：来源×实体绑定',
         '发现集按QK点积能量选32头内坐标，以原生V/O预测写入；另选256原生MLP单元并做原模型路径干预。',
         '256家族2048条；三个头两面板写入误差改善76.59%–97.70%。正确记录边联合排除使四选一下降25.52/21.74个百分点；B27H3选对记录却写入预测失败。',
         '随机32项也几乎保留答案；MLP范数匹配不合格。自主递推全词表top1一致率0，GLM逐格试跑失败，无跨模型机制确认。'),
        ('2907', 'source_formation_v1：Q/K形成与保留内容递推',
         '由较早状态预测晚层Q/K，分开评估向量误差、路由、原生写入、连续更新和匹配干预。',
         '每模型计划3072条；Qwen两留出B28H21/B29H10过局部写入门。新句式四个Q预测均差于发现均值；GLM新句式行为失败，未采该内部面板。',
         'Q/K向量MSE与实际竞争作用不等价；保留真实历史V仍是特权条件输入，输出闭合失败。'),
        ('2908', 'source_relay_v1/v2：查询条件与内容接力',
         '修复固定形状同前缀测量，用查询竞争损失拟合路由，分别使用真实V和预测V。',
         'V2实际发现768、确认1152条；确认两个头写入MSE相对原生岭模型降62.75%/35.99%。预测V的写入R降至−0.0666/−0.1281；全部自主递推top1一致率0。',
         '因事前功效检查缺失停止，只有探索结果，第二面板未完成；常量V也能保持答案，不能声称替换特异性。'),
        ('2912、2915、2918、2921–2927', '来源图谱保真、提示词与Q/K/V分工',
         '比较宽状态编辑与窄读取编辑，再正交检查语言、答案提示、问句后缀及Q/K/V。',
         '2912复核名字编辑40415次，图谱一致率98.33%，身份基线70.45%；R位置9216次替换全部保留原选择。2922统一Answer时英文/中文H整体复现86.36%/66.14%，V-only82.45%/62.59%，K-only接近0。',
         '一致率是原模型编辑诊断、受限四候选预测，不是从文本独立生成。提示与后续Q显著影响路径适用范围。'),
        ('2920、2923', 'recursive_branch14_v6：递归关系候选',
         '检验B26/U13099、B19/U64的关系解释在新词、新句式、新领域是否同向且胜过强基线，并回溯B39→B36。',
         '6144条、1024家族。两个候选均未过三类留出联合门；B26只有新句式一格过，B19部分反向；六项G预测均不如前缀平均embedding；两条回溯链未联合通过。',
         '这是具体解释、位置及算法的负结果；不能据固定快照推断训练期递归形成，也不否定所有分支复用。'),
        ('2924', 'branch_reuse14_v3：跨分支共同响应与三层组合',
         '对数值、标签、坐标分支检验同一单元组响应、直接/组合预测，以及原模型干预后果。',
         '1536条、384家族；坐标行为门失败，数值/标签部分准入。B19预选组7/8、B39组6/8有共同正响应，但完整联合门未过，有信息的16个因果预测格0个通过。',
         '保留共同响应及B36原生32单元局部评分作用；没有直接证明逐层产生更复杂特征。'),
        ('2928–2934', 'B36固定32单元与求值—匹配—代码读出',
         '控制标题/答案编码，干预固定组，比较足够状态预测与局部Jacobian传播；分解自由生成各环节。',
         '完整h37根预测13/14过门，加入前缀均值仅1/7有≥10%增量。改变ABC/XYZ可改变干预方向；2932的96家族672次生成只有10/14格通过联合行为门；2933的15内部条件均未改变184个确认视图首选。',
         '可计算扰动传播与语义机制分别成立与否；答案高余量、格式和停止问题限制正确率检验。'),
        ('2936', '非算术条件链与三类原生候选',
         '用条件链、保持/更新/报告分解检验候选关系响应是否超过词汇与embedding。',
         '给定真值保持可384/384，组合仅275/384与341/384；三类候选均未达晋级条件，匹配原模型消融未执行。',
         '给出中间真值不能代替模型自主形成与保持的验证。'),
        ('2937–2939', '自生成State的K/V、源位置与读取时刻',
         '同次自然生成分支内替换状态词或缓存V，以等幅控制、时刻/位置分离和原生恢复检验后继使用。',
         '2937两任务仅换V的目标改码755/768、761/768，等幅对照均0/768；自然回放1536视图全部通过。后续冻结4B头L21H19/H25与L20H5并分离读取时间。',
         '是受控范围内较强因果拼图；不证明所有推理依赖同一外显State，GLM格式门失败不能外推。'),
        ('2940–2944', '注意力剂量、动态Q与BF16可达性',
         '匹配实际注意力加权写入量，分开动态寻址、输入舍入、输出舍入和头路线选择。',
         '2940单头State−Note原始差在匹配后接近0；2942动态Q能复现主要注意力变化，192单元优于随机但完整覆盖仍未过。2944同留出14B旧路线原生投影后仅1/96家族可达，新路线恢复自然Q后96/96可达。',
         '数值门失败可来自剂量、注意力失衡及舍入，不能直接解释成语义机制不存在；可达不等于行为机制确认。'),
        ('2945–2955', '14B自产状态、事实、码本与最终答码',
         '区分首次State前E、已生成State位G、最终Code查询C，做来源阻断、K/V分离、恢复、真实换词和等长插入。',
         '2947–2949多条整链未过；2950的G事实+码本联合阻断、H18自产槽恢复等达到该支整链门，而首次真实换词链未过。2952联合阻断改变182条实际答码。2954原生增强预测未超过强文本/边际基线。',
         'G内容形成和E实际发词是不同命题；跨阶段最优指标不能拼成一次全链通过。2955等长插入仍无Receipt胜State，不支持简单距离阈值。'),
        ('2957、2959、2961–2962', '自然记录更新与授权最小对',
         '正交保留旧值/采用提议/码本读出，再替换批准拒绝词及记录设置。',
         '早期版本保留条件失败；2959试跑双侧联合85/96，行为门未过。2961词表与设置存在不同失效方向，冻结读出和新机制干预未进入后续。',
         '授权词效应、收件人与码本方向仍需拆分，不把方案或未执行阶段算成机制结果。'),
        ('2965、2967–2969、2971', '独立行号查表与B26H36紧凑预测',
         '独立设置行号、名字、来源和正文，先检验可读性，再用原生坐标预测QKV、写入与下游MLP，最后匹配阻断恢复。',
         '2967发现1024/1024、确认2048/2048行为正确；2968选定32坐标两位联合100%，随机最佳93.75%，未达10点特异优势。2971原生128坐标路由94.48%，写入/下游相对随机误差改善37.95%/41.45%，额外128家族因果预测方向96.09%。',
         '自然选择和局部预测有进展，但使用操作前真实内部状态与他头背景，尚非从文本独立模拟。'),
        ('2964、2970、2972–2979', '四类范式融合竞争与跨句式路径',
         '用同家族因子设计比较加法/二阶融合，逐头、残差窗口和MLP路径干预，配平位置、距离与范数。',
         '2972的1011条件二阶KL由0.9065降0.2607，1111有125/128家族变差；2973条件化规则总体KL0.07124，两基线0.20657/0.48918。候选路径跨句式方向反转，2979判定固定残差路径不能作为共同机制。',
         '受限条件的二阶统计收益不是普适融合律；格式开头不能代表内容来源与语法完成。'),
        ('2980–2981', '独立档案样式、内容和指定来源的自由生成',
         '2×2×2交叉档案代码/叙述样式、载荷与指定来源，从空回答自由生成并分开评分对象、数字和Python格式。',
         '4096个确认回答；敏感性口径两套新句式的代码相对叙述档案劫持效应为−19.92%、−7.42%，预设正向门不支持；锁箱128家族只哈希未运行。',
         '没有把旧格式干扰推广为普遍内容劫持；本轮未新增B26/H36因果测量。'),
        ('2982–2983', 'B26H36身份、印刷ROW与显示位置冲突',
         '用独立交换让姓名、标签和物理行位指向不同记录，按规则观察原生路由，比较Q/K坐标预算。',
         '2982身份/行规则在冲突中能改变读取方向。2983范围收窄到BEGIN下首行；确认自然姓名/标签金记录最高质量约99.02%/99.80%。512坐标Q/K方案虽胜随机与均值，完整准入仍失败，没有分支进入正式原模型干预。',
         '首行邻接边界仍需正交，不能外推末行与中间行；静态方差选列尚未解释条件选源的完整计算。'),
    ]

    table = '| Phase | 重要测试 | 原理 | 保存的关键结果 | 严格判断 |\n| --- | --- | --- | --- | --- |\n'
    for item in tests:
        table += '| ' + ' | '.join(item) + ' |\n'
    group_table = '| 组别 | 删除路径数 | 删除逻辑GiB | 去重文件GiB | 保留逻辑GiB |\n| --- | ---: | ---: | ---: | ---: |\n'
    for label, group in grouped.items():
        group_table += f'| {label} | {group["paths"]} | {group["logical"]/C.GIB:.6f} | {group["unique"]/C.GIB:.6f} | {group["kept"]/C.GIB:.6f} |\n'
    run_table = '| 结果目录 | 删除原场文件 | 删除逻辑GiB | 保留逻辑GiB |\n| --- | ---: | ---: | ---: |\n'
    for row in json.loads((OUT/'directory-summary.json').read_text(encoding='utf-8')):
        if row['delete_files']:
            run_table += f'| `{row["run"]}` | {row["delete_files"]} | {row["delete_logical_bytes"]/C.GIB:.6f} | {row["keep_bytes"]/C.GIB:.6f} |\n'

    formulas = r'''
以下区分原架构恒等式、提取算法、测量定义和候选接口，不把它们都称为新定律。

原生注意力与MLP计算（示意式；具体QK归一化、GQA与RoPE以各冻结实现为准）：

$$
\alpha_{ts}=\operatorname{softmax}_{s\leq t}(q_t^\top k_s/\sqrt{d}),\quad
d_{h,t}=W^O_h\sum_{s\leq t}\alpha_{ts}v_s,\quad
A_{l,j}=\operatorname{SiLU}(G_{l,j})V_{l,j},\quad m_l=W_l^{down}A_l.
$$

这里t/s是查询/来源token位置，h是头，l是块，j是MLP单元；头内Q/K坐标与残差坐标、MLP单元、标量权重不是同一索引空间。Phase2906在发现集按实际QK点积平方均值选32坐标，随后用选中项重新算softmax并接原生V/O；Phase2971按发现侧输入方差与投影列范数选原生列。它们都使用内部状态，不能称从文本独立推导。

局部误差改善与行为量的定义：

$$
R_H=1-\frac{\sum_{i\in H}\|y_i-\widehat y_i\|^2}{\sum_{i\in H}\|y_i-\overline y_D\|^2},\qquad
m_i=\ell_{i,correct}-\max_{a\ne correct}\ell_{i,a},\qquad
\Delta m_i=m_i^{edited}-m_i^{clean}.
$$

D为发现集，H为留出集，y是被预测组件的完整原生输出，均值仅由D估计。R是相对该基线的误差改善，不是全语言破解率；分母退化时不报告恢复。评分变化、真实argmax、完整生成与无关任务损伤分别记账。全词表保真使用D_KL(p_model∥p_prediction)，不能由四候选排序替代。

二因子交互与局部扰动传播：

$$
I_{12}=z_{11}-z_{10}-z_{01}+z_{00},\qquad
\delta h_{40}\approx J_{39}J_{38}J_{37}\delta h_{37}.
$$

I是指定坐标或logit尺度的析因差分；z为该条件响应，不是softmax概率上的任意“融合份额”。J是沿原生参考状态的局部导数，传播式是有限范围近似，既不解释语义命名，也不保证自由多步闭合。自产State任务里的s'=T(s,c)、code=π(s')首先是外部任务真值，须另有内部与因果证据才能称模型实现了该抽象算法。

RDC和全局图谱沿用Phase2822的接口，**本轮没有公式改进或新已证数学规律**：

$$
\mathcal T^D_{\ell,\tau}:\bigl(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p)\bigr)
\rightharpoonup\mathbf W_{\ell+1}(p),\qquad
\operatorname{Atlas}_D=(G_{external},G_{internal},E^D_{association}).
$$

p是因果可见前缀，L是类型化语言描述，粗体W是析因响应束而非模型权重矩阵，X是指定边界状态场，D与τ限定模型/材料和条件角色。箭头是部分定义的研究接口，不代表已经有独立可执行的统一机制。图的观察边、留出预测边、参数来源边与原模型干预边分别保留证据类型。
'''

    sections = f'''
**任务、覆盖与证据状态。** 本轮为历史研究审查和用户授权的数据清理，无模型加载、CUDA前向、训练、重采或正式证据晋级。读取并建立索引的原始11月记录为{len(plan['monthly_index'])}条，时间2026-11-03 01:30至2026-11-18 19:13，Phase2760–2983；原文件未找到11月1–2日记录，2775存在重号、2773没有对应标题，均按原样索引，不修改历史编号。`november-index.csv`逐项保存标题、原行号、正文字符数与SHA-256。这些条目包含工程、方案、复核、模型实验，不可当作224次独立科学实验。下表为大任务主线及重要结果；历史数字来自备忘录和对应保存报告，本轮没有重新运行历史模型确认。

主研究模型身份沿用历史冻结记录：Qwen3-4B BF16 `6be1034f0b55a6dfc9383b4f155a736433d822a21f3ab6178012c2c2e14739ad`，Qwen3-14B BF16 `1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce`，GLM4-9B BF16 `64baf4f7e4fb31aeb689b7cb8a63b2484be0aa1ca5ed7108f3c21f83c5879b3f`。不同运行、精度、题面和行为门不能合并成一个解释率。本轮只核对保存的身份材料与原场字节，没有重新哈希所有模型权重或独立节点复跑。

**重要测试、原理和结果。**

{table}

**数学对象与公式。**
{formulas}

**严格审视与路线判断。** 最强的增量是从“能读出信息”推进到部分可分离的条件选源、V/O写入、自产记忆读取与匹配恢复；仍有不少可保留的局部正结果。最大的缺口不是档案数量不足，而是很多预测依赖真实晚层状态、真实V、已给出的语义标签或其余原模型背景；缩成自己的连续更新后常失稳。强词汇/四状态/文本规则与随机原生组经常已经很强，不能用弱基线制造机制增量。大量干预有评分效应却没有真实选码变化；高余量、冗余、补偿和剂量均是竞争解释，当前数据未全部区分。行为门失败、对照失配、事前功效缺失、后验筛选、未完成面板和小样本停止各有不同含义，不混写为机制反证或复现成功。公式、权重复算及技术审计通过不能消除这些科学限制。自然语言、一般知识网络、长程语法、深组合、跨模型与独立节点的共同机制尚未闭合。

**智能理论的关键洞察与下一大阶段。** 值得检验的第一性命题是：固定共享参数在因果可见历史下，通过动态寻址、内容变换和状态写入复用有限运算。要解释可扩展组合，必须同时预测“取谁、取什么、写到哪里、以后如何使用”，并给出适用范围。单个坐标不是预设概念本体；同索引跨层/跨模型也不预设同功能。有限参数与无限长度的计算复用提供组合能力的可能性，不保证无限深度正确，更不能从局部LLM结果推出人脑或AGI已经解释。

下一阶段组织为“条件选源—内容更新—语法输出的统一可执行机制验收”，不是再逐个新增关键词小测试：

1. 建立同一套家族材料，将知识资料选择、推理中的保留/更新、语法与输出约束交叉；明确拆分身份、印刷标签、物理行位、句式、答案重编码和组合深度。先用发现材料校准完整行为与功效，冻结独立家族、比较预算和留出；保留本月失败作为已暴露回归材料。
2. 在相同信息预算下比较文本/位置/词汇强基线、原生坐标与MLP组、适用的字典/电路。围绕真实QK竞争间隔和完整V/O写入而非只看向量MSE选候选；分别报告状态预测、全词表KL、干预后果与成本。预先限定迭代与例外复杂度，无增量时停止该模型类或改变状态边界。
3. 对过门候选贯穿embedding、来源形成、自产记忆及最终读出的整链验收；逐步去掉真实未来V、真实中间状态与人工语义标签，进行原模型匹配阻断/恢复、跨家族新组合、真实自由生成与无关任务检查。范围内成功后再做独立节点和跨模型复现；形成机制的命题另需训练快照。当前任务不执行这些未来GPU工作。

**清理授权、磁盘检查与执行范围。** 用户明确选择“只保留脚本、题目、结果和复核记录，删除可重采原场”，并授权本任务在低于20GiB保留线时先写不超过10MiB维护审计。初始D盘约19.51GiB；该例外仅用于本次审计与清理，不改变`deploy/storage_policy.json`、不放行模型实验、不涉及C盘项目写入。此前12月10/13日的原场复查日期按本次用户选择终止对应保留用途。原始CSV只用来限定A1/A2的26个目录，**没有采用其粗略clean字段按扩展名删除**：其中还包含冻结源码ZIP、拟合器和关键复核数组。

本轮仅按已核实的生产器与数据角色删除自然前向的大体量G/V/A、HiddenState、Q/K/V、回放输入及逐块采集分片。保留全部原有脚本/源码归档、题目/分词身份、结果JSON和日志、拟合系数、逐题预测、热力图数据、参数账本、干预结果和反例；保留source目录部分MLP A复核数组、BF16匹配误差与权重统计，保留自产State的Value切片、B36接续检验和旧执行失配的少量证据。`qwen14_mlp_b28_b39_v2`及客户端共享原生文件没有清理。原目录保留，没有递归删除或跨shell拼接路径；13个实际受影响目录新增`RAW_STORAGE_STATUS_20261118.json`，原清单和旧审计不改写。

旧表的A1 80.66、A2 80.68采用GiB（2^30字节）口径，不是十进制GB。两组全部张量共161.341431GiB；若全部删除，硬链接关系下也不能回收同等物理量，本轮更未删除全部张量。实际执行如下：

{group_table}

删除逻辑总量{plan['delete_logical_bytes']:,}字节（{plan['delete_logical_bytes']/C.GIB:.6f}GiB），唯一文件大小{plan['delete_unique_file_bytes']:,}字节（{plan['delete_unique_file_bytes']/C.GIB:.6f}GiB）。A1约1.284989GiB为组内硬链接重复路径，未重复计作回收量；A2目录外约5.896214GiB的共享文件及其路径保持。删除前后D盘空闲由{receipt['free_before_bytes']:,}增至{receipt['free_after_bytes']:,}字节，**实测净增{receipt['observed_free_increase_bytes']:,}字节，约{receipt['observed_free_increase_bytes']/C.GIB:.6f}GiB（{receipt['observed_free_increase_bytes']/1e9:.3f}GB）**。删除后立即记录空闲{receipt['free_after_bytes']/C.GIB:.6f}GiB；最终报告等小文件略有占用。空闲变化含文件系统分配粒度，也可能受其他活动影响，不拿它代替逐文件账本。

{run_table}

**复核、复算入口与数据损失边界。** 删除前对26目录全部7,494个原有文件、174,306,941,512字节建立新SHA-256、大小、修改时间、卷/文件身份及硬链接数记录；删除对象另存NPY头/NPZ各成员形状、精度及首64数据字节（只是身份线索，不能恢复全场）。已有历史manifest可对应的2,039个删除路径哈希全部吻合；390个路径未找到匹配历史条目，明确登记缺覆盖，不补造历史证明。所有保留ZIP做成员CRC检查。所有删除操作前再次核对文件身份和角色；存在目录外硬链接的候选不准入。清理后2,429条计划路径均不存在，5,065个保留文件、20,716,126,693字节重新SHA-256全部一致，验证通过。此为执行与存储审计，不增加科学样本或独立模型复现。

完整明细：`tests/glm5/result/november_research_storage_audit_20261118/plan.json`、`directory-summary.json`、`historical-manifest-crosscheck.json`、`deletion-receipt.json`、`verification.json`与`progress.jsonl`。清理代码为`tests/glm5/november_storage_cleanup_20261118.py`，4项边界单测通过，覆盖原场准入、源码/结果/最小证据保护、模型/临时目录拒绝和26目录身份；测试输出有CSV只读文件的ResourceWarning，非断言失败，进程退出后句柄释放，未改冻结清理器。实际命令依次为项目venv下`python -B -u ... plan --maintenance-exception-10mib`、`execute --maintenance-exception-10mib`、`verify`，单测用`python -B -m unittest discover -s tests/glm5 -p test_november_storage_cleanup_20261118.py -v`。哈希阶段约182.2秒，保留文件复核约29.9秒；日志每25秒或逐批逐行刷新，启动首条进度已确认。此次无GPU占用与共享显存增长测量，不虚填模型资源。

原场已按用户选择删除，**没有另建完整原场备份**；“可重采”不等于当前可直接下载恢复，也不等于重采成本为零或跨软件版本逐位相同。保留的JSON/NPZ结果可检查原判定与反例，新的全坐标、全单元或改方法分析需要重采。恢复以各目录的冻结source ZIP、plan/prereg、materials、模型/tokenizer快照、种子与运行环境为准；递归/B36可从`tests/glm5/recursive_branch14_README.md`、`b36_composition14_README.md`开始，来源系列从`source_binding_design.py`及`source_formation_campaign.py`、`source_relay2_campaign.py`追溯。先恢复冻结源码并检查输出路径，在新的D盘目录登记预算及新运行身份，不能直接覆盖旧结果，也不能将旧确认材料重新叫作盲测。本轮不实际启动恢复，故精确重采和客户端运行行为未重新验收。客户端共享文件哈希保持，不声称重新完成浏览器/服务器验收；没有上传、部署、改正式中心或修改其他任务。

**通俗结论。** 本月已经找到一些“查哪条记录、把内容送到哪里、后来怎样读回来”的局部工作过程；尚未得到能解释整个语言系统的统一齿轮图。这次保留下实验题目、答案、失败案例和可追溯做法，去掉大部分可重新采集的逐层录像，释放约141.76GiB；以后若要从录像换角度分析，需要重新采集对应部分。
'''
    title = f'## Phase {phase}: 十一月研究全记录审查与A1/A2可重采原场清理 [{timestamp}]'
    report = '# 十一月研究审查与原场清理报告\n\n' + f'审计时间：{timestamp} America/Chicago；备忘录追加Phase：{phase}。\n\n' + sections.strip() + '\n'
    write_new('REPORT.md', report)
    structured = {'schema': 'november_research_review.v1', 'timestamp': timestamp,
                  'kind': 'historical_review_and_authorized_storage_maintenance', 'model_runs': 0,
                  'original_monthly_entries': len(plan['monthly_index']), 'memo_phase': phase,
                  'source_memo_before_sha256': plan['memo_before_sha256'],
                  'tests': [dict(zip(['phases', 'topic', 'principle', 'historical_result', 'limits'], t)) for t in tests],
                  'storage_groups': dict(grouped), 'deletion_receipt_sha256': C.sha(OUT/'deletion-receipt.json'),
                  'verification_sha256': C.sha(OUT/'verification.json'), 'scientific_conclusions_changed': False,
                  'new_mathematical_law': False}
    write_new('report.json', json.dumps(structured, ensure_ascii=False, indent=2)+'\n')
    write_new('engineering-checks.txt',
              'Command: .venv/Scripts/python.exe -B -m unittest discover -s tests/glm5 -p test_november_storage_cleanup_20261118.py -v\n'
              'Observed: 4 tests passed. Boundary cases protect frozen ZIPs, fitted predictors, intervention outputs, minimal evidence and client files; reject model/temp paths; check 26 approved directory identities.\n'
              'Observed warning: unclosed read-only CSV TextIOWrapper ResourceWarning in originals(); not an assertion failure. No model execution.\n')
    entry = ('\n\n'+title+'\n\n'+sections.strip()+'\n').encode('utf-8')
    check_write(C.MEMO, len(entry), operation='强制尾部追加研究审计')
    # Refuse to append stale bytes if another writer changed the memo during report construction.
    assert C.MEMO.read_bytes() == before, 'Concurrent memo writer; report saved, append stopped'
    with C.MEMO.open('ab') as stream:
        stream.write(entry)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    after = C.MEMO.read_bytes()
    assert after[:len(before)] == before and after[len(before):] == entry
    result = {'schema': 'append_only_memo_verification.v1', 'created_at': C.stamp(),
              'phase': phase, 'title': title, 'memo_path': str(C.MEMO),
              'original_prefix_bytes': len(before), 'original_prefix_sha256': hashlib.sha256(before).hexdigest(),
              'original_prefix_preserved': True, 'append_bytes': len(entry),
              'append_sha256': hashlib.sha256(entry).hexdigest(), 'memo_after_sha256': hashlib.sha256(after).hexdigest(),
              'report_sha256': C.sha(OUT/'REPORT.md'), 'free_bytes': shutil.disk_usage(C.BASE).free,
              'source_lines_of_key_original_phases': {str(n): source_lines[n] for n in [2906, 2907, 2908, 2923, 2924, 2928, 2937, 2950, 2971, 2983]}}
    write_new('memo-append-verification.json', json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    manifest = {p.name: {'bytes': p.stat().st_size, 'sha256': C.sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()}
    write_new('delivery-manifest.json', json.dumps({'files': manifest, 'self_excluded': True,
                                                 'cleanup_script_sha256': C.sha(ROOT/'tests/glm5/november_storage_cleanup_20261118.py'),
                                                 'closeout_script_sha256': C.sha(Path(__file__))}, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'phase': phase, 'prefix_preserved': True,
                      'report': str(OUT/'REPORT.md'), 'free_GiB': shutil.disk_usage(C.BASE).free/C.GIB,
                      'released_GiB': receipt['observed_free_increase_bytes']/C.GIB}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
