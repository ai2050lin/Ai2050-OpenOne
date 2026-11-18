"""Append-only scientific account, after all locked stages and visual/identity QA."""
import argparse,hashlib,re,sys,zipfile
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def pct(x):return f'{100*x:.2f}%'
def num(x):return f'{x:.4f}'
def ci(x):return '['+', '.join(num(v) for v in x)+']'

def archive():
    if (D.OUT/'source-archive.json').exists():
        old=D.read(D.OUT/'source-archive.json');assert D.sha(D.OUT/'source-and-protocol.zip')==old['sha256']
        assert all(D.sha(D.ROOT/name)==digest for name,digest in old['files'].items()),'Archived sources changed; preserve and version the bundle before retrying'
        return
    files=set((D.ROOT/'tests/glm5').glob('source_relay*.py'))
    files.update((D.ROOT/'tests/glm5').glob('test_source_relay*.py'))
    for name in ['plan.json','internal-freeze.json']:
        files.update(D.ROOT/p for p in D.read(D.OUT/name)['code_sha256'])
    for name in ['deploy/storage_policy.py','deploy/storage_policy.json','shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/natural_register_runner.py']:
        files.add(D.ROOT/name)
    files.update(D.OUT.glob('*.json'))
    for name in ['padding-diagnostic.json','prefix-numerics-audit.json','measurement-stop.json']:
        path=D.OUT.parent/'source_relay_v1'/name
        if path.exists():files.add(path)
    path=D.OUT/'source-and-protocol.zip';assert not path.exists()
    D.check_write(path,128<<20,operation='来源接力可复算源码归档');entries={}
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            name=p.relative_to(D.ROOT).as_posix();z.write(p,name);entries[name]=D.sha(p)
    with zipfile.ZipFile(path) as z:assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in entries.items())
    D.save('source-archive.json',{'files':entries,'sha256':D.sha(path),'scope':'Actual frozen sources, downstream reporting, protocols and numeric failure history; original native arrays and model weights remain separate.'})

def main(check_only=False):
    assert not (D.OUT/'closeout.json').exists(),'Closeout already recorded'
    D.verify_internal();r=D.read(D.OUT/'report.json');audit=D.read(D.OUT/'audit.json');assert audit['technical_identity_checks_passed']
    stopped=D.read(D.OUT/'measurement-stop.json');assert not audit['governance_passed']
    assert D.read(D.OUT/'visual-qa.json')['passed']
    D.read(D.OUT/'figure-manifest.json');cleanup=D.read(D.OUT/'cleanup.json')
    memo=D.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';before=memo.read_bytes()
    phase=max(map(int,re.findall(rb'^## Phase (\d+):',before,re.M)))+1
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    heads=', '.join(f'B{x["block"]}H{x["head"]}' for x in r['dual_panel_primary_pass_heads']) or '无'
    lines=[f'\n\n## Phase {phase}: 查询条件预测与内容接力的探索结果——固定形状校准、确认侧结果及运行准入缺口 [{stamp}]\n']
    lines.append('**执行与证据状态：本轮因事前功效检查缺失而停止，不能登记为合规的两面板确认。** 已保存发现768条、确认1152条完整行为及内部结果；新句式1152条中日志显示384条生成循环已完成，逐条结果未落盘，尚未采集内部状态。全部该面板材料记为已暴露，不填造行为统计、不把中止当成候选被否定。已完成结果按探索证据保留。\n')
    lines.append(f'两面板共同通过的头：**{heads}（第二面板未完成）**。确认侧的数值门比较保留真实当层V时的路由/写入误差，不代表独立内容预测、完整递推或机制必要性。\n')
    lines.append(r'''### 运行准入漏项与停止处置

本代理在开始本轮时漏执行了AGENTS.md第13节要求的“预期效应量→所需n→检验力≥0.8”检查。冻结plan的power字段只有样本数及bootstrap次数，没有预期效应量、所需样本数、计算器输出或功效通过判定。收尾复核发现后立即写入stop-request，正在运行的新句式行为在下一进度检查处退出；驱动器退出码1，没有继续原场、预测和干预。这是执行治理错误，不能用结果显著或事后计算补成事前合规。

主预测指标为连续写入误差，需另做与家族相关结构一致的功效设计；AGENTS中的配对二值计算器适用于准确率变化，不能把1152个相关查询冒充1152个独立家族。后续运行须在采集前同时明确连续主指标与二值干预指标的最小有意义效应、样本量和检验力；不为跨过门槛事后挑选大效应。

另发现结果provenance虽有git、seed、模型名称和联合快照哈希，但未使用第13节规定的model_path/model_revision完整字段与写入入口；实际路径可由冻结源码source_binding_design.MODELS追溯，联合快照见qwen14-identity.json，原结果不改写，不因此声称格式合规。总审计passed=False；已完成产物的身份与算术检查另记通过。停止记录、错误日志及原冻结版本完整保留，后续必须新版本、新材料和事前检查，不直接恢复这个运行。
''')
    lines.append(r'''### 当前总体成果、问题与本轮边界

已有研究从共享输出读出和可操纵的状态，推进到较早状态对部分后续Q/K及头写入的留出预测，并把若干局部计算追溯到原生残差坐标、头和权重。三图谱已有受控外部关系、原生响应记录和部分有类型的预测/计算来源边，尚未形成覆盖知识网络、推理与语法的通用关联算法。

上一轮硬伤包括：局部预测依赖真实未来V；新句式的Q向量预测退化；连续更新无法复现全词表输出；Qwen的MLP扰动匹配未过门。原生权重复算是计算来源证明，不是独立行为预测。来源屏蔽还有替代解释：模型可能改读错误记录，或信息已经转移到问题、指令与助手前缀。

本轮在本机Qwen3-14B原生BF16、CUDA上，执行了行为、发现原场、冻结拟合和确认侧，随后在新句式行为阶段停止。未修改正式中心、上传、发布或提交；不是dist_protocol v3正式分片，没有自动证据晋级。GLM拓展未执行，未执行的跨模型检验不记为失败或复现。

### 先修测量：V1停止与V2的新材料

V1发现侧的相同记录前缀K出现不一致。CPU复查发现：相同完整长度和批宽的1668次对照完全一致；完整长度不同的2136对出现约0.8%的平均相对K误差，最大约1.7%；另有36对在批宽变化时不一致。它指向运行形状/补齐相关的数值混杂，不能解释成decoder提前读到了未来问题，也没有在此定位到唯一CUDA内核原因。

原V1在确认行为开始、尚未完成首个完整前向时停止，所有已执行材料记为暴露。保留原日志、原场、拟合和错误，不以修改门槛补过关。随后用64组真实同前缀样本、128条输入执行独立数值对照：左补齐64/64对不同；固定全局宽度右补齐0/64对不同；删除未来问题后的右补齐前缀与完整输入的记录K也完全一致。该对照中六选一和全词表首选变化均为零，说明答案稳定不保证内部微小差异可解释。

V2排除全部V1来源名/实体名，重新生成完整三划分，保持候选头、预测目标、400步训练与门槛不变；固定右补齐执行宽度，按各句真实末位置读出。保存时将有效数组右对齐，仅为兼容分析索引；实际CUDA执行仍是右补齐。每个原场面板还必须通过所有同前缀记录K的逐位相等检查，及行为/原场基线精确重放。V1发现结果已被看见，V2不能称为完全没有前期探索的理论发现。

该数值问题使先前很小的条件效应需要专门复核，但不自动推翻所有历史行为或内部结论。校准只覆盖当前模型、精度、运行器与固定形状，其他模型和环境需另检。

### 材料、采集和冻结规则

- 计划发现64家族×12查询=768条；确认和新句式各96家族×12=1152条，共3072条；实际完整保存1920条。3来源×2实体×2呈现顺序；实体和来源顺序分别随机化。来源名与实体名跨划分不交叉，并排除前两研究阶段及V1已用名字。答案词汇和基础账本构式已有先前暴露；新句式仅指新的查询表述，不能称为全部文法或语义全新。
- 六候选值均在输入中，测试受控检索与绑定，不测试新增世界知识。行为门对所有S×E×O、句式、距离及code/tag条件逐格核对，格内n≥64、六选一与归一化完整生成各≥90%，合并各≥95%。最多生成4个token并要求终止，归一化只处理空白、大小写与末尾句点，严格格式另记。家族2000次bootstrap；操作性点估计门不等于置信下界保证。
- 固定hs20和两头B28H21、B29H10，来自前一阶段候选，不是本轮全模型重新发现。Qwen 40个Q头/8个KV头，正确KV索引分别4、2。采集4读出×41残差边界×5120全部坐标、完整前缀hs20、两头Q/K/V/注意力/真实头上下文，以及5角色在hs20–30和attention后的残差。BF16按uint16原位存储。
- 查询来源、实体两个中途读出仅作描述；主预测在完整查询已可见的最终读出。记录K在问题出现之前按因果构造不能依赖未来问题。保存各角色位置不意味着它们都适合读取所有变量。
- 主要对照为native128、random128、PCA128和embedding128；变换、坐标、中心和拟合系数只由发现侧确定。PCA虽同为128潜变量，却需要稠密输入变换，实际计算及参数预算更高；不能称为总预算完全相同。没有自然语料SAE/transcoder对照。

### 数学对象：拟合预测、原生运算和恒等式分开

令 \(x_s=h_{20,s}\) 为较早原生状态，\(I_q,I_k\) 是发现侧选择的128输入坐标，\(z\) 是发现侧中心与尺度。回归预测器为

$$
\widehat q=a_q+B_qz(x_t[I_q]),\quad
\widehat k_s=a_k+B_kz(x_s[I_k]),\quad
\widehat v_s=a_v+B_vz(x_s[I_k]).
$$

native输入按标准化输入与目标协方差平方和选择，随机输入和PCA为比较方法；岭系数λ=1，截距不惩罚。Q/K是原生QK归一化后、RoPE之前的向量。学习到的B不是原模型标量权重，不是已推导出的语言定律。

用原生RoPE和显式BF16舍入得到预测注意力与写入：

$$
\widehat p_s=\operatorname{softmax}_s((R_t\widehat q)^\top R_s\widehat k_s/\sqrt{128}),\qquad
\widehat w=W_O\sum_s\widehat p_s\,\widetilde v_s.
$$

\(\widetilde v_s=v_s\) 是保留真实未来内容的局部诊断；\(\widetilde v_s=\widehat v_s\) 才是不再读取未来真实Q/K/V的分支，但它仍观察原模型完整hs20前缀。past分支删除末位置V并重归一化，真past-V仍可能含问题/助手前缀已经形成的答案信息；删除self不等于移除全部特权内容。五个混合分支分别只恢复records、question、instruction、assistant_tail、other一组的真实V，最终位置始终预测。

查询条件模型从岭解起步，固定400步Adam、lr=0.003、batch=32，只更新回归系数，训练损失

$$
\mathcal L=\operatorname{KL}(p\|\widehat p)+0.1\,\frac{\|w-W_O\sum_s\widehat p_s v_s\|^2}{E_D\|w-\bar w_D\|^2}.
$$

训练中的真V与真注意力是监督信号。推断时Q/K仅由hs20产生，真V成绩和预测V成绩分别记账。全坐标写入误差可用 \(G=W_O^\top W_O\) 复算；审计另用全部5120输出坐标核对。定义 \(R=1-\sum\|w-\widehat w\|^2/\sum\|w-\bar w_D\|^2\)，允许为负，它不是整个模型的破解比例。

冻结主门：每个固定头在两留出面板均相对native岭模型降低注意力KL与真实V写入MSE至少5%，写入配对家族增益95%区间下界>0，两个头的单侧符号置换p作Holm校正后<0.05。其他多项比较为预定次要或描述性指标，不能挑选其中成功项替代主门；每个面板各自作两头校正，双面板共同满足是交集要求。

原生K的局部参数账本固定真实晚层Q、输入RMS尺度及K归一化尺度。若 \(h_b=h_{20}+\sum_{j=20}^{b-1}(\Delta a_j+\Delta m_j)\)，由原生K投影、归一化权重与位置旋转得到一个当次读向量g，则

$$
s_s=g_s^\top h_{20,s}+\sum_jg_s^\top\Delta a_{j,s}+\sum_jg_s^\top\Delta m_{j,s}+\epsilon_s.
$$

这里的残差增量来自存储边界之差，含BF16舍入；ε显式吸收归一化、投影、RoPE和复算算术的剩余误差。加上ε后精确相加是定义性恒等式，不是发现新定律。保留128×5120全部K权重及所有坐标贡献，不以Top-K删掉背景；但未对每个标量参数做因果干预，不能把账本某列命名成语义齿轮。

连续更新从真实hs20末状态开始，以自身预测状态输入下一个128坐标岭变换。比较直接预测下一状态、预测增量、以及拟合去掉真实两头写入的剩余量后加回预测写入。训练中可用真状态监督，留出递推不反馈真实晚层末状态；两头预测内容由初始hs20前缀计算，不再随中途预测状态重估。最后经原norm/unembedding比较全词表KL、首选一致率与六选一。它尚不是多token自回归闭合。

原模型干预从第20块起仅屏蔽最终查询到不同位置组的注意力，覆盖全部后续头；另只替换B28H21/B29H10的头上下文。位置屏蔽改变softmax竞争、没有匹配剂量，支持的是受限路径操作的总效应；两头替换保留其余原模型计算，成功不能证明两个头足够实现整个任务。
''')
    lines.append('\n模型与tokenizer联合快照：`'+D.read(D.OUT/'plan.json')['model_snapshot']+'`。运行器和库源码身份见runtime-environment.json，模型本体采用本地原生BF16权重，无4bit量化。\n')
    behavior=[]
    for split,p in r['panels'].items():
        if 'behavior' not in p:
            behavior.append([split,p['expected_rows'],'未汇总','未汇总','未汇总','未完成',p['reason']]);continue
        g=p['behavior'];overall=g
        behavior.append([split,overall['n'],pct(overall['forced_accuracy']),pct(overall['normalized_generation_accuracy']),pct(overall['strict_generation_accuracy']),str(g['passed']),str(p['failure_counts'])])
    lines.append('### 行为准入与错误\n'+table(['面板','N','六选一','完整归一化+终止','严格+终止','过门','错误分类'],behavior))
    lines.append('\n全部失败保留；逐条输入、生成和错误类别见report.json。不准入的面板记为未做内部检验，不能填零或称为机制否定。\n')
    prediction=[];primary=[];content=[];hybrids=[];causal=[];rollout=[];pars=[];comparisons=[]
    for split,p in r['panels'].items():
        if 'forecast' not in p:continue
        for h in p['forecast']['heads']:
            name=f'B{h["block"]}H{h["head"]}'
            for method,m in h['methods'].items():
                prediction.append([split,name,method,num(m['write_R']),num(m['attention_KL']),pct(m['record_accuracy']),pct(m['source_accuracy']),pct(m['entity_accuracy'])])
            a=h['primary'];primary.append([split,name,pct(a['relative_write_gain']),pct(a['relative_KL_gain']),ci(a['ci95_family']),num(a['holm_p']),str(h['passes']),f'{h["full_replay_relative_error"]:.3e}'])
            for method in ['native','aware_native']:
                m=h['methods'][method];content.append([split,name,method,num(m['write_R']),*[num(m['content_variants'][k]['write_R']) for k in ['predV','past_trueV','past_predV']]])
            for group,m in h['memory_hybrids'].items():hybrids.append([split,name,group,num(m['write_R']),ci(m['gain_vs_all_predicted']['ci95_family'])])
        for h in p['parameters']['heads']:
            pars.append([split,f'B{h["block"]}H{h["head"]}',f'{h["key_linearization_max_relative_error"]:.3e}',f'{h["explicit_rounding_residual_score_mean_abs"]:.3e}',f'{h["with_residual_reconstruction_max_error"]:.3e}',h['prefix_invariance']['nonidentical']])
        if split=='discovery':continue
        for arm,m in p['causal']['summary'].items():causal.append([split,arm,pct(m['forced_accuracy']),pct(m['top1_agreement']),num(m['KL_mean']),num(m['margin_change']),ci(m['margin_change_ci_family'])])
        for key,m in p['causal']['comparisons'].items():comparisons.append([split,key,pct(m['mean_gain']),ci(m['ci95_family'])])
        for method,m in p['update']['summary'].items():rollout.append([split,method,pct(m['forced_accuracy']),pct(m['top1_agreement']),num(m['KL_mean'])])
    lines.append('### 局部预测：所有方法在此表保留真实未来V\n'+table(['面板','头','方法','写入R','注意力KL','记录选中','来源选中','实体选中'],prediction))
    lines.append('\n发现侧是拟合内成绩，不是推广。记录选中以六记录注意力质量最大者计，即使记录总质量很小仍会产生一个排名；必须结合report.json的完整位置组质量，不把该排名等同最终模型答案。\n')
    lines.append('\n冻结主门与配对误差：\n'+table(['面板','头','写入误差降幅','KL降幅','写入绝对增益家族95%CI','Holm p','主门','真QKV复算相对误差'],primary))
    lines.append('\nFP32注意力复算与原生BF16头上下文存在数值差异，最后一列按全坐标写入范数衡量该复算误差；数值底噪附近的微小收益不能自动解释为新的语义机制。\n')
    lines.append('### 真V、预测V和位置组信息收益\n'+table(['面板','头','方法','真V R','预测V R','真past-V R','预测past-V R'],content))
    lines.append('\n仅恢复一组真实V，其他位置与最终V都预测：\n'+table(['面板','头','恢复组','写入R','相比全预测的MSE下降95%CI'],hybrids))
    lines.append('### 原生K参数来源账本\n'+table(['面板','头','K代数最大相对误差','score数值剩余平均绝对值','加剩余后最大误差','同前缀K不等数'],pars))
    lines.append('### 原模型中的位置路径与两头替换\n'+table(['面板','操作','六选一','首选一致率','全词表KL','margin变化','margin变化家族95%CI'],causal))
    lines.append('\n预定操作间的六选一配对差，区间为比例单位：\n'+table(['面板','比较','准确率增量','家族95%CI'],comparisons))
    lines.append('### 从hs20连续预测到输出\n'+table(['面板','方法','六选一','原模型首选一致率','KL(P原∥P预测)'],rollout))
    diagnostics=[]
    for split,p in r['panels'].items():
        if 'output_diagnostic' not in p:continue
        for method,m in p['output_diagnostic']['methods'].items():
            top=m['token_counts'][0];token=repr(top['text']).replace('|','∣').replace('\n','\\n')
            diagnostics.append([split,method,m['discovery_answer_vocabulary'],m['heldout_answer_vocabulary'],m['other'],f'{token}: {top["n"]}'])
    lines.append('\n首选token去向（确认后新增的CPU描述诊断；新句式开始前冻结同一配方，不改主门、不重评分）：\n'+table(['面板','方法','发现答案词次数','留出答案词次数','其他token次数','最常见token及次数'],diagnostics))
    lines.append('\n如果首选仍落在标记符或旧答案词上，即使六候选内部的排序有改善，也不能称为回答已被恢复。这是失真表现的描述，尚未定位导致失真的唯一层或计算路径。\n')
    lines.append(r'''
### 对已完成确认侧的实质解释

1. **查询条件损失有明确的局部预测增益，但仍待合规新复现。** B28H21与B29H10相对同128原生输入岭模型的真V写入MSE分别下降62.75%、35.99%，注意力KL下降58.28%、51.13%；家族配对区间为正，两头Holm p均0.0002。原生查询条件模型写入R为0.8277、0.5997；PCA查询条件模型为0.8668、0.6773，原生坐标法并非本轮所有方法中的最佳。这支持以查询竞争和下游写入评价表示的路线，不能证明发现了独特的原生语义算法。
2. **内容预测没有跟上路由预测。** 将真V换成hs20预测V，R降至−0.0666、−0.1281，低于发现侧均值基线。只恢复记录位置的真实V即可把R恢复到约0.8277、0.5960；恢复问题、指令或助手尾部的收益较小。该结论只针对这两个头的局部写入，不能据此否定全模型其他位置的信息接力。
3. **正确记录与全部记录屏蔽的差别很大。** 原模型六选一99.74%；从块20开始屏蔽最终查询到正确记录的路径后为4.51%，屏蔽全部记录则为99.65%，屏蔽全部过去位置为16.67%。全部记录屏蔽的全词表首选一致率只有70.66%，不能称为完整输出不变。此结果支持错误记录竞争及替代读取路径的解释，但尚未定位唯一接力位置；最终位置早于块20的计算仍保留，各操作没有等剂量匹配。
4. **常量V对照阻止了过强的因果推断。** 两头换成预测V后六选一仍99.74%，但换成常量V也为99.74%，常量V的KL0.001446还低于查询条件预测V的0.002576。其他原模型组件仍在工作，所以替换后保持作答不能证明预测器的特异性、必要性或充分性。局部可预测性和当前背景下的因果必要性不同。
5. **连续输出重建仍失败。** 自主状态递推的查询条件预测V分支六选一22.31%，真past-V分支38.02%，但所有方法对原模型全词表首选的一致率都是0%。前者1152条里1132条首选为标记符 `**`，没有一次落入留出答案词表。真实V也不能独自解决其余块更新和输出校准的问题；只报告受限六选一会掩盖这一失败。

以上均来自同一确认面板，不是五项独立复现；发现侧拟合、多个指标、数值校准及技术检查也不增加科学独立样本。尚未完成新句式检验，不能把“确认侧过数值门”写成“两面板预注册成功”。
''')
    lines.append(r'''
### 严格审视：哪些推断仍不成立

1. **信息边界仍然限制解释。** 预测V去掉了真实未来Q/K/V输入，却仍以原模型完整hs20为起点；hs20本身可能已经携带部分答案信息。局部回归的系数由监督拟合得到，尚未从embedding和原生标量参数推导出关系绑定算法。
2. **局部写入、原模型替换、独立递推是三个不同主张。** 前者保留真V，第二者保留其余原模型，第三者近似剩余20块。不能把不同分支最好的指标组合成一个并不存在的完整算法。六选一改善还需同时看全词表首选、KL和误差传播。
3. **位置屏蔽不能单独证实信息接力的唯一通路。** 全记录与正确记录屏蔽的不同可支持竞争/替代路径，但所有早于hs20的计算仍保留；最终位置可能在编辑前已编码答案。屏蔽问题或助手前缀也会改变归一化与注意力分配。还需特定写入边、时间窗口和副作用对照，才能区分信息位置与实际形成来源。
4. **参数账本不等于参数机制。** 条件g依赖真实晚层Q及归一化，原生K复算和余项闭合是算术。尚未得到单参数/单MLP单元的选择性、必要性或充分性证据。本轮没有用失败的旧MLP范数门补作因果结论。
5. **统计范围有限。** 每个留出只有96个家族，家族内12条不独立，名字是构造的音节组合，词表和构式范围有限。bootstrap条件于本次冻结拟合，不能代表自然语言全体或重新训练的变动。符号置换依赖零假设下家族差值的可交换/对称条件；随机坐标只取一个冻结集合，优化只用一个种子，尚未覆盖这些选择的变异。固定候选头、hs20和运行器来自已有探索；描述性多比较的显著项不能晋级成新的预注册主结论。
6. **三引擎未齐备。** 本轮仍属于受控对照与受限原模型干预，没有自然语料SAE/transcoder、跨语言、长推理、一般语法、多token干预生成、无关任务副作用或独立节点复算。跨模型坐标不能直接对应。本轮不证明无限组合、人脑机制、AGI或某条理论路线唯一。

### 智能理论工作洞察与下一阶段大任务

值得检验的第一性原理是：有限参数通过对当前位置、历史内容和竞争关系的条件计算，复用读写运算；要解释语言组合，须同时预测“选择谁、保存什么、写到哪里、下一步怎样使用”。表示距离只是其中一项；softmax竞争和V/WO会使同样大小的Q/K误差产生不同后果。以实际计算作用评价表示是候选方法原则，不能仅凭换损失函数就宣布新数学结构。

还有一个可由算式直接推出的识别边界：softmax对所有有效score加同一个常数不变；若两组注意力之差δp满足 \(\mathbf1^\top\delta p=0\) 且 \(W_OV^\top\delta p=0\)，它们产生完全相同的头写入。因此，写入保真本身一般不足以唯一确定路由，更不足以唯一确定Q/K或一个语义坐标。本轮同时检查注意力KL和写入误差可以约束这种歧义，但不能消除全部等效实现。上述是线性代数识别条件，不是声称已经在每个样本实测到了某个非零零空间方向；有限精度执行还需检查误差容差。

下一大阶段应围绕**关系信息首次形成、跨位置接力与可执行状态更新**组织完整任务包：

- A：固定形状校准推广到全部主模型/运行器；保留相同前缀不变量及基线重放作为采集准入。复核历史小效应的数值稳健性，分别记录可保留结论和受影响范围。
- B：从本轮全场追踪来源×实体信息在hs20之前何时出现，以及如何从记录写入问题/助手前缀。通过可见性受控的逐位置预测、原生读写参数与明确时窗检验竞争解释，避免将预先给定模板目标当成发现的内部算法。
- C：在全新家族上比较native坐标、SAE/transcoder与简单文本/embedding/位置基线；同时报告总参数、计算及输入信息预算，测预测V、未见组合与完整输出。先冻结候选，再用未暴露面板检验，不将本轮已暴露材料继续称为盲测。
- D：以可实现的BF16剂量检验具体写入和MLP/标量连接，增加同来源错实体、同实体错来源、匹配随机组件及无关任务。逐级去掉真实hs20和真实历史值依赖，再扩展组合深度与多token自然生成；跨模型串行、先过自己的行为门。

本轮产物提供的是上述任务的可重算起点，不能用完成条数或热力图数量计作语言破解比例。
''')
    res=r['resources'];lines.append(f'\n### 执行、审计、保留与清理\n\n已完成产物的身份及算术检查{audit["count"]}项通过；研究准入审计未通过，audit.passed=False，自动正式证据晋级=0。CUDA峰值{num(res["peak_cuda_GiB"])} GiB，DXGI进程共享显存峰值{num(res["peak_shared_MiB"])} MiB，最低可用RAM {num(res["min_available_RAM_GiB"])} GiB。进度按批/层逐行写入progress.jsonl，包含完成/总数、耗时、ETA与内存。共享GPU租约保证与其他任务串行；等待时间不计成实际GPU计算时间。\n')
    lines.append('脚本均在tests/glm5；本轮结果tests/glm5/result/source_relay_v2，旧失败与校准在source_relay_v1。实际启动命令 `.venv/Scripts/python.exe -B -u tests/glm5/source_relay2_campaign.py --bootstrap`；采集/拟合/预测/参数/干预/递推分别由同名前缀脚本执行，逐阶段日志保留。工程测试12项通过（test_source_relay2.py，含继承的原版本检查），并通过64组真实CUDA数值校准；不重复计作独立科学样本。冻结与不可覆盖保护保留，复算须另设输出目录并核对模型、材料和源码身份。source-and-protocol.zip逐文件哈希复核；feature-cards和typed-graph区分观察、预测、计算来源与受限干预。图表有原生全坐标原值/RMS两视图及完整数组，图像布局不表示因果。\n')
    lines.append(f'本轮仅清理本轮已无后续用途的可重建文件，实际释放{cleanup["bytes_removed"]}字节，明细见cleanup.json。V1诊断原场与V2已采集原场均保留，复审日期2026-12-10；不删其他任务或必要证据。技术审计不是独立科学复现。\n')
    lines.append('通俗说明：这一阶段把“能猜出模型去哪儿找信息”“能预测取出来的内容”和“能靠自己的状态继续算出答案”拆开检验；并先解决同一段文字因补齐方式而产生内部数值差异的问题。只有前后信息边界清楚、留出预测和原模型干预都可复核，才算向机制解释推进。\n')
    entry='\n'.join(lines).encode('utf-8')
    if check_only:
        print({'check_only':True,'phase':phase,'entry_bytes':len(entry),'tables_built':True,'memo_written':False});return
    archive();assert memo.read_bytes()==before,'Memo changed during preparation; recompute the next Phase before appending'
    D.check_write(memo,len(entry),operation='GLM5 append-only memo')
    with memo.open('ab') as f:f.write(entry)
    assert memo.read_bytes()[:len(before)]==before
    cm=D.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md';cb=cm.read_bytes();cp=max(map(int,re.findall(rb'^## Phase (\d+):',cb,re.M)))+1
    cross=f'\n\n## Phase {cp}: 来源接力探索与事前功效漏项停止索引 [{stamp}]\n\nQwen3-14B本轮计划3072条，已保存发现768条与确认1152条完整结果；新句式行为部分运行后，因发现AGENTS第13节事前功效检查遗漏而停止，未完成两面板检验。固定形状右补齐修复了本轮同前缀数值测量问题。确认侧两头查询条件真V写入预测改善，预测V和完整输出递推失败，常量V对照不支持替换特异性。已完成数据的身份及算术检查{audit["count"]}项通过，研究准入审计未通过，正式证据晋级=0。真实执行范围见 `tests/glm5/result/source_relay_v2/report.json`；完整原理、公式、结果、硬伤、治理错误、理论洞察与后续任务见 `research/glm5/docs/AGI_GLM5_MEMO.md` Phase {phase}。\n'.encode('utf-8')
    D.check_write(cm,len(cross),operation='Claude memo append index')
    with cm.open('ab') as f:f.write(cross)
    assert cm.read_bytes()[:len(cb)]==cb
    D.save('closeout.json',{'phase':phase,'claude_index_phase':cp,'timestamp':stamp,'memo_prefix_bytes':len(before),
        'memo_prefix_sha256':hashlib.sha256(before).hexdigest(),'memo_entry_sha256':hashlib.sha256(entry).hexdigest(),'claude_prefix_sha256':hashlib.sha256(cb).hexdigest(),
        'technical_identity_checks_passed':True,'research_admission_passed':False,'formal_evidence_promotion':0})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check-only',action='store_true');a=ap.parse_args();main(a.check_only)
