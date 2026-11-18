"""Append-only scientific closeout, after real evidence and client acceptance."""
import hashlib,re
from datetime import datetime
from pathlib import Path
import source_output_core as S
from source_output_core import C

def append(path,title,body):
    original=path.read_bytes();phase=max(int(x) for x in re.findall(rb'^## Phase (\d+):',original,re.M))+1
    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M');text=f'\n\n## Phase {phase}: {title} [{timestamp}]\n\n'+body+'\n'
    with path.open('ab') as stream:stream.write(text.encode('utf-8'))
    new=path.read_bytes();assert new[:len(original)]==original
    return {'phase':phase,'path':str(path),'original_bytes':len(original),'original_sha256':hashlib.sha256(original).hexdigest(),'new_sha256':hashlib.sha256(new).hexdigest()}

def main():
    assert not (C.OUT/'memo-append.json').exists();p=C.checked();d=C.read(C.OUT/'delivery-audit.json');assert d['passed'];r=C.read(C.OUT/'report.json');reg=C.read(C.OUT/'registration.json')
    fmt=lambda x:f'{x:.5g}' if isinstance(x,(int,float)) else str(x)
    behavior='|任务×答案符号|校准严格正确率|留出严格正确率|留出首词正确率|完整分支|\n|---|---:|---:|---:|---|\n'
    for key,v in r['behavior']['confirm']['branches'].items():behavior+=f"|{key}|{r['behavior']['calibration']['branches'][key]['accuracy']:.4%}|{v['accuracy']:.4%}|{v['top1']:.4%}|{'通过' if key in r['active_branches'] else '停止'}|\n"
    prediction='|分支|常数/末token嵌入 MSE|词袋 MSE|规则8格 MSE|原生8变量 MSE|原生/规则比|\n|---|---:|---:|---:|---:|---:|\n'
    for o in r['observations']:prediction+='|'+o['branch']+'|'+'|'.join(fmt(o['mse'][key]) for key in ['constant','token_hash','parsed8','native8'])+'|'+fmt(o['native_over_parsed'])+'|\n'
    if not r['observations']:prediction+='\n没有通过完整行为门的分支，未拟造内部预测结果。\n'
    effect='|分支 / 组|样本 / 家族|优势平均下降|无效原生方向对照|首词变化|作用门|\n|---|---:|---:|---:|---:|---|\n'
    for e in r['effects']:effect+=f"|{e['branch']} / {e['candidate']}|{e['n']} / {e['families']}|{e['mean_loss']:.5g}|{e.get('invalid_native_controls','不适用')}|{e['top1_changes']}|{'范围内通过' if e.get('supported') else '未支持/不适用'}|\n"
    if not r['effects']:effect+='\n整支行为门阻止后续作用检验；缺失不等于零作用。\n'
    interpretation=[]
    for o in r['observations']:
        probes=o['probes'];interpretation.append(f"{o['branch']}：B26固定32单元在映射前的来源/输出探针准确率为 {probes['0:target']['accuracy']:.4%}/{probes['0:output']['accuracy']:.4%}，答案前为 {probes['1:target']['accuracy']:.4%}/{probes['1:output']['accuracy']:.4%}；原生预测相对规则均值的改善区间 {o['native_gain']['frame_interval']}。")
    diagnostic_lines=[f"{x['branch']}留出{x['n']}条：去空格后正确代码{x['counts']['correct_after_trimming_whitespace']}，其中带额外空格{x['counts']['correct_code_with_extra_whitespace']}；字母任务仍返回编号{x['counts']['record_number_in_letter_task']}（编号与应选来源相符{x['counts']['correct_record_number_in_letter_task']}）" for x in r['failure_diagnostics']['branches'] if x['split']=='confirm']
    url=f'http://localhost:5173/workbench?lens=process&station=execution&run={C.RUN}&snapshot={C.SNAP}'
    body=f'''**任务与执行边界。** 继续上一阶段有效/过时记录选择研究，固定旧候选B29 H11/H15和B26的32个原生MLP单元，不重新选择层或单元。运行 `{C.RUN}`，本地Qwen3-4B，快照 `{C.SNAP}`，CUDA原生BF16、无量化、逐张量磁盘/显存字节校验，逐模型互斥。用户中途要求“先清理磁盘，等待另一研究完成14B测试后再测试”，本轮排队进程立即停止；随后仅进行CPU准备，确认其他研究的14B模型测试全部完成（其客户端与打包仍在收尾）后才恢复4B。详见 execution-hold/resume.json。没有远端发布或正式机制证据晋级。

**清理和保全。** 工作区盘点识别出关键原生全场与模型，不以目录名“temp/result”决定删除。两份看似重复的隐藏场/输出文件已是硬链接，保留。对旧原生数组尝试NTFS无损压缩：三个触及文件全部前后SHA一致，仅回收62,468,224字节，收益低后停止。另对 `D:/ai2050_offload/qwen3-14b` 的68个旧卸载张量逐字节比对保留的14B模型，登记索引、原模型偏移、形状和SHA；进一步核对当前14B入口直接读取模型且旧入口没有运行后，按已验证名单清理；原始模型、激活、反例和复算依据保留。清理缓存没有提前启动4B，模型测试仍等待另一研究完成。完整实际清理见 storage-cleanup-summary.json。原计划30GiB余量因另一授权研究占用共享C盘，在任何4B模型观测前另行记录12GiB资源下限修订，本轮3.8GiB预算与全部科学门不变，旧计划不覆盖。后来六个校准分支全部停止，原生与干预按原门均不执行；另一研究的13.27GiB源文件发布包占用C盘后，另记behavior-only资源修订：本轮目录总预算降至0.25GiB、磁盘保留5GiB，只完成固定留出行为与失败整理。旧资源规则、旧启动器和两次修订全部保留，评分与材料不变。 校准结束时C盘实测余量11.74GiB，低于当时12GiB预留；第一版只在阶段入口检查，行为循环没有逐批中止检查，这是明确的资源执行缺陷，不能用后来的修订追认合规。resource-deviation.json保留该事实；确认阶段始终高于其5GiB新界限，数据没有丢失。下版需把逐批空间检查和可恢复中止纳入运行器。

**可证伪设计与样本。** 5760条确切不同的新输入：48校准家族×24条件=1152，192留出家族×24=4608；校准6句式、留出12新句式，实体与查询措辞独立。每家族交叉“按有效状态/按最新时间/直接指定记录号”×“1/2或A/B”×“有效记录位置”×“答案映射方向”，状态和时间、要求位置的关系按家族均衡。108份生成式模板通过共享文法校验。输出映射位于选择规则之后；读出位置为映射出现前与答案前缀后。预注册规定仅在整支通过门后，于两位置采集全部37层HiddenState（含嵌入）、B26全部9728 gate/value/乘积单元、B29全部32头QK及输出，另保存H11/H15按四来源分解的V上下文。后者是局部计算来源，不能当作完整下游因果作用。

行为门以六个预声明 task×alphabet **完整分支**判定：每个“目标来源位置×映射”格和语言格均须 n≥48，且严格候选优势>0与完整词表top1正确率均≥90%。任何一格失败，则整支停止内部机制与干预，保留全部失败；没有确认后按家族正确与否筛选，没有修改提示重新冻结以救门。候选沿用旧冻结，校准拟合预测器，预测器在首次留出行为前冻结。

{behavior}

**首词失败诊断。** {'；'.join(diagnostic_lines)} 这是校准失败后对已保存argmax token的探索性分类，不是更换正式答案集；完整行为门与停止决定没有改变。\n\n**执行缺陷。** 冻结采集器在写准入记录前触发Path与字符串拼接TypeError，没有执行原生采集。保留冻结源码和失败记录，另用只能接受零合格分支的source_output_stopped.py写明停止状态；测试验证其遇到任一合格分支必定拒绝。该恢复入口不能执行采集或放宽行为门。今后复用该采集器须先修复路径表达式，并在新版本重新冻结。\n\n**数学对象和解释限度。** 令 s∈{{−1,+1}}为应选记录位置，m∈{{−1,+1}}为输出映射方向，o=sm为答案符号。对每个固定家族、任务和符号集，完整2×2设计可作恒等展开 x(s,m)=β₀+βₛs+βₘm+βₛₘsm，βₛ=平均(sx)、βₛₘ=平均(smx)。本轮展示βₛ与βₛₘ；它们是有明确定义的条件统计成分，不意味着隐藏空间有独立、正交或唯一的“来源模块”和“答案模块”。映射出现前的四个前缀逐token相同，故βₛₘ必须严格为零；这是因果可见性的负对照和采集校验，不是发现了模型主动屏蔽未来。最后一个token的输入嵌入相同，嵌入读出只对应校准常数基线。

对同一完整合格分支，用校准集拟合λ=1的标准化岭回归；原生8输入包括固定两头的局部读出、两来源质量差、固定单元组局部读出，以及可见映射、语言和长度。比较256维token哈希词袋、校准均值与按实际文本解析规则得到的8格均值。原生读出使用实际模型状态，规则参照使用已知任务类型；两者都不是仅靠图谱独立预测完整下一个token分布。所有MSE针对真实原模型候选logit差，而非任务正确答案标签。

{prediction}

{' '.join(interpretation)}

**预注册的作用、负对照与功效方案（仅合格分支执行）。** 每个留出家族按预声明元数据固定一种符号集和位置，保留两种映射，不按结果挑样本；每合格分支96家族、192输入。最后位置分别完整移除H29、U26及联合组；同层前两个旧冻结组做原始消融，并以其实际残差变化方向匹配候选扰动范数；另有随机等范数方向。范数误差>2%或原生方向缩放>4时保留无效登记，停止相应对照主张，禁止拿有效子集补过门。零剂量、恒等回放与移除后恢复检查单独记录。主门为平均正确答案优势下降≥0.2 nats，且相对随机与原生等幅方向的99.1667%句式聚类区间下端均>0。联合非加性不证明定向中介，B26先于B29，不能倒置成B29→B26。

{effect}

主效应的连续均值功效预设Δ=0.2 nats、家族SD=0.5、α=.05/6，96家族名义功效约0.90、所需76；仅12句式簇会限制推广。α按每个候选的六分支校正，未保证两个候选及全部辅助指标合并后的全局错误率，范围内通过不作正式多重检验确认。补充的二值副作用明确只取预声明m=+1的一条/家族，调用 tools/power_analysis.check_phase：20个百分点净准确率损害、25%预期不一致率时96家族名义功效0.90，所需76；这是大效应检测目标，不预测必然出现，也不为3–10个百分点小效应或“没有翻转即无作用”背书。精确McNemar及全部功效敏感性表见 secondary-binary-results/secondary-power-freeze。主表仍使用完整两映射，没有据此改变原门或删样本。

**严格审视与瓶颈。** {' '.join(r['summary'])} 所有失败见报告；响应、读出、局部代数写入和原模型作用分别记账。这里只检验旧候选，不足以否定其他原生组合；人工短模板和候选首词无法等同自然长对话。任务规则解析基线具有外部语义知识，原生读出依赖模型激活，两者都没有实现独立语言图谱模拟器。标准化线性探针的可读出性不能代替必要性或算法同一性；matched方向也不是日常分布内状态。未做跨模型、独立节点、SAE/transcoder、通用能力保持或完整中介链，不给出“新数学已经成立”或AGI结论。

**智能理论增量与下一大阶段。** 研究对象应从单个亮坐标改为“在变换下保留什么、按规则改变什么、该变换如何被原模型实现”。对来源选择与输出渲染，可以提出条件算子分解 f(记录,规则)→来源、g(来源,映射)→答案，但本轮只提供可检验的区分设计及相应结果，未证明两个独立模块。完整2×2展开只是数学恒等式，只有跨新规则预测、干预选择性与递推稳定性才是机制增量。

报告中的逐项限制同步如下：{'；'.join(r['limitations'])}

下一大阶段按三条并行证据线组织同一预注册任务，而非继续在已开封材料里换层、换词直到过门：①对本轮停止分支明确模型能力失败、格式读出失败及绑定失败的可区分预测，选真正新的语义域和输出体系，旧失败不改名；②从现存全场抽取规则切换、来源绑定、输出映射的跨位置转移，连接真实层序的源V/O与MLP写入，比较先选择再映射、联合路由及旁路三个替代模型；③在相同新家族和预算下比较原生条件算子、强文本解析/位置/n-gram基线及合适字典路线，要求未见全场和输出分布预测收益，再做定点干预与跨任务副作用。未能胜过简单基线的局部规律保留为有限拼图，停止升级为普遍编码定律；新增采集必须能区分这些替代解释。

**客户端、验证与保留。** 沿用现有工作区，增加“记录来源与输出符号”报告：整支行为与所有反例、两读出位置的原生全坐标、固定留出预测与基线、作用及无效负对照。坐标按原始索引保留，颜色按本阵列绝对最大值缩放，不能跨图按色深比较；可键盘查询索引。零结果与未测试明确分开，卡片保留真实组索引，没有伪造单坐标锚或因果链。

两个本机库逐对象相同：{reg['objects']}对象、{reg['cards']}卡、{reg['edges']}观察关联边，{d['full_coordinate_artifacts']}全坐标产物、每库{d['coordinate_values_per_library']}个FP32值，均与分析数组逐值一致。技术审计保留{d['native_inputs']}条原生输入、{d['causal_inputs']}条干预；{d['engineering_tests']}项针对模板/整支门/原生计算/冻结预测与停止登记的工程测试、{d['frontend_tests']}项前端测试、组件lint、构建及{d['browser_checks']}项真实浏览器检查通过，窄屏截图已检查。浏览未触发额外GPU请求。运行产物{d['retained_bytes']/2**30:.3f}GiB；WDDM峰值专用{d['wddm_peak_dedicated_bytes']/2**30:.3f}GiB、共享{d['wddm_peak_shared_bytes']/2**20:.3f}MiB，计数为分配量而非实测传输。收尾C/D余量{d['C_free_bytes']/2**30:.2f}/{d['D_free_bytes']/2**30:.2f}GiB。既有原生场继续保留；本轮停止分支没有生成原生场，不把缺失记为零作用。Junction不是第二份备份。

实际入口：source_output_design.py；source_output_execute.py behavior/collect calibration/confirm；source_output_analysis.py freeze/fit/report；source_output_causal.py run/aggregate（运行由execute互斥入口执行）；source_output_power.py freeze/report；source_output_publish.py audit/publish；verify_source_output.cjs；source_output_delivery.py。冻结源、材料、逐阶段日志、失败启动、清理与恢复记录均在 `tests/glm5/result/source_output_v1`，源码位于 `tests/glm5`，临时工具位于 `tests/glm5_temp`。详细命令与哈希在交付审计，未远端发布。

本机入口：[本轮报告]({url})，选择“研究报告与反例”。报告SHA `{reg['report_sha256']}`；恢复包 `{reg['bundle']}`；审计 `tests/glm5/result/source_output_v1/delivery-audit.json`。
'''
    glm=append(C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','记录来源与答案映射分离：整支行为门、冻结候选和作用边界',body)
    claude=append(C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','记录来源与输出符号的固定候选独立检验',
        f"运行 `{C.RUN}`。详见GLM5 Phase {glm['phase']}及 tests/glm5/result/source_output_v1/delivery-audit.json。5760新输入，固定B29 H11/H15和B26的32单元，{len(r['active_branches'])}/6完整分支过门；失败分支整支停止，无确认后家族筛选。来源/答案映射、强基线、实际等幅方向对照和失败均同步两个本机库。已按用户要求先清理保全、等待另一14B研究完成再测试4B。无正式机制晋级或远端发布。")
    C.save('memo-append.json',{'glm5':glm,'claude':claude,'append_only_prefix_verified':True});print({'glm5_phase':glm['phase'],'claude_phase':claude['phase']},flush=True)

if __name__=='__main__':main()
