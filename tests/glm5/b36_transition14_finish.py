"""Archive verified work and append one phase, after conditional native work."""
import json,re,zipfile,shutil,hashlib,importlib.metadata
import b36_transition14 as M
from b36_semantic14_memo import append
S=M.S

def run():
    S.configure();a=S.read(S.OUT/'analysis.json');plan=S.read(S.OUT/'plan.json');assert not (S.OUT/'memo-append-receipt.json').exists()
    native_path=S.OUT/'native-summary.json';native=S.read(native_path) if native_path.exists() else None
    assert not a['gate_passed'] or native is not None,'Counted interface passed: complete authorized native-path work before closeout.'
    assert not a['gate_passed'] or native['status'] in ['completed','numeric_admission_failed']
    for name,h in plan['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h,name
    assert a['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_transition14_analyze.py')
    for arm,h in a['raw_artifact_hashes'].items():assert S.file_sha(S.OUT/(arm+'.json'))==h,arm
    if (S.OUT/'native-plan.json').exists():
        for name,h in S.read(S.OUT/'native-plan.json')['source_hashes'].items():assert S.file_sha(S.ROOT/'tests/glm5'/name)==h,name
    if native and native['status']=='completed':assert native['analysis_sha256']==S.file_sha(S.OUT/'native-analysis.json')
    previous_stage=S.OUT/'behavior-stage-decision.json'
    if not previous_stage.exists():shutil.copyfile(S.OUT/'stage-decision.json',previous_stage)
    S.save('stage-decision.json',dict(status='behavior_passed_native_'+native['status'] if native else 'behavior_failed_native_not_run',
        behavior_gate_passed=a['gate_passed'],target_cells=S.read(previous_stage)['target_cells'],
        original_behavior_decision_sha256=S.file_sha(previous_stage),behavior_analysis_sha256=S.file_sha(S.OUT/'analysis.json'),
        native_summary_sha256=S.file_sha(native_path) if native else None,
        interpretation='Execution/admission status only; scientific effects and limitations remain in the analyses. Layer-vs-generation first formation is not established.'))
    source=list((S.ROOT/'tests/glm5').glob('b36_transition14*.py'))+list((S.ROOT/'tests/glm5').glob('test_b36_transition14*.py'))+[S.ROOT/'tests/glm5'/n for n in plan['source_hashes']]+[S.ROOT/'tests/glm5/b36_semantic14_memo.py',S.ROOT/'tools/power_analysis.py',S.ROOT/'tools/provenance.py',S.ROOT/'deploy/storage_policy.py',S.ROOT/'shared/dist_protocol/mechanism_trace.py']
    source=sorted(set(source));current_names={str(p.relative_to(S.ROOT)).replace('\\','/') for p in source}
    previous=S.ROOT/'tests/glm5/result/b36_map14_20261114_v1/source.zip'
    with zipfile.ZipFile(previous) as old,zipfile.ZipFile(S.OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in old.namelist():
            if n not in current_names:z.writestr(n,old.read(n))
        for p in source:compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');z.write(p,str(p.relative_to(S.ROOT)).replace('\\','/'))
    with zipfile.ZipFile(S.OUT/'source.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for p in source:assert z.read(str(p.relative_to(S.ROOT)).replace('\\','/'))==p.read_bytes()
    log=[json.loads(line) for line in (S.OUT/'progress.jsonl').read_text().splitlines() if line.strip()]
    stats=dict(cuda_peak_bytes=max(r.get('cuda_peak',0) for r in log),dxgi_shared_peak_bytes=max(r.get('shared_gpu_bytes',0) for r in log),generation_seconds=sum(S.read(S.OUT/(arm+'.json'))['seconds'] for arm in a['raw_artifact_hashes']),free_bytes=shutil.disk_usage(S.ROOT).free)
    S.save('delivery.json',dict(resource=stats,versions={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','scipy','safetensors','matplotlib']},scope='Local CUDA study; not independent-node reproduction. No publication or formal-center evidence registration. HTML static links checked; no browser layout claim.',engineering='4 material/parser/prefix tests; exact length/newline matching; all generated artifacts audited; frozen scientific sources unchanged.',native_status=native['status'] if native else 'Not run: counted interface gate failed.',source_archive_sha256=S.file_sha(S.OUT/'source.zip')))
    table=['条件 | 深度 | N | 步数正确 | 步骤全对 | 数值 | 行号 | 代码 | 格式 | EOS | 联合','--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---']
    for c in a['cells']:
        fields=['Steps','Value'] if c['arm']=='length_control' else M.FIELDS.get(c['arm'],['Steps','Value','Slot','Code']);na=set()
        if 'Steps' not in fields:na|={'step_count_correct','steps_correct'}
        if 'Slot' not in fields:na.add('slot_correct')
        if 'Code' not in fields:na.add('code_correct')
        table.append(' | '.join('不适用' if k in na else str(c[k]) for k in ['arm','depth','n','step_count_correct','steps_correct','value_correct','slot_correct','code_correct','format_correct','eos','joint']))
    comparisons='\n\n'.join(f"{r['left']}→{r['right']}，指标{r['metric']}：配对N={r['n']}，新增正确{r['gained']}，丢失正确{r['lost']}，差值{r['delta']}，准入{r['admitted']}，精确p={r['p_exact']}，Holm p={r['p_holm']}，描述性家族95%区间{r['descriptive_family95']}；{r['affine_clusters']}组相同终态仿射函数聚类的探索性区间{r['exploratory_affine_cluster95']}。" for r in a['primary'])
    examples='\n\n'.join(f"`{e['id']}`：真值{e['truth']}，正确行{e['expected_slot']}、代码{e['expected_code']}；输出 `{e['text'].replace(chr(10),' / ')}`；类型{e['parsed']['step_pattern']}。" for e in a['examples'][:8])
    native_text=native.get('memo_text','见native-summary.json的实际结果与限制。') if native else '明确计数接口未通过全部冻结条件，因此没有执行B36组消融、来源分区或后续层路径干预，也没有新原生HiddenState场。不能把未执行记为零效应，不能声称已判断复杂特征沿层还是沿生成时间形成。'
    native_outcome=''
    if (S.OUT/'native-review.json').exists():
        review=S.read(S.OUT/'native-review.json')
        assert review['analysis_sha256']==S.file_sha(S.OUT/'native-analysis.json')
        assert review['source_sha256']==S.file_sha(S.ROOT/'tests/glm5/b36_transition14_native_review.py')
        native_text+='\n\n**原生结果的严格解释（事后描述性复算）。** '+review['interpretation']+' 全坐标原值/逐层RMS图保留；另加asinh(20×原值均值/逐层RMS)的完整对称色标以显示低幅背景，不裁剪、不排序，也不用于特征选择或机制判定。'
        native_outcome=f" 原生确认的{len(review['top1_stability'])}个条件中，单条件改变首选token最多{max(r['changed_top1'] for r in review['top1_stability'])}/184；当前Code读出高度饱和，未建立固定32单元的正确率必要性或救援效应。"
    totals={arm:{k:sum(c[k] for c in a['cells'] if c['arm']==arm) for k in ['n','step_count_correct','steps_correct','value_correct','joint']} for arm in M.ARMS+['length_control']+M.HISTORY}
    headline=f"无表与仅加选项表的步骤数量正确分别{totals['no_table']['step_count_correct']}/96、{totals['table']['step_count_correct']}/96；增加Code字段后为{totals['code']['step_count_correct']}/96，再增加Slot字段为{totals['full']['step_count_correct']}/96。明确次数两种轮换的联合成功分别{totals['counted0']['joint']}/96、{totals['counted2']['joint']}/96。这里只能把差异归于对应的整组提示/输出要求改变，不能把文字步骤当作内部算法轨迹，字段说明的长度与内容也尚未完全正交。"
    history_verdict=(f"原样回放逐token一致{a['history_audit']['exact']}/{a['history_audit']['n']}，"+('达到冻结历史准入。' if a['history_audit']['admitted'] else '未达到冻结的96/96要求；全部历史主比较不准入确认，以p=1占据Holm校正位置，历史结果只作描述，没有放宽阈值或重跑替换。'))
    metadata_note=('原生分析执行前的静态审查发现特征卡和共享写入器重复提供model_snapshot键；单独的native_delivery.py入口仅移除值完全一致的重复键，再调用原写入器。冻结的native_analyze.py保持字节不变，数值计算、指标、阈值和哈希校验不变；实际执行组合及哈希见native-analysis-execution-amendment.json。该修订只涉及元数据写出，不是结果后调参。' if (S.OUT/'native-analysis-execution-amendment.json').exists() else '')
    body=f'''**任务与状态。** 承接Phase2932，按用户要求分别控制选项表、输出字段和自产历史，区分多打印一步、计算值不符、终态保持及输出接续。运行{S.RUN}。本轮96个家族，{a['n_natural']}次自然生成、{a['n_history']}次历史条件续写；多种视图不增加独立家族数。明确计数接口的预注册联合行为门{'通过' if a['gate_passed'] else '未通过'}。行为、历史输入干预、原生单元路径分别记账。{native_outcome}

**模型、材料与身份。** Qwen3-14B原始BF16，快照`{S.SNAPSHOT}`，14个权重及tokenizer文件逐一重新哈希；CPU读取safetensors、CUDA逐层执行标准HF贪心生成，batch96、SDPA、左padding、KV缓存，enable_thinking=False，不量化。本轮无4B/GLM4运行。全部96个实际可见操作序列不同，排除前轮108个生成候选、54个算术家族及Phase2932的96家族；初始值加语义前缀也不重合。ID只包含实际显示的程序与初始值，没有未显示的第三操作。初始10–29、中间0–99、加减1–9及乘2/3，两步与三步各48，正确选项行1–4各12。未见只指本系列登记材料，不声称预训练未见。不同配方仍可能具有相同仿射终态函数，统计以家族为单位并给聚类敏感性分析。两种深度不是同一程序的配对，不能据深度差作因果推断。

**对照设计。** no_table只输出Steps/Value；table仅在相同位置加入选项表，输出要求逐字相同；code增加Code字段，full再增加Slot字段，公共字段说明保持一致；no_steps从full去掉Steps；counted0/2额外明确更新次数与Steps项数，并轮换答案代码。代码字段说明为本轮统一版本，不与前轮完整提示逐字相同，所以不以跨阶段成绩差归因。明确计数同时改变指令内容和长度，只检验该接口的效果，不能把它当作已隔离的内部计数变量。

原方案在模型运行前冻结6项主要比较。执行中复核发现选项表引入长度差，追加length_control：在相同位置放四行无关文字，逐题精确匹配table的输入token数和换行数，保持Steps/Value要求。登记时只查看过no_table响应；table已完成生成，但未打开其响应内容。原方案及其哈希保留，补充方案单独保存，所有主要比较保守改为7项Holm校正，未改既有题目、评分或机制准入线。最初无关段落过长，CPU长度检查失败，缩短后才冻结；没有据模型输出调试该文字。重复完整句子的无关段落也可能影响语境，因此长度控制不能排除全部内容和格式替代解释。

**原理与数学对象。** 外部任务真值为s_k=T_k(s_(k−1))、v=s_d；选项行j满足V_j=v，代码c=π(j)。打印的步骤序列记u=(u₁,…,u_m)，m=d检验数量，u=(s₁,…,s_d)检验内容，二者分别评分。额外值恰等于T_d(v)、包含初始赋值、附加其他值及改变原正确前缀分别分类。Value恰等于再次执行末操作的结果是可见数值关系，不等于证明模型内部实际重复执行了该操作。

历史续写在full的实际输出中抽取第d个整数结束处，必须恰在真实token边界：replay原样回放自产前缀；boundary保持数值内容不变，仅注入换行与Value标签；gold在同样格式中替换正确步骤再进入Value；foil只将gold末值改成另一等位数整数，token长度逐题与gold相等。比较的是Pθ(y|x,u_d)与Pθ(y|x,u_d,b)或替换历史后的分布，θ始终不变，b是人工字段边界。取切口需要回看完整生成文本，所以不是一个在线预测未来的检测器。gold/foil包含人工信息，不能计作自主计算；boundary也是输入条件干预，不能替代神经元路径证据。

**冻结指标与准入。** 各自然条件分别记录步骤数量、步骤内容、Value、Slot、Code、格式及EOS；联合成功要求该条件所有必需字段正确且格式/EOS合格。明确计数接口counted0/counted2在两种深度×两种代码的4格，各需至少44/48联合成功；这是开跑前定义的新范围，其他诊断条件失败仍完整保留，不能用它替Phase2932原始接口补过。历史主比较要求96家族切口齐全且原样续写逐token全部复现；否则历史结果仅描述，未准入检验按p=1占据多重校正位置。实际历史审计：{json.dumps(a['history_audit'],ensure_ascii=False)}。

配对检验设新增正确g、丢失正确l、D=g+l，精确McNemar双侧p=min(1,2·Σ_(k=0)^min(g,l) C(D,k)/2^D)，D=0时定义p=1。七项Holm控制比较家族。规划Δ=.20、预期不一致率.30、α=.05/7，工具计算需94，实际96，功效.813；并不保证更小效应有充分功效。家族独立是近似，仿射函数聚类只作敏感性检查；逐格Wilson及重采样区间是描述性95%，非同时区间。零不一致对导致bootstrap区间[0,0]，不能据此认定总体等价。

'''+ '\n'.join(table)+f'''

**核心观察与历史限制。** {headline} {history_verdict}

**主要配对结果。**

{comparisons}

**错误与终态账本。** {json.dumps(a['breakdown'],ensure_ascii=False)}。

实际反例（完整输入、token及历史编辑见原始产物）：

{examples}

**B36实际路径与形成问题。** {native_text}

**工程修订披露。** {metadata_note}

**历史评分限制。** 历史条件的联合评分包含已由实验者固定的Steps字段。foil写入错误末值，因此其Steps全对和联合成功按构造为0，不能把这个0当成模型续写独立计算失败的证据；后续Value/Code是否纠正或跟随另行记账，并受本轮历史确认未准入的限制。

**严格审视。** 生成字段顺序不等于内部模块顺序，正确的打印步骤也不保证忠实反映内部计算。请求未来的Code字段在输入时已可见，它改变早期Steps生成不是未来token反向造成过去。提供历史或强制字段边界会改变可用信息与续写条件；恢复答案不能自动提升为必要性、充分性或唯一中介。材料只覆盖短整数程序、有限输出格式和单模型快照，尚非自然语言模式族的广泛验证；没有新跨模型坐标对齐、独立节点复算或同预算稀疏字典比较。共享显存由DXGI实际记录；系统提交内存包含其他进程，不能当作本任务独占量。

**三图谱、RDC与第一性原理。** 外部图谱获得程序、更新次数、选项关系、字段要求与历史条件的配对行为材料；内部及关联图谱仅按上述实际原生检验范围记账，未执行项不加因果边，也未登记正式中心。Phase2928传播拼图和格式边界、Phase2930–2932行为结果均按原限制保留。本轮没有改写RDC主体或全局图谱公式；状态递推和条件概率是实验定义，不包装成新智能数学定律。破解有限参数的可组合语言能力，需要解释任务要求怎样约束状态更新、终态保持和符号接续，并同时跟踪层索引l与生成时间t的h_(l,t)。二者可能共同参与，不能强行二选一。

**下一大阶段。** 组织“状态更新—停算边界—符号接续”的整体研究。第一步先修复测量闭环：从同一次自然生成的同一KV缓存和状态分支，核对无编辑续写在相同批次/位置设置下复现，再冻结历史实验，不能靠放宽本轮96/96阈值补过。第二步在未参与当前选择的新程序家族上配平输出字段、行数数字、长度与更新次数，区分重复执行末操作、初始赋值计数、数值复述与输出接续。第三步把干预位置扩展到第d次状态更新之前、进入Value之前和进入Code之前，比较固定历史重算与重新自由生成历史的不同结果，检验何时形成及如何传递；当前Code条件不能代替这些生成时刻。第四步在同一留出材料上比较完整原生坐标、固定B36组、匹配控制与可比预算稀疏特征，同时预测内部响应、后续输出及有限干预效应，检查后续attention/MLP的净作用和副作用，保留未解释残差。只有跨家族预测与原模型干预共同支持，才讨论可复用的复杂组合结构；本段为后续计划，不是本轮已完成结果。

**工程、资源与交付。** 4项材料、字段、评分和历史切口工程测试通过；全部实际输入分词/AST真值、可见表格、输出解码、评分、历史编辑和冻结身份审计；非零不一致对的p值由SciPy核对。补充长度与行数逐题精确相等。源码逐字节回读归档；图表人工检查，HTML只检查内容/本地链接，不声称桌面或手机浏览器渲染验收。未改共享协议及生产客户端，未重跑全套协议回归，没有发布。自然及历史生成累计{stats['generation_seconds']:.1f}秒，CUDA分配峰值{stats['cuda_peak_bytes']/2**30:.3f}GiB，DXGI日志采样的最大共享显存{stats['dxgi_shared_peak_bytes']/2**20:.1f}MiB（非连续监测的严格峰值）。全部新增数据D盘，按storage_policy保留20GiB余量，交付前空闲{stats['free_bytes']/2**30:.2f}GiB。主任务与长度补充使用同一GPU互斥锁串行运行。保留冻结计划、原始token、全部失败、补充登记、真实历史编辑、分析与源码；只清理本轮临时目录，实际字节见cleanup.json。研究记录只尾部追加并验证原字节前缀不变。

入口tests/glm5/b36_transition14*.py与test_b36_transition14.py。实际命令使用.venv/Scripts/python.exe -B -X utf8 -u：transition14.py冻结、test脚本检查、queue.py串行执行；length.py prepare/run冻结并排队补充；analyze.py复算、report.py作图；原生步骤使用native.py prepare/run，再由native_delivery.py执行原冻结native_analyze.py，实际执行范围按native-summary.json记账；finish.py归档。另完成两项原生分区/布尔分解工程测试和192个读出位置CPU核对，它们不计模型实验样本。结果tests/glm5/result/b36_transition14_20261114_v1/report.html，analysis.json、history-plan.json、length-plan.json、stage-decision.json、source.zip、manifest.json可逐项复核。

通俗说明：这次不是只看最后答案，而是把增加选项、增加输出要求、保留或改变已写出的步骤分开比较。它可以缩小多写一步发生的条件，并检查正确结果是否在转入下一字段时丢失；只有实际内部干预能进一步说明32个单元参与了哪条输出路径。
'''
    r1=append(S.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','终态接续分离：行为过门、回放失配与B36正确率负结果',body)
    r2=append(S.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','B36步骤数量与终态接续分离验收索引',f"详见GLM5 Phase {r1['phase']}，运行{S.RUN}。96新实例且实际可见操作序列与既有材料不重合；{a['n_natural']}自然生成、{a['n_history']}历史条件续写，视图不增加独立N。主方案与长度补充分别冻结，7项Holm；明确计数接口行为门{'通过' if a['gate_passed'] else '未通过'}。原样历史回放{a['history_audit']['exact']}/{a['history_audit']['n']}逐token一致。人工边界/正确或扰动历史不等于自主计算，也不自动证明神经元路径。原生工作状态：{native['status'] if native else '未执行，行为准入失败'}。完整限制、资源、图表与源码见tests/glm5/result/b36_transition14_20261114_v1/report.html。RDC公式未改，无发布。")
    S.save('memo-append-receipt.json',dict(records=[r1,r2]))
    target=S.TEMP.resolve();assert target==(S.ROOT/'tests/glm5_temp/b36_transition14_20261114_v1').resolve() and target.drive.upper()=='D:'
    paths=list(target.rglob('*'));assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(target) for p in [target]+paths)
    size=sum(p.stat().st_size for p in paths if p.is_file());shutil.rmtree(target);S.save('cleanup.json',dict(removed_directory=str(target),removed_bytes=size,scope='Only current-run scratch/cache; all raw generated tokens, frozen plans, history edits, audit and necessary native evidence retained.'))
    report=S.OUT/'report.html';text=report.read_text(encoding='utf8')
    import html
    if native:
        outcome='原生路径实验已完成，具体效应与限制见后文。' if native['status']=='completed' else '原生数值准入未通过，已停止受影响的机制确认。'
        text=text.replace('预注册的明确计数接口已过行为门；还需要原生路径实验。','预注册的明确计数接口已过行为门；'+outcome)
    addition=''
    addition+='<p>历史评分限制：foil条件的Steps末值由实验者固定为错误值，因此其Steps全对和联合成功按构造为0；不能把该0解释为续写阶段的独立计算失败。后续Value/Code另行评分，历史比较本轮仅作描述。</p>'
    if native and '<h2>B36原生来源与后续路径</h2>' not in text:
        addition+='<h2>B36原生来源与后续路径</h2><p>'+html.escape(native_text)+'</p><p><a href="native-summary.json">原生步骤状态</a></p>'
    addition+='<p><a href="source.zip">源码归档</a> · <a href="manifest.json">文件清单</a></p>'
    if (S.OUT/'native-feature-card.json').exists():addition+='<p><a href="native-feature-card.json">固定单元组特征卡</a> · <a href="native-analysis-execution-amendment.json">元数据写出修订记录</a></p>'
    assert '</html>' in text
    text=text.replace('</html>',addition+'</html>');report.write_text(text,encoding='utf8')
    for name in re.findall(r'(?:href|src)="([^"]+)"',text):assert (S.OUT/name).is_file() or name=='manifest.json',name
    entries=[dict(path=str(p.relative_to(S.OUT)).replace('\\','/'),bytes=p.stat().st_size,sha256=S.file_sha(p)) for p in sorted(S.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json']
    S.save_json(S.OUT/'manifest.json',dict(run_id=S.RUN,files=entries,provenance=S.read(S.OUT/'delivery.json')['provenance']))
    assert all(S.file_sha(S.OUT/e['path'])==e['sha256'] for e in entries)
    print('Appended GLM5',r1['phase'],'Claude',r2['phase'],'verified files',len(entries),flush=True)

if __name__=='__main__':run()
