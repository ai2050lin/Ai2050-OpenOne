"""Complete a failed behavioral gate with audited, append-only research record."""
import json,shutil,zipfile,importlib.metadata,re
import b36_map14 as M
from b36_semantic14_memo import append
S=M.S

def run():
    S.configure();a=S.read(S.OUT/'analysis.json');assert not a['gate_passed'],'Gate passed: continue to authorized native-path work.'
    assert not (S.OUT/'memo-append-receipt.json').exists()
    p=S.read(S.OUT/'plan.json');progress=[json.loads(s) for s in (S.OUT/'progress.jsonl').read_text().splitlines() if s.strip()]
    stats=dict(cuda_peak_bytes=max(r.get('cuda_peak',0) for r in progress),dxgi_shared_peak_bytes=max(r.get('shared_gpu_bytes',0) for r in progress),generation_seconds=sum(S.read(S.OUT/(arm+'.json'))['seconds'] for arm in M.ARMS),free_bytes=shutil.disk_usage(S.ROOT).free)
    S.save('delivery.json',dict(resource=stats,versions={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','scipy','safetensors']},tests='3 material/rotation/parser tests pass; all672 rendered materials, token decodes, metrics and hashes checked; Nonzero-discordance McNemar values checked by scipy binomtest; zero discordances give p=1 by definition.',new_mechanism_interventions=0,new_native_fields=0,source_hashes_verified=True,publishing='Not published or registered as formal center evidence.',scope='One local node; not an independent implementation replication. No new batch/unpadded equivalence or browser layout claim.'))
    table=['任务 | 步数 | N | 步骤全对 | 数值对 | 行号对 | 代码对 | 格式 | EOS | 联合成功','--- | --- | --- | --- | --- | --- | --- | --- | --- | ---']
    for c in a['cells']:
        irrelevant={'steps_correct','value_correct'} if c['arm'].startswith(('match','encode')) else set()
        if c['arm'].startswith('encode'):irrelevant.add('slot_correct')
        if c['arm']=='compute':irrelevant|={'slot_correct','code_correct'}
        table.append(' | '.join('不适用' if k in irrelevant else str(c[k]) for k in ['arm','depth','n','steps_correct','value_correct','slot_correct','code_correct','format_correct','eos','joint']))
    primary='\n\n'.join(f"{x['left']}→{x['right']}：N={x['n']}，新增正确{x['gained']}、丢失正确{x['lost']}，准确率差{x['delta']:+.4f}；精确p={x['p_exact']:.6g}，Holm p={x['p_holm']:.6g}；描述性95%家族重采样区间{x['descriptive_bootstrap95']}，按相同终态仿射函数聚类的探索性敏感性区间{x['exploratory_affine_cluster95']}（{x['exploratory_affine_cluster_count']}组）。" for x in a['primary_comparisons'])
    examples='\n\n'.join(f"`{e['id']}`：真值{e['truth']}，选项值{e['values']}，代码{e['codes']}，正确行{e['expected_slot']}、代码{e['expected_code']}；实际输出 `{e['text'].replace(chr(10),' / ')}`，可见分类{e['kind']}。" for e in a['examples'][:5])
    body=f'''**任务与状态。** 承接Phase2931“步骤正确但最终选错项”，按用户要求把数值计算、选项匹配、答案代码拆开，在96个独立新程序家族上完成672次真实生成。运行{S.RUN}。联合行为门{sum(c['passed'] for c in a['cells'])}/14格通过，尚未满足全部冻结条件，因此未启动B36固定32单元的机制正例干预或层间/生成时间形成验收。部分成功条件不能代替整个联合门，未执行不等于干预零效应。

**模型与材料身份。** 主模型Qwen3-14B原始BF16，快照`{S.SNAPSHOT}`，14个权重/tokenizer文件重新逐一哈希，计划同时冻结评分器、运行器、原权重装载器和资源模块哈希。CPU读取safetensors、CUDA逐层执行，HF标准贪心生成、SDPA、左padding、batch96、KV缓存；非thinking模式但允许模型自行写步骤。无量化、无外部模型、无4B/GLM4新运行。原创程序，采用整数加减及乘2/3，初始10–29，中间值0–99；构造器按完整三操作候选排除前轮108个生成候选和54个算术家族，但两步题只显示前两项。审计发现实际显示的操作序列共93种，其中9个家族沿用旧操作序列；所有96个初始值加实际操作实例及语义前缀均与旧材料不重合。本轮因此是新实例确认，不能称为全部未见操作组合的推广；冻结计划中的新程序措辞按此限定。不同程序可得到相同的终态仿射函数，额外报告按x→a·x+b聚类的探索性敏感性区间；主检验仍为原冻结程序家族口径，不能把不同程序数当作不同数学规律数。基础操作量扩展到1–9，是范围变化，不用跨阶段成绩差推断因果。新96家族按两步/三步各48分配，每一深度正确行1–4各12。所有代码视图、提供真值的对照均归原家族，672不作为独立N。

**设计与数学。** 真值s_k=T_k(s_(k−1))，结果v=s_d；匹配算子j=M(v,V)满足V_j=v，代码c=π(j)。V是四个互异候选数值的有序表，π是行号到A/B/C/D的双射。这些是任务定义而非内部机制定律。每家族7个配对条件：compute只自行输出Steps和Value；integrated0/2自行计算并输出Steps、Value、Slot、Code；match0/2给定正确数值，只要求找行号和代码；encode0/2给定正确行号，只要求代码。两代码条件保持原程序、选项数值、行序不变，只将行的代码循环移两位。原始旋转在家族间配平，覆盖四个循环映射。提供正确值或行号的条件是分离环节的控制，不能计为模型独立完成原始计算。所有模型输出均为无强制token的自然贪心续写。

compute/integrated/match/encode生成上限分别48/64/24/16个新token，EOS与截断分别记录。联合成功为该任务要求的所有字段正确∧格式合规∧实际EOS。14个任务×深度格每格48，全部至少44/48才准入B36联合机制研究。所有672条运行完再判断，不见到前面失败就筛掉其余格。预注册两项主要配对检验：integrated0与match0代码正确率、match0与encode0代码正确率，96家族，精确McNemar并用Holm校正两检验。预期效应16个百分点、翻转率.20、每检验规划α=.025，tools/power_analysis.py给出所需75、实际96、功效.897；此假设不保证小效应有功效。深度细分、另一映射和错误分类主要作描述，不事后增设通过阈值。各格Wilson区间及配对家族bootstrap区间为描述性95%，不是同时区间。

'''+ '\n'.join(table)+f'''

**主要发现。** {a['interpretation']}

**主要比较。**

{primary}

配对推断依赖家族独立的近似，相同终态仿射函数聚类只是敏感性检查，不能证明消除了所有相关性。若两项对照均全部正确，则无不一致对，p=1；相应非参数重采样区间会退化为[0,0]，不能据此证明总体完全等价或不存在小差异。

**可见错误的分层账本。** {json.dumps(a['output_error_breakdown'],ensure_ascii=False)}。

记录中间步骤是否能解析、是否等于真值，Value是否正确，Slot是否与模型打印的Value相符，Code是否与模型打印的Slot相符。若数值正确而行号错，与行号正确而代码错分开；未解析/未完成与明确算错分开。所谓错误位置只指可观察输出字段，不足以断言错误最早产生在哪个神经层。实际样例：

{examples}

**轮换与替代解释。** {json.dumps(a['rotation_descriptions'],ensure_ascii=False)}。代码旋转下的稳定错误值或稳定行号只排除有限的固定字母解释。提供正确数值/行号还改变提示长度、计算负担及历史，正确率提高只能在该提供信息的对照范围内解释，不能当作内部状态的纯因果中介。输出正确中间步骤不是忠实内部推理的保证；Value/Slot/Code的新增显式输出也可能改变原来隐式选择过程。两步和三步由不同家族承担，深度差不能解释为同程序的配对因果效应。本轮比前轮更大且材料独立，但未覆盖自然语料、更多语言、所有24种代码排列或其他模型。

**理论增量和未完成工作。** 为外部图谱提供96家族的带真值操作、选项表、代码双射及逐字段失败边界材料，未登记为正式中心图谱；内部图谱无新验证B36节点，关联图没有新增神经元因果边。Phase2928的原生传播/格式反例、Phase2930的行为范围和Phase2931的输出分离全部按原限制保留。RDC主体和全局图谱公式不改，上述M、π是外部任务定义，没有新统一数学理论或AGI结论。行为门失败后未运行路径干预，不能宣称已检验复杂特征沿层间或生成过程形成。

**第一性原理与下一大阶段。** 求值、关系匹配与符号读出可能共享或复用内部计算；要破解编码机制，须把保持数值关系的代码置换与真实内部响应对应起来，并区分神经层l和生成时间t的h_(l,t)。接续任务应围绕本轮错误谱联合组织，而不是继续只换提示词：①围绕终态保持与输出接续组织新材料上的完整因子实验：先保持Steps/Value输出不变，只增添选项表；再在同一选项表下逐项增减Steps和Slot字段；最后区分模型自产前缀与人工提供前缀，检查错误是否随额外更新、字段负担或历史变化。保留原端到端格式和本轮全部失败，不把简化子任务通过冒充原联合门通过。②对行为可靠的冻结范围，采集全部层的原生坐标与多个因果可见位置，明确区分模型自产历史和人工提供历史；比较代码置换下语义关系保持、数值身份基线与历史复述。③之后对固定B36组及匹配组作预注册强度改变，测末端输出和副作用，分别阻断/恢复后续attention与MLP及历史来源；以净输出中介而不是能读出或响应强度定位路径。④只有跨家族状态预测、有限改动预测与真实路径操作同时支持，才判断某个组合结构是在层内接续还是借助生成历史形成。计划未执行的部分不计入本轮结果。

**资源、工程与交付。** 首次启动在GPU互斥关卡停止；保留generation.log，等待其他研究任务正常释放设备，未绕过锁或终止其他进程，恢复日志与时间见admission-resume.json。源码中一个非raw正则字符串触发SyntaxWarning，Python当前解释为预期反斜杠，材料/解析正负例测试通过；冻结后未为消除警告改科研源码。三项材料、代码轮换、错误分离测试通过；全部672输入分词、程序AST独立真值、题面表格、输出token解码、指标与产物身份复核，存在不一致对的精确p值用SciPy独立核对；无不一致对时按定义p=1。未改共享协议或生产客户端，未重跑全套协议回归。无新batch/无padding逐token等价测试，继承的历史HF四题对照不能替代本轮全范围复核。HTML只作静态链接/内容检查，不声称浏览器桌面/手机渲染验收。CUDA分配峰值{stats['cuda_peak_bytes']/2**30:.3f}GiB，共享峰值{stats['dxgi_shared_peak_bytes']/2**20:.1f}MiB，生成各批累计{stats['generation_seconds']:.1f}秒。所有新增数据D盘，预算300MiB、保留20GiB余量，交付前空闲{stats['free_bytes']/2**30:.2f}GiB；逐token进度行记录剩余估计及资源。保留材料、完整输出、统计、源码、失败与恢复日志；仅清理本轮独占临时缓存，字节见cleanup.json。没有服务器发布或正式证据登记。

脚本tests/glm5/b36_map14*.py、test_b36_map14.py；命令为.venv/Scripts/python.exe -B -X utf8 -u相应脚本：b36_map14.py冻结，test_b36_map14.py工程检查，b36_map14_run.py或queue.py串行执行，analyze.py审计分析，finish.py交付。结果tests/glm5/result/b36_map14_20261114_v1/report.html及analysis.json、plan.json、source.zip、manifest.json。记录尾部追加，验证原内容字节前缀未变。

通俗说明：把一道题拆成“算出多少”“这个数在哪一行”“这一行用哪个字母回答”，再分别给模型提供前一步正确结果。这样能看出它在哪种条件下开始输出不一致；仍须通过内部干预才能确定真正的神经元路径。
'''
    r1=append(S.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','数值求值—选项匹配—代码读出的96家族独立分解验收',body)
    r2=append(S.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','B36数值到代码分解验收索引',f"详见GLM5 Phase {r1['phase']}，运行{S.RUN}。96新程序家族、672配对生成，区分自行求值、给定数值匹配行号、给定行号读代码，代码轮换保持行和值不变。联合行为门{sum(c['passed'] for c in a['cells'])}/14格，未全部通过，B36路径干预未启动。两项主要McNemar检验按Holm校正，逐字段错误及完整反例保留；输出分离不等于内部模块分离，RDC公式不变。先因GPU互斥等待后恢复，无干扰其他进程。全部D盘，审计与源码归档见tests/glm5/result/b36_map14_20261114_v1/report.html。")
    S.save('memo-append-receipt.json',dict(records=[r1,r2]))
    source=sorted(set(list((S.ROOT/'tests/glm5').glob('b36_map14*.py'))+[S.ROOT/'tests/glm5/test_b36_map14.py',S.ROOT/'tools/power_analysis.py']+[S.ROOT/'tests/glm5'/n for n in p['source_hashes']]))
    old=S.ROOT/'tests/glm5/result/b36_generated14_20261113_v1/source.zip'
    current_names={str(q.relative_to(S.ROOT)).replace('\\','/') for q in source}
    with zipfile.ZipFile(old) as previous,zipfile.ZipFile(S.OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in previous.namelist():
            if name not in current_names:z.writestr(name,previous.read(name))
        for q in source:compile(q.read_text(encoding='utf8'),str(q),'exec');z.write(q,str(q.relative_to(S.ROOT)).replace('\\','/'))
    with zipfile.ZipFile(S.OUT/'source.zip') as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        for q in source:assert z.read(str(q.relative_to(S.ROOT)).replace('\\','/'))==q.read_bytes()
    target=S.TEMP.resolve();assert target==(S.ROOT/'tests/glm5_temp/b36_map14_20261114_v1').resolve() and target.drive.upper()=='D:'
    paths=list(target.rglob('*'));assert all(not q.is_symlink() and not q.is_junction() and q.resolve().is_relative_to(target) for q in [target]+paths)
    size=sum(q.stat().st_size for q in paths if q.is_file());shutil.rmtree(target)
    S.save('cleanup.json',dict(removed_bytes=size,removed_directory=str(target),scope='Only current-run scratch; all frozen results/logs/raw generated tokens retained.'))
    entries=[dict(path=str(q.relative_to(S.OUT)).replace('\\','/'),bytes=q.stat().st_size,sha256=S.file_sha(q)) for q in sorted(S.OUT.rglob('*')) if q.is_file() and q.name!='manifest.json']
    S.save_json(S.OUT/'manifest.json',dict(run_id=S.RUN,files=entries,provenance=S.read(S.OUT/'delivery.json')['provenance']))
    assert all(S.file_sha(S.OUT/e['path'])==e['sha256'] for e in entries)
    for link in re.findall(r'href="([^"]+)"',(S.OUT/'report.html').read_text(encoding='utf8')):assert (S.OUT/link).is_file(),link
    print('Appended Phase',r1['phase'],'Claude',r2['phase'],'verified manifest',len(entries),flush=True)

if __name__=='__main__':run()
