"""Append-only scientific closeout, evidence manifest and mandated provenance."""
import hashlib,json,re,subprocess
from datetime import datetime
from pathlib import Path
import mlp_puzzle_campaign as P
from tools import provenance

def append_phase(path,title,body):
    before=path.read_bytes();text=before.decode('utf-8-sig')
    assert title not in text,'Already appended; refuse duplicate phase'
    n=max(int(v) for v in re.findall(r'^## Phase (\d+):',text,re.M))+1
    date=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    addition=f'\n\n## Phase {n}: {title} [{date}]\n\n{body}\n'
    with path.open('ab') as f:f.write(addition.encode('utf-8'))
    after=path.read_bytes();assert after[:len(before)]==before
    return {'phase':n,'path':str(path),'old_bytes':len(before),'old_prefix_sha256':hashlib.sha256(before).hexdigest(),
            'appended_bytes':len(addition.encode('utf-8')),'prefix_unchanged':True}

def main():
    plan=P.checked();r=P.read(P.OUT/'client-report.json');reg=P.read(P.OUT/'registration.json');qa=P.read(P.OUT/'client-acceptance.json')
    assert qa['passed'] and not qa['errors'] and not qa['gpuRequests']
    assert P.sha(Path(reg['bundle']))==reg['bundle_sha256']
    scripts=['mlp_puzzle_campaign.py','mlp_puzzle_causal_checked.py','mlp_puzzle_causal_failed.py','mlp_puzzle_baseline_audit.py',
             'mlp_puzzle_audit_publish.py','test_mlp_puzzle_campaign.py','mlp_puzzle_http_check.py','verify_mlp_puzzles.cjs','mlp_puzzle_closeout.py']
    dependencies=['tests/glm5/mlp_condition_design.py','tests/glm5/mlp_condition_stream_load.py','tests/glm5/mlp_condition_run.py',
                  'shared/dist_protocol/mechanism_trace.py','frontend/src/components/v8/MlpPuzzleReport.jsx',
                  'frontend/src/components/v8/ResearchInterventionReport.jsx','frontend/src/components/v8/ModelStructure.jsx']
    source_hashes={p:P.sha(P.ROOT/p) for p in [*[f'tests/glm5/{s}' for s in scripts],*dependencies]}
    provenance.RESULTS=P.OUT/'provenance'
    provenance._git=lambda *args:subprocess.run(['git','-c',f'safe.directory={P.ROOT.as_posix()}',*args],cwd=P.ROOT,capture_output=True,text=True).stdout.strip()
    provenance.model_revision=lambda _:P.SNAPSHOT
    artifact=provenance.write_result('native_puzzle_closeout',{'run_id':P.RUN,'registration':reg,'source_hashes':source_hashes,
        'formal_evidence_promotion':False,'admissibility':r['audit']['admissibility'],'browser_acceptance':qa,
        'cleanup':P.read(P.OUT/'cleanup.json'),'retention':'Native 640 family arrays retained for scientific reuse; no previous experiment deleted',
        'dependency_scope':'Main experiment source frozen before discovery; dependency hashes recorded at closeout, not falsely described as pre-run frozen'},
        model_path=P.MODEL,seed=P.SEED,tool='mlp_puzzle_closeout')
    assert not provenance.check_provenance(P.read(artifact))
    effects='\n'.join(f'| {e["group"]} | {e["mean_margin_loss"]:.6f} | {e["control_mean_margin_loss"]:.6f} | {e["effect_prediction_mse_ratio"]:.6f} | {e["valid_controls"]}/256 | {"满足方向性数值门" if e["use_supported"] else "未满足"} |' for e in r['effects'] if e['dose']==1)
    baseline=r['audit']['baselines'];g={g['id']:g for g in r['groups']}
    body=f'''**任务与证据身份。** 按用户要求，以既有 Qwen3-4B B28—35 原生 MLP 归档为起点，完成代表单元/组的解释、反例、未见材料预测、原模型作用测试和本机客户端同步。运行 `{P.RUN}`；BF16、CUDA、SDPA，唯一模型快照 `{P.SNAPSHOT}`；块 B 从0编号，对应 hs[B+1]。旧归档已暴露，只用于选取探索性代表，不作为新盲测。未加载其他模型、未修改模型权重、未提交Git或部署网站。结果位于 `tests/glm5/result/mlp_puzzle_v1/`。

**设计、采集与真实规模。** 从每块选6个原生单元：旧词身份预测、高双负能量、稀有强响应、高方差各1，加预抽随机2，共48；它们是描述性代表，不逐一宣称语义确认。在128个发现家族的前96个上，逐块选择8单元的绑定组和语言组，共16组，同层随机对照按激活尺度匹配；后32个发现家族选重点块，然后在128家族重拟合并冻结。重点作用对象为 B34/{{2359,842,2406,1189,2569,7114,4321,3236}}、B28/{{179,370,9480,7038,7956,564,4005,2800}}、单个 B34/U2359、旧稀有候选 B33/U1521。

人工材料给两个对象分别记录红/蓝，变化查询对象q、记录映射r、输出语言L，完整2×2×2交叉；q=+1查询首对象，r=+1首对象红，L=+1英文，−1中文。试跑16家族×8=128条，发现128×8=1024条，留出512×8=4096条。留出换记录ID、对象名词、4种句式，家族完整切分。三批首token正确率均100%，各句式强制选择也100%；64条确定性短生成得到red/blue/红色/蓝色各16次。模型确实完成了本受控任务，不能外推到开放翻译、风格或递归。另256家族各取一个配平变体，4对象×2剂量×目标/随机组/等范数方向，构成作用检验；512道已暴露选择题仅作副作用检查。

采集保留全部8块末位置A；发现集全G/V，留出仅冻结组、对照和代表单元的G/V；全部37个HiddenState在记录末端和最终读出两个位置，含hs[0]及最终归一化边界。发现集保留全词表logits；留出全词表KL当场计算后不重复保存logits。640个NPZ的哈希、5120条发现/留出输入唯一性核查通过。所有这些输入最终token的embedding逐位相同，但末层响应随条件变化。此事实排除了读出位置的单token身份解释，不能排除整个前缀的词序、长度等因素。全模型398张量按流式加载逐张量核对，避免既有Windows映射加载故障。

**数学对象与可执行解释。** 原生计算为 z=RMSNorm(residual_after_attention)，G_j=w_gate,j·z，V_j=w_up,j·z，A_j=SiLU(G_j)V_j，MLP写回 Σ_j w_down,j A_j。真实hook核对 A 与 SiLU(G)⊙V 一致。条件预测采用 X=(1,q,r,L,qL,rL,qr,qrL)，β=(XᵀX)⁻¹XᵀA，Â=Xβ；绑定弱基线去掉qr、qrL，语言弱基线去掉含L项。这是饱和八条件均值表，不是新数学结构；答案颜色c=qr由作者提供的关系运算给出，不是算法从无标签文本发现。预测时不读待预测A或未来token，但人工解析变量已经包含很强的任务知识。

具体例子：新家族0的 B34/U2359 在红色相关条件中 G约6.53—7.50、V约−4.78—−3.22，A约−31.25—−23.25；蓝色相关条件G约−3.17—−1.99、V约−3.91—−3.27，A约+0.418—+0.891。两种输入颜色始终同时出现，变化的是查询与记录关系。负门控并非硬关闭，SiLU负值乘负V可以得到正A。同一权重和同一原生单元能产生条件化、有符号的不同写入，但A符号不直接等于“支持/反对答案”。

**未见响应结果与强替代解释。** 16/16组满足原冻结数值门：误差比≤0.8、家族胜率≥0.65、按16组校正的bootstrap改善下界>0，实际每组512/512家族优于弱基线。B34绑定组误差比 {g['binding-B34']['relative_mse']:.6f}，同层随机组 {g['binding-B34']['random_group_relative_mse']:.6f}；B28语言组 {g['language-B28']['relative_mse']:.6f}，随机组 {g['language-B28']['random_group_relative_mse']:.6f}。48个旧档案代表中5个平均误差不优于加法基线，不能从旧形态直接升级为此任务功能。

随后明确标为事后挑战，增加更强简单基线：仅用(c=qr,L)四状态均值表，即β的(1,L,qr,qrL)四列；系数仍只由发现集拟合。16组中11组四状态比八状态预测更好；B34绑定组八/四状态误差比 {baseline['groups']['binding-B34']['eight_over_four']:.6f}，B28语言组 {baseline['groups']['language-B28']['eight_over_four']:.6f}。全词表平均KL：八状态 {baseline['output_KL']['full']:.9f}，四状态 {baseline['output_KL']['four_state']:.9f}，弱加法 {r['output_prediction']['additive_KL']:.6f}，常数 {baseline['output_KL']['constant']:.6f}。输出只有四种强确定状态，而且预测器直接拟合logits；这些极小KL不构成单元图谱独立生成，也不能当作整体破解率。应保留四状态响应拼图，撤回“需要更复杂八条件机制”的解释优势。

**原模型作用与反例。** 原生A减半/置零，只动答案前最后位置；重算完整后续层。4个对象的后缀重算与完整原模型置零逐位一致；256家族取消编辑后原logits逐位恢复，属于实现校验，不是路径救援。下表为完全置零，下降为原正确优势−编辑后正确优势，正数表示削弱正确答案；语言组比较正确语言和另一语言的同色首词，其他组比较同语言的红蓝首词。

| 对象 | 正确优势下降 nats | 等范数方向对照 | 效应预测MSE/不变基线 | 有效对照 | 冻结方向性数值门 |
| --- | ---: | ---: | ---: | ---: | --- |
{effects}

B28组超过等范数对照的改善区间约[0.4116,0.6231] nats；同层随机组平均−0.01318，也显著小于目标0.50098。但分层显示效应主要在中文：红中文0.5293、蓝中文1.5234，红英文−0.0508、蓝英文0.0020，因此不能称为一般双语控制器。B34绑定组置零后平均正确优势反而增加0.16162；其红中文下降+0.3652、红英文下降−0.7266，符号随条件改变。冻结的“一贯正向帮助绑定答案”解释未过门，反向、条件化贡献作为新候选保留，不改名后充当原假设通过。B33/U1521置零效应约0.00586，预测误差比1.0028，未胜过不变预测；旧稀有强相关不自动提供此任务功能。所有对象与剂量的首token改变均0/256，且分布KL都很小，没有建立行为必要性或充分性。

历史512题原模型正确370；B34组置零原对改错4、原错改对2，B28组2、0。无法据此宣称普遍能力保持。效应预测需要原模型基线logits加发现集校准增量，与仅凭输入条件预测logits分别计账。

**工程失败、协议硬伤与正式准入。** 第一轮作用测试在第一个家族发现：把已经实现的BF16差值再次转BF16会改变随机对照，尚无完成的因果记录即停止。保留failed脚本和日志；`control-amendment.json`记录修正函数、时间、哈希和范围，只取消这次二次量化，不改候选、种子、剂量或科学阈值。重跑2048次实际对照差值逐位一致；2%范数门不合格的样本如实留在分母中。

更重要的是，运行前虽写连续效应MDE近似，却没有执行仓库第13节要求的二值行为功效准入，也没有完整三栏。事后调用工具，在n=256、p_d=0.10、α=.0125下，检测2/3/5个百分点分别需2788/1240/447题，当前功效约.069/.164/.513；事后计算不能补成预注册通过。原始plan.provenance缺git_dirty，原始逐家族JSON通过plan哈希溯源而未各自带完整五字段；保留原文件，客户端出版物及本次helper写出的审计补齐五字段，不倒填冻结。主脚本有事前哈希，完整导入依赖只有收尾哈希。plan字段说绝对效应，但冻结执行代码用“正向优势下降≥0.2”，本报告遵守执行口径并公开差异。**这些准入缺口使本轮不晋级正式证据，新增正式机制确认数为0。** 数值门结果保留为待审单节点观察和局部作用数据。

另外只有4种留出句式；名词与句式未独立交叉，bootstrap按记录家族而非句式簇；因果子集只取每家族一个条件，语言与句式相关，分层结果描述性。没有自然语料SAE/transcoder平行比较、注意力来源分解、单参数干预、全生成链因果验证或独立节点复现。共享显存计数不可读，日志null不等于0；CUDA峰值约9134MiB，未出现OOM。这些限制阻止将结果外推到知识网络、聊天/辩论融合、泛化递归或AGI。

**客户端交付与验收。** 本机两个研究库已登记64张卡（48代表单元+16组）和16条明确标为观察/关联的语言映射边，未增加因果边等级。新报告五页：能保留的拼图、作用与副作用、反例与失败、代表单元、材料与审计。可查8条件热力图、真实G/V/A、四状态强基线、两剂量及对照、分层符号反转、原始输入、失败理由和完整哈希。组内任一成员可从统一空间详情跳回组证据，显示“组有效不等于每个成员独立有效”。浏览器8项验收通过、错误0、GPU请求0；两张截图已目视核对；Python科学/精度测试4项、前端56项通过，生产构建成功，保留既有大分块警告。最初证据链接验收暴露开发端口相对URL问题，已改用项目apiUrl并重跑。正式远端、安装版及离线包尚未验证或发布。

当前客户端报告哈希 `{reg['report_sha256']}`，图哈希 `{reg['graph_sha256']}`，可上传包 `{Path(reg['bundle']).name}`，包哈希 `{reg['bundle_sha256']}`。旧出版修订保留，但同一run按最新版读取，不当成新样本或复现。

**理论增量、停止条件与下一大阶段。** 有限参数下可扩展组合的一个必要问题是：模型如何先算出关系状态，再把状态转成有条件、有符号的写入。当前已找到末层对(c,L)的稳定响应及条件化作用；尚未找到从原始token计算c=qr的上游机制。由此转向“状态形成与条件读写追踪”大阶段，而非继续给末层热区加语义名：

1. 先复用本轮已保留37层HiddenState，探索四状态从哪层、哪个合法读出位置开始出现，冻结候选层和原生坐标组合；此处已暴露材料只作发现。
2. 新文法让实体、≥16种句式、颜色词、语言、位置和关系深度独立交叉，加入否定/覆盖旧记录等真正反例；比较四状态查表、词袋/位置/n-gram和适用的字典/电路方法。所有材料和依赖哈希、二值与连续功效预算必须在运行前自动过门；3个百分点二值效应至少遵守工具所需1240有效配对并另考虑句式聚类，不能直接沿用256。
3. 在冻结的上游候选中分别追踪QK选择与V/O写入、归一化和G/V门控，测试来源遮断、真实路径恢复与匹配负控制。目标是用较小计算关系预测未见关系组合的内部状态及输出，而非给预测器直接提供已算出的答案状态。
4. 若复杂候选在同一未见材料上不优于四状态/简单基线，停止增加条件项；若响应能读出但作用符号不稳定，转为条件化贡献命题；若只有近饱和答案无翻转，先改善试验辨别力和功效，不无休止放大剂量。负结果按对象/条件/指标登记，只否定相应命题。

**复算、资源与清理。** 实际命令：`python -B -u tests/glm5/mlp_puzzle_campaign.py --stage freeze/pilot/discovery/select/calibrate/holdout/capability/generation/report`（各阶段分别调用）；作用用 `mlp_puzzle_causal_checked.py --stage causal`；强基线用 `mlp_puzzle_baseline_audit.py`；核验/登记用 `mlp_puzzle_audit_publish.py audit/publish`；`python -B -m unittest tests/glm5/test_mlp_puzzle_campaign.py`；`node tests/glm5/verify_mlp_puzzles.cjs`；frontend内vite build及vitest run。均使用本地.venv，GPU阶段严格顺序，最长单次留出采集约146秒，逐批换行进度。测试产物约2.52GiB，保留原生数组、logits发现场、冻结预测器、输入、日志与反例，供后续研究，不删除旧研究数据。仅清理本轮可重建客户端构建目录，范围与释放字节见cleanup.json。全部源码收尾哈希及五字段审计：`{artifact.relative_to(P.ROOT).as_posix()}`。后续不得把这次事后审计、重复上传或重新打开同一留出集算作新增盲测。
'''
    result=append_phase(P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','Qwen3-4B尾层原生拼图闭环、四状态强基线与作用反例',body)
    cross=append_phase(P.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','尾层拼图闭环与强基线审计',
        f'详见 GLM5 Phase {result["phase"]}，运行 `{P.RUN}`。48代表单元、16组，512×8未见输入；16组超过弱基线，但四状态强基线在11组更好。B28语言组有条件化局部作用，B34组合组呈反向/符号变化，稀有单元未过作用预测，全部首词不变。已同步64卡、16观察边到本机客户端，浏览器8项、Python4项、前端56项和构建通过。功效准入、原始溯源及冻结依赖存在已记录缺口，因此正式机制晋级0；没有远端发布。下一阶段转向上游状态形成与条件读写。保留2.52GiB证据，清理仅本轮客户端构建。')
    P.save('memo-append.json',{'glm5':result,'claude':cross,'provenance_artifact':str(artifact)})
    print(json.dumps({'phase':result['phase'],'claude_phase':cross['phase'],'prefix_unchanged':True,'provenance':str(artifact)},ensure_ascii=False))

if __name__=='__main__':main()
