"""Archive actual calibration, append evidence boundary, and clean owned scratch."""
import json,hashlib,shutil,zipfile,importlib.metadata
import b36_generated14 as G
from b36_semantic14_memo import append
S=G.S

def run():
    S.configure();a=S.read(S.OUT/'analysis.json');assert not a['gate_passed'],'Passing pilot requires further work, not a failure closeout.'
    assert not (S.OUT/'memo-append-receipt.json').exists()
    rows=S.read(S.OUT/'chunk-000.json')['results'];p=S.read(S.OUT/'plan.json');by={r['id']:r for r in p['rows']}
    logs=[json.loads(x) for x in (S.OUT/'progress.jsonl').read_text().splitlines()]
    resource=dict(cuda_peak_bytes=max(r.get('cuda_peak',0) for r in logs),shared_peak_bytes=max(r.get('shared_gpu_bytes',0) for r in logs),elapsed_seconds=S.read(S.OUT/'chunk-000.json')['elapsed_seconds'],free_bytes=shutil.disk_usage(S.ROOT).free)
    S.save('delivery.json',dict(resource=resource,packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','safetensors']},model_identity_reference='plan.json/model_files',mechanism_interventions=0,heldout_responses=0,source_checks=dict(original=S.file_sha(S.ROOT/'tests/glm5/b36_generated14.py')==p['source_sha256'],amended=S.file_sha(S.ROOT/'tests/glm5/b36_generated14_batch72.py')==S.read(S.OUT/'execution-amendment.json')['source_sha256']),tests='3 unittest cases passed; full72 output-token, truth and parser recomputation.',limitations='No new unpadded/batch equivalence measurement or browser layout test. Prior phase HF audit is a limited inherited check, not a new generation equivalence test.'))
    t=['深度 | 格式 | N | 步骤可解析 | 步骤全对 | 最终答案对 | 格式合规 | EOS结束 | 联合成功','--- | --- | --- | --- | --- | --- | --- | --- | ---']
    for c in a['cells']:t.append(' | '.join(str(c[k]) for k in ['depth','format','n','steps_parseable','steps_correct','final_correct','format_correct','eos','joint']))
    examples=[]
    for r in rows:
        if not r['metrics']['joint'] and len(examples)<4:examples.append(f"样例 `{r['id']}`：真值{by[r['id']]['truth']}，最终目标{by[r['id']]['expected']}；实际生成 `{r['text'].replace(chr(10),' / ')}`。分类：{json.dumps(r['metrics'],ensure_ascii=False)}。")
    body=f'''**任务和状态。** 承接Phase2930，校准独立新材料上的模型自产中间状态、最终答案与停止行为。运行{S.RUN}。已完成12个新程序家族×两/三步×直接数字及两种代码映射，共72次完整贪心生成。72次中间步骤全部正确、格式及EOS全部合格；直接数字24/24，代码答案43/48。5次错误集中于3个家族，都是正确打印步骤之后选错最终项。冻结联合门5/6格通过，尚未全部过门；B36固定32单元的实际输出路径追踪、层间与生成过程中形成的比较均未启动。不能把行为失败当作B36干预零效应。

**额外的描述性反例。** 两个家族在code0和code2下均选择相同的错误数值，只是字母由D变B：最终真值9却选11，真值25却选26。另一次真值27却选28。故固定偏好某一个字母不足以解释全部错误；“步骤值正确→最终选项错误”是可见输出的分离，不足以定位内部语义、选项读取还是格式处理路径，也不能把打印的步骤当作忠实内部推理。详见error-description.json，事后描述，不增加独立N。

**模型、材料与冻结。** Qwen3-14B原始BF16快照`{S.SNAPSHOT}`，14个原始权重/tokenizer文件重新核对哈希。CUDA逐层送入原权重，HF标准生成与KV缓存，左padding、batch72、SDPA、do_sample=False、enable_thinking=False、max_new_tokens=64。非thinking模式依然允许模型生成所要求的中间状态，不提供任何中间正确值或答案示范，不teacher-force、不强制答案token。没有4B或GLM4新运行。材料从未测完整操作程序选取，排除上阶段已测54个算术家族，初始数10–29，因此与前轮初始0–9的语义前缀不重合；完整程序、初始状态及所有代码视图按家族保留。选择108个候选中前12为发现试跑，余96未送入模型；沿用的旧文法候选曾有划分标签，本轮按新数值任务重新登记，不能同时算旧协议的未暴露确认结果。原创构造数据，无真实个人信息。

**原理与评分。** 任务真值s_k=T_k(s_(k−1))，a=π(s_d)。要求第一行Steps列出每次更新后数值，第二行Final给数字或答案代码，随后EOS。这些真值是外部评估定义，不意味着模型内部按同样步骤实现。中间步骤全部可解析且逐项相等记C，最终答案等于目标记A，恰合两行格式记F，64token内实际产生EOS记E，联合指标J=C∧A∧F∧E。每个深度×格式格需至少11/12联合成功，六格全部通过才继续确认和机制；没有结果后改门。分别记录可解析、计算错误、正确步骤后错误最终答案、格式、EOS；解析失败归未知，不能简单等同计算失败。没有EOS而耗尽预算记截断，不推断模型永远不会停止。小样本为发现筛查，无必要性/正确率干预的功效主张。

'''+ '\n'.join(t)+f'''

完整结果见analysis.json与report.html，原始token与全文见chunk-000.json。达到门的条件数{sum(c['passed'] for c in a['cells'])}/6，不允许选取答对题补过。样例为实际输出：

'''+ '\n\n'.join(examples)+f'''

**实现与资源调整披露。** 初始batch12在无已提交结果时停止，原因是逐生成步重新搬运14B权重的吞吐明显较单token评分慢。保持原始脚本和日志，单独保存execution-amendment.json及batch72执行脚本；只变批次及进度日志频率，材料、贪心规则、64token上限、指标、阈值未变。修改前未查看成功/失败结果。一次命令因PowerShell引号错误未创建执行脚本，未产生模型结果；改用明确的脚本生成器恢复。批次变更可能造成数值差别，没有新增batch12/72逐token等价检查，不能声称消除了所有实现替代解释。继承上一阶段四题HF核对仅作背景证据。

**严格审视与理论边界。** 本轮只检查12个程序家族，代码视图不是独立样本；未跑96候选确认，不能作总体性能估计。数值范围与上阶段不同，不能把跨阶段成绩变化完全归因于多生成了中间步骤。完整答案须依固定可解析格式评分；格式外自然解释可能包含正确结果，保留全文供审查，但不事后放宽门。模型写出的步骤是可观察文本，并不保证忠实表征内部计算；即便步骤全对，也不能推断特征沿对应神经层形成。未采集本轮机制原场、未执行B36组消融或救援、未比较同预算字典，不存在新必要性、充分性、净中介或跨任务机制结论。

**三图谱和RDC。** 外部图谱增加自产状态序列与最终输出映射的失败/成功格和停止信息。内部图谱与关联机制图无新验证节点/因果边；B36的32单元仍是原冻结候选，Phase2928既有传播拼图边界保留，Phase2930格式观察保留。RDC没有新公式、新数学运算或AGI结论，J及状态递推只是实验定义。

**第一性原理与下一大阶段。** 要区分层间形成与生成过程中形成，需要把网络深度l与生成时间t分开，记录h_(l,t)及模型真实自产历史，避免把输入到下一步的已生成答案视为独立语义发现。下一大阶段应先定位当前主要失败类型，再冻结有限的校准修订：将计算真值、代码映射和结束符的评估分离，加入只复制已给定序列与只转换答案代码的匹配控制，确定失败是状态更新还是读出接口。控制过门后，用更大新家族确认；再对同一材料比较未生成中间文本的末位状态与已生成历史条件，在不提供人工真值的前提下选择读出位置、保存原生场，分别检验B36写入、后三层attention/MLP与历史token的阻断和恢复。须预注册模型自身输出历史与实验者提供历史的不同证据级别，不能再次仅换提示句式宣称理论突破。该后续为计划，未执行。

**验收、保存与清理。** 三项材料/解析器工程测试通过，72条原始输出token解码、材料真值、输入分词与指标逐项重算。未改共享协议或生产客户端，未跑全套协议回归，无发布。日志逐生成步记录时间、预计剩余和CUDA/DXGI资源，实际运行{resource['elapsed_seconds']:.1f}秒；CUDA分配峰值{resource['cuda_peak_bytes']/2**30:.3f}GiB，共享峰值{resource['shared_peak_bytes']/2**20:.1f}MiB。全部新增项目数据在D盘，冻结新增峰值300MiB，保留20GiB余量。保存源代码、原始/修订执行计划、全部结果、失败日志和源码包；仅清理本轮独占临时目录，实际字节见cleanup.json。收尾清单逐文件哈希验证，记录尾部追加并核对原前缀。结果tests/glm5/result/b36_generated14_20261113_v1/report.html，执行入口tests/glm5/b36_generated14.py与b36_generated14_batch72.py；命令为.venv/Scripts/python.exe -B -X utf8 -u入口，prepare先生成冻结材料，run生成，analyze重算，finish交付。

通俗说明：这次让模型把每一步真正写出来，再写最终答案并停下。可以检查它从哪一步开始错、是否只是选错编号或没有按要求结束，但这些文字本身还不是内部齿轮的证据。联合可靠性不足时不继续给32个单元下机制结论。
'''
    first=append(S.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md','模型自产中间状态、最终答案与停止行为的联合校准',body)
    second=append(S.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md','B36自产步骤行为校准索引',f"详见GLM5 Phase {first['phase']}，运行{S.RUN}。12新程序家族、72次原始BF16 CUDA生成，独立记录步骤、最终答案、格式与EOS，六格联合门未过。96候选家族未送入模型，B36路径干预未启动，未检验复杂特征层间/生成时间形成。无新RDC公式。保留真实输出、批次调整与全部失败；入口tests/glm5/result/b36_generated14_20261113_v1/report.html。")
    S.save('memo-append-receipt.json',dict(records=[first,second]))
    sources=list((S.ROOT/'tests/glm5').glob('b36_generated14*.py'))+[S.ROOT/'tests/glm5/test_b36_generated14.py']
    with zipfile.ZipFile(G.OLD/'delivery-source.zip') as old,zipfile.ZipFile(S.OUT/'source.zip','w',zipfile.ZIP_DEFLATED) as out:
        for name in old.namelist():out.writestr(name,old.read(name))
        for path in sources:compile(path.read_text(encoding='utf8'),str(path),'exec');out.write(path,str(path.relative_to(S.ROOT)).replace('\\','/'))
    with zipfile.ZipFile(S.OUT/'source.zip') as z:assert z.testzip() is None
    target=S.TEMP.resolve();assert target==(S.ROOT/'tests/glm5_temp/b36_generated14_20261113_v1').resolve() and target.drive.upper()=='D:'
    files=list(target.rglob('*'));assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(target) for p in [target]+files)
    size=sum(p.stat().st_size for p in files if p.is_file());shutil.rmtree(target)
    S.save('cleanup.json',dict(removed_bytes=size,removed_directory=str(target),retained='All original inputs, output tokens, metrics, logs, source and identity. Only current-run scratch removed.'))
    entries=[dict(path=str(q.relative_to(S.OUT)).replace('\\','/'),sha256=S.file_sha(q),bytes=q.stat().st_size) for q in sorted(S.OUT.rglob('*')) if q.is_file() and q.name!='manifest.json']
    S.save_json(S.OUT/'manifest.json',dict(files=entries,provenance=S.read(S.OUT/'delivery.json')['provenance'],run_id=S.RUN))
    assert all(S.file_sha(S.OUT/e['path'])==e['sha256'] for e in entries)
    print('Finished Phase',first['phase'],'index',second['phase'],'manifest entries',len(entries),flush=True)

if __name__=='__main__':run()
