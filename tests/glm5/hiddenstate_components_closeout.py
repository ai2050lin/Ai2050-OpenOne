"""Verify every published coordinate, preserve provenance, append research notes."""
import base64,hashlib,json,re,shutil,tarfile
from datetime import datetime
import numpy as np
import hiddenstate_components as H
from deploy.research_library import Library,canonical


def main():
    plan=H.checked();reg=H.read(H.OUT/'registration.json');report=H.read(H.OUT/'report.json')
    assert reg['model_snapshot']==plan['model_snapshot']
    acceptance=H.read(H.OUT/'client-acceptance.json');assert acceptance['passed'] and not acceptance['errors'] and not acceptance['gpuRequests']
    assert 'OK' in (H.OUT/'unit-tests.log').read_text(encoding='utf-8-sig')
    assert '56 passed' in (H.OUT/'frontend-tests.log').read_text(encoding='utf-8-sig')
    assert 'built in' in (H.OUT/'build.log').read_text(encoding='utf-8-sig')
    libs=[Library(path) for path in reg['local_libraries']];arrays=np.load(H.OUT/'analysis-arrays.npz');verified=[]
    with tarfile.open(reg['bundle'],'r:gz') as tar:
        manifest=json.load(tar.extractfile('MANIFEST.json'));assert len(manifest['objects'])==reg['objects']==45
        for obj in manifest['objects']:
            digest=obj['sha256'];raw=tar.extractfile(f'objects/{digest}.json').read()
            assert hashlib.sha256(raw).hexdigest()==digest;doc=json.loads(raw)
            for lib in libs:assert canonical(lib.artifact(digest))==raw
            assert not H.provenance.check_provenance(doc)
            if doc['schema']=='hiddenstate_components_coordinates.v1':
                layer=doc['hidden_index'];decoded=np.frombuffer(base64.b64decode(doc['array']['data']),'<f4').reshape(doc['array']['shape'])
                expected=np.stack([arrays[key][layer][:,[3,6,7],:] for key in ['discovery_beta','evaluation_beta']])
                np.testing.assert_array_equal(decoded,expected);verified.append(layer)
    assert sorted(verified)==list(range(37))
    snap=plan['model_snapshot'];cleanup=H.read(H.ROOT/'tests/glm5_temp/hiddenstate-cleanup.json')
    assert cleanup['removed'] and not H.Path(cleanup['path']).exists()
    H.save('cleanup.json',cleanup,snap)
    (H.ROOT/'tests/glm5_temp/hiddenstate-cleanup.json').unlink()
    sources=['tests/glm5/hiddenstate_components.py','tests/glm5/hiddenstate_components_publish.py','tests/glm5/hiddenstate_components_closeout.py',
        'tests/glm5/test_hiddenstate_components.py','tests/glm5/verify_hiddenstate_components.cjs',
        'frontend/src/components/v8/HiddenStateComponentsReport.jsx','frontend/src/components/v8/hiddenstate_components.css',
        'frontend/src/components/v8/ResearchInterventionReport.jsx']
    result={'schema':'hiddenstate_components_delivery.v1','run_id':H.RUN,'model_snapshot':snap,
        'registration':reg,'two_libraries_all_objects_verified':45,'full_coordinate_layers_verified':verified,
        'coordinate_values_per_library':37*2*2*3*2560,'scientific_unit_tests':6,'frontend_tests':56,
        'browser_checks':len(acceptance['checks']),'build':True,'component_lint':True,
        'visual_review':['desktop full-layer curve','native 2560-coordinate grid','390px viewport with keyboard-scrolling chart'],
        'source_hashes':{name:H.sha(H.ROOT/name) for name in sources},
        'results_hashes':{f.name:H.sha(f) for f in H.OUT.iterdir() if f.is_file()},
        'cleanup':cleanup,'gpu_used':False,'remote_published':False,'independent_new_confirmation':0,
        'free_disk_MiB':shutil.disk_usage(H.ROOT).free/2**20,
        'unverified':['new GPU or causal experiment','fresh unseen language/style/logic/grammar combinations','independent node','installed desktop/offline pack','remote deployment']}
    H.save('delivery-audit.json',result,snap)
    memo=H.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';index=H.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    old=memo.read_bytes();oldindex=index.read_bytes();assert H.RUN.encode() not in old,'This run is already recorded; do not append twice'
    phase=max(map(int,re.findall(rb'^## Phase (\d+):',old,re.M)))+1
    iphase=max(map(int,re.findall(rb'^## Phase (\d+):',oldindex,re.M)))+1
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    entry=rf'''

## Phase {phase}: 全层HiddenState成分形成、条件基准与状态更新的回溯检验 [{stamp}]

**任务与完成边界。** 用户要求继续Phase 2882的成分/状态更新方案。本轮完整交付第一里程碑：复用Phase 2880已保存的Qwen3-4B全层场，回答“输出语言和关系颜色何时可读出、条件之间如何交互、简单状态能否预测后续变化”，同步真实客户端及反例。运行 `{H.RUN}`。这是新分析，不是新模型采集；0次GPU前向，0次新增盲测，0次原模型干预，0项正式机制晋级。风格、一般逻辑、句法、跨组合深度与原生来源追踪尚未完成。同期Phase 2885的坐标4/100及MLP作用是另一运行的证据，不合并成此次颜色/语言的因果解释。

**模型、材料与准入。** 本地models/hf/qwen3-4b，BF16来源，模型/tokenizer完整快照 `{snap}`。核验原640个npz的SHA，128个发现家族×8条件、512个已暴露评估家族×8条件，共5120个唯一分词输入。原始形状为每家族[8条件,37状态,2位置,2560坐标]；逐文件流式读取，不重写原场。q是查询对象，r是记录映射，L是要求的输出语言（+英文、−中文），颜色c=qr（+红、−蓝）。所有提示主体为英文，不能称为输入中英文分离试验。两个位置为record_end及answer_prefix；前者看不到未来q/L，后者全部条件可见。hs[0]是embedding，hs[1..35]是B0..B34后的原始残差，hs[36]含B35与最终RMSNorm，不作为B35纯写入。

先冻结plan及4个依赖源码哈希、原材料/旧审计哈希、模型身份与资源上限，再分析。五字段溯源用tools.provenance.write_result写入用户指定tests/glm5/result/hiddenstate_components_v1/provenance；原文件身份缺口如实继承。功效工具check_phase(Δ=.10,n=512,p_d=.10,α=.05/74)给所需180、实际512、功效1.0并通过运行门；仅作独立家族假设下的二值敏感度预算，不能覆盖4个句式簇、连续向量误差或事后层选择。全部评估材料早已暴露，128/512只是原始划分，不因本次先拟合再读取而恢复盲性。发现估计冻结文件与系数哈希都在重新读取旧评估激活前保存。

**算法与数学对象。** 每家族八条件正交设计X的列为(1,q,r,L,qL,rL,qr,qrL)，满足XᵀX=8I。各层各位置全2560维的条件系数为β_f=XᵀH_f/8；发现集平均β拟合预测器，旧评估均值只作观察。保留全部坐标，不用Top-K或降维定义主干。常数、q/r/L加法、四状态、八条件表四种方法在同一数据上比较原生全状态MSE，预测时不减去测试家族真实均值。

$$
\hat h_\ell(c,L)=\mu_\ell+c\,a_\ell+L\,b_\ell+cL\,d_\ell,\qquad
\text{{语言对照方向}}=b_\ell+c\,d_\ell,\quad
\text{{条件中心}}=\mu_\ell+c\,a_\ell.
$$

该式是四状态经验拟合，不是新定理，不是从token算出c的机制。单因子读出用发现集中心、条件方向及逐坐标方差固定权重，w∝β_factor/max(var,10⁻⁶)，不扫正则或调阈值。比较直接颜色、先分别读q/r再相乘、仅向量范数，以及跨语言的颜色读出、跨颜色的语言读出。条件中心/方向由对应发现条件拟合，但尺度使用整个发现集方差，不能声称训练时完全未接触另一条件。可读出起点按连续3个原始残差状态准确率≥90%描述，排除最终归一化状态；不是信息首次存在或因果写入时刻。

状态更新只预测一步：给真实h_ell，比较原样复制、发现集平均增量、人工提供(c,L)的四状态增量、由源h_ell读出(c,L)再选增量。即ĥ_(ell+1)=h_ell+Δμ+ĉΔa+L̂Δb+ĉL̂Δd，所有增量由发现集拟合，读出仅使用源层。不输入目标层；但真实源层已经含有原网络计算，不能算输入到输出的闭环替代模型。最后35→36因RMSNorm边界不一致排除。另通过原生最终RMSNorm及tied embedding行计算FP32四答案logit lens；终点与旧BF16分数最大误差0.124992nats，低于冻结0.25门。中间lens只是诊断读出，非剩余网络实际输出；本轮未测完整词表KL。

**结果一：不同成分出现于不同阶段，弱背景不能删除。** answer_prefix的语言从hs[15]（B14后）达到99.9512%，颜色从hs[24]（B23后）达到94.4336%，hs[25..35]颜色100%。hs[24]跨语言颜色：英文拟合→中文99.4141%，中文拟合→英文97.9004%。但4个句式的直接颜色准确率分别100%、100%、82.4219%、95.3125%；整体94.4%不能掩盖句式差异。hs[24]颜色仅范数51.1230%，q/r读出相乘55.3955%；后者未达到连续3层90%标准。这只否定所测读出方式，不能断言模型没做关系组合或r不存在。

record_end的未来语言/查询条件在所有37层逐位一致；颜色和语言读出全为50%。answer_prefix的hs[0]在所有5120输入也逐位一致，颜色/语言机会水平，四种状态预测MSE都是0，因此误差比例显示未定义。这个负对照确保我们没有把未来信息或最后token身份误认为上下文机制。

**结果二：交互与条件基准一起变化，整体读出准确不保证规则迁移。** 颜色×语言交互系数占配平条件响应方差的能量：hs[25]约0.2455%，hs[35]约12.2065%。这是实验内响应能量比例，不是模型总信息的语义占比。红蓝两条件语言方向余弦由hs[24]的0.996835变为hs[33]的0.606020、hs[34]的0.614644；方向有变化，但没有反转。整体语言读出在hs[33..35]仍100%，红色条件拟合的固定阈值迁到蓝色却分别50%、50%、50.0488%。

事后代数诊断说明不能把这直接叫作语言信息消失：hs[34]迁移后英语平均得分1.028924、中文0.159197，阈值0，两类均值同为正；目标中心偏移0.594061，大于语言半间隔0.434864。条件中心不适配已足以解释整体失败方向，语言区分仍存在。此处只分析均值，没有用测试标签重新调阈值，更没有声称修正后准确率已经通过。可保留的新候选是“成分读出需要检验方向、条件中心及读者三者”，尚未证明每种语义共享此结构。

**结果三：简单状态基线限制了复杂解释。** hs[24]八条件/四状态MSE=.888743，八条件更好；到了hs[33/34/35/36]，比例依次1.007418、1.024215、1.044026、1.008410，四状态反而更好。hs[35]四状态/常数=.194380，说明此受控任务的全状态MSE能大幅降低，但不能解释成破解80.6%的语言机制。八条件在早层有用而尾层多项冗余，支持研究状态压缩和条件读写的阶段差异，不支持一律加复杂项。

hs[24]→hs[25]一步预测：平均增量MSE=.264062、推断状态增量=.168449，误差降低36.21%；hs[34]→hs[35]分别5.018874与1.283309，降低74.43%。在源状态解码已全对时，推断状态与提供真实状态得到相同预测，这是按构造预期的结果，不是另一次机制发现。早期四状态可能更差，例如hs[0]→hs[1]平均增量约3.1253e−5，推断状态约3.1535e−5。净颜色增量最大的是hs[32]→hs[33]，但它受幅度增长影响，并不等于最早或最关键颜色计算层。

**科学审计与硬伤。** 独立实现显式求和/重建与Parseval核对4个真实家族×6层×2位置，并重算四种预测误差；全部通过。全部74行与逐家族数组对账、零方差、末层归一化和前缀负对照通过。实现细节和数学合格不等于机制正确。新plan没有逐个冻结512个旧logit JSON旁文件哈希，虽与旧plan身份一致，只有收尾记录其SHA；终点校准因此保留这一追溯缺口，不补写冻结。新数组/报告/估计器均有哈希链。相同节点复算不叫独立节点复制。

仍只有4个句式且名词与句式相关，无新增自然语料、SAE/transcoder公平比较、全背景协方差白化、注意力QK/VO/MLP分界、全token轨迹或参数级干预。原生坐标对角方差探针是已知标签的受控检验，不是无监督特征发现；其起点依赖读出器，不能当作统一机制层。平均方向和全状态MSE也会受幅度、家族中心与位置影响。RMSNorm可能改变几何，因此hs[36]跨条件读出恢复100%不能被单独归因于归一化（还包含B35）。全部负结果只否定相应固定读出/预测命题；没有新增关于风格竞争、泛化递归或AGI的实证结论。

**客户端与可追溯交付。** 复用研发实验记录和空间成果库，无新增重复空间。报告四页为全层形成、完整原生坐标、状态更新、边界与反例；展示37×2边界、发现/旧评估两套全2560维系数、实值查询、读出/强基线、阈值迁移失败及原始反例。矩阵按原生索引，色标说明仅当前向量缩放；弱值保留。11张全向量观察卡、11条语言响应关联边，单坐标只作导航锚点，明确不等于一个概念神经元，没有新增因果边。两个本机研究库共45种相同内容产物逐对象SHA回读；每库37个数组全部1136640个浮点值与分析原件逐位一致。

报告SHA `{reg['report_sha256']}`，图SHA `{reg['graph_sha256']}`；可恢复上传包SHA `{reg['bundle_sha256']}`。页面入口为/workbench?lens=process&station=execution&run={H.RUN}&snapshot={snap}。本轮只登记本机两个库，未上传正式远端、未部署网页、未打包安装版。研究包保留供后续受控同步，不把同一包多次登记算新样本。

6项科学计算测试、56项前端测试、11项浏览器验收通过；检查真实旧输入、原生数值、两库一致、缺数据/恢复、归一化边界和手机键盘滚动，页面错误0、模型运行请求0。桌面曲线/原生网格/390px实际视口截图已目视核对。新增组件lint和生产构建通过，保留既有大分块及模块类型提示。浏览器缺数据为隔离请求拦截，未改正式科学产物。未改共享协议/runner，因此没有重跑305项协议测试；安装版和离线包未验收。

**第一性原理与下一大阶段。** 当前最清楚的约束是：同一最后token的初始embedding完全一样，却能得到不同答案，故“仅用最后token向量、无前缀历史的确定性自循环”不可能解释这个任务。候选整体更新必须显式包含前缀各位置/KV历史，或证明某个压缩状态足够；当前一步拟合还未闭合。RDC本轮没有新定理或统一公式升级，只有四状态拟合、条件基准和阶段差异的可计算候选。内部图谱新增全层候选系数，外部图谱仍是旧颜色/语言设计，两图谱新增带失败范围的观察映射。

下一大阶段围绕“条件状态的形成与相对读写”，不继续给末层热区命名：(1)冻结hs[14..25]形成窗口和hs[32..35]条件基准变化窗口，在新实体、独立句式、否定/覆盖记录上预测；比较固定中心、条件中心、四状态及词袋/位置等预算可比基线，失败后收缩具体命题，不调同一锁箱过门。(2)行为校准后加入一项风格和一项句法约束，把输入语言和输出语言分别处理，逻辑用关系操作检验；不能用更多旧模板变体冒充新语义覆盖。(3)采集多个合法位置的attention后/MLP后残差，先判断前缀选择、写入与归一化分别解释多少，再做冻结原模型作用和匹配负控制；复用Phase 2885的分子/分母分工方法，但不直接移植其坐标语义。(4)与字典/电路方法同样材料比较，只有未见组合预测、来源与作用带来新增信息才升级复杂度。若条件中心已解释迁移失败，先保留简单解释；如果只有读出没有更新预测，转查来源而不是反复扩大方向。

**资源、复算与恢复。** 核心CPU流式分析约2分钟，无新增GPU负载；每16/32家族写换行进度，最终640/640，所有结果在tests/glm5/result/hiddenstate_components_v1。冻结预算40MiB；新成果与两份本机CAS在预算内，原2.52GiB来源证据保持不动。清理本轮可重建构建目录60文件、9529754字节，路径与归属核验后删除；保留原场、脚本、日志、系数、审计、截图及小型同步包。当前D盘可用空间仍不足大型新采集。例如48条件×1000家族×37边界×5位置×2560×BF16，单残差数组约42.3GiB（仅预算示例，不是已确定样本量），不能在百MiB余量下开跑。后续需明确实际存储卷和容量，再按行为/聚类功效冻结新采集；不删除旧证据腾空间。

执行入口：hiddenstate_components.py --stage freeze/analyze；hiddenstate_components_publish.py audit/publish；python -B -m unittest tests/glm5/test_hiddenstate_components.py；node tests/glm5/verify_hiddenstate_components.cjs；frontend内vitest run、组件ESLint与vite build；hiddenstate_components_closeout.py完成全产物回读及append。详细源文件/产物哈希、未验证范围及清理见delivery-audit.json、audit.json、cleanup.json，运行计划和失败边界不可被后续修改覆盖。通俗结论：已看到“用哪种语言说”与“该回答哪种颜色”在不同阶段变得可读，且它们的读取规则会相互影响；下一步必须解释这些状态怎么被算出来，不能把分出两类当成破解整套编码。
'''
    cross=f'\n\n## Phase {iphase}: 全层HiddenState的成分、条件基准与更新检验 [{stamp}]\n\n详见GLM5 Phase {phase}。运行{H.RUN}，复用640家族/5120条件的37层×2位置原生场，无GPU/新盲测/干预。语言hs[15]、颜色hs[24]达到连续三层90%读出标准；颜色hs[25]起全对。尾层红色拟合的语言读出迁到蓝色接近机会水平，事后几何发现条件中心偏移与方向变化并存；不能解释为语言信息消失。四状态在hs[33..36]胜过八条件，源状态条件增量改善一步预测，均非完整机制。客户端11观察卡/11关联边、45产物同步两个本机库，37全坐标数组逐值回读；科学6项、前端56项、浏览器11项、lint/构建通过。继承旧准入缺口且全为已暴露材料，正式晋级0；未上传远端。清理仅本轮构建约9.09MiB。下一步为新材料上的条件状态形成、QK/VO/MLP来源与作用；磁盘容量是新全场采集的实际瓶颈。\n'
    assert memo.read_bytes()==old and index.read_bytes()==oldindex,'Memo changed during preparation; retry from latest tail'
    with memo.open('ab') as f:f.write(entry.encode('utf-8'))
    with index.open('ab') as f:f.write(cross.encode('utf-8'))
    assert memo.read_bytes().startswith(old) and index.read_bytes().startswith(oldindex)
    H.save('memo-append.json',{'phase':phase,'index_phase':iphase,'timestamp':stamp,
        'memo':str(memo),'previous_bytes':len(old),'previous_sha256':hashlib.sha256(old).hexdigest(),
        'entry_sha256':hashlib.sha256(entry.encode('utf-8')).hexdigest(),'append_only_verified':True},snap)
    print(json.dumps({'phase':phase,'index_phase':iphase,'verified_arrays':len(verified),'objects':45,'free_MiB':shutil.disk_usage(H.ROOT).free/2**20}),flush=True)


if __name__=='__main__':main()
