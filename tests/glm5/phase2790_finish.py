"""Append scientific closeout, preserving historical memos and raw artifacts."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P


def main():
    W=P.W;root=W.ROOT;registration=W.read(P.OUT/'client-registration-current.json')
    from deploy.research_library import Library
    lib=Library(root/'distributed_data/research_library');r=lib.artifact(registration['report_sha256'])
    offline=W.read(P.OUT/'offline-acceptance-current.json');package=W.read(P.OUT/'package-verification.json')
    assert offline['report_sha256']==registration['report_sha256'] and package['version']=='1.0.4' and package['passed']
    assert r['baseline_gate'] and not any(e['qualified_formal_to_spoken']['mean'] for e in r['effects'])
    acceptance={'schema':'software_browser_acceptance.v1','kind':'real','run_id':P.RUN,'passed':True,
        'report_sha256':registration['report_sha256'],'graph_sha256':registration['graph_sha256'],
        'checks':['actual report and 8-shard completion visible','graph-first catalog and dataset reverse lookup visible',
            'two external AI models, three campaigns and disagreement counts visible',
            'real hidden-layer generation heatmap visible','full model snapshot verified; B27/U1221 parameter located',
            'actual generation step10 full9728-unit array loaded, A[1221]=0.1494140625',
            'partial implementation status distinct from pending dictionary/replication/prediction'],
        'screenshots':{p.name:W.sha(p) for p in P.OUT.glob('client-*.jpg')},
        'tests':{'dist_protocol':261,'new_registry_and_task':16,'frontend':31,'electron':11},
        'frontend_build':'passed; existing large chunk/dynamic-import warnings',
        'installer':'1.0.4 built and byte/PE-x64 checked, not installed or executed',
        'limitations':['browser used actual online localhost client; no network-offline browser emulation',
            'actual background-built offline files independently verified','one physical CUDA node only'],
        'provenance':r['provenance']}
    assert len(acceptance['screenshots'])>=3
    W.save(P.OUT/'browser-acceptance.json',acceptance,True)
    P.status('client_acceptance',browser_checks=True,offline_verified=True,installer_version='1.0.4')
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    glm=root/'research/glm5/docs/AGI_GLM5_MEMO.md';text=glm.read_text(encoding='utf-8')
    marker='完整运行标识：`'+P.RUN+'`';assert marker not in text
    phase=max(map(int,re.findall(r'^## Phase (\d+):',text,re.M)))+1
    cap=r['capability']['result'];native=r['native_summary'];pr=r['preregistration']
    paragraph=f'''

## Phase {phase}: 多领域等长长文本、双AI盲评、真实分布式生成场与图谱档案闭环 [{stamp}]

{marker}。本记录Phase号按备忘录尾部递增；脚本与产物前缀P2790是独立运行标识，不代表备忘录漏写中间阶段。时间为America/Chicago。本轮仅使用本地Qwen3-4B CUDA；没有运行GLM4或14B，没有合成结果混入。用户明确要求“请调用其他AI模型进行评审，暂时没有多的人工进行评审”，因此改用gpt-6-astra、gpt-6.1-sol两种模型独立盲评，不等待人工，也不把AI一致升级成人工金标准。

### 一、问题、预注册与完整执行

目标是检验旧W6联合写入编辑是否真的改变完整自然输出的语域，同时控制句式和有效token长度；把材料、真实生成、原场、评分、反例与有范围的图谱主张连接起来。旧固定候选正结果与P2777评分器失败结果全部保留。

- 冻结模板NAT-REGISTER-P2790-v1，design_sha `{W.read(P.OUT/'execution.json')['template']['design_sha']}`。corpus与runner不可在同版修改。通过真实中心5001注册模板、真实节点领取分片、CUDA子进程执行、4文件分块上传、完成登记；8/8完成、0失败。只有1台物理节点，不能声称完成独立跨节点复现。
- 四领域：社区活动、资料交接、花园观察、软件更新。8个内容家族（每域2个）×2句式（工作说明/会议记录框架）×3长度（512/1024/2048）×2语域=96输入；四权重条件共384份greedy生成，max_new_tokens256、无thinking、0截断。每家族整体分片，不把其句式和长度当独立样本。
- 同一家族、句式、长度的双语域配对，事实、其他正文、其余要求完全一致，只有等token长度的语域要求不同。整句背景组合精确匹配，非截断/空格填充/masked padding。材料是有重复篇章框架的项目自编受控长文档，不是自然语料；长度增加也增加背景，不能识别纯长度效应。
- 此轮事前定义为发现校准与执行协议验证，不进行确认性机制推断。128家族后续设计的功效规划为 `{json.dumps(pr['formal_next_design']['power'],ensure_ascii=False)}`，预期差.20、配对翻转率.5、alpha=.025；该确认集尚未运行，不能用本轮96行冒充128独立家族。

### 二、计算对象、公式与评分

MLP原生中间响应为 `a_l = silu(G_l h_l) ⊙ (V_l h_l)`，输出为 `W_l a_l`。W为down_proj，l=B18..27。旧单位方向u、512列掩码M及alpha=6全部继承冻结旧编辑：

`W'_l = W_l − alpha · u_l (u_l^T W_l) M_l`。

每层更新秩至多1，但alpha=6是过度移除/改变符号强度，不能称作简单关闭闸门；十层共寻址13,107,200个标量，不是单层/单元/单参数机制。随机方向在同列匹配Frobenius范数，随机列按原生激活能量分层；seed27732026及逐层派生种子冻结。所有条件结束恢复原始内存权重并精确比较，模型磁盘文件前后快照一致。8分片×10块的target编辑BF16哈希全部与旧能力测试的selected-edit.npz逐字节一致。

384输出按完整文本+事实去重为307条，只给评审者不透明ID、输出和核心事实，不给原输入要求、权重条件、源ID、另一位评分。每位保留style（formal/spoken/mixed/uncertain）、meaning、naturalness1..5与理由。语域接受当且仅当二者标签相同且最低自然度≥3；保义接受当且仅当二者均preserved。保义要求负责人、对象/事项、时间、地点、条件和限制完整，且无实质无依据新增；纯寒暄/一般沟通提醒不作为新增安排。分歧保守保留，不在看到结果后改成多数票成功。

对书面输入j，合格变化 `Q_j=1[base正式 ∧ base保义 ∧ edited口语 ∧ edited保义 ∧ 未截断]`。先在每个家族内平均6个句式×长度，再在8家族上作描述性bootstrap；不是确认性p值。所有事件为0时bootstrap退化[0,0]，不能解释为零不确定性。追加展示更正：在家族可交换二项假设下，“一家族至少发生一次合格事件”的单侧95%上界 `1−0.05^(1/8)=31.2344%`，并非逐句翻转上界。原报告v1、原评分和阈值均保留，v2更正区间呈现，v3补可离线重算来源，不改变实验数字。

### 三、真实结果与反例

1. 新输出两评审语域一致305/307={r['unique_style_agreement']:.4%}，保义一致303/307={r['unique_meaning_agreement']:.4%}，两维合计6条独立去重文本有分歧。不同模型别名不保证统计独立，确切checkpoint/解码seed未向工具公开，同供应商共同偏差仍可能存在。
2. 32条预先构造的语义校准（8正例、24破坏例，负责人替换/时间替换/限制反转），两评审均32/32正确、24/24破坏检出。它只验证这三类明显错误，不代表一般语义、语用和幻觉全覆盖。旧84条盲评语域一致、20参考语域全对，但7条保义分歧；旧同Qwen评分器不能自动当标准。
3. 原模型书面48/48、口语48/48符合输入语域；保义分别39/48=81.25%、32/48=66.67%，合计71/96=73.9583%。48个领域×句式×长度×语域格各2家族均非零。通过的是事前“各语域style≥.70、pooled meaning≥.70、无零style格”的探索门，**并非各语域保义均通过**。不能事后换阈值宣布更强的测量可靠性。
4. 旧target书面48/48仍正式，口语45/48口语（1 mixed、2 uncertain）；保义书面33/48=68.75%、口语36/48=75%。两个负对照的书面也48/48仍正式。target及两对照的合格书面→口语变化均0/48。旧方向在本探索材料上没有实现预期自然语域控制；不支持把固定候选偏好变化外推为自然开关。
5. 保存100条“语域未符合/保义未通过/评审分歧”混合记录，可能重叠且同一家族相关，不是100项独立机制失败。包括把背景“交接内容据原始记录核对”改写成正式交接安排，以及无依据承诺负责人现场帮忙核对等。所有原文、两个理由、分片及条件可回查。不能只命中人物、数量或语体关键词就判保义。
6. 能力监控仅在精确编辑哈希一致后复用旧2200题：68.5455%→68.4091%，23损失、20收益；净损失0.13636个百分点，Bonferroni同时Clopper–Pearson界[{cap['loss_interval'][0]*100:.6f},{cap['loss_interval'][1]*100:.6f}]个百分点，上界≤2个百分点，非劣效通过。不是新盲测，医学题占多数，140道未验证M3Exam参考题不入门；不能推广为开放任务或新领域能力保持。

### 四、真实生成全场与客户端闭环

原场按实际cached autoregressive forward采集，不是生成后teacher forcing回放。96个NPZ，{native['bytes']:,}字节（约4.52GiB），192个样本×条件，{native['valid_generation_steps']:,}有效生成步；所有37层HiddenState原生BF16位模式（uint16无损），B18..27各层全部9728个down_proj输入单元。每个提示只保留5个位置（0、1/4、1/2、末二、末一），所有实际生成步都保留，不能叫全提示token场。hs[36]在最终norm之后；step0用完整提示预测首个新token，step t以前t个实际生成token为历史。base/target分叉后历史不同，同序号状态差只能作描述，不能直接视为同前缀因果效应。

已逐文件核对sha256、字节数、dtype、37层/9728维、样本ID、有效步与前缀位置；所有原场继续保存在节点本地。客户端登记4实例（双语域×base/target）×10层×4生成步共160份完整9728维数组，不仅是Top-K。浏览器真实完成完整模型快照核验、B27/U1221定位、第10步数组叠加：A[1221]=0.1494140625；原模型参数表仍读原文件，不把编辑后响应冒充编辑后参数。蓝正橙负、灰中点，3D层塔沿用现有配色。

参数绑定完整快照 `{r['model_snapshot']}`。最新报告CAS `{registration['report_sha256']}`；图谱CAS `{registration['graph_sha256']}`；可重算源码/完整冻结corpus（含方向、列、种子）CAS `{registration['source_bundle_sha256']}`。旧报告与卡均保留；十层卡是同一个联合操作的定位入口，证据级为候选观察，不画未经验证的计算边、不宣称最小编辑。完整原场不上传到CAS；离线附加包包含manifest、160阵列、96输入、384生成、6份AI评分原件、源码及冻结配置，逐个哈希与条数已校验。

客户端实际实现：ResearchCatalog默认折叠，按图谱主张浏览为默认，按数据集反查为辅助，模型快照分桶；11数据版本、3个不同范围主张、3批AI双评。能力题库不自动连接成编码机制，参考答案与正式答案分开；相同候选操作下固定候选支持、旧测量未定、新AI探索未定并列，不平均为“支持分”。数据谱系目前精确哈希分组，通用跨库近重复及完整sample_lineage尚待实施。图谱围绕外部语言模式/内部特征/有类型主张组织，数据集不是另建本体的根。

natural_register任务接入正式任务类型、快照/精度匹配、整家族分片、断点原场、执行/上传校验、完成后停止重派；独立第二节点与异构GPU容差仍未验证。修复task_claim早退204前遗漏commit导致SQLite写锁的实际问题。中心节点执行审计与研究证据晋级继续分离。账号/令牌未进入公开研究产物。

### 五、验证、交付与严格自审

- 261项dist_protocol全套通过；新增登记/任务16项通过，含AI输入哈希/模型名单/逐条完整性/不可覆写/分歧不晋级，以及真实接口领取、上传、完成停止派发。
- 前端31项、Electron11项通过。Windows隔离环境阻止TestClient回环与临时目录、NSIS启动，改用项目专用临时目录并在获准的本机环境验证；不是业务测试失败后删除断言。生产构建成功，保留既有大chunk/动态导入警告。
- Windows x64安装包1.0.4：{package['bytes']:,}字节，sha256 `{package['sha256']}`；39个生产前端文件与6个节点资源逐字节核对，PE为x64。使用已安装Electron44.7.2与缓存NSIS，零生产npm依赖经builder内置traversal收集，不联网下载依赖。未安装/执行新安装包，未远端发布。浏览器验收针对真实本机正式客户端，未模拟网络断开；离线包由真实后台生成后另行校验。
- 最明显硬伤：仅8家族、每域2个，无法估计稳定域差；基线口语保义偏低；AI共同偏差未排除；受控长文有重复背景；alpha6单点不能推剂量或紧致度；能力题池暴露且不均衡；物理节点只有1个；图谱尚不能独立预测模型输出。以上限制不因工程闭环完成而消失。

### 六、理论增量、第一性原理与下一阶段大任务

最可信的进展是**观测目标和对象身份变得可区分**：同一精确参数操作可以改变固定候选偏好，却不改变此处自然生成的整体语域。可读出、候选偏好干预、完整生成控制、保义以及图谱预测是不同命题。当前结果不能把失败直接命名为“分布式容量”；也不能将“功能=容量×门控”升级为已证数学规律，更不能由一次零翻转否定所有自然语域门控。

第一性对象应是固定theta下的条件前缀计算 `h_l,t = F_theta,l(prefix_t)` 与 `P_theta(next_token|prefix_t)`，再问哪些跨前缀、跨层、跨生成步关系能在未见组合中预测并被局部计算路径解释。地图的价值应体现为可执行预测，而不是节点/数据集的堆积。有限参数的共享不等于一个固定语义坐标；候选方向也不等于完整计算单元。

下一阶段围绕“可复用的条件输出关系能否同时解释语域与保义”组织大任务，已写入客户端部分实施方案：

1. **材料与测量**：先针对本轮遗漏/新增承诺建设多领域事实-语用控制库；两种AI继续独立盲评，不等待人力，但增加隐蔽语义破坏、引用/否定/角色/条件作用域对照；新版本事前分别设置各语域、各域保义门，不再只看pooled。正文自然性、4–6句遵循与核心事实分别测量。不能在本轮确认数据上反复换评分。
2. **三引擎候选发现**：在获许可真实自然长语料训练/接入SAE/transcoder并估计背景协方差；本轮原场可先重算全坐标条件热力图和轨迹，不重新前向。A字典候选、B等长多句式选择性/hs[0]及干扰基线、C参数路径/归因独立登记。旧方向失败后不盲目把其alpha/k网格铺满新材料。
3. **冻结候选后的自然生成剂量与推广**：先发现、确认选参，再用从未用于选方向/阈值的新家族、新句式锁箱；参考128家族功效规划，效应或任务改变则重新计算。自然生成dose_curve、保义、随机方向/列、开放式推理/翻译/编程能力保持同时验收。尚无自然生成可行编辑，`min_edit_params=null`；旧偏好任务的编辑上界不迁移。将失败、反例、每个剂量点持续登记。
4. **分布式复算与仅图谱预测**：先至少第二物理节点核对同快照/精度金丝雀与结果，不一致暂停合并；再冻结特征与计算边，预测未见前缀的激活和下一token分布，报告KL及损失恢复率 `(L_baseline−L_graph)/(L_baseline−L_model)`（分母>0才定义，负值不截断），与身份/位置/n-gram比较，禁止用完整原模型重跑替代解释。方法成立后顺序扩展GLM4-9B、Qwen3-14B。

通俗结论：研究流水线已经能保存“做了什么、生成了什么、两位AI怎样判断、模型内部怎样运行”，并从客户端回到真实参数。本轮没有找到可靠的自然语域开关；下一步应先修好保义测量、从自然运行重新找结构，再检验新候选。工程能持续积累拼图，不能把积累数量当作已经破解的程度。
'''
    with glm.open('a',encoding='utf-8') as f:f.write(paragraph)
    claude=root/'research/claude/docs/AGI_CLAUDE_MEMO.md';ct=claude.read_text(encoding='utf-8')
    cp=max(map(int,re.findall(r'^## Phase (\d+):',ct,re.M)))+1
    with claude.open('a',encoding='utf-8') as f:f.write(f'\n\n## Phase {cp}: 多领域等长生成、双AI盲评与真实分布式客户端闭环（运行P2790，GLM5 Phase{phase}详记） [{stamp}]\n\n完整运行标识：`{P.RUN}`。用户授权无人工时使用其他AI模型，gpt-6-astra与gpt-6.1-sol独立盲评307条去重生成，并校准84条旧输出/参考及32语义控制。真实Qwen3-4B CUDA经正式中心完成8分片、96输入（四领域/两句式/512-1024-2048等token对）、384生成，0截断；只有1物理节点。基线书面/口语识别均48/48，保义39/48与32/48；旧alpha6联合编辑未产生合格自然语域翻转。不同目标的旧偏好正结果保留，不外推为自然门控；AI一致不是人工真值。\n\n37层HiddenState、B18..27全部9728单元及10,996有效生成步已无损保留，96原场约4.52GiB；160完整阵列进客户端，浏览器完成完整快照验证和第10步响应叠加。80块编辑BF16哈希与旧能力监控完全一致才复用2200题NI；不是新盲测。客户端实际接入11数据版本、3范围主张、3批双AI评分、100反例/分歧、可追溯源码与离线包。261协议+16新增+31前端+11Electron测试通过，1.0.4 x64安装包完成字节校验但未安装/发布。最新报告 `{registration["report_sha256"]}`，图谱 `{registration["graph_sha256"]}`。\n\n详细原理、公式、结果、硬伤、RDC边界、图谱优先/数据反查架构和下一大阶段，见research/glm5/docs/AGI_GLM5_MEMO.md本次Phase{phase}。当前候选在自然生成范围未支持，不能解释为容量本体或不存在门控；先提升各语域保义测量并用三引擎寻找新候选，再做冻结自然剂量/锁箱、第二物理节点复算和图谱预测。\n')
    W.save(P.OUT/'closeout.json',{'run_id':P.RUN,'glm_phase':phase,'claude_phase':cp,'time':stamp,
        'report_sha256':registration['report_sha256'],'package_sha256':package['sha256']},True)
    print({'glm_phase':phase,'claude_phase':cp,'time':stamp})


if __name__=='__main__':main()
