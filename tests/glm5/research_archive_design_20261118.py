"""Save a reviewable archive-design proposal; no server/model/data-cleanup actions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timedelta, timezone

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import checked_dir, check_write

OUT=ROOT/'tests/glm5/result/research_archive_design_20261118_v1'
MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
SOURCES=[
    'shared/dist_protocol/lang_map/structure.json','shared/dist_protocol/lang_catalog.json',
    'shared/dist_protocol/lang_tree.json','shared/dist_protocol/lang_map.py','scripts/build_lang_map.py',
    'deploy/research_library.py','shared/dist_protocol/research_registry.py',
    'deploy/research_ingest_bundle.py','deploy/distributed_service.py',
    'tests/glm5/run_routine_test.py','deploy/storage_policy.json','design/platform_spec_v1.md',
]

PROPOSAL={
    'schema':'experiment_archive_design.proposal.v1',
    'status':'proposal_only_not_implemented',
    'model_runs':0,'server_writes':0,'deletions':0,
    'decision':'Use the three capability domains as versioned browsing views; store reusable datasets, immutable experiment designs, separate execution attempts, per-case results and content-addressed artifacts with many-to-many links.',
    'capability_view':{
        'primary_domains':['KN','RS','GR'],
        'allow_cross_membership':True,
        'version_existing_sources':SOURCES[:3],
        'facets':['language','task','style','length','composition_depth','method','model_snapshot','precision','readout','execution_status','scientific_status','availability'],
        'rule':'Tag the controlled or assessed capability; do not label every sentence as a grammar experiment merely because it has syntax.',
    },
    'records':{
        'capability_node':{'identity':'Stable existing language item ID plus classification revision','links':['primary_parent','also_in','legacy_alias']},
        'dataset_revision':{'identity':'Immutable dataset ID/version and exact material digest','contains':['case_id','family_id','raw_text','expected_answer_or_rule','source','split','exposure','transform_links']},
        'experiment_revision':{'identity':'Stable experiment ID plus immutable protocol digest','contains':['research_question','purpose','principle','hypotheses','alternative_explanations','dataset_refs','methods','measurement_boundary','baseline_and_controls','criteria','resource_budget','retention_contract','prior_work_and_novelty']},
        'execution_attempt':{'identity':'New run_id for every real attempt, even equivalent reruns','contains':['experiment_revision','dataset_revision','model_and_tokenizer_identity','actual_prompt_and_token_hashes','code_bundle','environment','config','seed','node','time','progress','exit_state','artifact_manifest','archive_receipt']},
        'result_analysis':{'identity':'Run reference plus analysis revision and input artifact hashes','contains':['per_case_outputs','per_family_metrics','all_conditions','baselines','controls','uncertainty','failures','counterexamples','scientific_status','scope','unexplained_residual']},
        'artifact':{'identity':'Full SHA256 of bytes, distinct from semantic metadata digest','contains':['role','required_by','availability','location','size','encoding','retention_class','rebuild_recipe','integrity']},
    },
    'retention':{
        'core':'Always archive each attempt, actual questions or exact frozen material, real outputs, summaries, failures, code/config and necessary reproducibility records.',
        'essential_evidence':'Retain artifacts needed to recompute registered metrics/claims or reproduce important hard-to-recreate failures; negative and positive evidence use the same rule.',
        'managed_reuse':'Owner, reader/dependency, byte quota, expiry and rebuild recipe; explicit review, not indefinite retention.',
        'scratch':'Disposable only after required records are preserved and no live dependency remains; audit every deletion.',
        'important_results_include':['learned dictionaries/transcoders','frozen selection and fitted predictors','counterexamples','needed output distributions and matched-control records'],
    },
    'status_axes':{
        'execution':['draft','frozen','queued','running','completed','failed','interrupted'],
        'archive':['pending','local_ready','uploading','server_verified','missing_artifacts'],
        'scientific':['not_measured','candidate_observation','supported_in_scope','not_supported','inconclusive'],
        'availability':['present','remote_only','retired_not_archived','recapture_required','historically_missing'],
        'publication':['private','restricted','published'],
    },
    'finalization':[
        'Persist minimal start record before execution; stream actual case/step results and essential failure context in committed batches.',
        'Finalize required manifests even on error, interruption, timeout or failed behavior gates.',
        'Upload missing blobs and metadata, verify full hashes, completeness and references, then atomically commit the archive revision.',
        'Read back the committed report and required artifacts; record a receipt with store identity and manifest digest. Backup verification follows retention policy.',
        'Derive an explicit unreferenced disposable-file list; preserve required local pending records if remote archive fails.',
        'Clean only owned, idle, identity-checked paths and record new availability and actual disk change.',
    ],
    'deduplication':{
        'bytes':'Deduplicate identical immutable artifact bytes by hash; never count uploads as independent runs.',
        'execution_key_fields':['protocol_digest','dataset_digest','model_snapshot_and_edit_digest','tokenizer_digest','runner_and_analysis_code_digests','numeric_environment_profile','precision','configuration','seed','capture_and_readout'],
        'repeat_rule':'An equivalent intention key is a preflight lookup, not proof of identical output. Preserve each actual attempt; require repeat purpose and retain independence metadata.',
        'similarity':'Suggest related studies and near-duplicate families for review; never automatically merge scientific claims, splits or evidence.',
    },
    'migration':[
        'Inventory memo references, actual files, frozen bundles, current server objects and previous cleanup ledgers without model execution.',
        'Import verified originals with source hashes; mark synopsis-only and missing output separately.',
        'Map legacy Phase/run IDs as aliases without renumbering or inventing data; record unavailable raw fields honestly.',
        'If recovery requires a new run, create a new execution linked to the old study; do not backdate or replace the missing historical result.',
    ],
    'rollout':[
        {'stage':'A','deliverable':'Approve minimum archive contract and verify a representative end-to-end pilot','gate':'Success, failure, interruption, mixed NPZ, reused data and retired raw availability can all be represented truthfully'},
        {'stage':'B','deliverable':'Connect routine, distributed and custom producers to the same registration/finalization contract','gate':'Upload interruption, disk full, checksum mismatch, duplicate upload, crash recovery and last-copy protection tests pass'},
        {'stage':'C','deliverable':'Backfill historical catalog and deduplicate data without rerunning models','gate':'Every discovered legacy attempt has provenance, classification state and explicit completeness/availability'},
        {'stage':'D','deliverable':'Browse/compare by capability, question, protocol, run and case in existing UI; enable audited retention','gate':'From summary to real question/output/code/artifact and back; negative results and missing data remain visible'},
    ],
}

BODY=r'''
**本轮任务与结论。** 用户要求先确认“每次测试有目标、原理和结果，系统保存到服务器，重要数据可查看与重现，非重要数据清理”的方案，并讨论知识网络、分析推理、语法规则三类逐步细分到题目的组织方式。本轮为源码/记录审查及方案设计；没有新增模型、服务器上传、部署、数据库迁移、历史结果回填或数据清理。下面的结构、状态与流程是拟采用的合同，不是已经完成的功能。

建议保留三大类作为主要浏览入口，但底层使用“能力图谱＋版本化题库＋实验档案＋运行记录＋内容寻址文件”。不能只按能力树建文件夹，也不能把三类实验的下一层都称为语法：知识应按实体/属性/关系/来源组织，推理按条件/操作/组合/更新组织，语法按构词/词类/句法/角色/约束组织。一个实验或题目可以有主归属及明确的交叉归属；分类是研究视角，不是模型内部必然存在三套独立模块的证明。

**现有实现与实际缺口。** 已核对以下本地入口，未实查在线服务器，因此不把本地代码能力等同于当前公网部署状态：

- `shared/dist_protocol/lang_map/structure.json`已有KN知识网络、RS推理功能、GR语法规则三个领域及各自子领域；`lang_tree.json`已有主类型与also_in，明确分类只是展示组织；`lang_catalog.json`与词表参与生成细项。`lang_map.py`和`scripts/build_lang_map.py`已有“领域→子领域→语言项→词”的构建逻辑。方案应扩展这些ID与版本映射，不另外造一棵互相冲突的树。
- `deploy/research_library.py`已有SQLite索引、规范JSON内容寻址、报告修订、工作包、特征/关系和读取校验；`research_registry.py`已有数据版本、范围化主张、别名与审核。它们不是自动保存所有本地实验的总入口。当前sync主要扫描约定research路径、特定报告格式及已进入中心的结果，并不自动收集每个`tests/glm5/result/*`目录。
- `deploy/distributed_service.py`有冻结design_sha、结果上传init/chunk/finish、去重与审计；`research_ingest_bundle.py`有研究包上传、哈希检查及读回。现有机制优先复用，泛化到其他测试类型仍需扩展合同，不能把新方案说成现成可用。
- `tests/glm5/run_routine_test.py`已保存脚本、问题、review/conclusion与日志尾部，并检查本地小记录后清理本轮临时目录；实际还没有把服务器完整归档及读回回执作为这次清理的前提。只保存单个脚本、Git HEAD和日志尾部，也不保证脏工作区依赖、逐题结果及重要故障上下文完整可重现。它是优先接入归档闭环的入口，不在本轮静默改动。
- 现有真实研究库拒绝synthetic，研究图谱有独立准入。工程测试、数值校准与合成夹具也应可归档，但放在明确隔离的测试种类/命名空间，不得为“统一保存”而放松真实研究或证据晋级条件。

因此，当前主要缺口是统一的实验对象、生产者注册/结束合同、逐题结果与依赖清单、可靠服务器确认和可审计清理，而不是缺少更多目录层级。

**一、三大类怎样细分。** 建议优先沿用现有领域和细项ID，按研究需要扩充，不一次设计尽所有语言现象，也不让分类调整改变实验身份。

| 领域 | 第二层示例 | 继续细分时的核心对象 |
| --- | --- | --- |
| 知识网络 | 实体与类别、属性、实体关系、事件、数量时空、词义、知识来源 | 例如属性→属性归属→同名实体区分→跨来源属性绑定 |
| 分析推理 | 逻辑运算、条件与推断、变量/角色绑定、证据选择、状态更新、多步组合 | 例如证据选择→来源有效性→冲突记录→过时/新记录选择 |
| 语法规则 | 构词与书写、词类、短语组合、句式、论元角色、语序标记、时体情态与依存约束 | 例如论元→施受角色→主动/被动→角色交换最小对 |

语言、风格、任务形式（问答/翻译/代码等）、长度、组合深度是并列筛选维度，不必在每类下面完整复制一遍。同一跨来源绑定实验可以主挂“知识来源”，关联“条件选择”；若显式操纵否定或论元结构，再关联相应语法节点。所有句子都有语法，不意味着所有实验都应被标为语法实验。保留分类来源、审定状态和taxonomy版本；历史ST/SQ/TK等标签保留兼容映射，不按新树重写旧结论。

浏览上可以“领域→子领域→语言现象→研究问题/实验→题目家族→具体题目”，但实际数据不是单一父子的目录树：一个题目家族可用于多个实验，一个实验可覆盖多个能力节点，一个能力也会由许多题库和不同方法研究。词层也不能代替题目层：词、义项、token和具体输入分别保存。

**二、归档的核心对象与关系。** 最少区分以下六组记录，细项可作为子记录或引用实现，不要求一次建成许多独立服务：

| 对象 | 解决的问题 | 必须保留的身份与关系 |
| --- | --- | --- |
| 能力节点 | 研究什么语言现象、在哪里浏览 | 稳定语言项ID、分类版本、主归属/交叉归属 |
| 版本化题库 | 测什么，同族/改写/翻译是否泄漏 | dataset版本、case_id、family_id、实际文本、真值或评分规则、来源、划分与暴露状态 |
| 实验方案 | 为什么测、怎样测、怎样区分竞争解释 | 实验ID、研究问题、目标、原理、方法、基线/对照、指标、阈值、资源与保留合同、冻结修订 |
| 运行批次 | 哪次实际执行，由谁用什么配置完成 | 每次新的run_id、协议/材料/代码/模型/tokenizer身份、精度/环境、种子、节点、时间、进度、退出与归档状态 |
| 结果与分析 | 实际发生什么、支持什么、有哪些反例 | 全部已执行题目的真实输出/分数、有效样本/家族数、全部条件格、基线与对照、区间、失败、范围化结论、分析版本 |
| 文件产物 | 数据在哪里、重要性如何、能否复核/恢复 | 完整字节SHA-256、角色、编码/形状/索引、被谁依赖、可用性、保留等级、重建方式与回执 |

总体关联是“能力节点↔实验方案→多次运行→逐题结果”，方案/运行共享引用“题库版本→题目家族→具体题目”，所有对象引用同一套哈希文件。研究问题可把多版方案组织成同一大任务；不再把每个小改动变成互不相连的新研究。

所谓“一个测试”必须明确是在指方案、一次执行还是单个题目。Phase、论文式报告、测试函数数量和上传次数不能混作独立科学实验数量。保留现有run_id/Phase作为兼容别名，不能直接改名破坏客户端与历史链接；新对象关系通过附加映射连接。

**三、每个测试卡需要什么。** 所有类型都要记录“目标、原理、实际输入、实际输出、执行身份、状态、结论范围与产物”。确认性机制研究再要求假设/替代解释、行为门、因果可见位置、基线和匹配对照、指标/阈值、功效与冻结；工程检查则保存断言与故障复现，不强迫它填写不适用的模型和统计功效字段。探索可明确标探索，不能为了归档而事后伪造预注册。

每张实验卡默认回答：要解决什么未知；为什么这项设计能回答；具体改变/固定什么；用哪些题目与划分；方法和读出边界；实际完成多少、失败多少；比较哪些基线/对照；看到哪些结果与反例；支持什么、不支持什么；与哪些历史实验重复/修正/复现；哪些数据现存、已退役或缺失；怎样复算指标以及怎样重新执行模型。数学公式按需要定义变量和计算对象，不要求每次硬造新理论。

模型身份用权重与tokenizer快照，不能只用模型名称或config哈希；参数干预还需编辑清单。实际执行代码与必要本地依赖打包，脏工作区不能只留Git提交号。保留实际渲染的prompt、token IDs或其受控存储及哈希、生成参数和打分代码版本。原始文件字节哈希与规范JSON语义哈希分别记录，不能通过重排文本/换行悄悄改变模型输入。

逐题数据按家族和固定大小分片压缩保存，索引给出case_id到分片/行的映射；数据库只存必要检索字段及对象引用，不给每题复制脚本、题库和大文件。原始结果必须逐批提交，异常和中断也保存已完成记录、最后进度和必要故障上下文，不能等整轮结束才写总报告。

**四、怎样既防重复又允许复现。** 分三层处理：

1. 文件相同：按完整SHA-256存一份字节，多处引用；重复上传幂等，不增加样本或独立节点数。
2. 执行意图相同：运行前用协议/题库版本、模型与tokenizer、编辑、执行与分析代码、数值环境配置、精度、种子及采集/读出生成等价键，先查询已有完整结果。相同键只说明计划相近，不保证非确定性运算输出逐位相同。确需漂移检查、独立节点复现、随机波动或错误恢复时，仍创建新的run_id，记录重复原因与独立性。
3. 科学问题相近：按问题、受控因素、能力节点、数据家族和方法找相关实验，由研究者确认新增信息。只提示相似，不自动把相似题面、同名概念或不同模型结论合并。改写/翻译/实体替换保留家族关系，不能靠随机切分把近重复包装为新盲测。

建议新运行必须填写一句“相比已有结果，这次新增什么”：新假设、强基线、修复测量、未见组合、不同模型/精度、独立复算或必要重跑都可以；没有新增信息且已有可用产物时优先复用或重新分析，停止无目的重复。结果数、独立家族数、独立运行数、节点数、上传数分别计数。

**五、服务器保存什么，本地清理什么。** 不以显著性、正负结论、目录名或NPZ扩展名判定重要性。

| 等级 | 数据角色与例子 | 建议保存策略 |
| --- | --- | --- |
| 核心档案 | 每次运行的目标原理、冻结题目、实际输出、完整判定、代码/配置、关键日志、失败/中断与反例 | 所有运行都入服务器；轻量但长期保留，失败同样保存 |
| 必要证据/研究成果 | 支撑主张复算的逐家族量、对照与干预记录、必要输出分布/原生切片、选出的结构和拟合器、训练好的SAE/transcoder等 | 随实验声明required_artifacts，按内容哈希保存并作为依赖固定；不能当普通缓存清理 |
| 登记复用缓存 | 近期确有读取者的原场、训练激活、可重建中间量 | 限容量、有负责人/用途/读取入口、到期复查；无明确任务不默认长期留存 |
| 临时数据 | 已验证重复副本、无后续用途暂存、可重建绘图/打包缓存、不属于必要证据的原场 | 核心和必要证据已妥善保留、无活动依赖后按明细清理，保存删除账本 |

最小复核记录必须按具体主张定义：均值/方差不能复算所有非线性、置换或因果指标；若声称全词表KL，就要保存足够的输出分布或明确登记需模型重采的依赖；只有Top-K截图不能替代完整科学依据。大原场可短期保留后退役，但展示包和必要切片不是完整原场备份。客户端在用数据和已学习的字典是正式依赖/结果，不因体积大自动删除。

应明确三个重现层级：看报告与原题；不运行模型从已存数据复算指定指标；按代码、材料、模型和环境重新执行模型。每个实验标记能达到哪一级，允许数值容差及未验证范围。模型权重通常共享引用一次并保存文件清单/哈希与恢复来源，不每轮重复上传；必要原始语料/权重能否恢复也影响重现状态。服务器需要保存的不只是网页截图或文字总结。

**六、可靠归档与清理的闭环。** 采用现有上传/CAS思路扩展，拟流程为：

`运行前注册最小卡片 → 冻结方案/题库 → 逐批落盘 → 成功/失败/中断都结算 → 上传缺失对象 → 服务端校验完整性与引用并原子提交 → 从服务器读回核验并拿到回执 → 执行本轮可丢弃文件清单 → 登记数据可用性变化`。

回执至少绑定run_id、协议/产物manifest哈希、归档位置身份及完整性判定；HTTP成功或“已发送”不等于入库、可读与可重现。核心档案需要备份和恢复抽查；同一磁盘的另一目录不算独立备份。正式归档、发布给公众、科学审计、独立模型复算和主张成立是不同事件。上传的重要模型结果先存为有来源的档案，不能自动作为正式机制证据。

上传失败或服务器缺空间时保留必需的本地待传包和原始结果，显示待处理原因；不因日志有结论而把最后一份必需数据删掉。已经验证可丢弃的纯缓存可按合同清理，避免所有无用大场因断网无限滞留。成功后也不是整目录删除：从必需结果、学习到的模型、展示依赖、盲测/复用登记和未完成上传的引用出发列出可删除项，再检查归属、文件身份、联接/硬链接、占用与实际空间变化；哈希共享对象要考虑全部存活引用及并发提交，不能删除仍被别的实验引用的最后副本。

服务器同样要配空间配额、上传/解包峰值预算、残缺上传过期清理、分层保留和审计；不能只把本地磁盘失控搬到服务器。先复用现有持久数据目录、索引和结果CAS，不为本方案更换整套存储产品，也不在程序升级时覆盖数据目录。大文件按分片下载，默认网页只加载摘要与当前题目，避免每次浏览下载整场。

所有必需产物依赖应在跑前声明并在结束时核对：如果全部实际结果只能从某个大文件中重建，该文件在完成必要成员拆分并校验之前仍是重要数据。不能把上传需求、清理条件和证据等级混成一个completed状态。

**七、状态、访问与展示。** 最少分开显示执行状态、归档完整性、科学判定及数据可用性；公开状态另记。执行成功但结论未支持、运行失败但记录完整、历史有总结但逐题结果缺失，都必须表达。缺失统计不能填0，未执行不能写失败，技术验收不能替代科学确认。工程/合成材料隔离，图谱只引用通过对应准入的真实研究。

未开封确认/锁箱可以加密或受权限控制地完整归档，但不能因“方便查看”自动向所有参与者公开；查看/下发事件记录暴露范围。普通页面浏览公开摘要，原题、原始输出、图表、代码、哈希、复算说明按需展开；身份权限与任务令牌等不打进研究包。每张能力页显示研究问题、已有证据、失败边界、缺失数据与下一任务，提供按模型/方法/语种/版本比较。优先扩展现有语言图谱、研究库和研发任务页，不另起一套互不相通的测试系统。

**八、历史测试怎样整理。** 先建立历史账本：Phase、脚本、实际题库、run_id、结果文件、服务器对象、依赖和清理记录之间逐项关联。历史编号保留，不为了新分类重编号。将“原件完整且可校验”“仅有总结、未找到原始结果”“原场已退役但结果在”“执行中断/部分结果在”“当前无法判断”分别登记；缺什么就明确写什么。AI可提出分类和目标原理草稿，但要附原始依据并标待核对，不能从备忘录反写出从未保存的数值或伪造旧源码。

重复上传/复制文件去重，真实重复运行保留；科学上有增量的复现、负结果和反例不因“相似”被删除。Phase2984/2985的原场退役应导入availability账本，状态可为retired_not_archived或recapture_required，不能在迁移时宣称原场已上传服务器。若需要重新运行补结果，必须作为新的run_id链接旧实验，保留“历史结果缺失”的事实，不能回填成旧日期的原实验。本轮不执行历史重跑或任何上传。

**九、分阶段落地及验收。**

| 大阶段 | 主要交付 | 进入下一阶段的依据 |
| --- | --- | --- |
| A：确认归档合同与试点 | 冻结对象关系、最小卡片、结果/文件角色、回执与清理条件，用少量有代表性的成功/失败/中断/混合NPZ/旧原场退役材料验证 | 能准确表达实际状态并从报告追到原题、结果和源码，不编造缺件，不先铺开整个细分类树 |
| B：所有新测试先不再丢记录 | 日常、分布式和自定义实验入口接入统一开始/逐批/结束/归档合同，隔离工程与真实研究 | 断网、断点、崩溃、磁盘不足、坏哈希、缺文件、重复上传、并发引用和最后副本保护等故障路径通过 |
| C：历史回填与去重 | 建立全部已发现测试的缺件/版本/别名账本，归档可核实原件，补能力节点映射和近重复提示 | 每条找到的历史测试有来源和完整性/可用性状态；不能把回填条目数当作独立测试或机制进展 |
| D：统一浏览、比较与生命周期 | 从三类能力到研究问题、实验、运行、题目和文件一条链回查，同题比较与可审计清理 | 负结果、数据缺失、适用边界和已退役原场可见；索引/对象/备份恢复可核对 |

阶段A/B优先于把所有分类分得极细：漂亮的目录不能补救持续丢失新结果。开始可在现有SQLite索引与CAS上增量扩展，不需要先做复杂知识图数据库或重新造客户端。具体字段、配额、批次与备份策略在试点依据实际规模冻结，不在没有服务器容量信息时承诺成本或工期。

**公式、理论边界与通俗总结。** 归档结构可记为

$$
\mathrm{Run}\rightarrow(\mathrm{ProtocolRevision},\mathrm{DatasetRevision},\mathrm{ModelIdentity},\mathrm{CodeConfig},\mathrm{Artifacts}),\quad
\mathrm{CaseResult}\rightarrow(\mathrm{Run},\mathrm{Case},\mathrm{Condition},\mathrm{Output}),\quad
\mathrm{Experiment}\leftrightarrow\{\mathrm{CapabilityNode}\}.
$$

这只是数据关系定义，不是新的智能数学结构。RDC与语言机制结论沿用已有研究，本轮没有新模型测量或公式改进。第一性研究的增量应仍看未见材料的预测、原模型因果检验与反例；归档的作用是让这些命题、输入、运算、结果和限制可以积累、比较与修正，不能用分类覆盖率或保存数量当破解率。

通俗地说：三大类像图书馆的分类导航；实验卡像一本研究档案；一次运行是一份有日期和配置的实测记录；题目与文件被多份研究共享引用。先把每次真实测量留住，再让人能从任何能力点查到问题、做法、答案、失败和证据。重要数据按是否支撑复核和后续使用判断，而不是按“结果好不好看”判断。
'''


def main():
    checked_dir(OUT,2*2**20,operation='实验归档方案与记录')
    if (OUT/'proposal.json').exists():raise RuntimeError('Proposal already saved; use a new revision')
    before=MEMO.read_bytes()
    phase=max(map(int,re.findall(rb'^## Phase (\d+):',before,re.M)))+1
    now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-6)))
    stamp=now.strftime('%Y-%m-%d %H:%M')
    evidence=[]
    for name in SOURCES:
        path=ROOT/name;data=path.read_bytes()
        evidence.append({'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    proposal={**PROPOSAL,'created_at':now.isoformat(timespec='seconds'),'memo_phase':phase,'source_evidence':evidence}
    for name,obj in [('proposal.json',proposal),('source-evidence.json',{'sources':evidence,'live_server_checked':False})]:
        path=OUT/name;payload=(json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
        check_write(path,len(payload),operation='方案产物')
        with path.open('xb') as f:f.write(payload)
        assert json.loads(path.read_text(encoding='utf-8'))==obj
    title=f'## Phase {phase}: 语言图谱关联的实验档案、服务器归档与数据保留方案（设计） [{stamp}]'
    entry=('\n\n'+title+'\n\n'+BODY.strip()+'\n\n**本轮实际交付与核对。** 完整结构化方案保存在`tests/glm5/result/research_archive_design_20261118_v1/proposal.json`，源码依据哈希见同目录`source-evidence.json`。该方案未实现或部署；实际执行仅为本地源码/记录读取、方案保存和本条追加。生成JSON读回一致，来源文件未修改，备忘录原前缀逐字节核对。没有创建模型缓存或待清理大文件，释放空间0。\n').encode('utf-8')
    check_write(MEMO,len(entry),operation='尾部追加归档设计')
    assert MEMO.read_bytes()==before,'Concurrent memo writer; stop before append'
    with MEMO.open('ab') as f:f.write(entry);f.flush();os.fsync(f.fileno())
    after=MEMO.read_bytes();assert after[:len(before)]==before and after[len(before):]==entry
    for row in evidence:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
    record={'passed':True,'phase':phase,'title':title,'memo_line':before.count(b'\n')+3,
            'original_prefix_bytes':len(before),'original_prefix_sha256':hashlib.sha256(before).hexdigest(),
            'original_prefix_preserved':True,'append_sha256':hashlib.sha256(entry).hexdigest(),
            'source_files_unchanged':len(evidence),'model_runs':0,'server_writes':0,'deletions':0,
            'proposal_sha256':hashlib.sha256((OUT/'proposal.json').read_bytes()).hexdigest()}
    with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
