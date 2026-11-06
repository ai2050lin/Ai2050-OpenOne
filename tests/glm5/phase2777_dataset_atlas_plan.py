"""Create a reviewable data/atlas specification AFTER the real run is complete."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import html
import json
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_confirm as C
import phase2773_w6_loop as W


SOURCES=[
 ('CEVAL','C-Eval','综合知识与能力保持；按学科等权汇总，保留每域损伤，不作为内部特征本体。','https://github.com/hkust-nlp/ceval'),
 ('MMLU','MMLU','与中文题池并列分层，不能把考试分数直接等同于语言机制。','https://github.com/hendrycks/test'),
 ('GRAMMAR','CLiMP / BLiMP','中文、英文最小语法对照；原短句作为测量基线。扩成长文必须另建文法、核验含义和行为门，不能只补背景就称自然长文。','https://aclanthology.org/2021.eacl-main.242/'),
 ('BLIMP','BLiMP','英语语法对照与构式覆盖，结合自主标点、角色、否定和长距离文法。','https://github.com/alexwarstadt/blimp'),
 ('REASON','GSM8K','多步算术行为及内容保持；另建可控组合深度文法，不把现成解题标签当作内部电路。','https://github.com/openai/grade-school-math'),
 ('CODE','HumanEval','执行单元测试的代码行为保持；固定执行环境与沙盒，题目、变体按题族一起切分。','https://github.com/openai/human-eval'),
 ('TRANSLATION','FLORES+','句子对齐的跨语言行为与表征比较；锁定HF revision，不把短句自动当作长篇翻译。','https://huggingface.co/datasets/openlanguagedata/flores_plus'),
 ('NLI','XNLI','多语言蕴含、矛盾和中立的受控语义视角；不是因果推理全集。','https://github.com/facebookresearch/XNLI'),
 ('LONG','LongBench v1','自然较长上下文的外部效度补充；按原任务、文档来源、实际token长度和模型能力门筛选。v2很长且为选择题，不替代自然生成语域验收。','https://github.com/THUDM/LongBench'),
 ('MEDICAL','CMExam','医学专业分层监测，限制在综合能力指标中的权重；不以大量医学题制造全域能力保持。','https://github.com/williamliujl/CMExam'),
]


def main():
    from deploy.research_library import Library
    r=W.read(C.OUT/'result.json');registration=W.read(C.OUT/'client-registration-current.json')
    cap=r['hypotheses']['legacy512_a6']['capability'];effect=r['hypotheses']['legacy512_a6']['qualified_flip']
    now=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    principles=[
      ('主入口围绕图谱，数据集作为证据维度','研究对象是语言规律与内部计算关系；同一规律应承接多种材料。同一数据集又可服务知识、推理、语法多个问题，不能各建一棵互不相连的本体。保留按数据集管理材料、质量与执行进度的反向入口。'),
      ('三个图层，两个浏览方向','外部层：词义、角色、关系、构式、任务、语境。内部层：模型特定的原生位置、字典特征、条件响应与计算路径。证据层：外部变量到内部对象、内部到输出的有类型主张。默认“规律→证据→数据”；研发支持“数据→任务→证据→图谱”。'),
      ('三种能力是主分类，语域等是交叉条件','知识网络、分析推理（包括翻译、编程）、语法体系作为稳定导航。语言、语域、文体、领域、上下文长度和组合深度作为可重叠过滤项，不强制给每张卡唯一分类。'),
      ('对象身份与实验声明分离','原生定位对象由完整模型快照、精度、层、边界、原生索引确定；提取特征还须绑定方法、方向或字典内容哈希。运行ID属于证据记录，不应不断生成“同一神经元的新本体”。同一单元可参与多个特征，不能因索引相同就合并语义。'),
      ('保留范围不同的正、负结果','固定候选偏好与完整自然输出是不同观测目标。本次新主张不覆盖旧偏好结果；以同一候选写入操作关联两者，各自保存任务、指标、对照、反例与状态。联合10层的定位卡按1个独立干预计数。'),
      ('破解程度最终靠留出预测衡量','执行率、特征数、数据集数、图谱覆盖都不是破解百分比。最终检验只用提取图谱预测未见前缀的激活及下一token分布，相比词汇身份、位置、n-gram等简单基线增加多少保真度。模型原始输出是核对目标，完整重跑模型不是图谱解释。'),
    ]
    contracts={
      'dataset_version':['dataset_id','upstream_revision','source_url','license','redistribution_scope','content_sha256','schema_version','label_status','task_facets','available_languages'],
      'sample_lineage':['sample_id','origin_document_id','source_group_id','content_sha256','parent_sample_ids','transform_version','raw_split','research_split','actual_token_ids','model_tokenizer_hash','readout_spans','factor_labels','annotation_source'],
      'feature_identity':['model_snapshot_sha256','precision','layer','boundary','native_index','method_version','direction_or_dictionary_sha256','canonical_feature_id','native_locator_id'],
      'claim':['claim_id','subject_feature_ids','external_pattern_ids','operational_scope','outcome_metric','design_sha256','run_ids','dataset_versions','independent_source_groups','split_type','effect_and_interval','negative_controls','content_gate','capability_gate','dose_curve','counterexample_artifacts','evidence_type','claim_status','audit_status'],
      'artifact':['sha256','schema','producer_version','model_snapshot','run_id','source_lineage','storage_uri','size','available_offline','recompute_recipe'],
      'migration':'append canonical aliases and claim mappings to existing immutable CAS; never rewrite old CAS or merge different model buckets; original links remain resolvable',
      'status_semantics':'supported_in_scope / not_supported / inconclusive / not_measured stay separate; execution/audit/evidence are separate dimensions; contradictory requires genuinely incompatible scoped propositions',
      'dedup':'same document, translations, paraphrases and derivative datasets share source_group_id; splits and effective evidence counts use groups, not rows or dataset filenames',
    }
    dataset_roles=[
      ('自然发现库','经许可的真实长段落、对话、报告、代码及双语文本','A引擎字典学习、激活普查、背景协方差','按原文档与来源组切分；512/1024/2048起步，保留上下文完整性。SAE训练量按激活token预算与复现质量逐步扩大，不能用少量模板替代自然语料。'),
      ('受控机制库','本项目生成式文法及经人审的语义等价重述','B引擎变量映射、长度/句式/词汇混杂检验','记录族、词表、构式、领域分别留出；自然语域必须先通过输出行为门，长度在同一家族内交叉，不只跨不同记录比较。'),
      ('外部行为库','知识、推理、翻译、编程与语法公开基准','能力保持与外部效度','按任务分别计分、域内配对NI、域间等权宏平均；正式验收题不参与方向或剂量选择。保留污染未知、答案来源和许可状态。'),
      ('反例与复算库','本轮全部失败、未知判定、原对改错、独立节点复算','驱动新文法与C引擎因果检验','反例发现后只能形成新版本预注册；同一确认集不能反复改阈值。重复使用的能力监测池明确标注已暴露。'),
    ]
    stages=[
      ('A · 数据契约与身份整理','先建立dataset_version / sample_lineage / claim索引，对既有CAS追加映射。为本轮偏好与自然生成两条主张建立共同候选操作引用。',
       '旧报告与参数定位链接不失效；跨来源近重复不能重复算独立证据；未知许可、参考答案与真实金标准可区分。'),
      ('B · 语域测量与自然长文行为','在日常安排、解释说明、工作交流、叙事、知识介绍等领域建设真实连贯的双语体材料。先让独立双人盲评校准书面/口语/混合、保义与自然度，冻结分歧处理；不把关键词命中当主判。',
       '未编辑模型各域、各句式、各长度行为门通过后才选内部正例。以同一内容家族交叉512/1024/2048有效token，多句式与领域留出；最终家族量按试跑方差和预定最小效应做功效规划，不能用增样掩盖无效测量。'),
      ('C · 发现、映射与生成轨迹机制','在同一冻结材料上并行比较字典特征、全坐标条件统计与候选参数路径。采提示与自回归生成步的全层原生场，检验旧编辑是只动初始读出、被后续恢复，还是评分器漏检。',
       'A候选须经B留出复现，再按需C检验参数与计算边；保留hs[0]、位置/长度、随机标签、能量匹配对照。自然生成剂量曲线重新预注册；本轮只有0与6两个强度，不能继承旧候选分数曲线冒充自然生成曲线。'),
      ('D · 分布式复算与图谱预测','新增natural_register_intervention.v1任务适配器，以内容家族分片，独立节点复算；形成图谱独立的状态/输出预测器，并跨模型检验结构。',
       '先Qwen3-4B，再GLM4、Qwen3-14B逐个执行。大模型按实测显存安排合适节点或另桶offload/量化；不将16GB卡视作可直接容纳14B bf16。原始张量进入大文件CAS，中心仅索引。图谱预测优于简单基线后，才能报告解释增量。'),
    ]
    pages=[
      ('进展','总览 → 三种能力卡 → 点击看冻结分母和研究矩阵','第一屏最多三张能力卡，只突出计划执行率。展开后分开显示有效行为格、留出复现、因果主张、反例及预测保真度；无测试显示“未测”，无分母显示“—”。'),
      ('空间','语言图谱 / 参数结构 / 样本响应','默认围绕语言模式与候选机制浏览；顶部模型快照、证据范围、语言/长度、数据来源过滤。点击节点开右栏，默认一句主张、证据状态和一个实例，详情再展开。参数页保留所有head/MLP阵列，布局距离不能冒充真实计算关系。'),
      ('研发','计划与执行 / 数据与模板 / 证据审核','数据集列表负责来源、版本、质量、切分、任务队列；详情有“查看关联图谱”反向跳转。统一实时分析入口，通过样本与模型上下文打开空间响应，不再复制一套分析页面。'),
      ('配置','模型与节点 / 算力贡献 / 账号','资源与托管配置留在这里，研究材料和图谱不进入配置。贡献算力只执行注册任务类型、资源门与冻结任务包；本轮本机运行不标成多节点已复算。'),
    ]
    monitor=W.read(C.OLD/'capability-inputs.json')
    used={key:sum(q['gate_eligible'] and q['source'].startswith(prefix) for q in monitor)
        for key,prefix in [('CEVAL','C-Eval'),('MMLU','MMLU'),('MEDICAL','CMExam')]}
    dataset_status=lambda key:f'本轮复测既有监测题 {used[key]} 道；未用作图谱发现语料' if key in used else '建议接入；本轮未下载或运行'
    spec={'schema':'research_dataset_atlas_plan.v1','kind':'proposal','revision':'usage-status-corrected-v2','run_id':C.RUN+'-DATA-ATLAS-PLAN',
        'created_at':now,'decision':'graph-centric navigation with dataset provenance and reverse dataset workflow',
        'implementation_status':'proposal; real natural-register report/graph/client renderer already implemented; canonical registry and new distributed job are future work',
        'principles':[{'title':a,'description':b} for a,b in principles],'contracts':contracts,
        'roles':[{'role':a,'materials':b,'purpose':c,'rule':d} for a,b,c,d in dataset_roles],
        'dataset_candidates':[{'id':a,'name':b,'role':c,'primary_source':d,'status':dataset_status(a)} for a,b,c,d in SOURCES],
        'stages':[{'stage':a,'work':b,'acceptance':c} for a,b,c in stages],
        'client_pages':[{'page':a,'secondary':b,'behavior':c} for a,b,c in pages],
        'current_case':{'report_sha256':registration['report_sha256'],'graph_sha256':registration['graph_sha256'],
            'natural_flip':effect,'capability':cap,'prompts':512,'independent_families':128,
            'generated_outputs':2048,'joint_interventions':1,'parameter_locator_cards':10,'new_multi_node_replication':False},
        'job_contract_proposal':{'job_kind':'natural_register_intervention.v1','unit':'whole source family across frames/styles/lengths',
            'input':['pinned_model_snapshot','frozen_corpus','style_rater_version','direction_sha256','selected_columns','dose_grid','controls','capability_monitor','resource_limits'],
            'output':['full_output_text_and_token_ids','rater_scores','content_and_capability_scores','native_manifest','canary','edit_manifest','restoration_verification','provenance'],
            'merge':'same design,model,precision,method; dedup shard; repeat=replica check, not new sample; disagreement quarantined',
            'evidence_boundary':'audit/transport success never promotes an intervention into a general language mechanism'},
        'fidelity_definition':{'kl':'mean_t KL(p_model(.|prefix_t) || p_atlas(.|prefix_t)) on heldout source groups',
            'loss_recovery':'(L_simple_baseline - L_atlas)/(L_simple_baseline - L_model), only defined if denominator>0; no clipping negative values',
            'no_leakage':'predictor may access prefix and extracted graph only; cannot use target state, future answer tokens or a complete forward replay as explanation'},
        'sources_checked_date':now.split(' ')[0]}
    path=W.ROOT/'research/phase2777/dataset-atlas-plan-v2.research.json';W.save(path,spec,True)
    ref=Library(W.ROOT/'distributed_data/research_library').ingest(spec,str(path.relative_to(W.ROOT)),kind='local_report')['sha256']
    esc=html.escape
    def table(headers,values):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in values)+'</tbody></table></div>'
    role_table=table(['材料角色','材料','研究用途','规则'],[[esc(v) for v in row] for row in dataset_roles])
    candidate_table=table(['来源','使用状态','研究用途'],[[f'<a href="{url}" target="_blank" rel="noreferrer">{esc(name)}</a>',esc(dataset_status(key)),esc(role)] for key,name,role,url in SOURCES])
    page_table=table(['一级入口','保留的二级分组','显示方式'],[[esc(v) for v in row] for row in pages])
    stages_html=''.join(f'<details><summary>{esc(a)}</summary><p>{esc(b)}</p><p><b>验收：</b>{esc(c)}</p></details>' for a,b,c in stages)
    principles_html=''.join(f'<details><summary>{esc(a)}</summary><p>{esc(b)}</p></details>' for a,b in principles)
    loss=', '.join(f'{x*100:.2f}' for x in cap['loss_interval'])
    verdict='测量门失败，未验收' if not r['baseline_gate'] else '范围内支持' if r['hypotheses']['legacy512_a6']['supported'] else '未通过自然生成主判'
    proposal=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数据与图谱组织方案 · AI2050</title>
<style>:root{{--blue:#184f95;--ink:#1e293b;--muted:#64748b;--line:#dce4ed;--bg:#f4f6f9}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.7 "Segoe UI","Microsoft YaHei",sans-serif}}header{{background:#fff;border-bottom:1px solid var(--line);padding:12px 24px;display:flex;gap:16px;align-items:center}}header b{{color:var(--blue)}}header span{{color:var(--muted);font-size:12px}}main{{max-width:1260px;margin:auto;padding:24px}}h1{{font-size:25px;line-height:1.4;margin:0 0 8px}}h2{{font-size:18px;margin:0 0 10px}}h3{{font-size:15px}}p{{margin:8px 0}}button,select,.link{{font:inherit;border:1px solid var(--line);border-radius:6px;background:#fff;padding:6px 12px;color:var(--ink);cursor:pointer}}button.on,.link{{color:var(--blue);border-color:#aec7e5;background:#eff6ff}}a{{color:var(--blue)}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}}.card{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:20px;margin-bottom:16px}}.muted{{color:var(--muted);font-size:12px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stats b{{display:block;font-size:28px;color:var(--blue)}}.stats small{{color:var(--muted)}}.workspace{{display:grid;grid-template-columns:170px minmax(400px,1fr) 260px;gap:16px}}.side button{{display:block;width:100%;text-align:left;margin:6px 0}}.chip{{font-size:11px;border-radius:20px;padding:3px 8px;background:#eef3f9;color:#35577e;display:inline-block}}.claim{{padding:12px;border:1px solid var(--line);border-radius:7px;margin:12px 0}}.negative{{color:#925423;background:#fff8ef}}.scope{{border-left:3px solid #95b7dd;padding-left:12px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{text-align:left;border-bottom:1px solid var(--line);padding:12px;vertical-align:top}}th{{color:var(--muted);font-weight:500}}.scroll{{overflow:auto}}details{{border-top:1px solid var(--line);padding:12px 0}}summary{{cursor:pointer;color:#334f73;font-weight:500}}[hidden]{{display:none!important}}svg{{width:100%;height:auto}}svg text{{font-family:inherit;fill:#334155}}footer{{color:var(--muted);font-size:12px;padding:16px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:12px;font-size:12px}}@media(max-width:900px){{.workspace{{grid-template-columns:1fr}}.stats{{grid-template-columns:1fr}}header{{flex-wrap:wrap}}}}</style>
<header><b>AI2050</b><span>数据与图谱组织方案 · 只读预览</span><span style="margin-left:auto">{now} · 方案不冒充已上线功能</span></header><main>
<div class="card"><span class="chip">研究对象优先 · 数据来源可追溯</span><h1>围绕图谱积累知识，围绕数据验证证据</h1><p>默认从语言模式与内部计算关系进入；数据集作为来源、筛选和复现维度，同时保留研发中的数据管理视角。</p><p class="muted">本页是新的信息架构方案。W6真实报告、反例与参数碎片已进入正式客户端；统一数据注册表、新分布式任务适配和下述导航整合尚未实施。</p></div>
<nav aria-label="方案视图"><button class="on" data-view="space">空间 · 图谱为主</button><button data-view="datasets">研发 · 数据视角</button><button data-view="progress">进展 · 证据矩阵</button><button data-view="plan">整体方案与执行路线</button></nav>
<section data-panel="space"><div class="workspace"><aside class="card side"><h3>语言图谱</h3><p>知识网络</p><p>分析推理</p><p>语法体系</p><hr style="border:0;border-top:1px solid #dce4ed"><span class="chip">交叉条件：语域</span><p class="muted">三种能力可重叠；语域不另建一棵独立本体。</p><p class="muted">模型</p><p>Qwen3-4B · bf16</p><p class="muted">快照 {r['model_snapshot'][:12]}</p></aside>
<div class="card"><h2>语域 · 候选写入操作</h2><label class="muted" for="source">证据来源 </label><select id="source"><option value="all">全部来源</option><option value="preference">P2773 · 固定候选偏好</option><option value="natural">P2777 · 自然生成</option></select>
<svg viewBox="0 0 650 245" role="img" aria-label="方案关系图，布局不表示模型几何"><path d="M145 120H300M400 100L530 58M400 140L530 192" fill="none" stroke="#9eb7d4" stroke-width="2"/><rect x="10" y="83" width="150" height="74" rx="9" fill="#eef5fe" stroke="#b9cee8"/><text x="85" y="114" text-anchor="middle" font-size="14">外部语域条件</text><text x="85" y="137" text-anchor="middle" font-size="11">操作定义与材料范围</text><rect x="275" y="78" width="150" height="88" rx="9" fill="#f4f7fb" stroke="#b9cee8"/><text x="350" y="110" text-anchor="middle" font-size="14">候选写入方向</text><text x="350" y="135" text-anchor="middle" font-size="11">B18–27 联合编辑</text><rect x="495" y="24" width="145" height="66" rx="9" fill="#eef5fe" stroke="#b9cee8"/><text x="568" y="52" text-anchor="middle" font-size="13">固定候选偏好</text><text x="568" y="74" text-anchor="middle" font-size="11">范围内效应保留</text><rect x="495" y="163" width="145" height="66" rx="9" fill="#fff8ef" stroke="#dfc7a6"/><text x="568" y="190" text-anchor="middle" font-size="13">自然生成充分性</text><text x="568" y="213" text-anchor="middle" font-size="11">{verdict}</text></svg>
<p class="muted">方案关系示意；距离不表示模型几何。两条主张的指标不同，不相互覆盖，也不平均成一个“支持分”。</p>
<div class="claim" data-source="preference"><span class="chip">旧证据 · 不覆盖</span><p>固定候选的成对偏好衰减 7.037 nats；它本身不证明自然口语控制。</p></div>
<div class="claim" data-source="natural"><span class="chip negative">本轮 · {verdict}</span><p>自然输出合格翻转 {effect['mean']*100:.2f}% · 128 个家族 · 2 种留出句式。</p><p class="muted">256 对输入严格等长，384 / 512 有效token；包括未支持结果与未知判定。</p></div></div>
<aside class="card"><h3>证据详情</h3><p class="scope">先显示主张、范围和一条实例；再展开完整文本与原场。</p><p>能力监测：{esc(cap['status'])}</p><p class="muted">{cap['n']}题，损失区间 [{loss}] 个百分点；这一结果不代替语域主判。</p><a class="link" href="{registration['client_url']}" target="_blank">打开正式研究报告 ↗</a><p class="muted">10张参数定位卡 = 1个联合干预，不是10项独立机制。</p><details><summary>来源与版本</summary><p class="muted">完整报告 {registration['report_sha256']}</p><p class="muted">图谱 {registration['graph_sha256']}</p></details></aside></div></section>
<section data-panel="datasets" hidden><div class="card"><h2>数据用于什么研究，而不是数据集各自画一张孤立图</h2><p class="muted">数据管理保留独立入口：来源与许可 → 版本与切分 → 行为门 → 冻结任务 → 关联证据 → 回到同一图谱。</p>{role_table}</div><div class="card"><h2>材料分工与接入状态</h2><p class="muted">现有监测与尚未接入的材料分别标注；公开基准短句/选择题不能直接充当自然长篇机制正例。下载前核对所用revision、原始数据许可、答案质量及训练污染风险。</p>{candidate_table}</div></section>
<section data-panel="progress" hidden><div class="stats"><div class="card"><b>512</b><span>本轮冻结输入</span><br><small>128家族，2留出句式</small></div><div class="card"><b>2,048</b><span>实际自由输出</span><br><small>原模型、旧编辑、两种负对照</small></div><div class="card"><b>1</b><span>独立联合干预</span><br><small>不是破解百分比</small></div></div><div class="card"><h2>研究进展分开显示</h2>{table(['执行完成','行为可识别','范围内干预','图谱预测保真度'],[['本轮4条件已保存', '书面行为门：'+('通过' if r['baseline_gate'] else '未通过')+'；两向试跑失败',verdict,'未测；不能用节点数代替']])}<details><summary>三种能力卡怎样显示</summary><p>每张卡先显示冻结计划的执行百分比与分母，点击才显示数据来源矩阵、有效行为格、留出类型、反例和未测部分。不同模型、精度和设计不混算。</p></details></div></section>
<section data-panel="plan" hidden><div class="card"><h2>组织原则</h2>{principles_html}</div><div class="card"><h2>客户端层级</h2>{page_table}</div><div class="card"><h2>阶段性执行方案</h2>{stages_html}</div><div class="card"><h2>记录契约与判据</h2><details><summary>数据、特征、主张与产物字段</summary><pre>{esc(json.dumps(contracts,ensure_ascii=False,indent=2))}</pre></details><p>预测保真度在留出前缀计算 KL；损失恢复比例为 (L基线 − L图谱) / (L基线 − L模型)，分母为正才定义，负恢复不截断。预测输入不得泄漏目标状态或未来答案。</p></div></section>
<footer>方案产物 {ref} · 全量合同与路线保存在同名 research.json。已实现与待实现明确分开，所有研究声明保留来源及适用范围。</footer></main>
<script>document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-view]').forEach(x=>x.classList.toggle('on',x===b));document.querySelectorAll('[data-panel]').forEach(x=>x.hidden=x.dataset.panel!==b.dataset.view)}});document.getElementById('source').onchange=e=>{{document.querySelectorAll('[data-source]').forEach(x=>x.hidden=e.target.value!=='all'&&x.dataset.source!==e.target.value)}};</script></html>'''
    p=W.ROOT/'research/phase2777/dataset-atlas-plan-v2.html';p.write_text(proposal,encoding='utf-8')
    registration={'sha256':ref,'html':str(p),'created_at':now}
    W.save(C.OUT/'dataset-atlas-plan-registration-v2.json',registration,True)
    W.save(C.OUT/'dataset-atlas-plan-registration-current.json',registration)
    C.status('dataset_atlas_plan','proposed',plan_sha256=ref)
    print(json.dumps({'plan_sha256':ref,'html':str(p)},ensure_ascii=False))


if __name__=='__main__':main()
