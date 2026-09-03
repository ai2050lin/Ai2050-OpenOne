export const STATE_HEATMAP_ROUTE = {
  id: 'embedding-hidden-state-heatmap-v1',
  title: 'Embedding + HiddenState Heatmap',
  resultType: 'state_heatmap',
  embeddingEvent: 'embedding',
  hiddenStateEvents: ['residual2', 'residual_output', 'residual1', 'residual_input'],
  maxDimensions: 12,
  sourceSchema: 'real_component_trace.v1',
  boundary:
    '只显示同一次 Run 采集的稀疏 top-k：左侧为词嵌入，右侧为按 Layer 的 HiddenState；深色格表示未采样，不表示零值。',
};

import { researchAssetUrl } from '../config/researchAssets';

export const RELATION_CONTRAST_HEATMAP_ROUTE = {
  id: 'relation-contrast-heatmap-v1',
  title: 'Relation Contrast Heatmap',
  resultType: 'relation_contrast_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c097_relation_contrast_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'relation_contrast_heatmap.v1',
  boundary:
    '显示的是三个共享原始单元的关系四格对比之均值 G_C；它是诊断性对比几何，不是纯语义、独立三重复或因果机制。',
};

export const GRAPH_WALSH_HEATMAP_ROUTE = {
  id: 'graph-walsh-heatmap-v1',
  title: 'Directed Graph Walsh Heatmap',
  resultType: 'graph_walsh_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c100_graph_walsh_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'graph_walsh_heatmap.v1',
  boundary:
    '显示的是受控有向图中、对答案代码取平均后的 xy 路径交互。它是单一 Qwen3 任务场的观察结果，不是通用语义图编码、固定神经元或因果闭合。',
};

export const C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE = {
  id: 'c101-activation-coordinate-heatmap-v1',
  title: 'C101 Activation Coordinate Field',
  resultType: 'activation_coordinate_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c101_activation_coordinate_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c101_activation_coordinate_heatmap.v1',
  boundary:
    '格子是 Qwen3-4B 的 2560 个激活坐标：state0 为词嵌入，后续 state 为 Hidden State；它们不是权重参数、独立神经元或已确认的语义机制。',
};

export const C102_COORDINATE_BARCODE_HEATMAP_ROUTE = {
  id: 'c102-coordinate-barcode-heatmap-v1',
  title: 'C102 Coordinate Barcode Field',
  resultType: 'coordinate_barcode_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c102_coordinate_barcode_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c102_coordinate_barcode_heatmap.v1',
  boundary:
    '8/8 冻结条码在新材料中复现；C105 纠正候选顺序后，5/8 族通过双分区受控干预。格子是词嵌入或 Hidden State 的激活坐标，不是权重参数、稀疏语义神经元或已经闭合的语言机制。',
};

export const C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE = {
  id: 'c104-upstream-role-barcode-heatmap-v1',
  title: 'C104-C107 Upstream Truth / Task Field',
  resultType: 'upstream_role_barcode_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c104_upstream_role_barcode_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c104_upstream_role_barcode_heatmap.v1',
  boundary:
    '4/4 冻结上游条码在新材料中复现。C107 重算后，属性绑定与施事-受事只在 raw truth-direction 判据达到四格受控；code-aligned task 判据没有任何 K 达到四格。128/256 是复用数据上的首次受控响应规模，不是最小、必要或功能充分联盟。格子是激活坐标，不是模型权重或语义神经元。',
};

export const C109_ROLE_STATE_FIELD_ATLAS_ROUTE = {
  id: 'c109-role-state-field-atlas-v1',
  title: 'C109-C155 Relation-Role-State and Type-Graph Transfer Atlas',
  resultType: 'role_state_field_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c109_role_state_field_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c109_role_state_field_atlas.v1',
  boundary:
    '显示C109-C150的全坐标关系-角色-状态场、析因响应、五语言族观测、转移预测审计和局部可预测窗口。全部2560列均为embedding或HiddenState激活坐标，不是参数权重。C142有9/35个效应复现；C143完整轨迹预测失败；C144只通过聚合一阶重建；C146没有两模型共同接口；C150窗口是回溯性观察，不改写旧门也不授权因果。尚未建立最小必要结构、完整传输图、跨模型内部不变量、语义神经元、因果闭合或新数学。',
};

export const C157_C166_LOCAL_FIELD_HEATMAP_ROUTE = {
  id: 'c157-c166-local-field-coordinate-heatmap-v1',
  title: 'C157-C166 Local Field and Coordinate Transmission Atlas',
  resultType: 'local_field_coordinate_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c157_c166_local_field_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c157_c166_local_field_heatmap.v1',
  boundary:
    '显示C159-C165的自然/伪词响应、接收者单侧预测、q24->q25局部传动和语言程序项。每列是Qwen3-4B的embedding或HiddenState物理激活坐标，不是权重参数、独立语义神经元或跨模型同名坐标。',
};

export const C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE = {
  id: 'c167-c168-relation-residual-coordinate-heatmap-v1',
  title: 'C167-C168 Relation-Conditioned Transmission Atlas',
  resultType: 'relation_residual_coordinate_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c167_c168_relation_residual_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c167_c168_relation_residual_heatmap.v1',
  boundary:
    '每行源编号是q24关系角色的物理激活坐标，每列是q25目标角色的物理激活坐标。fresh关系残差通过预测与错源坐标控制，但尚未得到最小电路、自然必要性、Attention/MLP归因或完整语言机制。',
};

export const C170_ROLE_CHECKPOINT_HEATMAP_ROUTE = {
  id: 'c170-role-checkpoint-coordinate-heatmap-v1',
  title: 'C170 Role / Checkpoint Relation Transport Atlas',
  resultType: 'role_checkpoint_coordinate_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c170_role_checkpoint_coordinate_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c170_role_checkpoint_coordinate_heatmap.v1',
  boundary:
    '同一16坐标联盟在q23-q25的relation源角色稳定、query部分复用、primary不复用。这里显示的是下一检查点激活响应，不证明primary/query不存在自己的最优坐标，也不等于最小自然因果电路。',
};

export const C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE = {
  id: 'c183-natural-response-ecology-heatmap-v1',
  title: 'C183 Natural Relation Response Ecology',
  resultType: 'natural_response_ecology_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c183_natural_response_ecology_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c183_natural_response_ecology_heatmap.v1',
  boundary:
    '完整资产保存七个自然关系族、六个角色在embedding与HiddenState检查点的全部2560个激活坐标，以及relation源q24到q25的局部有符号响应。query/relation响应跨新词汇复现，但固定目标边与坐标对没有复现；这些坐标不是模型权重、语义神经元或唯一因果电路。',
};

export const C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE = {
  id: 'c189-new-material-response-scaffold-heatmap-v1',
  title: 'C189 Generic Scaffold / Phrase-Conditioned Field',
  resultType: 'new_material_response_scaffold_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c189_new_material_response_scaffold_heatmap.json'),
  maxDimensions: 12,
  sourceSchema: 'c189_new_material_response_scaffold_heatmap.v1',
  boundary:
    '完整资产保存27个新词汇/新释义关系单元的q24到q25目标能量与有符号平均响应，每行均含2560个物理激活坐标。通用传播骨架跨条件稳定，但细粒度关系族指纹随措辞重构；它们不是权重、语义神经元或唯一因果电路。',
};

export const C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE = {
  id: 'c191-response-equivalence-atlas-v1',
  title: 'C191 Missing-Aware Response Equivalence Atlas',
  resultType: 'response_equivalence_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c191_response_equivalence_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c191_response_equivalence_atlas.v1',
  boundary:
    '52个行为合格单元的最近邻主要保留任务内关系族，但关系族与关系措辞仍有关联，且成对中位差较小。每列是q25物理激活坐标；这不是抽象语义本体、参数权重、语义神经元或因果电路。',
};

export const C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE = {
  id: 'c193-program-centered-response-residual-v1',
  title: 'C193 Program-Centered Response Residual',
  resultType: 'program_centered_response_residual_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c193_program_centered_response_residual.json'),
  maxDimensions: 12,
  sourceSchema: 'c193_program_centered_response_residual.v1',
  boundary:
    '显示的是从每格响应谱中减去“同程序、其他关系族”均值后的有符号L1归一化残差。关系词变化会显著削弱同族近邻，程序中心化只部分恢复且未通过探索门；这是失败分解，不是对C192的救援或抽象语义确认。',
};

export const C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE = {
  id: 'c202-signed-operator-campaign-v1',
  title: 'C194-C202 Signed Operator Campaign',
  resultType: 'signed_operator_campaign_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c202_signed_operator_campaign.json'),
  maxDimensions: 12,
  sourceSchema: 'c202_signed_operator_campaign.v1',
  boundary:
    '显示Qwen3-4B词嵌入、HiddenState基线和q23干预后的q24/q25有符号响应，共保留全部2560个物理激活坐标。C195确认局部轨迹可观测，但C196-C199预测门失败、C200因类型条件不足未做因果实验；C201跨模型原始拓扑余弦受近均匀基线抬高，不能视为共同内部代码。',
};

export const C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE = {
  id: 'c215-response-interval-composition-atlas-v1',
  title: 'C205-C215 Response Interval and Composition Atlas',
  resultType: 'response_interval_composition_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c215_response_interval_composition_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c215_response_interval_composition_atlas.v1',
  boundary:
    '显示九种语言程序在embedding/q23/q24/q25的基线、dose-1奇偶干预响应，以及两跳路径组合的实测、加性预测和交互残差；全部2560列都是Qwen3-4B物理激活坐标。自然晚层响应可复现，但固定线性齿轮、双构造组合闭合、类型化救援和三模型共同不变量均未通过冻结门。',
};

export const C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE = {
  id: 'c220-response-state-minimality-atlas-v1',
  title: 'C220 Response-State Minimality Atlas',
  resultType: 'response_state_minimality_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c220_response_state_minimality_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c220_response_state_minimality_atlas.v1',
  boundary:
    '显示C216冻结模板与C219共享接口新词汇在q24/q25关系及答案边界角色上的有符号响应。最小观测梯子的新鲜集为0.80，坐标置乱与因子交换均为0.20，但能量负控仍有0.65；这不是唯一最小状态、上下文无关语义代码或因果机制。',
};

export const C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE = {
  id: 'c222-surface-conditioned-response-atlas-v1',
  title: 'C222 Surface-Conditioned Signed Response Atlas',
  resultType: 'surface_conditioned_response_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c222_surface_conditioned_response_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c222_surface_conditioned_response_atlas.v1',
  boundary:
    '显示C216与C221在q24/q25关系及答案边界角色上的全部2560个有符号激活坐标。同一C221表面跨新词汇为20/20，原始场预测NRMSE为0.579；跨表面模板失败，面板中心化仅是回溯分解，不是单样本解码器、因果结构或普适关系代码。',
};

export const C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE = {
  id: 'c233-surface-transport-composition-atlas-v1',
  title: 'C223-C233 Surface Transport and Composition Atlas',
  resultType: 'surface_transport_composition_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c233_surface_transport_composition_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c233_surface_transport_composition_atlas.v1',
  boundary:
    '保存五个语言族在embedding/q23/q24/q25的表面条件响应护照，以及运输和组合锁箱的预测、真值、误差或交互；每行含全部2560个Qwen3-4B物理激活坐标。表面运输、广泛组合、因果、三模型共同拓扑和新数学门均未通过，局部通过项不能解释为固定语义坐标或完整机制。',
};

export const C243_CONDITIONAL_EVENT_ATLAS_ROUTE = {
  id: 'c243-conditional-event-atlas-v1',
  title: 'C234-C243 Conditional Event Atlas',
  resultType: 'conditional_event_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c243_conditional_event_atlas_compact.json'),
  maxDimensions: 12,
  sourceSchema: 'c243_conditional_event_atlas_compact.v1',
  boundary:
    '保存五个语言族、三个析因效应、embedding与全部36层、六个角色的lockbox+fresh有符号均值；每行含全部2560个Qwen3物理激活坐标。未见事件仅2/5族、组合仅1/5族通过，因果未测试；三模型正结果只是受脚手架混杂约束的粗因素-角色-相对深度候选。',
};

export const C244_INDEPENDENT_EVENT_REPLICATION_ROUTE = {
  id: 'c244-independent-event-replication-v1',
  title: 'C244 Independent Event Replication',
  resultType: 'independent_event_replication_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c244_independent_event_replication.json'),
  maxDimensions: 12,
  sourceSchema: 'c244_independent_event_replication.v1',
  boundary:
    '两种新表面、六套新词汇的独立Qwen3复验；每行保留全部2560个物理激活坐标。态度-事件、转折和比较通过冻结事件门；类型图与翻译被长度匹配控制击败。它不是语义神经元、最小坐标联盟或因果路径。',
};

export const C245_CONFIRMED_EVENT_CORE_ROUTE = {
  id: 'c245-confirmed-event-core-v1',
  title: 'C245 Confirmed Signed Event Core',
  resultType: 'confirmed_event_core_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c245_confirmed_event_core.json'),
  maxDimensions: 12,
  sourceSchema: 'c245_confirmed_event_core.v1',
  boundary:
    '显示C237 discovery规则与C244独立材料同号稳定交集：-1/0/+1分别表示下降、缺失和上升事件。确认核仍依赖家族、效应、检查点、角色、阈值和受控任务，不是权重、语义神经元、最小路径或因果机制。',
};

export const C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE = {
  id: 'c254-tri-material-event-atlas-v1',
  title: 'C246-C254 Tri-Material Event and Full-Token Atlas',
  resultType: 'tri_material_event_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c254_tri_material_event_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c254_tri_material_event_atlas.v1',
  boundary:
    '三材料角色行显示同坐标同号事件，full-token行显示精确token对齐后的逐坐标事件符号平衡。角色行仍含span平均，token行不包含无法一一对齐的编辑片段；坐标是激活维度，不是权重、语义神经元或唯一因果电路。',
};

export const C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE = {
  id: 'c260-output-path-causal-atlas-v1',
  title: 'C256-C260 Output-Sensitive Early Path Atlas',
  resultType: 'output_path_causal_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c260_output_path_causal_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c260_output_path_causal_atlas.v1',
  boundary:
    'q0是词嵌入，q1-q16是block后pre-norm HiddenState；每列是Qwen3-4B物理激活坐标。前缀/角色阶梯只在冻结事件掩码内定位充分干预路径，不证明坐标最小性、自然必要性、自由生成闭合、语义神经元或Attention/MLP电路。',
};

export const C262_GENERATION_SPECIFICITY_ATLAS_ROUTE = {
  id: 'c262-generation-specificity-atlas-v1',
  title: 'C261-C262 Coverage and Full-Word Specificity Atlas',
  resultType: 'generation_specificity_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c262_generation_specificity_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c262_generation_specificity_atlas.v1',
  boundary:
    '75%是冻结哈希覆盖阶梯中最早通过点，不是最小坐标联盟。正确路径虽能16/16生成目标词，但检查点掩码反序控制也为16/16，因此自然词生成特异性门失败；C260只保留为带空格token-logit控制结果。',
};

export const C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE = {
  id: 'c272-state-conditioned-operator-atlas-v1',
  title: 'C263-C272 Full-Coordinate State Operator Atlas',
  resultType: 'state_conditioned_operator_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c272_state_conditioned_operator_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c272_state_conditioned_operator_atlas.v1',
  boundary:
    '每行保存Qwen3-4B全部2560个物理激活坐标；q0是词嵌入，q1-q36是HiddenState检查点。逐坐标状态护照、滚动预测、局部稀疏因果与双向生成门均未通过；跨模型通过的是匿名角色-相对深度功能拓扑，不是共同物理坐标或因果电路。',
};

export const C273_RESPONSE_ECOLOGY_ATLAS_ROUTE = {
  id: 'c273-response-ecology-atlas-v1',
  title: 'C273 Full-Coordinate Event Failure Ecology',
  resultType: 'response_ecology_failure_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c273_response_ecology_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c273_response_ecology_atlas.v1',
  boundary:
    '逐坐标事件分类显示持续、反转、新生、护照漏报和错报。它解释冻结护照为什么失败，但不识别唯一因果来源，也不证明某种联合状态模型正确。',
};

export const C275_CROSS_ROLE_REUSE_ATLAS_ROUTE = {
  id: 'c275-cross-role-reuse-atlas-v1',
  title: 'C275 Cross-Role Same-Sign Reuse Atlas',
  resultType: 'cross_role_reuse_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c275_cross_role_reuse_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c275_cross_role_reuse_atlas.v1',
  boundary:
    '数值表示目的角色新生事件前，同一物理坐标已在来源角色以同号出现的比例。两材料六族重复支持结构性共现；C276前瞻预测未超过持续性，因此不能称为自然运输或有向因果边。',
};

export const C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE = {
  id: 'c289-joint-response-campaign-atlas-v1',
  title: 'C289 Joint Response Event Automaton Atlas',
  resultType: 'joint_response_event_automaton_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c289_joint_response_campaign_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c289_joint_response_campaign_atlas.v1',
  boundary:
    '每行保留Qwen3全部2560个物理激活坐标。C280-C281确认的是六族可前瞻预测的离散事件自动机；组合残差仅为描述，C291的合格局部坐标干预未通过，不能称为唯一因果电路或连续HiddenState闭合。跨模型结果只比较匿名角色拓扑。',
};

export const C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE = {
  id: 'c308-conditional-hypergraph-campaign-atlas-v1',
  title: 'C308 Conditional Hypergraph Campaign Atlas',
  resultType: 'conditional_hypergraph_campaign_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c308_conditional_hypergraph_campaign_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c308_conditional_hypergraph_campaign_atlas.v1',
  boundary:
    '每行保存Qwen3全部2560个物理激活坐标，并明确区分embedding、HiddenState、转移预测、幅值、组合与资格掩码。C302支持六族锁箱场组合预测，C307支持匿名跨模型拓扑；C306因果补丁失败，因此不能解释为唯一超图、连续模拟器或参数级因果电路。',
};

export const C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE = {
  id: 'c335-dual-axis-response-atlas-v1',
  title: 'C310-C335 Dual-Axis Response and Graph-Depth Atlas',
  resultType: 'dual_axis_response_atlas_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c335_dual_axis_response_atlas.json'),
  maxDimensions: 12,
  sourceSchema: 'c335_dual_axis_response_atlas.v1',
  boundary:
    '每行保留Qwen3全部2560个物理激活坐标，并区分embedding、HiddenState与跨检查点平均图深度算子。六族二阶残差具有预测特异性，改名图上的浅层增量仍可迁移但明显衰减；这不是模型参数、语义神经元、唯一因果电路、功能双模拟或新数学定理。',
};

export const C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE = {
  id: 'c360-single-sample-operator-field-v1',
  title: 'C336-C360 Single-Sample Operator Field',
  resultType: 'single_sample_operator_field_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c360_single_sample_operator_field.json'),
  maxDimensions: 12,
  sourceSchema: 'c360_single_sample_operator_field.v1',
  boundary:
    '每行保留一个冻结确认样本的全部2560个Qwen3-4B物理激活坐标。A/B只有局部单样本预测增益，二阶交互I未通过联合门；图递归和因果中介未获资格，粗跨模型响应候选也不是功能双模拟。',
};

export const C390_LANGUAGE_OPERATION_FIELD_ROUTE = {
  id: 'c390-language-operation-full-coordinate-v1',
  title: 'C369-C390 Typed Language Operation Field',
  resultType: 'language_operation_full_coordinate_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c390_language_operation_full_coordinate.json'),
  maxDimensions: 12,
  sourceSchema: 'c390.language_operation_full_coordinate.v1',
  boundary:
    'Rows retain all 2560 Qwen3-4B physical activation coordinates for typed operation responses and one complete token field. They are observations, not model weights, semantic neurons, universal operators, or a causal language algebra.',
};

export const C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE = {
  id: 'c398-independent-construction-lockbox-v1',
  title: 'C391-C398 Independent Construction Lockbox',
  resultType: 'independent_construction_lockbox_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c398_independent_construction_lockbox.json'),
  maxDimensions: 12,
  sourceSchema: 'c398.independent_construction_lockbox.v1',
  boundary:
    'Each row retains all 2560 Qwen3-4B physical activation coordinates for fresh-construction interaction centroids. The within-material prediction gains are small, only negation weakly transfers the old atlas, and family decoding is embedding-confounded; these rows are descriptive observations, not causal semantic coordinates or a language algebra.',
};

export const C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE = {
  id: 'c414-output-sensitive-language-field-v1',
  title: 'C399-C414 Output-Sensitive Language Field',
  resultType: 'output_sensitive_language_field_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c414_output_sensitive_language_field.json'),
  maxDimensions: 12,
  sourceSchema: 'c414.output_sensitive_language_field.v1',
  boundary:
    'Rows retain all 2560 Qwen3-4B physical activation coordinates. Cross-construction transfer was sparse, q0 prediction did not pass, and no writer branch qualified; these are construction-conditioned observations rather than universal operators or causal semantic coordinates.',
};

export const C433_AXIS_LOCKBOX_FIELD_ROUTE = {
  id: 'c433-axis-lockbox-field-v1',
  title: 'C426-C433 Axis Lockbox Interaction Field',
  resultType: 'axis_lockbox_interaction_field_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c433_axis_lockbox_field.json'),
  maxDimensions: 12,
  sourceSchema: 'c433.axis_lockbox_field.v1',
  boundary:
    'Rows retain all 2560 Qwen3-4B physical activation coordinates. Higher-order attitude interactions improved some unseen-construction predictions, but the preregistered pair gate and dynamic-donor gate failed; the field is observational and not a causal circuit.',
};

export const C26801_RESIDUAL_STATE_OPERATOR_FIELD_ROUTE = {
  id: 'c26801-residual-state-operator-field-v1',
  title: 'C26801 Residual State Operator Field',
  resultType: 'residual_state_operator_field_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c26801_residual_state_operator_field.json'),
  maxDimensions: 12,
  sourceSchema: 'c26801.residual_state_operator_field.v1',
  boundary:
    'Rows retain all 2560 Qwen3-4B physical coordinates for an exact token embedding, HiddenState checkpoints, Attention/MLP/total family residuals, fitted diagonal slopes, and held-out matched-vs-mismatch gains. They are activations or fitted parameters, not model weights, semantic neurons, a universal code, or a causal language mechanism.',
};

export const C32561_LANGUAGE_ENCODING_FIELD_ROUTE = {
  id: 'c32561-language-encoding-field-v1',
  title: 'C32561-C39440 Full-coordinate Language Trajectory, Semantic Passport, VJP, and Finite-Effect Field',
  resultType: 'semantic_encoding_output_field_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c32561_semantic_encoding_output_field.json'),
  maxDimensions: 12,
  sourceSchema: 'c32561.semantic_encoding_output_field.v1',
  boundary:
    'Rows retain all 2560 Qwen3-4B physical coordinates for concrete token/output embeddings, event-aligned embedding activations and HiddenState, semantic/lexical signed trajectories, fresh-family update passports, coordinate RMS, fitted local slopes, and output contributions. Fixed coordinate and event identity improve local prediction, but Chinese/family transfer and the internal-to-output compiler remain open; no causal gear or closed language mechanism is claimed.',
};

export const C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE = {
  id: 'c42641-output-conditioned-crossmodel-field-v1',
  title: 'C39761-C160512 Full-coordinate Q/K/V, Region, and Staged Compiler Field',
  resultType: 'output_conditioned_crossmodel_field_heatmap',
  sourcePath: researchAssetUrl('research_kernel/c42641_output_conditioned_crossmodel_field.json'),
  maxDimensions: 12,
  sourceSchema: 'c42641.output_conditioned_crossmodel_field.v1',
  boundary:
    'Each panel uses a model-local physical axis: residual/embedding coordinates, attention-head indices, K/V head coordinates, token positions, token regions, and relative functional stages are never conflated. Phase2539-2547 add all-token embedding/HiddenState, exact Q/K/V and weighted-value coordinates, Attention/MLP-to-next-K/V writes, autonomous interventions, BF16 cross-model stage tests, and an independent value-token region replication. They support a repeated content-to-address-to-downstream-to-output functional skeleton, not a shared physical basis, minimal route, universal semantic gear, or closed language mechanism.',
};

export const HEATMAP_RESULT_TYPES = [
  STATE_HEATMAP_ROUTE.resultType,
  RELATION_CONTRAST_HEATMAP_ROUTE.resultType,
  GRAPH_WALSH_HEATMAP_ROUTE.resultType,
  C101_ACTIVATION_COORDINATE_HEATMAP_ROUTE.resultType,
  C102_COORDINATE_BARCODE_HEATMAP_ROUTE.resultType,
  C104_UPSTREAM_ROLE_BARCODE_HEATMAP_ROUTE.resultType,
  C109_ROLE_STATE_FIELD_ATLAS_ROUTE.resultType,
  C157_C166_LOCAL_FIELD_HEATMAP_ROUTE.resultType,
  C167_C168_RELATION_RESIDUAL_HEATMAP_ROUTE.resultType,
  C170_ROLE_CHECKPOINT_HEATMAP_ROUTE.resultType,
  C183_NATURAL_RESPONSE_ECOLOGY_HEATMAP_ROUTE.resultType,
  C189_NEW_MATERIAL_RESPONSE_SCAFFOLD_HEATMAP_ROUTE.resultType,
  C191_RESPONSE_EQUIVALENCE_ATLAS_ROUTE.resultType,
  C193_PROGRAM_CENTERED_RESIDUAL_HEATMAP_ROUTE.resultType,
  C202_SIGNED_OPERATOR_CAMPAIGN_HEATMAP_ROUTE.resultType,
  C215_RESPONSE_INTERVAL_COMPOSITION_ATLAS_ROUTE.resultType,
  C220_RESPONSE_STATE_MINIMALITY_ATLAS_ROUTE.resultType,
  C222_SURFACE_CONDITIONED_RESPONSE_ATLAS_ROUTE.resultType,
  C233_SURFACE_TRANSPORT_COMPOSITION_ATLAS_ROUTE.resultType,
  C243_CONDITIONAL_EVENT_ATLAS_ROUTE.resultType,
  C244_INDEPENDENT_EVENT_REPLICATION_ROUTE.resultType,
  C245_CONFIRMED_EVENT_CORE_ROUTE.resultType,
  C254_TRI_MATERIAL_EVENT_ATLAS_ROUTE.resultType,
  C260_OUTPUT_PATH_CAUSAL_ATLAS_ROUTE.resultType,
  C262_GENERATION_SPECIFICITY_ATLAS_ROUTE.resultType,
  C272_STATE_CONDITIONED_OPERATOR_ATLAS_ROUTE.resultType,
  C273_RESPONSE_ECOLOGY_ATLAS_ROUTE.resultType,
  C275_CROSS_ROLE_REUSE_ATLAS_ROUTE.resultType,
  C289_JOINT_RESPONSE_CAMPAIGN_ATLAS_ROUTE.resultType,
  C308_CONDITIONAL_HYPERGRAPH_CAMPAIGN_ATLAS_ROUTE.resultType,
  C335_DUAL_AXIS_RESPONSE_ATLAS_ROUTE.resultType,
  C360_SINGLE_SAMPLE_OPERATOR_FIELD_ROUTE.resultType,
  C390_LANGUAGE_OPERATION_FIELD_ROUTE.resultType,
  C398_INDEPENDENT_CONSTRUCTION_LOCKBOX_ROUTE.resultType,
  C414_OUTPUT_SENSITIVE_LANGUAGE_FIELD_ROUTE.resultType,
  C433_AXIS_LOCKBOX_FIELD_ROUTE.resultType,
  C26801_RESIDUAL_STATE_OPERATOR_FIELD_ROUTE.resultType,
  C32561_LANGUAGE_ENCODING_FIELD_ROUTE.resultType,
  C42641_OUTPUT_CONDITIONED_CROSSMODEL_FIELD_ROUTE.resultType,
];
