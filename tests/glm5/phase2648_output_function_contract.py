"""Prospective contract for a complete operation x output-function campaign."""
from transformers import AutoTokenizer
from phase2648_output_function_material import *

OUT=RESULT/'phase2648_output_function_contract'

def main():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    cases=build(tok);save(OUT/'material/cases.json',cases)
    p={'plan':{2648:'8192语言操作×输出功能材料与冻结合同',2649:'单prompt BF16自然生成及4096全坐标原场',2650:'初始2048条件双读出FP32逐坐标与全token V因子',2651:'固定输出词与实体变化分离的完整坐标响应图',2652:'独立实体2048扩大图谱和冻结算法复核',2653:'不同输出功能的1088固定单参数条件验证',2654:'证据审计、参数客户端、存储清理与完整交付'},
        'scope':'8families x2languages x16new name-pairs x2forms x2target entities x2mention orders x4mode cells =8192 prompts',
        'mode_cells':MODES,'initial_units':INITIAL_UNITS,'confirmation_units':CONFIRM_UNITS,
        'full_coordinate_scope':'BF16 all-layer full-token H streamed into all-coordinate maps, H3anchors and MLP boundary for4096 cases; FP32 same selected cases H3anchors, MLP boundary, native/common boundary adjoints and four-layer alltoken V factors. Full-token raw H for64 published exemplars only. NoTopK, no donor intervention.',
        'common_readouts':'canonical entityA-first-token minus entityB-first-token forname/cloze; fixedYes-No or是-否 fortruth. Actual native IDs separately recorded; unavailable collisions explicit.',
        'signed_response':'semantic-target A-minus-B observed difference; truth-oriented difference has a declared -1 factor fortruth_b. Preserve BOTH, never silently answer-align all gradients.',
        'frozen_envelope':'target RMS>order RMS and>form RMS percoordinate, inherited2647, no threshold optimization; compare across new entities, across output functions and truth probe reversal; all physical coordinates retained',
        'finite_scalar':'same8 literal V scalar coordinates from2646 scalar_selection.json;64confirmation cases with8families x2languages x4modecells, unit12/form0/target0/order0; +/-0.2 matrixRMS roundedBF16 targets in calibratedFP32 core; allweightrestoration hashes',
        'models':'local Qwen3-4B only for this frozen campaign; prior crossmodel arithmetic checks do not establish these new semantic patterns in other models; sequentialCUDA nonquantized BF16 behavior and same-valued FP32 numeric fields',
        'storage':'single-model jobs only.8GiB disk safety. All-token hidden values streamed for every recorded case, not compressed coordinates; only full exemplar raw H persisted. Delete manifested unpublished anchor/factor packs only after complete analysis and client checks.',
        'limits':['All operations still identify a unique person; yes/no changes output function and queried proposition, not a perfect surgical isolation of semantics.',
            'Cloze assistant sentence prefix is supplied by experimenter, not autonomous full-sentence production.',
            'New person names but reused body templates and sense/fruit catalog; not all lexical content is heldout.',
            'First-token contrast differs from whole answer, leading whitespace may change native cloze token identities.',
            'Fixed output rows can create shared geometry; high gradient cosine alone not semantic reuse.',
            '4096 field cases share4initial and4confirmation name-pairs perlanguage; not4096 independent semantic units.'],
        'prior_terminal_sha256':sha(RESULT/'phase2647_matched_operation_delivery/analysis/terminal_audit.json')}
    save(OUT/'protocol/frozen.json',p)
    collisions=[{'case_id':r['case_id'],'words':r['common_readout_words'],'ids':r['common_readout_ids']} for r in cases if not r['common_readout_available']];save(OUT/'analysis/first_token_collisions.json',collisions)
    checks={'8192_unique_prompts':len(cases)==len({r['prompt'] for r in cases})==8192,'4096_field_cases':sum(r['field_set']!='behavior_only' for r in cases)==4096,
        '64_published_cases':sum(r['published'] for r in cases)==64,'disjoint_field_entities':not set(INITIAL_UNITS)&set(CONFIRM_UNITS),
        'single_gpu_prefix_length_safe':max(len(r['prompt_ids']) for r in cases)<=106}
    assert all(checks.values())
    finish(2648,'8192语言操作与输出功能交叉材料及完整合同',OUT,{'provenance':str(Path(__file__)),'summary':{'cases':len(cases),'field_cases':4096,'token_length_range':[min(len(r['prompt_ids']) for r in cases),max(len(r['prompt_ids']) for r in cases)],'common_collisions':len(collisions),'contract':p},'checks':checks},
        '沿2647幅度较稳定、方向不稳定的证据，交叉相同语言事实与人名问答、给定前缀事实续写、两种询问实体的自然真假判断。名字变化与固定是/否输出头行分开观察，先冻结全坐标规则，再看新内部场。',
        r'x=(f,\ell,e,s,v,o,m);\quad N=8\cdot2\cdot16\cdot2\cdot2\cdot2\cdot4=8192;\quad D_{semantic}=H(v=0)-H(v=1),\quad D_{truth}=(-1)^pD_{semantic}.',
        '八族、双语、各16对新人名；两句式、两个正确目标、双提及顺序；name/cloze/truth_a/truth_b四模式格。truth_a与truth_b询问不同实体，真假由v和p共同决定。完整body、陈述、事实前缀、IDs、跨度均留档。',
        '固定是/否读出不再随人名更换头矩阵行，有助于检验上轮固定方向不稳定的来源；但输出功能也改变了问题本身，仍不能把差异全部归因读出矩阵。原生与外部固定读出分账，语义目标方向与真值方向分账。',
        '使用新名字但旧句型与词义材料，只有一个模型；所有任务仍基于唯一人物属性，不等于开放组合语言。给定assistant事实前缀是实验支架，不能冒称自主整句生成。全token H流式分析、原包仅64示例；其余锚点与参数因子范围明确。',
        '完整执行2649—2654并记录阴性：自然行为、初始全场、条件分离、独立扩大、真实固定标量前向、客户端与清理。相同目标沿现有自动续研继续，不再只比较末层相似度。')

if __name__=='__main__':main()
