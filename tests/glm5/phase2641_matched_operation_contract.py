"""Crossed operation material and complete same-entity versus independent-entity campaign."""
from collections import Counter
from transformers import AutoTokenizer
from phase2641_matched_operation_material import *

OUT=RESULT/'phase2641_matched_operation_contract'

def main():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    cases=build(tok);save(OUT/'material/cases.json',cases)
    contract={'phases':{2641:'同实体八操作全析因材料与完整合同',2642:'4096条单prompt自然生成与未干预BF16全token场',2643:'初始512条件FP32原生/共同读出双伴随全坐标图谱',2644:'同实体操作复用、提及顺序/句式混杂和独立实体分析',2645:'另一512条件独立实体FP32扩大图谱',2646:'扩大图谱复核与固定单V参数真实前向验证',2647:'客户端、存储清理及完整交付'},
        'model':'local Qwen3-4B nonquantized; BF16 natural behavior, FP32 same-value numerical atlas; sequential loads',
        'material':'8families x2languages x32disjoint name pairs x2forms x2correct entities x2mention orders =4096 unique prompts',
        'families':FAMILIES,'initial_field_units':INITIAL_UNITS,'initial_discovery_units':[0,1],'initial_entity_holdout_units':[30,31],'confirmation_field_units':CONFIRM_UNITS,
        'coverage':'BF16 full-token all-layer hidden states for1024 field cases and all-layer MLP boundary values; FP32 same cases full-token hidden and both-output all-coordinate hidden/MLP boundary adjoints plus all-token V factors at0,5,17,35. Every coordinate participates; no Top-K or reduced-coordinate representation.',
        'readouts':{'native':'frozen natural BF16 first top1/top2 token IDs; same output identity matched explicitly','common':'fixed entityA-first-token minus entityB-first-token, independent of correct target; external readout, not native discovered semantics','collision':'common first-token collisions unavailable, never replaced with zero or secretly resolved with answers'},
        'comparisons':'within same entity pair across operations; answer switch, mention-order reversal and form crossed; independent entities separately; raw and low-amplitude coordinates kept; arithmetic observation not donor transfer',
        'finite_validation':'64 confirmation prefixes, four V matrices, two fixed coordinates perlayer chosen before this material from prior numeric calibration, +/-0.2 matrixRMS; no coordinate optimization on language data; observe both readouts',
        'storage':'keep full published exemplar packs and complete all-coordinate derived maps; only manifest-listed unshown raw fields cleaned after completed analysis and publication checks',
        'limits':['32 name-pairs reused across tasks are not4096 independent semantic units','tasks all select a named person; this common output function is not general language encoding','English8 vs Chinese4 word-sense lexical entries are not strict translations','first-token outputs do not equal complete names; full natural name accuracy separately recorded','semantic content is not held identical across all families; entities/answers/order are crossed controls, not a magic causal isolation']}
    save(OUT/'protocol/frozen.json',contract)
    collisions=[{'case_id':r['case_id'],'entities':[r['entity_a'],r['entity_b']],'ids':r['common_readout_ids']} for r in cases if not r['common_readout_available']]
    summary={'cases':len(cases),'by_family':dict(Counter(r['family'] for r in cases)),'fullfield_cases':sum(r['field_set']!='behavior_only' for r in cases),
        'common_firsttoken_collision_cases':len(collisions),'collision_entities':sorted({tuple(r['entities']) for r in collisions}),
        'token_length_min':min(len(r['prompt_ids']) for r in cases),'token_length_max':max(len(r['prompt_ids']) for r in cases),'contract':contract}
    save(OUT/'analysis/firsttoken_collisions.json',collisions)
    checks={'all4096_prompts':len(cases)==4096,'all_prompts_unique':len({r['prompt'] for r in cases})==4096,
        'all1024_field_cases':sum(r['field_set']!='behavior_only' for r in cases)==1024,
        'entity_split_disjoint':not set(INITIAL_UNITS)&set(CONFIRM_UNITS),'prior2640_complete':read(RESULT/'phase2640_paired_precision_atlas/analysis/terminal_audit.json')['all_checks_passed']}
    assert all(checks.values())
    finish(2641,'同实体八语言操作4096全析因材料与全坐标研究合同',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '固定同一对自然人名，交叉八种语言操作、目标实体、先提及实体和两种句式。相同实体是匹配对照，不作为独立复现；不同实体另设留出及扩大。分开实际原生输出对与外部A/B共同读出。',
        r'x=(f,\ell,e,s,v,o);\quad N=8\cdot2\cdot32\cdot2\cdot2\cdot2=4096;\quad m_c=z_{\mathrm{first}(A)}-z_{\mathrm{first}(B)}\ne m_n=z_{\mathrm{top1}}-z_{\mathrm{top2}}.',
        '时序、类别、词义、施事、否定、数量比较、回指、标点八族；中英各32对人名。原始文本/IDs/实体跨度全部保存，所有目标首token碰撞显式列为不可用；初始与确认各512条件。',
        '上一轮不同index过滤只剩4个跨族同输出对，不表示同材料操作匹配不存在。新材料将同实体跨操作与独立实体泛化拆成两个问题。首先测绘条件响应，任何“选择实体”的晚层公共读出不能升级成全部语义共享主干。',
        '人名选择输出较窄，语料为受控合成短句，不代表开放语言。相同实体并未保证不同任务的命题内容完全一致；词义材料及句法模板数量有限。共同读出由实验者给定，必须与模型自然输出对分账；firsttoken碰撞不得用零填补。',
        '完整执行2642—2647：先自然生成和全场，再双读出全坐标结构、独立材料扩大、固定单参数前向、客户端和清理，不因单一行为或因果门失败停止。')

if __name__=='__main__':main()
