"""Freeze broad tokenized material and operation edges before new model outputs."""
import itertools,shutil
from collections import defaultdict
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2686_independent_role_material import build,NAMES,FAMILIES
from phase2683_explicit_answer_audit import self_test

OUT=RESULT/'phase2686_independent_role_contract'
AXES=('family','language','unit','content_instance','form','roster_order','mention_order','target_index','output_function')

def material_audit(rows):
    assert len(rows)==len({r['prompt'] for r in rows})==8192
    by_axis={};edges={}
    for axis in ('roster_order','mention_order','target_index','form'):
        groups=defaultdict(list)
        for row in rows:groups[tuple(row[k] for k in AXES if k!=axis)].append(row)
        assert len(groups)==4096 and all(len(rr)==2 for rr in groups.values())
        ee=[]
        for rr in groups.values():
            rr.sort(key=lambda r:r[axis]);a,b=rr;assert a[axis]==0 and b[axis]==1
            if axis=='target_index':assert a['target']!=b['target'] and a['gold_semantic_target']!=b['gold_semantic_target']
            else:assert a['target']==b['target'] and a['gold_semantic_target']==b['gold_semantic_target']
            if axis in ('roster_order','mention_order'):assert a['facts_in_canonical_order']==b['facts_in_canonical_order']
            ee.append([a['case_index'],b['case_index']])
        edges[axis]=ee;by_axis[axis]={'pairs':len(ee),'expected_target_change':axis=='target_index'}
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in AXES if k!='output_function')].append(r)
    same_prefix=0
    for rr in groups.values():
        assert len(rr)==4 and len({r['body'] for r in rr})==1
        prefixes=[tuple(r['prompt_ids'][:r['body_end_token']+1]) for r in rr]
        same_prefix+=len(set(prefixes))==1
    assert len(groups)==2048
    summary={'conditions':8192,'unique_prompts':8192,'language_family_cells':16,'cases_per_language_family':512,
        'operation_pairs':by_axis,'same_body_four_function_groups':2048,'same_token_body_prefix_groups':same_prefix,
        'actual_token_range':[min(len(r['prompt_ids']) for r in rows),max(len(r['prompt_ids']) for r in rows)],
        'published_allH_examples':sum(r['published'] for r in rows),'published_full_MLP_examples':sum(r['parameter_published'] for r in rows)}
    return summary,edges

def main():
    assert read(RESULT/'phase2685_native_attention_contract/analysis/final.json')['all_checks_passed']
    assert not (OUT/'analysis/final.json').exists()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    reports={};materials={};free=shutil.disk_usage(RESULT).free;assert free>9*1024**3
    for split in ('initial','confirmation'):
        rows=build(tok,split);summary,edges=material_audit(rows);reports[split]=summary
        path=OUT/f'material/{split}.json';save(path,rows);save(OUT/f'material/{split}_operation_edges.json',edges)
        materials[split]={'case_sha256':sha(path),'edge_sha256':sha(OUT/f'material/{split}_operation_edges.json')}
        print('2686 TOKENIZER AUDIT',split,summary,flush=True)
    initial=read(OUT/'material/initial.json');fresh=read(OUT/'material/confirmation.json')
    previous=read(RESULT/'phase2681_fresh_source_confirmation/material/cases.json')
    oldnames={r[k] for r in previous for k in ('entity_a','entity_b')}
    names={split:{name for language in ('en','zh') for name in NAMES[split][language]} for split in NAMES}
    checks={'each8192beforeformaloutputs':all(r['conditions']==8192 for r in reports.values()),
        'initial_confirmation_no_entity_overlap':not names['initial']&names['confirmation'],
        'new_entities_disjoint_from2681':not (names['initial']|names['confirmation'])&oldnames,
        'initial_confirmation_no_prompt_overlap':not {r['prompt'] for r in initial}&{r['prompt'] for r in fresh},
        'all_actual_prompt_lengths_fit256':max(r['actual_token_range'][1] for r in reports.values())<=256,
        'parser50selftests':self_test()==50,'all_operation_edges_verified':True}
    assert all(checks.values()),checks
    parser_path=ROOT/'tests/glm5/phase2683_explicit_answer_audit.py'
    plan={'material':materials,'reports':reports,'field_fixed_length':256,'native_precision':'BF16 nonquantized',
        'formal_outputs_not_collected_in_this_phase':True,'formal_initial_conditions':8192,'formal_confirmation_conditions':8192,
        'generation':'Natural cachegreedy separately from paddedfield; initialQ4 max_new_tokens32 predeclared, not increased after inspectingcorrectness. Emit originalIDs/text/EOS and finalchannel explicitly. Crossmodels independently calibrated later, not inheritQ4budget.',
        'scoring':{'strict':'exactcasefoldwhole-string','legacy_normalized':'limited punctuation/Chinesealias normalization; NOT semanticaccuracy',
            'explicit_final':'frozen lastline parser, no substring/gold-guidedselection; allunparsed/ambiguous/empty/no_boundary retained','parser_sha256':sha(parser_path),'original2683_parser_was_posthoc':'Now frozen BEFORE all2686newformaloutputs; this does not retroactively make2683blind.'},
        'fullcoordinate_algorithm':'For each of4operation edges, nativeallH/allMLP coordinate response signs/zeros/amplitudes; conditions grouped byfamily/language andcrossfunction, roster/factorder/target separated. No TopK, semantic label or donor shift. QKV allactualtokens/allheads in8predeclaredlayers; incoming parameter terms on full2560coordinates ofallfixedrowwindows.',
        'claim_limits':['8192 conditions reuse256 base family-language-entity-content instances across32 interface/factor combinations; not8192independentfacts.',
            'Roster is explicit taskscaffold; r changes roster order, not a hidden psychological entity-slot variable. Fact order separate. Some relationtemplates and structures remain shared acrossinitial/confirmation.',
            'Word-sense form1 sometimes adds a reporting clause rather than changing syntacticstructure; factual-role examples are hand-designed, not independently expert-labelled or open-domain.',
            'Taxonomy and quotation tasks can be solved by lexical patterns; no gate based onlyonanswer accuracyor deletion decides routeclosure.',
            'Candidate windows are not universal circuits. Qwen14all5directions need independent>=4096confirmation plusfullbackground.',
            '256padded analysis and naturalcachegeneration are distinct numericalprotocols; actualnewshape noopchecks required beforefieldanalysis.'],
        'resources':{'free_before':free,'disk_floor':8*1024**3,'storage':'Detailed8192field storage budget must be audited before2687 modelrun; streamunpublishedraw, retainallcoordinatecharts and publishedpacks. No earliercleanup withoutnewcampaignfinalQA.'},
        'real_cases':[{'case_id':r['case_id'],'text':r['text'],'prefill':r['prefill'],'target':r['target']} for r in initial if r['unit']==r['content_instance']==r['form']==r['roster_order']==r['mention_order']==r['target_index']==0 and r['output_function']=='name']}
    save(OUT/'protocol/frozen.json',plan)
    finish(2686,'8192发现与8192独立确认材料：名单顺序、事实顺序、关系目标和输出功能分开冻结',OUT,
        {'provenance':str(Path(__file__)),'summary':{'material_audits':reports,'initial_and_confirmation_entities':NAMES,'formal_model_outputs_in_this_phase':0,'actual_case_examples':plan['real_cases']},'checks':checks},
        '在看任何新正式输出前冻结八族双语材料。名单先后顺序、事实行顺序、关系指向和表达形式分别有明确配对边；预期目标改变与目标保持由构造规则区分。只做基础材料与分词审计，不把这些审计当模型能力。',
        r'N=8\times2\times4\times2\times2\times2\times2\times2\times4=8192;\quad y(v\oplus1)\ne y(v),\quad y(r\oplus1)=y(o\oplus1)=y(f\oplus1)=y;\quad E_a=\{(i,j):x_{i,-a}=x_{j,-a},a_i=0,a_j=1\}.',
        'C0018192发现+8192新实体/新关系填充确认条件；C002每套四种操作各4096配对边及语义目标变化/保持核对；C003每套2048同正文四输出功能组与真实token前缀检查；C004两套分词长度、全prompt唯一、16实体名跨套/对2681不交叠；C005在正式生成前冻结旧严格/有限归一化分数及50自测的末行解析器；C006逐族实际测试用例写入协议和本记录。',
        '外部操作的相对编码需要把名字提及、事实顺序与关系目标分开；同一目标在表面改动后是否复用、目标变更时如何分化由不同边观察，不预先要求所有情况下同号。全部失败与低幅值坐标保留。',
        '本Phase没有模型正式生成或新能力结果；8192不是8192独立语义事实。新字符串不等于新抽象规律，名单本身是人工支架。事实模板由程序人工构造，未获独立专家盲标。truth仅探问固定canonical实体A；不能据此声称一般量词/逻辑机制。',
        '继续2687在256固定场形状和自然生成双账下采集全部H/MLP背景与八层全来源QKV；先完成存储预算和新形状无操作检查，再做2688真实全输入坐标参数项、2689实际标量、2690确认和2691顺序跨模型。')

if __name__=='__main__':main()
