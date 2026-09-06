"""Integrity and limitation audit, never equating procedural checks to mechanism success."""
import argparse,json,re,sys
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';RESULT=TESTS/'result';sys.path.insert(0,str(TESTS))
import phase2605_c676097_c692480_singleprompt_source_patch as io
import phase2608_c725249_c741632_autonomous_source_band as p8
OUT=RESULT/'phase2616_campaign_evidence_audit';MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'

def preflight():
    checks={};details={}
    for c in p8.CONDITIONS:
        records=io.read_jsonl(p8.OUT/f'behavior/{c}.jsonl')
        logits=np.load(p8.OUT/f'field/first_logits_{c}.float16.npy',mmap_mode='r')
        states=np.load(p8.OUT/f'field/greedy_prefill_{c}.float16.npy',mmap_mode='r')
        expected=0 if c=='baseline' else 1 if c.startswith('source') else 60 if c.startswith('kv') else 30
        mismatches=sum(int(np.argmax(logits[i])!=r['generated_ids'][0]) for i,r in enumerate(records))
        details[c]={'next_argmax_mismatches':mismatches,'hook_count_errors':sum(r['hook_sites']!=expected for r in records),
            'logits_finite':bool(np.isfinite(logits).all()),'states_finite':bool(np.isfinite(states).all())}
        checks[c]=mismatches==0 and details[c]['hook_count_errors']==0 and details[c]['logits_finite'] and details[c]['states_finite']
    result={'checks':checks,'details':details,'all_checks_passed':all(checks.values())};io.save_json(OUT/'analysis/measurement_preflight.json',result)
    print(json.dumps(result),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args();measurement=preflight()
    if a.preflight:return
    finals={}
    for phase in range(2600,2616):
        paths=list(RESULT.glob(f'phase{phase}_*/analysis/final.json'))
        if len(paths)!=1:raise RuntimeError((phase,paths))
        finals[phase]=io.load_json(paths[0])
    headings=[int(v) for v in re.findall(r'^## Phase (\d+):',MEMO.read_text(encoding='utf-8-sig'),re.M)]
    own=[n for n in headings if 2600<=n<=2615]
    checks={'measurement_capture_valid':measurement['all_checks_passed'],'all16_results':len(finals)==16,
        'memo_contiguous_no_duplicates':own==list(range(2600,2616)),
        'all_procedural_checks':all(f.get('all_checks_passed',False) for f in finals.values())}
    checks['cross_model_fields_finite']=all(np.isfinite(np.load(path,mmap_mode='r')).all()
        for phase,key in ((2610,'qwen14'),(2611,'glm4'),(2612,'ds7'))
        for path in (RESULT/f'phase{phase}_{key}_natural_replication/field').glob('*.npy'))
    result={'phase':2616,'timestamp':datetime.now().astimezone().isoformat(),'checks':checks,'completed_phases':list(finals),
        'measurement':measurement,'language_mechanism_closed':False,
        'milestones':{'open_greedy_2608':finals[2608]['conditions'],'train_only_2609':finals[2609]['conditions'],
            'qwen14':finals[2610]['conditions'],'glm4':finals[2611]['conditions'],'ds7':finals[2612]['conditions'],
            'chat_followup_2615':finals[2615]['conditions']},
        'limitations':['pair-oracle patches are diagnostics not the extraction algorithm','short controlled language tests admit copying shortcuts',
        'core overlap remains in 2603 splits; 2609 event-vocab split fixes one dimension only',
        'raw 24-token and chat 64-token results not interchangeable','2615 reuses 2609 heldout materials: interface calibration, not independent replication',
        'source_wrong in 2608 patches one prior token, not span-size-matched when source has multiple tokens',
        'single fixed roll is not all perturbation directions','2602/2604 old logit-lens final state was normalized again and is not primary evidence',
        'failure to close a mechanism does not prove existing mathematical frameworks are fatally defective'],
        'next_scientific_target':'event- and surface-conditional coordinate mapping with independent new semantic cases; output identity and temporal relation must be separated'}
    result['crossmodel_dual_natural_success_conditional']={}
    for phase,key in ((2610,'qwen14'),(2611,'glm4'),(2612,'ds7')):
        directory=RESULT/f'phase{phase}_{key}_natural_replication/behavior'
        rows={c:io.read_jsonl(directory/f'{c}.jsonl') for c in ('baseline','natural_donor','source1','source_roll')}
        eligible=set(r['pair_id'] for r in rows['baseline'] if r['recipient_correct']) & set(r['pair_id'] for r in rows['natural_donor'] if r['recipient_correct'])
        result['crossmodel_dual_natural_success_conditional'][key]={'n_pairs':len(eligible),'selection':'both natural variants correct; conditional view, not replacement for all-60 analysis',
            'conditions':{c:p8.metrics([r for r in rows[c] if r['pair_id'] in eligible]) if eligible else None for c in rows}}
    result['all_checks_passed']=all(checks.values());io.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=rf'''

## Phase 2616: 大阶段证据终审、测量校验与条件编码研究边界 [{stamp}]

**原理、用例与公式。** 逐项检查Phase2600—2615完整性，并将1440条greedy的第一个token与保存的完整词表argmax核对，核验每一干预实际调用站点数与完整坐标有限性。过程检查通过不等于科学假说通过：

$$\hat y_0=\arg\max_v z_v,\qquad \text{{measurement integrity}}\ne\text{{mechanism closure}}.$$

**结果汇总。** 检查={json.dumps(checks,ensure_ascii=False)}。Phase2608全部外测120pair的KV生成另一侧答案率={finals[2608]['conditions']['kv']['donor_correct']:.6f}，KV-roll={finals[2608]['conditions']['kv_roll']['donor_correct']:.6f}；这不能与2607经过行为选择45pair的93.3%候选转向率直接相减。完整三模型、训练独立方向与chat自动续研结果全部写入`final.json/milestones`。

**相关文件。** `{OUT}`含测量preflight、16Phase索引与终审final；客户端数据Phase2614；各Phase原始输出与公式保留在既有MEMO。仅本MEMO新增Markdown记录。

**勘误与硬伤。** Phase2608错token条件只有前一token，source多token时总位移并不匹配，不能把该对照当严格等量位置因果证据；roll使用同source位置，是相应方向对照。Phase2602/2604 logit lens末状态已经final norm又被归一化，不作为本轮关键证据。source平均不保留词内次序；“六操作”指六受控材料族，并未排除复制、格式、标签捷径。重复Phase未闭合不逻辑蕴含现有范式必然致命缺陷。未获得普遍机制，更不能据此宣布需要新数学定理。

**理论进展与第一性原理。** 有限权重能够被不同上下文反复调用，是架构事实；要解释语言能力，需要找出外部操作如何改变条件化状态、哪些坐标响应在未见事件/表述中保持可预测，以及这些响应如何被后续层读取。现有证据把多层通路、oracle干预、训练规则、生成接口分开，避免一次因果失败使研究归零。

**下一阶段大任务。** 以事件身份×关系词形×位置×输出协议的交叉材料为独立样本单位，先全坐标刻画条件化响应，再检验训练得到的坐标预测与实际误差，最后用自然生成判断是否被使用。必须把同词形复用、跨同义词复用、跨事件和跨语言分别列账；每个重要阳性再扩充全新事件复验，不用旧测试反复调剂量宣称外推。本轮已自动执行同目标Phase2615接口校准；进一步通用提取器仍是未解决研究目标。

**结论。** 本次有限研究合同及其接口续研均有独立产物与连续记录；程序完成不等于语言编码破解。当前最可靠进展是受控source信息的多层可读出性与强烈条件依赖边界。
'''
    if '## Phase 2616:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
