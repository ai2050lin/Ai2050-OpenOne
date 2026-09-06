"""Separate exact answer, formatting, leading entity, truncation, and native readout kind."""
import json,re
from collections import Counter
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2649_output_function_behavior import OUT,MATERIAL

def first_entity(text,a,b,lang):
    clean=text.lstrip(' \n\t*_`\"\'');found=[]
    for i,name in enumerate((a,b)):
        pattern=re.escape(name)+(r'(?=$|[^A-Za-z])' if lang=='en' else '')
        if re.match(pattern,clean,re.I):found.append(i)
    return found[0] if len(found)==1 else None

def main():
    rows=[json.loads(s) for s in (OUT/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines()];assert len(rows)==8192
    material={r['case_index']:r for r in read(MATERIAL/'material/cases.json')};tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    assert first_entity(' **Adam**.','Adam','Brian','en')==0 and first_entity('Isabell.','Isabel','Kevin','en') is None
    results=[]
    for r in rows:
        m=material[r['case_index']];text=r['generated'];plain=text.strip().strip(' .。').replace('**','').replace('__','').strip()
        first=first_entity(text,m['entity_a'],m['entity_b'],m['language']) if r['output_function']!='truth' else None
        vocab=m['common_readout_words'];sets=[]
        for word in vocab:
            variants={word,' '+word}
            if r['output_function']=='truth' and m['language']=='en':variants|={word.lower(),word.upper(),' '+word.lower(),' '+word.upper()}
            sets.append({tok.encode(w,add_special_tokens=False)[0] for w in variants})
        classes=[]
        for token in r['native_ids']:
            found=[i for i,s in enumerate(sets) if token in s];classes.append(found[0] if len(found)==1 else None)
        if all(x is not None for x in classes):kind='different_answer_firsttokens' if classes[0]!=classes[1] else 'same_answer_surface_variants'
        elif any(x is not None for x in classes):kind='answer_vs_other_token'
        else:kind='neither_answer_firsttoken'
        results.append({'case_index':r['case_index'],'case_id':r['case_id'],'mode':r['mode'],'language':r['language'],'family':r['family'],
            'primary_strict_correct':r['strict_correct'],'primary_content_correct':r['content_correct'],'eos':r['eos'],'hit_generation_limit':not r['eos'],
            'whole_name_after_markdown_only_correct':plain.casefold()==m['target'].casefold() if r['output_function']!='truth' else None,
            'first_named_entity_index':first,'leading_named_entity_correct':first==m['target_index'] if first is not None else False,
            'native_pair_kind':kind,'native_pair_firsttoken_classes':classes,'native_ids':r['native_ids'],'canonical_common_ids':r['common_ids'],
            'native_first_in_canonical_common':r['native_ids'][0] in r['common_ids']})
    groups={}
    for group in sorted({r['mode']+'/'+r['language'] for r in results}):
        rr=[r for r in results if r['mode']+'/'+r['language']==group]
        groups[group]={'n':len(rr),'primary_content_correct':sum(r['primary_content_correct'] for r in rr),'hit_generation_limit':sum(r['hit_generation_limit'] for r in rr),
            'leading_named_entity_correct':sum(r['leading_named_entity_correct'] for r in rr) if not group.startswith('truth') else None,
            'whole_name_markdown_normalized_correct':sum(bool(r['whole_name_after_markdown_only_correct']) for r in rr) if not group.startswith('truth') else None,
            'native_pair_kinds':dict(Counter(r['native_pair_kind'] for r in rr)),'native_first_in_canonical_common':sum(r['native_first_in_canonical_common'] for r in rr)}
    save(OUT/'analysis/output_interface_rows.json',results);report={'groups':groups,
        'boundary':'Post-observation diagnostic only. Primary fixed scoring unchanged. First named entity does not certify later generated facts or complete-name-only compliance; missingEOS is a16token truncation limit, not failed linguistic computation. Native pair may distinguish spaces/formatting for same answer. Canonical standalone-name common readout in cloze is an artificial fixed-row control, NOT total probability of a naturally emitted answer with leading spaces/markdown/multiple tokens.'}
    save(OUT/'analysis/output_interface_audit.json',report);print(json.dumps(report,ensure_ascii=True),flush=True)
    marker='**Phase2649 输出接口审计补充**'
    if marker not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+' ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+']。本次审计不改写原定完整答案评分。中文cloze原定正确42/1024，先写对人名929/1024，其中655条在16token上限截断；英文cloze原定909/1024、先写对957/1024。加粗、解释和截断不能直接叫作语言能力崩解；先写对人名也不能认证其后整句话。英文cloze1024条的原生首token无一属于无前导空格的固定名字读出集合；因此该共同读出仅是跨功能固定头行的人工对照，不是自然答案总概率。原生top1/top2还可能是同一答案的大小写/空格/格式变体，完整分类见analysis/output_interface_rows.json。此补充在初始FP32采集中追加，不改变既定材料、输出对或坐标规则。\n')

if __name__=='__main__':main()
