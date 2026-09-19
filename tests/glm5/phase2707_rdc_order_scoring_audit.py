"""Pilot-only correction: byte-piece field onset and parsed trace versus requested formatting."""
import re
from rdc_conditional_common import *
from rdc_order_material import score,norm
OUT=CAMPAIGN/'m_order'


def trace_content(row,text):
    s=norm(text);expected=norm(row['expected_fields']['Trace']);f=row['family']
    if f=='quantity_update':
        pairs=re.findall(r'\[([0-9]+),([0-9]+)\]',s);epairs=re.findall(r'\[([0-9]+),([0-9]+)\]',expected)
        remaining=re.sub(r'\[[0-9]+,[0-9]+\]','',s)
        return [tuple(map(int,p)) for p in pairs]==[tuple(map(int,p)) for p in epairs] and bool(re.fullmatch(r'[;=>→\-]*',remaining))
    if f=='handover':
        return re.split(r'->|→|=>',s)==expected.split('->')
    # Positive/negative edge type is substantive, not just a separator format.
    def edges(value):
        ids=list(re.finditer(r'c[0-9]+[a-z]',value));links=[]
        for a,b in zip(ids,ids[1:]):
            connector=value[a.end():b.start()];links.append('negative' if connector=='-/>' else 'positive' if connector in ('->','→','=>') else 'unknown')
        remainder=value[:ids[0].start()]+value[ids[-1].end():] if ids else value
        return [m[0] for m in ids],links,remainder
    return edges(s)==edges(expected)


def score_v2(row,text,eos,truncated):
    old=score(row,text,eos,truncated);parsed=old['parsed_fields'];new=dict(old,score_version=2,result_exact=old['result_correct'],trace_exact=old['trace_correct'])
    trace=parsed.get('Trace',[])
    new['trace_correct']=len(trace)==1 and trace_content(row,trace[0])
    new['trace_requested_format']=old['trace_correct']
    if row['family']=='quantity_update' and len(parsed.get('Result',[]))==1:
        def quantities(value):
            s=norm(value);found=re.findall(r'([^=;,]+)=([0-9]+)',s)
            return [(name,int(n)) for name,n in found]
        expected=quantities(row['expected_fields']['Result']);actual=quantities(parsed['Result'][0])
        new['result_correct']=actual==expected
    new['all_content_correct']=bool(new['result_correct'] and new['trace_correct'] and new['neutral_correct'])
    new['limits']='Parsed named/numeric fields, path order, and category-edge polarity. Outer field labels still strict; malformed LABEL/CONTENT or 标签 prefixes are format failures, not silently repaired. Correct numeric trace with alternate separators can have true content but false requested-format.'
    return new


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);special=set(tok.all_special_ids)
    rows=read(OUT/'prefixes.json')
    immutable(OUT/'scoring_alignment_protocol_v2.json',{'source_sha':sha(Path(__file__)),
      'discovery':'Only first36training-entity0 pilot outputs examined before formal expansion.17old missing boundaries mix byte-piece detection misses and real schema failures. Native rawfields andgenerationcommits remain unchanged.',
      'field_onset':'First captured Result-label boundary candidate whose next token is nonspecial and decoded piece is not whitespace-only, INCLUDING replacement characters from incomplete UTF8 byte pieces. This is onset of the Result field, not necessarily a completeword or first decision-bearing number.',
      'preservation':'Existing capture already persists Q/K/V/P andfullH at every Result-label candidate. Correct alignment using those storedfields without recapture or choosing by expectedanswer. Version1 lexicalflag andbehavior retained; correctedbehavior andmaterial savedseparately.',
      'scoring':'Parse trace quantities/path and categoryedgepolarity separately from requested separator format. Preserve result_exact/trace_exact andouterlabels/order/format. Same reference andno new facts. Missing malformedResultlabels remainmissing andall suchoutputs stayinbehavior denominators.',
      'limits':'Boundary analyses condition on format-valid field onset. Report coverage byorder/family; numeric task onset is before firstname, not firstnumericdisambiguation. Outer-schema failures are not proof of lost semantic knowledge.'})
    for r in rows:
        s=score_v2(r,r['reference'],True,False);assert all(s[k] for k in ('result_correct','trace_correct','all_content_correct','field_order_correct','format_structure'))
        if r['family']=='quantity_update':
            alt=r['reference'].replace(r['expected_fields']['Trace'],r['expected_fields']['Trace'].replace(' -> ','; '));s=score_v2(r,alt,True,False)
            assert s['trace_correct'] and not s['trace_requested_format']
            bad=r['reference'].replace(r['expected_fields']['Trace'],'[999,999] -> [999,999] -> [999,999]');assert not score_v2(r,bad,True,False)['trace_correct']
        if r['family']=='category_chain' and not r['truth']:
            bad=r['reference'].replace('-/>','->');assert not score_v2(r,bad,True,False)['trace_correct']
    changes=[];selected=set();count=0;missing=[]
    for r in rows:
        bp=OUT/f'behavior/{r["sample_id"]}.json'
        if not bp.exists():continue
        b=read(bp);chosen=None
        for sid in b['boundary_candidates']:
            step=read(OUT/f'steps/{sid}.json')
            if step['next_token_id'] not in special and step['next_token'].strip():chosen=sid;break
        if chosen:selected.add(chosen)
        else:missing.append(r['sample_id'])
        s=score_v2(r,b['generated'],b['scores']['eos'],b['scores']['truncated'])
        changed={k:{'v1':b['scores'][k],'v2':s[k]} for k in ('result_correct','trace_correct','all_content_correct') if b['scores'][k]!=s[k]}
        if chosen!=b['result_boundary_state'] or changed:changes.append({'sample_id':r['sample_id'],'old_boundary':b['result_boundary_state'],'new_boundary':chosen,'score_changes':changed})
        save(OUT/f'behavior_scored/{r["sample_id"]}.json',dict(b,scores=s,result_boundary_state=chosen,original_behavior_sha=sha(bp),scoring_protocol_sha=sha(OUT/'scoring_alignment_protocol_v2.json')));count+=1
    material=[dict(r,result_field_onset=r['sample_id'] in selected,alignment_version=2) for r in read(OUT/'material.json')]
    save(OUT/'material_scored.json',material)
    save(OUT/'scoring_alignment_audit.json',{'timestamp':stamp(),'references_checked':288,'passed':True,'rescored_prefixes':count,'valid_result_field_onsets':len(selected),'missing':missing,'changes':changes,
      'raw_preserved':True,'source_sha':sha(Path(__file__))})
    print('ORDER_SCORING_ALIGNMENT_V2',count,len(selected),len(missing),len(changes),flush=True)


if __name__=='__main__':main()
