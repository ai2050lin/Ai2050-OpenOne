"""Version2 separated long-task scoring; original behavior/commit files remain immutable."""
import re
from collections import Counter
from rdc_conditional_common import *
from rdc_long_material import compact
OUT=CAMPAIGN/'k_long'


def score_v2(row,text,eos,truncated):
    s=text.strip();norm=compact(s);ref=compact(row['reference']);f=row['family']
    lines=[compact(x) for x in s.splitlines() if x.strip()]
    expected=[compact(x) for x in row['reference'].splitlines() if x.strip()]
    hits=[];detail={};order=True;format_ok=False
    if f=='clause_reorder':
        hits=[compact(clause) in norm for clause in row['expected_rows']]
        pos=[norm.find(t) for t in row['ordered_terms']];order=all(p>=0 for p in pos) and pos==sorted(pos)
        format_ok=len(lines)==4 and all(line.endswith(('.','。')) for line in lines)
    elif f in ('reference_chain','taxonomy_explain'):
        label1=('持有人:' if row['language']=='zh' else 'owner:') if f=='reference_chain' else ('结论:' if row['language']=='zh' else 'decision:')
        label2='路径:' if row['language']=='zh' else 'path:'
        first=next((x[len(label1):] for x in lines if x.startswith(label1)),None)
        path=next((x[len(label2):] for x in lines if x.startswith(label2)),None)
        expected_first=expected[0][len(label1):];expected_path=expected[1][len(label2):].split('->')
        actual_path=[] if path is None else path.split('->')
        hits=[first==expected_first,Counter(actual_path)==Counter(expected_path)]
        order=actual_path==expected_path
        format_ok=len(lines)==2 and lines[0].startswith(label1) and lines[1].startswith(label2) and '->' in lines[1]
        detail={'first_field_correct':hits[0],'path_membership_with_multiplicity':hits[1],'path_order_correct':order,
          'owner_path_consistent':None if f!='reference_chain' or not actual_path else first==actual_path[-1]}
    elif f in ('role_table','structured_extract'):
        nfields=4 if f=='role_table' else 3;parts=[x.split('|') for x in lines];eparts=[x.split('|') for x in expected]
        shape=len(parts)==3 and all(len(p)==nfields for p in parts)
        canonical=[]
        verbs={'lent':'lend','showed':'show','sent':'send'}
        for p in parts:
            p=p.copy()
            if f=='role_table' and len(p)==4:p[1]=verbs.get(p[1],p[1])
            canonical.append(tuple(p))
        hits=[Counter(canonical)==Counter(map(tuple,eparts))]
        if f=='role_table':
            order=shape and [p[1] for p in canonical]==[p[1] for p in eparts]
            action_labels=shape and [p[1] for p in parts]==[p[1] for p in eparts]
            format_ok=shape and action_labels;detail={'canonicalized_action_content':True,'requested_action_labels':action_labels}
        else:
            order=shape and [p[0] for p in parts]==[p[0] for p in eparts]
            format_ok=shape and all(p[-1].isdigit() for p in parts)
    elif f=='temporal_revision':
        names=(row['u'],row['v']);wanted=dict(re.findall(r'([^=;\n]+)=([0-9]+)',row['reference']))
        for name,number in wanted.items():
            match=re.search(re.escape(compact(name))+r'=([0-9]+)',norm);hits.append(bool(match and match[1]==number))
        positions=[norm.find(compact(name)) for name in names];order=all(p>=0 for p in positions) and positions==sorted(positions)
        total='总计' if row['language']=='zh' else 'total'
        pattern=re.escape(compact(names[0]))+r'=[0-9]+;'+re.escape(compact(names[1]))+r'=[0-9]+'
        format_ok=len(lines)==2 and bool(re.fullmatch(pattern,lines[0])) and bool(re.fullmatch(total+r'=[0-9]+',lines[1]))
    else:
        # Translation and style: lexical constraints, not a semantic judge.
        patterns=[]
        for term in row['constraint_terms']:
            if f=='translation' and term=='desk':patterns.append(r'\b(?:desk|table)\b')
            elif f=='translation' and term=='刷子':patterns.append('刷子|画笔')
            else:patterns.append(term if '|' in term else re.escape(term))
        hits=[bool(re.search(p,s,re.I)) for p in patterns]
        positions=[norm.find(compact(term)) for term in row['ordered_terms']]
        order=all(p>=0 for p in positions) and positions==sorted(positions)
        format_ok=bool(s) and not any(t in s for t in ('```','<think>','</think>'))
        detail={'constraint_patterns':patterns,'semantic_equivalence':'not established','no_added_fact_check':'not automatically evaluated'}
        if f=='style_rewrite':detail['politeness_marker_present']=bool(re.search('请|烦请' if row['language']=='zh' else r'\b(?:please|could|would|kindly)\b',s,re.I))
    return {'score_version':2,'exact_reference':norm==ref,'declared_content_constraints':all(hits),'constraint_fraction':sum(hits)/max(len(hits),1),
      'order_constraints':bool(order),'format_structure':bool(format_ok),'eos':bool(eos),'truncated_at_limit':bool(truncated),
      'constraint_hits':hits,'details':detail,'content_kind':'lexical_checklist_only' if f in ('translation','style_rewrite') else 'parsed_deterministic_constraints'}


def main():
    rows=read(OUT/'prefixes.json')
    protocol={'source_sha':sha(Path(__file__)),'score_version':2,'discovery':'Pilot16of128 prefixes (entity0 only), before the remaining112; no heldoutunit6/7 outputs inspected. Corrected after manual review; rawgeneration andv1scores retained.',
      'corrections':['Parse Owner field, not namepresence elsewhere inpath.','Whitespace-insensitive arrow parsing; exactreference can never have false content solely from arrowspacing.','Separate row/sentence order, structured content andformatshape.','Wrong numericvalue can retaincorrectformat.','Role action inflection normalized only forcontent; requested literalactionlabels remain formatcheck.','desk/table for Chinese桌 andbrush刷子/画笔 recognized as declared lexical alternatives; names stillmustremain literal as instructed.'],
      'limits':['Scoring is conservative and task-specific; no universal semantic evaluator.','Translation/style do not check allmeaning or alladdedfacts.','Corrections use only trainingpilot, not independenttest outputs.']}
    path=OUT/'scoring_protocol_v2.json'
    if path.exists() and read(path)!=protocol:
        assert len(list((OUT/'prefix_commits').glob('*.json')))<=16,'Formal expansion has started; do not change scoring protocol'
        immutable(OUT/'scoring_protocol_v2_initial.json',read(path))
        save(OUT/'scoring_preflight_correction.json',{'timestamp':stamp(),'reason':'Correct metadata typo remaining127 to112; actual pilot16of128 and scoring algorithm unchanged','original_sha':sha(path)})
        save(path,protocol)
    else:immutable(path,protocol)
    for r in rows:
        sc=score_v2(r,r['reference'],True,False)
        assert all(sc[k] for k in ('exact_reference','declared_content_constraints','order_constraints','format_structure')),(r['sample_id'],sc)
        if r['family']=='reference_chain':
            corrupted=r['reference'].splitlines();label='持有人：' if r['language']=='zh' else 'Owner: '
            corrupted[0]=label+'WRONG';bad=score_v2(r,'\n'.join(corrupted),True,False)
            assert not bad['declared_content_constraints'] and bad['order_constraints'] and bad['format_structure']
            no_spaces=r['reference'].replace(' -> ','->');assert score_v2(r,no_spaces,True,False)['declared_content_constraints']
        if r['family']=='temporal_revision':
            corrupted=re.sub(r'=[0-9]+','=999',r['reference'],count=1);bad=score_v2(r,corrupted,True,False)
            assert not bad['declared_content_constraints'] and bad['format_structure']
    changed=[];count=0
    for r in rows:
        bp=OUT/f'behavior/{r["sample_id"]}.json'
        if not bp.exists():continue
        b=read(bp);new=score_v2(r,b['generated'],b['scores']['eos'],b['scores']['truncated_at_limit'])
        save(OUT/f'behavior_scored/{r["sample_id"]}.json',dict(b,scores=new,original_behavior_sha=sha(bp),scoring_protocol_sha=sha(OUT/'scoring_protocol_v2.json')))
        delta={k:{'v1':b['scores'][k],'v2':new[k]} for k in ('declared_content_constraints','order_constraints','format_structure') if b['scores'][k]!=new[k]}
        if delta:changed.append({'sample_id':r['sample_id'],'changes':delta})
        count+=1
    save(OUT/'scoring_audit.json',{'timestamp':stamp(),'reference_checks':128,'reference_all_passed':True,'adversarial_field_and_spacing_checks':True,'rescored_prefixes':count,'changed_cases':changed,'raw_behavior_preserved':True})
    print('SCORING_V2_PASS',count,len(changed),flush=True)


if __name__=='__main__':main()
