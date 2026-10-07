"""Independent graph entailment, seals, numerical anchors and material verification."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from phase2752_context_interaction import ROOT,OUT,sha,write,now


def closure(edges):
    reach=set(edges)
    while True:
        extra={(a,d) for a,b in reach for c,d in reach if b==c}
        new=reach|extra
        if new==reach:return reach
        reach=new


def entails(graph,prop):
    edges=graph['edges']
    family=graph['family']
    if family=='category':
        m=re.fullmatch(r'(\w+) is a (\w+)',prop) or re.fullmatch(r'(\w+) belongs to the (\w+) category',prop)
        assert m,prop
        subject,target=m.groups()
        facts=closure([(e['source'],e['target']) for e in edges if e['relation'] in ('member_of','subclass_of')])
        if (subject,target) in facts:return True
        assert any((subject,e['source']) in facts and e['target']==target for e in edges if e['relation']=='disjoint_from'), 'Unknown, not false'
        return False
    if family=='role':
        m=re.fullmatch(r'(\w+) was the (initial giver|final recipient)',prop)
        assert m,prop
        ordered=sorted(edges,key=lambda e:e['event_index'])
        subject,role=m.groups()
        return subject==(ordered[0]['source'] if role=='initial giver' else ordered[-1]['target'])
    facts=closure([(e['source'],e['target']) for e in edges])
    if family=='spatial':
        m=re.fullmatch(r'(\w+) is to the (left|right) of (\w+)',prop)
        assert m,prop
        a,direction,b=m.groups()
        return (a,b) in facts if direction=='left' else (b,a) in facts
    m=re.fullmatch(r'(\w+) is inside (\w+)',prop)
    if m:
        a,b=m.groups()
        return (a,b) in facts
    m=re.fullmatch(r'(\w+) contains (\w+)',prop)
    assert m,prop
    a,b=m.groups()
    return (b,a) in facts


def validate_material(directory,assertion=False):
    material=json.loads((directory/'material.json').read_text(encoding='utf-8'))
    graphs={g['world']:g for g in json.loads((OUT/'external_graphs.json').read_text(encoding='utf-8'))['graphs']}
    for r in material['rows']:
        if assertion:
            m=re.search(r'\nStatement: It is (really|not) the case that (.*?)\.\nReturn ',r['text'])
            assert m,r['id']
            word,prop=m.groups()
        else:
            question=r['text'].split('\nQuestion: ')[1].split('\n')[0]
            m=re.search(r'(really|not) (?:true that|be correct to say that|conclude that) (.*?)\?$',question)
            assert m,r['id']
            word,prop=m.groups()
        value=entails(graphs[r['world']],prop)
        expected='yes' if value != (word=='not') else 'no'
        assert expected==r['expected'],r['id']
    seal=json.loads((directory/'pre_capture_seal.json').read_text(encoding='utf-8'))
    assert seal['material_sha256']==sha(directory/'material.json')
    for source in seal['files']:
        assert sha(ROOT/source['snapshot'])==source['sha256']
    return dict(rows=len(material['rows']),graph_entailment_label_checks=len(material['rows']),
        interpretation='Independent transitive closure / event-order query interpreter on generator graph, reading actual proposition text; not an unrestricted natural-language parser or proof that polar-question pragmatics matches logic.')


def verify(final=False):
    result=dict(created_utc=now(),primary=validate_material(OUT),assertion=validate_material(OUT/'assertion_control',True),captures={})
    for label,dest in [('4B',OUT/'4B'),('14B',OUT/'14B'),('assertion4B',OUT/'assertion_control/4B')]:
        if not (dest/'capture_done.json').exists():
            assert not final, 'Capture not finished: '+label
            continue
        done=json.loads((dest/'capture_done.json').read_text(encoding='utf-8'))
        execution=json.loads((dest/'execution.json').read_text(encoding='utf-8'))
        assert sha(ROOT/execution['source']['snapshot'])==execution['source']['sha256']==done['code_sha256']
        ids=[]
        finite=True
        for path in sorted(dest.glob('chunk_*.npz')):
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            with np.load(path) as z:
                assert z['hidden'].shape[0]==len(meta)
                finite &= bool(np.isfinite(z['hidden']).all())
            ids.extend(r['id'] for r in meta)
        assert len(ids)==done['count'] and ids==execution['selected_ids'] and finite
        checks=dict(count=len(ids),all_finite=finite,execution_snapshot_matches=True,material_sha256_matches=execution['material_sha256']==sha(dest.parent/'material.json'))
        if final:
            qa=json.loads((dest/'quality_audit.json').read_text(encoding='utf-8'))
            checks['pilot_repeat_max']=max(qa['repeated_pilot_max_differences'].values())
            assert checks['pilot_repeat_max']==0
            summary=json.loads((dest/'prediction_summary.json').read_text(encoding='utf-8'))
            selection=json.loads((dest/'validation_selection.json').read_text(encoding='utf-8'))
            assert summary['alpha_choices']==selection['choices']
            with np.load(dest/'prediction_metrics.npz') as z:
                assert np.all(z['zero_interaction'][:,done['last_norm_index']]==1)
            checks['baseline_and_selection_validated']=True
        result['captures'][label]=checks
    write(OUT/('verification_final.json' if final else 'verification_preliminary.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--final',action='store_true')
    verify(p.parse_args().final)
