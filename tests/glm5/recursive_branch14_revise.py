"""Create a new protocol after discovery-only behavior diagnosis.

Keeps family IDs and filler partitions fixed. Renaming family_id would silently
reshuffle grammar.assign_splits and expose old discovery items as new holdouts.
"""
import copy
import numpy as np
import recursive_branch14 as R
from recursive_branch14_neutral import variant
from shared.dist_protocol import grammar


def prepare():
    from transformers import AutoTokenizer
    R.configure();old=R.OUT.parent/'recursive_branch14_20261113_v5'
    diagnosis=R.read(old/'neutral/report.json')
    assert R.file_sha(R.ROOT/'tests/glm5/recursive_branch14_neutral.py')==diagnosis['source_sha256']
    assert diagnosis['passed'],'Neutral reference calibration failed; do not tune further'
    chosen='neutral_reference_explicit'
    assert R.OUT!=old and not (R.OUT/'materials.json').exists()
    assert not any((old/s/'collection.json').exists() for s in R.SPLITS[1:]),'Never recycle opened holdouts'
    olddata=R.read(old/'materials.json');oldrows=olddata['rows'];audit=R.read(old/'materials-audit.json')
    tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True);rows=[]
    for r in oldrows:
        n=variant(r,tok);n['id']=r['id']+'|relation-v6';n['source_item_id']=r['id']
        n.pop('diagnostic_variant',None);n.pop('original_id',None)
        encoded=tok(n['prompt'],add_special_tokens=False,return_offsets_mapping=True)
        n['content_end']=max(i for i,(a,b) in enumerate(encoded['offset_mapping']) if b<=len(n['body'])+1 and b>a)
        start=n['prompt'].index('\n',n['prompt'].index('The file title'))+1
        n['content_start']=next(i for i,(a,b) in enumerate(encoded['offset_mapping']) if a>=start)
        assert n['tokens'][n['content_end']]==624 and n['content_end']<n['answer_pos']
        assert n['family']==r['family'] and n['split']==r['split'] and n['metadata']==r['metadata']
        rows.append(n)
    new_grammars={};generated={};grammar_audits=[]
    for name,sha in audit['grammar_sources'].items():
        spec=R.read(R.ROOT/name);assert R.digest(spec)==sha
        previous=copy.deepcopy(spec);spec['grammar_version']=6
        spec['description']='Direct description/counterfactual relation probe. Applicability reasoning is outside the admitted behavioral claim.'
        for frame in spec['frames']:
            h='structure' if '-structure-' in spec['family_id'] else 'procedure'
            if frame['levels']['role']=='quoted':
                fr=int(frame['levels']['wording'])
                opening=['An editor quotes the following instruction for proofreading only: "','A rejected proposal states: "','An example printed for discussion reads: "'][fr]
                closing=['" It has not been adopted. No other rule for the current case is provided.',
                         '" This proposal is not in force, and the current specification supplies no replacement rule.',
                         '" This example does not govern the current case; no applicable instruction has been given.'][fr]
                new_open=['A reference book describes this rule: "','A technical note records the following relationship: "','An explanatory example states: "'][fr]
                frame['text']=frame['text'].replace(opening,new_open).replace(closing,'"')
            question=('If the pairing described in the passage were used, which item would be paired with {X}?' if h=='structure' else
                      'If the ordering procedure described in the passage were carried out, what would its output be?')
            frame['probe']['prompt']=frame['text']+'\n'+question+'\nA. {C0}\nB. {C1}\nC. {C2}\nGive only the option letter.\nAnswer:'
            operative=frame['levels']['role']!='inventory'
            frame['probe']['answer']='{AOP}' if operative else '{ANON}'
            frame['probe']['contrast']=['{ANON}' if operative else '{AOP}','{AFOIL}']
        for fl in spec['fillers']:
            for k in ['C0','C1','C2']:
                fl['slots'][k]=fl['slots'][k].replace('Not specified by an applicable rule','No pairing is specified' if '-structure-' in spec['family_id'] else 'No output procedure is specified')
            if 'explicit' in chosen and '-procedure-' in spec['family_id'] and fl['metadata']['operation']=='rotate':
                fl['slots']['STEP0']='The output consists of the second entry, then the third entry, then the first entry.'
        assert grammar.assign_splits(spec)==grammar.assign_splits(previous),'Family leakage from reshuffling'
        assert not grammar.validate(spec)
        path=R.ROOT/'shared/dist_protocol/grammars'/(spec['family_id']+'-revision6.json')
        assert not path.exists(),'Do not overwrite an existing grammar revision'
        R.check_write(path);R.save_json(path,spec);new_grammars[str(path.relative_to(R.ROOT))]=R.digest(spec)
        corpus=grammar.generate(spec,include_canary=False);generated.update({r['id']:r for r in corpus['items']})
        grammar_audits.append(dict(family_id=spec['family_id'],grammar_version=6,grammar_sha256=grammar.spec_sha(spec),
            validation_errors=[],visibility=corpus['visibility'],warnings=corpus['_warnings'],preview=grammar.preview(spec,sample=1)))
    for r in rows:
        g=generated[r['source_item_id']]
        assert g['text']==r['body'] and g['probe']['prompt']==r['prompt']
        assert 'ABC'.index(g['probe']['answer'].strip())==r['gold']
    assert len({r['prompt'] for r in rows})==len(rows)==6144
    R.save('materials.json',dict(schema='recursive_branch14_materials.v6',rows=rows,seed=R.SEED,
        description='Original controlled language; a counterfactual question tests the described relation, not applicability status.',
        scope=olddata['scope'],prior_material_sha256=R.file_sha(old/'materials.json')))
    newaudit={k:v for k,v in audit.items() if k not in ['grammar_sources','grammar_audits','min_tokens','max_tokens','mean_tokens']}
    newaudit.update(grammar_sources=new_grammars,grammar_audits=grammar_audits,min_tokens=min(len(r['tokens']) for r in rows),
        max_tokens=max(len(r['tokens']) for r in rows),mean_tokens=float(np.mean([len(r['tokens']) for r in rows])),
        all_family_partitions_unchanged=True,selected_discovery_diagnostic_variant=chosen,
        behavior_scope='The output of the described relation under hypothetical execution; quoted and operative descriptions both have a specified output.')
    newaudit['limitations']=audit['limitations']+['After discovery-only applicability failures, the probe question changed meaning; new success does not rescue the original applicability claim.']
    R.save('materials-audit.json',newaudit)
    R.save('material-revision.json',dict(kind='prospective_protocol_revision',previous_run=R.read(old/'plan.json')['run_id'],
        prior_material_sha256=R.file_sha(old/'materials.json'),prior_pilot_sha256=R.file_sha(old/'pilot.json'),diagnostic_sha256=R.file_sha(old/'diagnosis/report.json'),neutral_calibration_sha256=R.file_sha(old/'neutral/report.json'),
        selected_variant=chosen,selection_rule='Original applicability and four diagnostic forms failed. Final discovery calibration removes applicability reasoning and tests neutral described relations; stop if any cell fails. This is a narrower task, not a repair of the old behavioral claim.',
        original_applicability_failure_retained=True,all_family_partitions_unchanged=True,heldout_responses_exposed=False,
        formal_conclusion_scope_changed=True,source_sha256=R.file_sha(__file__)))
    print('Revised materials',len(rows),'max tokens',newaudit['max_tokens'],'variant',chosen,flush=True)


if __name__=='__main__':prepare()
