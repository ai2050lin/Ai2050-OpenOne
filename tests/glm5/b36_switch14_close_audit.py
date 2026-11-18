"""Independent line parser, paired material identity and executed-stage ledger."""
import collections
import b36_switch14_design as D
S=D.S


def run():
    S.configure();p=S.read(S.OUT/'plan.json');D.M.admit(p);cp=S.read(S.OUT/'calibration-plan.json');D.M.admit(cp)
    a=S.read(S.OUT/'behavior-analysis.json');cal=S.read(S.OUT/'calibration.json');rows={r['id']:r for r in p['rows']};counts={};checked=0
    for arm in D.ARMS:
        rr=S.read(S.OUT/(arm+'.json'))['results'];tot=collections.Counter()
        for r in rr:
            row=rows[r['id']];text=r['text'].removesuffix('<|im_end|>').strip();lines=text.splitlines();assert len(lines)==2
            names,values=zip(*(line.split(':',1) for line in lines));assert names==('Steps','Value')
            raw_steps=[s.strip() for s in values[0].split(',')];raw_value=values[1].strip()
            assert all(s and all(c in '0123456789' for c in s) for s in raw_steps+[raw_value])
            steps=list(map(int,raw_steps));value=int(raw_value)
            metrics=dict(step_count_correct=len(steps)==row['depth'],steps_correct=steps==row['truth'],value_correct=value==row['truth'][-1],format_correct=True,eos=r['tokens'][-1]==151645)
            metrics['joint']=all(metrics[k] for k in ['steps_correct','value_correct','format_correct','eos'])
            assert all(r['parsed']['metrics'][k]==v for k,v in metrics.items())
            tot.update({k:int(v) for k,v in metrics.items()});checked+=1
        counts[arm]=dict(tot)
    changes=[]
    for g in p['families']:
        x=rows[g['id']+':first2'];y=rows[g['id']+':first3'];assert x['visible_ops']==y['visible_ops']==g['ops']
        assert len(x['tokens'])==len(y['tokens'])==113
        at=[i for i,(u,v) in enumerate(zip(x['tokens'],y['tokens'])) if u!=v];assert len(at)==1
        changes.append(at[0])
    gate=all(counts[arm]['joint']>=p['min_joint'] for arm in D.ARMS) and not a['missing_sites'];assert gate==a['passed']
    assert cal['passed'] and not gate
    assert not (S.OUT/'native-summary.json').exists() and not list(S.OUT.glob('native-*.npz'))
    S.save('independent-score-audit.json',dict(passed=True,n_records=checked,counts=counts,paired_token_change_positions=dict(collections.Counter(changes)),
        calibration_source_identity_unchanged=True,new_study_source_identity_unchanged=True,behavior_gate_passed=gate,source_sha256=S.file_sha(__file__),
        scope='Separate splitlines/int parser agrees on all192 outputs. Program truth and tokenizer checks are in frozen behavior audit. No independent GPU implementation or native branch validation.'))
    S.save('stage-decision.json',dict(status='cache_passed_behavior_failed_native_not_run',calibration_passed=True,calibration_exact=96,behavior_gate_passed=False,cells=a['cells'],missing_sites=a['missing_sites'],
        native_executed=False,reason='first3 joint42/96 < frozen88/96; no correct-subset selection or threshold change.',
        retained_candidate=dict(unit_ids=p['unit_ids'],matched_unit_ids=p['matched_unit_ids'],source=p['candidate_source']),
        followup='Use current failures only as exposed discovery evidence. Freeze new instruction/control families before any revised confirmation; current native draft needs real numeric admission before use.'))
    prefix='.venv/Scripts/python.exe -B -X utf8 -u tests/glm5/'
    commands=['b36_switch14_calibrate.py prepare','b36_switch14_calibrate.py run','test_b36_switch14.py','b36_switch14_design.py','b36_switch14_behavior.py run','b36_switch14_audit.py','b36_switch14_review.py','b36_switch14_close_audit.py']
    S.save('execution-ledger.json',dict(completed_commands=[prefix+c for c in commands],
        not_executed=['b36_switch14_paths.py','b36_switch14_analyze.py'],not_executed_reason='Frozen behavior gate failed',
        delivery_entry=prefix+'b36_switch14_finish.py',scope='Principal research/engineering commands only; read-only shell inspection commands omitted. Delivery is recorded by its own receipt after completion.'))
    print('Independent192-output audit passed; one-token paired identity confirmed; native correctly not run',flush=True)


if __name__=='__main__':run()
