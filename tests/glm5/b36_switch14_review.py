"""Descriptive output error audit; never changes frozen admission or primary tests."""
import collections
import b36_switch14_design as D
S=D.S


def run():
    S.configure();p=S.read(S.OUT/'plan.json');D.M.admit(p);rows={r['id']:r for r in p['rows']};out=[];examples=[]
    for arm in D.ARMS:
        records=S.read(S.OUT/(arm+'.json'))['results'];counter=collections.Counter()
        for r in records:
            row=rows[r['id']];q=r['parsed'];truth=row['truth'];visible=row['visible_truth'];counter[q['step_pattern']]+=1
            flags=dict(correct_steps_wrong_value=q['metrics']['steps_correct'] and not q['metrics']['value_correct'],
                value_matches_last_printed=q['metrics']['value_matches_last_printed_step'] is True,
                extra_steps_with_correct_value=bool(q['steps'] and len(q['steps'])>row['depth'] and q['metrics']['value_correct']),
                missing_updates_but_printed_prefix_correct=bool(q['steps'] and len(q['steps'])<row['depth'] and q['steps']==truth[:len(q['steps'])]),
                missing_updates_but_value_correct=bool(q['steps'] and len(q['steps'])<row['depth'] and q['metrics']['value_correct']))
            if arm=='first2':
                flags.update(printed_all_three_visible_values=q['steps']==visible,value_equals_ignored_third_result=q['value']==visible[-1],
                    correct_two_step_prefix_but_extra=bool(q['steps'] and q['steps'][:2]==truth and len(q['steps'])>2))
            for k,v in flags.items():counter[k]+=int(v)
            if not q['metrics']['joint']:examples.append(dict(id=r['id'],text=r['text'],truth=truth,visible_truth=visible,flags=flags))
        out.append(dict(arm=arm,n=96,counts=dict(counter)))
    metric_limits=None;displacements=None
    if (S.OUT/'native-analysis.json').exists():
        a=S.read(S.OUT/'native-analysis.json');observed={r['id']:r for arm in D.ARMS for r in S.read(S.OUT/(arm+'.json'))['results']}
        metric_limits={arm:sum(r['conditions']['baseline']['correct'] and not observed[r['id']]['parsed']['metrics']['value_correct'] for r in a['events'] if r['arm']==arm and r['site']=='report') for arm in D.ARMS}
        displacements={}
        for condition in ['remove','matched','double']:
            norms=[r['conditions'][condition]['actual_after36_displacement_norm'] for r in a['events']]
            displacements[condition]=dict(n=len(norms),nonzero=sum(v>0 for v in norms),min=min(norms),max=max(norms),mean=sum(norms)/len(norms))
    left=S.read(S.OUT/'first2.json')['results'];right=S.read(S.OUT/'first3.json')['results']
    identical=sum(a['tokens']==b['tokens'] for a,b in zip(left,right))
    S.save('output-review.json',dict(cells=out,failures=examples,identical_full_token_sequences_across_instructions=identical,first_digit_correct_despite_wrong_full_value=metric_limits,actual_residual_displacements=displacements,source_sha256=S.file_sha(__file__),
        scope='Descriptive visible-output audit only, no additional confirmatory tests. Equal-to-ignored-third value does not prove the model internally executed that update. Flags can overlap; ignored-third and repeated-last can coincide. No success filtering, no changed gates.'))
    print(out,flush=True)


if __name__=='__main__':run()
