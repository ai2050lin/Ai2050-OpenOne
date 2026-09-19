"""Append-only-derived conservative scoring; original trajectory commits untouched."""
import argparse
from collections import defaultdict
from rdc_update_common import *
from rdc_update_scoring import score,checks
from rdc_update_terminal_audit import enrich_score,checks as terminal_checks


def summary_tables(records):
    groups=defaultdict(list)
    for r in records:
        for granularity,cohort in (('kind',r['kind']),('cohort_style',r['cohort']+'/'+str(r['answer_style']))):
            groups[r['mode'],r['branch'],granularity,cohort].append(r)
    summaries=[]
    for (mode,branch,granularity,cohort),rr in sorted(groups.items()):
        scored=[r for r in rr if r['kind']!='natural'];stoppedparsed=[r for r in scored if r['answer_scoring']['EOS'] and r['answer_scoring']['conservative_final_answer'] is not None]
        summaries.append({'mode':mode,'branch':branch,'granularity':granularity,'cohort':cohort,'rows':len(rr),
          'source_groups':len({r['source_group'] for r in rr}),'tokens_mean':float(np.mean([r['length'] for r in rr])),
          'EOS':sum(r['answer_scoring']['EOS'] for r in rr),'censored':sum(r['answer_scoring']['censored'] for r in rr),
          'scorable_rows':len(scored),'terminal_parsed':sum(r['answer_scoring']['conservative_final_answer'] is not None for r in scored),
          'parsed_and_stopped':len(stoppedparsed),'correct_and_stopped':sum(r['answer_scoring']['parsed_and_stopped_correct'] for r in scored),
          'wrong_parsed_and_stopped':sum(not r['answer_scoring']['conservative_final_correct'] for r in stoppedparsed),
          'success_source_cluster':clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in scored],[r['source_group'] for r in scored]) if scored else None,
          'scope':'Correct+stopped/all is an operational success yield, not accuracy after assuming unparsed/censored outputs wrong. Natural continuations have no unique free-generation gold.'})
    paired=[];by={(r['mode'],r['branch'],r['sample_id']):r for r in records}
    for mode in ('own_history','long_answers'):
      for branch in sorted({r['branch'] for r in records if r['mode']==mode and r['branch']!='native'}):
        for kind in ('controlled_program','controlled_language'):
            rr=[r for r in records if r['mode']==mode and r['branch']==branch and r['kind']==kind]
            if not rr:continue
            delta=[];improved=[];worsened=[]
            for r in rr:
                n=by[mode,'native',r['sample_id']];a=r['answer_scoring']['parsed_and_stopped_correct'];b=n['answer_scoring']['parsed_and_stopped_correct']
                delta.append(int(a)-int(b))
                if a and not b:improved.append(r['sample_id'])
                if b and not a:worsened.append(r['sample_id'])
            paired.append({'mode':mode,'branch':branch,'kind':kind,'rows':len(rr),'operational_yield_delta':clustered(delta,[r['source_group'] for r in rr]),
              'new_correct_stopped_ids':improved,'lost_correct_stopped_ids':worsened,'scope':'Operational parse+stop change, not a judgment that every prior unparsed case was semantically wrong.'})
    return summaries,paired


def main(final=False):
    start=time.monotonic();out=BASE/'behavior_analysis';tests=checks();format_tests=terminal_checks()
    protocol=read(BASE/'terminal_format_audit/protocol.json')
    assert protocol['auditor_sha256']==sha(ROOT/'tests/glm5/rdc_update_terminal_audit.py')
    materials={r['sample_id']:r for name in ('natural_material.json.gz','program_material.json.gz','language_material.json.gz') for r in gzread(BASE/name)}
    records=[];files={};completion={};changes=[];format_changes=[]
    for mode in ('own_history','same_history','long_answers','qwen4','qwen14','glm4'):
        folder=BASE/('scale/'+mode if mode in ('qwen4','qwen14','glm4') else mode)
        completion[mode]=(folder/'result.json').exists()
        if not completion[mode]:continue
        paths=(folder/'commits').glob('*.json' if mode in ('qwen4','qwen14','glm4') else '*/*.json')
        for path in sorted(paths):
            r=read(path)
            if 'generated_ids' not in r:continue
            row=materials[r['sample_id']];ids=r['generated_ids'];cap=1024 if mode=='long_answers' else 128
            stopped=bool(r.get('stopped_by_EOS',r.get('EOS',r.get('answer_scoring',{}).get('EOS',False))))
            s=score(row,r['generated_text'],ids,{ids[-1]} if stopped and ids else set(),cap)
            if 'answer_scoring' in r:assert s==r['answer_scoring'],(path,s,r['answer_scoring'])
            secondary=enrich_score(row,r['generated_text'],s)
            rec={k:row.get(k) for k in ('sample_id','source_group','cohort','kind','split','family','language','answer_style','representation','target')}
            rec.update(mode=mode,branch=r.get('branch','native'),length=len(ids),answer_scoring=s,format_aware_scoring=secondary,
              raw_record=path.relative_to(BASE).as_posix(),first=r.get('first',{}),first_divergence=r.get('first_divergence_step'))
            records.append(rec);files[path.relative_to(BASE).as_posix()]=sha(path)
            if secondary['conservative_final_answer']!=s['conservative_final_answer']:
                format_changes.append({'mode':mode,'sample_id':row['sample_id'],'branch':rec['branch'],
                  'primary':s,'format_aware':secondary,'raw_record':rec['raw_record']})
            old=r.get('parsed_answer_correct',r.get('parsed_accuracy'))
            if mode=='own_history' and row['kind']!='natural' and old!=s['conservative_final_correct']:
                changes.append({'sample_id':row['sample_id'],'branch':rec['branch'],'old_leading_parser':old,'terminal_parser':s['conservative_final_correct'],'stopped':stopped})
    if final:assert all(completion.values()),completion
    summaries,paired=summary_tables(records)
    format_summaries,format_paired=summary_tables([r|{'answer_scoring':r['format_aware_scoring']} for r in records])
    prefix='final' if final else 'preliminary'
    record_name=f'{prefix}_format_{protocol["auditor_sha256"][:12]}_records_{len(records)}.json.gz'
    compressed(out/record_name,records)
    result={'timestamp':stamp(),'source':snapshot(__file__),'scorer':snapshot(ROOT/'tests/glm5/rdc_update_scoring.py'),'parser_unit_tests':tests,
      'complete_collections':completion,'final':final,'records_file':record_name,'trajectories':len(records),'summaries':summaries,'paired':paired,
      'format_aware_summaries':format_summaries,'format_aware_paired':format_paired,'format_audit_changes':format_changes,
      'format_audit_regression':format_tests,'format_audit_protocol_sha256':sha(BASE/'terminal_format_audit/protocol.json'),
      'legacy_changes':changes,'original_commit_sha256':files,'seconds':time.monotonic()-start,
      'correction':'Original own_history leading Yes/No parser did not enforce explain-style final Answer marker. It remains diagnostic only. New scoring does not alter prompts, labels, generated IDs or original commits.',
      'scope':'Finite conservative terminal grammar, EOS and cap distinguished; reasoning correctness ungraded. Repeated diagnostics are not extra independent data.'}
    path=out/('result.json' if final else 'preliminary.json')
    if path.exists():save(out/'summary_history'/f'{sha(path)}.json',read(path))
    save(path,result);ledger('formal_behavior_scoring_final' if final else 'formal_behavior_scoring_preliminary',result['seconds'])
    print('BEHAVIOR_ANALYSIS',len(records),'legacy_changed',len(changes),'final',final,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
