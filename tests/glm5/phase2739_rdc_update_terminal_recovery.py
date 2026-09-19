"""Freeze and audit a secondary terminal-format repair without changing trajectories."""
import argparse
from rdc_update_common import *
from rdc_update_terminal_audit import enrich_score,checks


def main(stage,amend=False):
    out=BASE/'terminal_format_audit';start=time.monotonic();test=checks()
    if stage=='freeze':
        previous=None
        if (out/'protocol.json').exists():
            if not amend:return
            previous=sha(out/'protocol.json');save(out/'protocol_versions'/f'{previous}.json',read(out/'protocol.json'))
        observed={p.name:len(list(p.glob('*.json'))) for p in (BASE/'long_answers/commits').iterdir() if p.is_dir()}
        protocol={'timestamp':stamp(),'source':snapshot(__file__),
          'auditor':snapshot(ROOT/'tests/glm5/rdc_update_terminal_audit.py'),
          'auditor_sha256':sha(ROOT/'tests/glm5/rdc_update_terminal_audit.py'),
          'primary_scorer_sha256':sha(ROOT/'tests/glm5/rdc_update_scoring.py'),'unit_checks':test,
          'existing_long_commits_at_freeze':observed,
          'trigger':'Read first completed native branch: explicit Final Answer in boxed math/code fence and a final requested-variable assignment were correctly preserved as unparsed by the narrower primary grammar. Secondary grammar is an announced post-outcome diagnostic, not pre-registered confirmation.',
          'rules':'On stopped, uncensored controlled programs only: recover an otherwise unparsed complete marked Markdown/LaTeX literal, terminal boxed literal, or exact last-line value statement of the frozen requested variable. No arbitrary last digit, no code execution, no selecting rules by whether the answer equals gold.',
          'nonmutation':'Do not edit running long-answer/primary scorer, prompts, gold, generated IDs, original commits or primary summary. Retain and report primary and secondary metrics together.',
          'previous_protocol_sha256':previous,
          'amendment':'Handle bold closing delimiters around Answer headings and exact requested-variable value statements; regression includes wrong-variable rejection. Old grammar and protocol retained. This remains post-outcome scoring recovery.' if amend else None}
        if amend:save(out/'protocol.json',protocol)
        else:immutable(out/'protocol.json',protocol)
        print('TERMINAL_FORMAT_AUDIT_FROZEN',observed,test,flush=True);return
    r=read(BASE/'behavior_analysis/result.json');assert r['final'] and r['trajectories']==1464
    assert read(out/'protocol.json')['auditor_sha256']==sha(ROOT/'tests/glm5/rdc_update_terminal_audit.py')
    original={x['sample_id']:x for x in gzread(BASE/'program_material.json.gz')}
    records=gzread(BASE/'behavior_analysis'/r['records_file']);checked=0
    for item in records:
        raw=read(BASE/item['raw_record'])
        if item['kind']=='controlled_program':
            expected=enrich_score(original[item['sample_id']],raw['generated_text'],item['answer_scoring'])
            assert expected==item['format_aware_scoring'];checked+=1
        assert sha(BASE/item['raw_record'])==r['original_commit_sha256'][item['raw_record']]
    summary={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'parser_regression':test,
      'all_trajectory_records':len(records),'controlled_program_records_checked':checked,
      'recovered_records':len(r['format_audit_changes']),'changes':r['format_audit_changes'],
      'primary_summaries':r['summaries'],'format_aware_summaries':r['format_aware_summaries'],
      'format_aware_paired':r['format_aware_paired'],'protocol_sha256':sha(out/'protocol.json'),
      'primary_scorer_unchanged':sha(ROOT/'tests/glm5/rdc_update_scoring.py')==read(out/'protocol.json')['primary_scorer_sha256'],
      'seconds':time.monotonic()-start,
      'scope':'Secondary post-observation parsing audit of the same outputs, not a new model run, semantic improvement or reasoning-chain evaluation. Unparsed outcomes remain distinct from wrong answers.'}
    assert summary['primary_scorer_unchanged'];save(out/'result.json',summary);ledger('secondary_terminal_format_audit',summary['seconds'])
    print('TERMINAL_FORMAT_AUDIT_PASS',summary['recovered_records'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','audit']);parser.add_argument('--amend',action='store_true')
    args=parser.parse_args();main(args.stage,args.amend)
