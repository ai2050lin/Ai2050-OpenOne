"""Post-observation agent terminal adjudication, not another fitted parser."""
import argparse
from copy import deepcopy
from rdc_update_common import *
from phase2739_rdc_update_behavior_analysis import summary_tables


def main(stage):
    out=BASE/'manual_terminal_audit'
    behavior=read(BASE/'behavior_analysis/result.json')
    assert behavior['final'] and behavior['trajectories']==1464
    records=[r for r in gzread(BASE/'behavior_analysis'/behavior['records_file']) if r['mode']=='long_answers']
    assert len(records)==192
    inventory=[]
    for r in records:
        s=r['format_aware_scoring']
        if s['EOS'] and not s['censored'] and s['conservative_final_answer'] is None:
            raw=read(BASE/r['raw_record'])
            inventory.append({'sample_id':r['sample_id'],'branch':r['branch'],'raw_record':r['raw_record'],
              'raw_sha256':sha(BASE/r['raw_record']),'target':r['target'],'generated_text':raw['generated_text'],
              'EOS':s['EOS'],'censored':s['censored']})
    protocol={'scope':'Entire residual unparsed, stopped, uncensored long-answer set under the unchanged secondary grammar; no random subset or winner selection.',
      'timing':'Post-outcome agent review. The need for this audit was discovered after reading residual Chinese number-only instructions and English printed-digit declarations in completed branches. Not preregistered, blinded or independently human-replicated.',
      'rules':'Read the actual complete text and explicit terminal declaration. Quote the supporting terminal span exactly; assign one literal digit only when unambiguous, otherwise abstain. Do not execute generated code, score the reasoning chain, or change either automated grammar.',
      'grader':'Main research agent, unblinded to gold; correctness is computed from the independently frozen target after the terminal answer is recorded.',
      'ungraded':'Censored outputs and unsupported/ambiguous declarations remain unresolved. This limited terminal audit does not establish semantic reasoning validity or generalization.'}
    if stage=='inventory':
        immutable(out/'inventory.json',{'timestamp':stamp(),'source':snapshot(__file__),
          'behavior_sha256':sha(BASE/'behavior_analysis/result.json'),
          'secondary_protocol_sha256':sha(BASE/'terminal_format_audit/protocol.json'),
          'protocol':protocol,'rows':inventory})
        print('MANUAL_TERMINAL_INVENTORY',len(inventory),flush=True)
        for r in inventory:print(json.dumps(r,ensure_ascii=False),flush=True)
        return
    start=time.monotonic()
    frozen=read(out/'inventory.json')
    assert frozen['rows']==inventory and frozen['behavior_sha256']==sha(BASE/'behavior_analysis/result.json')
    annotations=read(out/'annotations.json')
    assert annotations['reviewer']=='Main research agent'
    lookup={r['raw_record']:r for r in annotations['rows']}
    assert len(lookup)==len(inventory) and set(lookup)=={r['raw_record'] for r in inventory}
    results=[];adjudicated=[]
    for r in records:
        rec=deepcopy(r);s=dict(r['format_aware_scoring'])
        item=lookup.get(r['raw_record'])
        if item:
            raw=read(BASE/r['raw_record']);quote=item['quoted_terminal_text'];answer=item['answer']
            assert sha(BASE/r['raw_record'])==item['raw_sha256']
            assert quote and raw['generated_text'].rstrip().endswith(quote.rstrip()),r['raw_record']
            assert answer is None or answer in tuple('12345678')
            assert item['reason'] and item['whole_output_read'] is True
            if answer is not None:
                s.update(conservative_final_answer=answer,conservative_final_correct=answer==str(r['target']),
                  parsed_and_stopped_correct=answer==str(r['target']))
            adjudicated.append(item|{'target':r['target'],'correct':None if answer is None else answer==str(r['target'])})
        rec['answer_scoring']=s;results.append(rec)
    summaries,paired=summary_tables(results)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'protocol':protocol,
      'inventory_sha256':sha(out/'inventory.json'),'annotations_sha256':sha(out/'annotations.json'),
      'primary_scorer_sha256':sha(ROOT/'tests/glm5/rdc_update_scoring.py'),
      'secondary_scorer_sha256':sha(ROOT/'tests/glm5/rdc_update_terminal_audit.py'),
      'reviewed_residual_outputs':len(adjudicated),'resolved_outputs':sum(r['answer'] is not None for r in adjudicated),
      'unchanged_generation_count':192,'adjudications':adjudicated,'summaries':summaries,'paired':paired,
      'seconds':time.monotonic()-start,
      'limits':'This is a third, explicitly unblinded post-observation measurement layer on the same192 outputs. It is not a new generation, independent confirmation, parser improvement, training benefit or reasoning-chain correctness audit. Primary and secondary outputs and all raw commits remain unchanged.'}
    assert result['secondary_scorer_sha256']==read(BASE/'terminal_format_audit/protocol.json')['auditor_sha256']
    save(out/'result.json',result);ledger('manual_terminal_annotation_validation',result['seconds'])
    print('MANUAL_TERMINAL_AUDIT_PASS',result['reviewed_residual_outputs'],result['resolved_outputs'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inventory','audit']);main(p.parse_args().stage)
