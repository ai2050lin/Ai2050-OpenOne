"""Supplemental unblinded terminal review; original frozen scores remain untouched."""
from rdc_formation_common import *
from phase2747_rdc_program_own_history import BRANCHES


def main():
    start=time.monotonic();folder=OUT/'program_own_history/terminal_review';finish=folder/'result.json'
    if finish.exists():return read(finish)
    source=OUT/'program_own_history/analysis/audited_records.json.gz';rows=gzread(source)
    assert len(rows)==192 and read(OUT/'program_own_history/analysis/result.json')['all_passed']
    # The main agent inspected the entire residual stopped/unparsed set before
    # authoring this exact receipt. This is not a newly trained parser or an
    # independent blinded reviewer. Never extend the set silently.
    candidates=[r for r in rows if r['answer_scoring']['EOS'] and r['answer_scoring']['conservative_final_answer'] is None]
    assert [(r['branch'],r['sample_id']) for r in candidates]==[('code_identity','u2736_7cda4cfd7a507d3c153a')]
    r=candidates[0];quote='So, the value of `v_184_4` is **3**.'
    assert r['generated_text'].rstrip().endswith(quote) and r['target']=='3' and not r['answer_scoring']['censored']
    record_path=OUT/'program_own_history/records'/r['branch']/(r['sample_id']+'.json')
    review={'branch':r['branch'],'sample_id':r['sample_id'],'source_group':r['source_group'],
        'record_sha256':sha(record_path),'generated_text_sha256':hashlib.sha256(r['generated_text'].encode()).hexdigest(),
        'exact_terminal_quote':quote,'requested_variable':'v_184_4','terminal_value':'3',
        'terminal_value_matches_gold':True,'EOS':True,'strict_answer_only':False,
        'original_frozen_parser_correct_and_stopped':False,
        'reason':'Complete final sentence explicitly gives the requested variable as3 and then EOS. Frozen pattern did not cover the value-of phrasing with a backtick-quoted variable. Not inferred from an arbitrary leading/last digit.',
        'reviewer':'Main research agent; entire input and complete output read after outcomes were available, not blinded or independent.',
        'boundary':'Terminal result only; strict requested output format still fails. This receipt does not establish general reasoning-chain correctness or change any original scoring rule.'}
    native={v['sample_id']:v for v in rows if v['branch']=='native'};summary=[]
    for branch in BRANCHES:
        rr=[v for v in rows if v['branch']==branch];assert len(rr)==32
        def audited(v):return bool(v['answer_scoring']['parsed_and_stopped_correct'] or
            (v['branch'],v['sample_id'])==(review['branch'],review['sample_id']))
        summary.append({'branch':branch,'groups':32,'frozen_parser_correct_and_stopped':sum(v['answer_scoring']['parsed_and_stopped_correct'] for v in rr),
            'supplemental_terminal_correct_and_stopped':sum(audited(v) for v in rr),
            'paired_supplemental_change':clustered([int(audited(v))-int(audited(native[v['sample_id']])) for v in rr],[v['source_group'] for v in rr])})
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'scope':'Post-outcome unblinded terminal-content audit, separate from the unchanged frozen primary measurement.',
        'complete_residual_EOS_unparsed_set_size':len(candidates),'reviews':[review],'summary':summary,
        'all192_original_records_retained':True,'audited_records_sha256':sha(source),
        'original_program_result_sha256':sha(OUT/'program_own_history/result.json'),
        'original_program_analysis_sha256':sha(OUT/'program_own_history/analysis/result.json'),
        'new_generation_or_parser_training':False,'seconds':time.monotonic()-start}
    immutable(finish,value);ledger('phase2747_program_terminal_review',value['seconds'])
    print('FORMATION_PROGRAM_TERMINAL_REVIEW',len(candidates),flush=True);return value


if __name__=='__main__':main()
