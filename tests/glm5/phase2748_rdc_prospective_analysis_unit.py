"""Synthetic causal-prefix alignment cases, before any prospective results."""
from rdc_question_common import *
from phase2748_rdc_prospective_analysis import sequence_alignment


def main():
    cases=[([1,2,3],[1,2,3],3,3,None),([1,2],[9,2],0,1,0),
        ([1,2,3],[1,9,8],1,2,1),([1],[1,2],1,1,1),([1,2],[1],1,1,1),
        ([1,2,3],[1,2,9],2,3,2)]
    checks=[]
    for left,right,common,valid,divergence in cases:
        value=sequence_alignment(left,right)
        assert value['common_output_prefix_tokens']==common and value['common_causal_prediction_steps']==valid
        assert value['first_output_divergence_index']==divergence
        assert all(left[:i]==right[:i]for i in range(valid))
        if valid<min(len(left),len(right)):assert left[:valid]!=right[:valid]
        assert sequence_alignment(right,left)==value
        checks.append({'left':left,'right':right,'result':value,'all_valid_states_have_identical_prior_tokens':True})
    value={'timestamp':stamp(),'all_passed':True,'analysis':snapshot(Path(__file__).with_name('phase2748_rdc_prospective_analysis.py')),
        'checks':checks,'scope':'Synthetic indexing, not real language or CUDA evidence.'}
    save(OUT/'unit/prospective_analysis_current.json',value)
    print('NATURAL_PROSPECTIVE_ANALYSIS_UNIT_PASS',len(checks),flush=True)


if __name__=='__main__':main()
