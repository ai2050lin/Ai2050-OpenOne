"""CPU-only synthetic identities for frozen six-run comparison arithmetic."""
from rdc_question_common import *
from phase2748_rdc_learning_analysis import state_metrics,first_divergence
from phase2748_rdc_fit_analysis import bootstrap


def main():
    checks=[]
    native=np.arange(28,dtype=float).reshape(4,7)
    metrics,coords=state_metrics(native,native.copy())
    assert np.array_equal(metrics['first_state_change_MSE'],np.zeros(4))
    assert np.array_equal(metrics['within_state_change_MSE'],np.zeros(4))
    assert np.array_equal(metrics['native_within_response_energy'],metrics['learned_within_response_energy'])
    checks.append('Unchanged vectors: both state-change terms exactlyzero, amplitudes equal')
    shift=np.arange(7,dtype=float)
    metrics,coords=state_metrics(native,native+shift)
    assert np.array_equal(metrics['first_state_change_MSE'],np.repeat(np.mean(shift**2),4))
    assert np.array_equal(metrics['within_state_change_MSE'],np.zeros(4))
    checks.append('Common context shift: nonzero absolute change, exactlyzero within-question change')
    delta=np.arange(4)[:,None]*np.arange(7)[None,:]
    learned=native+delta;metrics,coords=state_metrics(native,learned)
    centered=delta-delta.mean(0)
    assert np.array_equal(metrics['within_state_change_MSE'],np.mean(centered**2,axis=1))
    assert np.array_equal(coords['within_state_change_MSE_by_coordinate'],np.mean(centered**2,axis=0))
    checks.append('Question-dependent shift: independent full-coordinate centering agrees byquestion andbycoordinate')
    assert first_divergence([1,2],[1,2])is None
    assert first_divergence([1,2],[1,2,3])==2
    assert first_divergence([1,2],[1,3])==1
    assert first_divergence([1],[2])==0
    checks.append('Own-history divergence includes firstchangedtoken andprefix-length divergence; unchangedisNone')
    rows=[{'group_id':c+str(g),'cohort':c}for c in ['drop','quoref']for g in range(2)for q in range(4)]
    right=np.arange(16,dtype=float);left=right+.25
    result=bootstrap(rows,left,right,2748005)
    for cohort in ['drop','quoref','equal_cohort']:
        assert result[cohort]['mean_left_minus_right']==.25
        assert result[cohort]['paired_context_bootstrap_95_percent_interval']==[.25,.25]
    checks.append('Whole-context paired bootstrap preserves knownconstantdelta inbothcohorts andequalcohortsummary')
    value={'timestamp':stamp(),'source':snapshot(__file__),'analysis':snapshot(Path(__file__).with_name('phase2748_rdc_learning_analysis.py')),
        'all_passed':True,'checks':checks,'scope':'Synthetic arithmetic only; no native model, optimizer update or learned effect observed.'}
    immutable(OUT/'unit'/('learning_analysis_'+str(time.time_ns())+'.json'),value)
    save(OUT/'unit/learning_analysis_current.json',value)
    print('NATURAL_LEARNING_ANALYSIS_UNIT_PASS',len(checks),flush=True)


if __name__=='__main__':main()
