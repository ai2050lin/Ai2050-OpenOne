"""Read-only prediction summaries and bounded-campaign continuation handoff."""
import argparse,py_compile,shutil
from rdc_conditional_common import *


def main(final=False):
    p=CAMPAIGN/'p_token_conditioned';result=read(p/'result.json');reports={r['model']:r for r in result['reports']};rows=read(p/'selected_rows.json')
    with np.load(p/'features.npz') as z:ids=z['current_input_token_ids'];actual=z['attention'].astype(np.float64)
    tr=np.array([i for i,r in enumerate(rows) if r['word_split']=='train']);te=np.array([i for i,r in enumerate(rows) if r['word_split']=='test'])
    known=np.isin(ids[te],ids[tr]);summaries=[]
    for name,r in reports.items():
        with np.load(p/f'predictions/{name}.npz') as z:
            assert np.array_equal(te,z['test']);prediction=z['prediction'].astype(np.float64)
        error=np.mean((prediction-actual[te])**2,axis=1);assert abs(error.mean()-r['mse'])<1e-7
        summaries.append({'model':name,'mse':float(error.mean()),'known_current_token_mse':float(error[known].mean()),
          'unseen_current_token_mse':float(error[~known].mean()),'by_test_entity':{str(u):float(error[[rows[i]['unit']==u for i in te]].mean()) for u in range(12,16)}})
    save(p/'summary_audit.json',{'timestamp':stamp(),'passed':True,'prediction_rows_recomputed':27,'known_current_token_states':int(known.sum()),
      'unseen_current_token_states':int((~known).sum()),'reports':summaries,'scope':'Stored prediction arithmetic checked; current-token strata and four correlated entity groups are descriptive, not independent confirmation.'})
    frozen_files=list((p/'models').glob('*.npz'))+[p/name for name in ('features.npz','input_grams.npz','selected_rows.json','protocol.json')]
    immutable(p/'frozen_followup_predictors.json',{'files':{str(file.relative_to(p)):sha(file) for file in sorted(frozen_files)},
      'source_sha':sha(ROOT/'tests/glm5/phase2710_rdc_token_conditioned_attention.py'),'purpose':'Freeze current exploratory candidates for a future untouched confirmation; no such capture is claimed completed.'})
    total=sum(x.stat().st_size for x in CAMPAIGN.rglob('*') if x.is_file());ceiling=30*1024**3;minimum_next=2508796179
    decision={'timestamp':stamp(),'completed_phases':list(range(2703,2711)),'same_scientific_goal':True,'campaign_bytes':total,'campaign_ceiling_bytes':ceiling,
      'remaining_campaign_bytes':ceiling-total,'free_disk_bytes':shutil.disk_usage(CAMPAIGN).free,'free_disk_floor':8*1024**3,
      'next_comparable_capture_estimated_bytes':minimum_next,'estimate_source':'O32-prefix pilot projected512 capture plus0.55GiB analysis reserve; an estimate, not a guarantee',
      'decision':'Bounded campaign completed; do not launch the next comparable independent confirmation beyond its frozen30GiB allocation. Scientific goal remains open; this is not a physical-disk-full or model-failure claim.',
      'why_no_smaller_substitute':'The next uncertainty is frozen transfer of P, not more fitting on the same observed material. Silently reducing material merely to fill the remaining space would not meet the comparable multi-family confirmation design.',
      'next_stage':{'status':'not_executed','question':'Does the P H23 compiler transfer under separately controlled operation, lexical identity, source order, and generation position?',
        'freeze':['P full-context linear/quadratic factor, H23 compiler and direct-head weights/ridges/scales','token additive/product negative controls','native arithmetic and train-mean baselines'],
        'material':'At least a cost-piloted comparable512 fresh prefixes across8operations and16newentitygroups; independently vary wording and source order; new units, relation compositions and longer depth evaluated separately.',
        'analysis':['Preserve all native H coordinates, all source KV/head probabilities, all unit outputs at declared steps','Measure state/attention error separately from fully generated content, format and EOS','Decompose current-entry predictor error versus native architectural compilation and historical-cache dependence','If frozen transfer fails, compare typed source-position/role-equivalence input relations against exact-position kernels using equal-input/capacity controls'],
        'model_policy':'Start nonquantized Qwen3-4B CUDA; only after trustworthy capture, sequential Qwen14 CPU offload then GLM4; no concurrent CUDA model.',
        'resource_requirement':'New explicitly configured finite allocation or an approved archive/retention decision; rerun physical cost pilot before expansion. No automatic infinite queue.'},
      'retention':{'deleted_files':[],'reason':'All retained H/native fields are queryable by the client or required evidence/recomputation. Per-case all-token moments preserve nonpanel evidence; no disposable full fields identified.'}}
    assert total<ceiling and decision['free_disk_bytes']>8*1024**3 and minimum_next>ceiling-total
    save(CAMPAIGN/'continuation_decision.json',decision)
    compiled=[]
    for pattern in ('phase270[3-9]*.py','phase2710*.py','rdc_conditional*.py','rdc_long_material.py','rdc_order_material.py','rdc_attention_transfer_material.py'):
        for file in (ROOT/'tests/glm5').glob(pattern):py_compile.compile(str(file),doraise=True);compiled.append(str(file.relative_to(ROOT)))
    py_compile.compile(str(ROOT/'server/rdc_feature_service.py'),doraise=True)
    save(CAMPAIGN/'python_compile_audit.json',{'passed':True,'files':compiled+['server/rdc_feature_service.py'],'timestamp':stamp()})
    if final:
        from phase2707_rdc_delivery_audit import prefix_audit
        audit=read(CAMPAIGN/'delivery_audit.json');assert audit['passed'] and not audit['partial'];assert read(CAMPAIGN/'client_verification.json')['passed']
        for rel,digest in audit['source_files'].items():assert sha(ROOT/rel)==digest,(rel,'changed since full integrity audit')
        save(CAMPAIGN/'terminal.json',{'timestamp':stamp(),'state':'bounded_campaign_complete','scientific_goal_solved':False,'phases':list(range(2703,2711)),
          'memo_final':prefix_audit(),'integrity_audit_sha':sha(CAMPAIGN/'delivery_audit.json'),'client_verification_sha':sha(CAMPAIGN/'client_verification.json'),
          'continuation_decision_sha':sha(CAMPAIGN/'continuation_decision.json'),'source_files_unchanged_since_integrity':True,
          'pending_model_jobs':[],'model_weights_modified':False,'deleted_data':[],'resume':'continuation_decision.json next_stage; no launch within exhausted comparable-capture allocation'})
    print('DELIVERY_SUMMARY_COMPLETE','final' if final else 'prepare',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
