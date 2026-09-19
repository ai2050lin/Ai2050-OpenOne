"""Publish bounded delivery and explicit remaining scientific questions; no background queue."""
from rdc_mechanism_common import *

def main():
    audit=read(CAMPAIGN/'delivery_audit.json');assert audit['memo_original_prefix_preserved']
    b=CAMPAIGN/'b_relations';c=CAMPAIGN/'c_generation';d=CAMPAIGN/'d_generalization'
    stress=read(d/'result.json');prefix=read(d/'causal_prefix_result.json')
    assert len(stress['results'])==120 and len(prefix['pairs'])==32
    warning='Phase2698测量边界：自然不同长度前向中，问题之前U/V的差异不能解释为未来语义回写。32对等形右补齐检查中，公共前缀全层全坐标差异全部为0。C位于问题之后，仍可读取关系；本结果不证明纯语义或通用机制。'
    save(b/'extension_result.json',dict(stress,measurement_warning=warning,causal_prefix_audit=prefix))
    for run,n in [('b_relations',512),('c_generation',384)]:announce(run,state='complete',completed=n,total=n)
    events('b_relations','same_goal_followup_complete',phase=2698,comparisons=120,shape_pairs=32)
    events('c_generation','delivery_complete',phase=2697,states=384)
    scripts=sorted((ROOT/'tests/glm5').glob('phase269[5-8]_rdc*.py'))
    keepfiles=scripts+[ROOT/'tests/glm5/rdc_mechanism_common.py',ROOT/'tests/glm5/rdc_relation_material.py',ROOT/'server/rdc_feature_service.py',ROOT/'frontend/src/components/app/RdcFeatureAtlas.jsx']
    delivery={'timestamp':stamp(),'status':'bounded_campaign_complete','phases':[2695,2696,2697,2698],
        'science_status':'candidate readability and native source ledgers; NOT language mechanism closure',
        'completed':{'review_corrections':13,'old_cases_reused':1024,'reader_controls':44,'native_mlp_unit_layers':3,
            'new_relation_cases':512,'relation_comparisons':129,'generation_prefixes':128,'generation_states':384,
            'cross_family_language_stress_fits':120,'same_shape_prefix_pairs':32,'additional_shape_forward_calls':64},
        'source_sha':{str(p.relative_to(ROOT)):sha(p) for p in keepfiles},
        'retention':{'new_raw_bytes':sum(r['raw_bytes'] for r in audit['artifacts']),'old_raw_bytes_referenced':18049992623,
            'reason':'Client queries and later structure research use all raw fields; same-shape diagnostic stores all-coordinate error summaries only.',
            'deleted_files':0,'deleted_bytes':0},
        'client':'http://localhost:5173/rdc; real committed fields, native scalar parameters, MLP units, all sources, generation steps; no task-start endpoint',
        'historical_deferred':'Phase2691 four-protocol crossmodel campaign remains deferred, not falsely marked complete.',
        'next_phase':2699,'next_started':False,'no_unbounded_background_queue':True}
    save(CAMPAIGN/'delivery.json',delivery)
    save(CAMPAIGN/'handoff.json',{'timestamp':stamp(),'next_phase':2699,'state':'current_bounded_delivery_complete',
        'same_scientific_goal':True,'continuation_already_executed':'Phase2698 expanded whole-family/language stress and64 causal-prefix calibration forwards after A/B/C.',
        'next_large_task':['Freeze an independent relation corpus with equal execution shapes and whole-family/lexical/wording disjoint groups; compare H24C support and requested-answer readers to native natural outputs.',
            'Trace retained C relation readout into native L23→L35 computations with correct-history/outcome distinctions; keep all units and low values, not highest-contribution pruning.',
            'Fit task-stage-conditioned earlier-layer predictors of the actually relevant next-token target, not reuse answer Yes/No contrast to score punctuation/EOS.',
            'Only after new corpus preflight and measured resource estimate, serial nonquant Qwen14/otherlocal model confirmation; coordinate indices not aligned across models.'],
        'prerequisites':'New schema/frozen material and revised numerical execution contract before expanding scans. Do not repeat old datasets as fresh confirmation.',
        'recompute_entrypoints':[str(p.relative_to(ROOT)) for p in scripts],
        'resource_boundary':'A/B/C plus120 stress fits and64 shape forwards completed; no infinite loop or unattended CUDA campaign created.'})
    print('FINALIZED',delivery['completed'],flush=True)

if __name__=='__main__':main()
