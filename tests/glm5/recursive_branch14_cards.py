"""Feature cards keep response, prediction, and intervention evidence distinct."""
import recursive_branch14 as R


def run():
    R.configure();p=R.admission();sel=R.read(R.OUT/'selection.json')
    response=R.read(R.OUT/'response-report.json');effects=R.read(R.OUT/'intervention-report.json')
    examples=R.read(R.OUT/'counterexamples.json');cards=[]
    for h,(b,u) in R.CANDIDATES.items():
        rr=[x for x in response['results'] if x['hypothesis']==h]
        ee=[x for x in effects['results'] if x['hypothesis']==h]
        cards.append(dict(
            feature_id=f'{R.SNAPSHOT}:B{b}:U{u}:G:content-end:recursive-branch14-v6',
            model='Qwen3-14B',snapshot=R.SNAPSHOT,block=b,unit=u,native_index_base=0,
            source='Historical natural-corpus candidate; prospective controlled response test',
            hypothesis=p['hypotheses'][h],extraction='Natural forward G,V,A at common content-end token, before question/options; actual BF16 down-projection input.',
            original_computation=dict(G='gate_proj.weight[u] dot normalized MLP input',V='up_proj.weight[u] dot normalized MLP input',
                A='native SiLU(G)*V',write='A times down_proj.weight[:,u]'),
            natural_corpus_evidence=dict(status='Historical exploratory contexts; not independent evidence from this campaign',
                report='../recursive_generalization_review_20261113/report.html'),
            discovery_group=sel['groups'][h],heldout_response_and_prediction=rr,
            prospective_response_hypothesis_passed=response['hypothesis_passed'][h],
            interventions=ee,counterexample_ids=[e['row']['id'] for e in examples['examples'] if e['hypothesis']==h],
            token_trajectories=f'each split/{h}-all-token-GVA.npy',
            position_diagnostic=dict(path='position-diagnostic.json',status='Exploratory analysis after primary test; no change to feature selection, interventions or confirmatory verdict.'),
            joint_state_diagnostic=dict(path='joint-state-summary.json',status='Descriptive post-primary G/V covariance and sign-quadrant accounting; no revised hypothesis test.'),
            unembedding_readout=dict(status='No new isolated unembedding projection claim',
                measured='Original-model final vocabulary KL and competing-answer margins under unit/group changes'),
            upstream_downstream=dict(status='Candidate-specific pathway not traced',
                separate_experiment='B39 roots and B38/B37/B36 predictive groups are separate candidates; not these units\' proven parents'),
            evidence_boundary='Heldout association/prediction and range-limited original-unit effects are separate. No training-time branching, necessity, sufficiency, unique mechanism, or complete graph claim.',
            input_hashes={name:R.file_sha(R.OUT/name) for name in ['plan.json','selection.json','response-report.json','intervention-report.json','counterexamples.json']}))
    R.save('feature-cards.json',dict(schema='recursive_branch14_feature_cards.v1',cards=cards,
        source_sha256=R.file_sha(__file__),formal_service_registration=False))


if __name__=='__main__':run()
