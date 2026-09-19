"""Audit the new attachments against committed evidence and freeze a finite plan."""
from rdc_query_common import *

ATTACHMENTS=[Path('C:/Users/Admin/.codex/attachments/02571e86-581c-4370-8a60-710bd579f131/pasted-text.txt'),
 Path('C:/Users/Admin/.codex/attachments/79e95a22-97c6-4e7c-b849-efc24ff7f2ef/pasted-text.txt')]

def main():
    if (BASE/'contract.json').exists():print('QUERY_CONTRACT_ALREADY_FROZEN');return
    import re, psutil
    start=time.monotonic();raw=MEMO.read_bytes();last=int(re.findall(rb'^## Phase (\d+):',raw,re.M)[-1]);assert last==2739,last
    for name in ('verification/final.json','manual_terminal_audit/result.json','verification/scientific_integrity.json'):
        assert read(PRIOR/name)['all_passed']
    manifest=read(PRIOR/'delivery_manifest.json')
    checks=[]
    for p,h in manifest['required_artifact_sha256'].items():
        actual=sha(PRIOR/p);assert actual==h,p;checks.append({'path':p,'sha256':actual})
    attachments=[]
    for i,p in enumerate(ATTACHMENTS,1):
        content=p.read_text(encoding='utf-8');assert content
        dest=BASE/'attachments'/f'attachment_{i}.txt';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        attachments.append({'source':str(p),'sha256':sha(p),'saved':str(dest.relative_to(BASE)),'characters':len(content)})
    r={'timestamp':stamp(),'result_ceiling_bytes':12*1024**3,'disk_floor_bytes':12*1024**3,'compute_ceiling_seconds':21600,
       'per_model_process_seconds':7200,'maximum_concurrent_CUDA_models':1,
       'basis':'New user-requested finite campaign; reuse the previously documented six-hour/12GiB engineering ceiling. Pilot before sizing, preserve all previous evidence, do not interpret automatic continuation as an unlimited budget.',
       'host_available_bytes_at_start':psutil.virtual_memory().available,'disk_free_bytes_at_start':shutil.disk_usage(ROOT).free}
    immutable(BASE/'resources.json',r)
    assertions=[
      ['native_factor_accounting','retain','Complete source/coordinate/unit/parameter accounting is implemented; a conditioned allocation identity is not a unique causal semantic gear.','native_paths/result.json'],
      ['transpose_means_no_direction','correct','Common transposition preserves a Gram matrix; a single transposition changes antisymmetric cross-inner-products. Direction labels require external calibration. Not proof that all gain is only amplitude.','fresh_graph/result.json'],
      ['query_only_means_no_information','correct','Query-only already contains the current full native state/embedding; it is not a no-information baseline.','graph/frozen.json'],
      ['content_format_projection','correct','The 192-direction rule is a ridge residual, not exact orthogonal projection; the batch rule is locally orthogonal only to its batch mean. C is conditional digit competition, F digit probability mass, neither equals general reasoning/style.','learning/finite_result.json'],
      ['low_initial_digit_means_cannot_reason','correct','Initial direct-digit accuracy, eventual terminal value and reasoning-chain validity are different. Native31/32 eventual correct-and-stop on only8semantic groups; does not establish arbitrary long reasoning.','manual_terminal_audit/result.json'],
      ['no_cross_model_isomorphism','correct','Centered correlations do not prove absence of every conditional/nonlinear/functional correspondence; matching coordinates was never established.','scale_analysis/result.json'],
      ['KV_absolute_separation','correct','Same token/position history: last MLP lies after all KV writes. Different chosen future tokens can change subsequent KV. Middle-block changes can propagate downstream; not every possible change must avalanche.','same_history/result.json'],
      ['identity_effect_disappeared','correct','Zero eligible matched controls means not identifiable in this material, not an observed zero effect.','language_identity/result.json'],
      ['moments_all_compression_impossible','correct','A specified finite-order ambient summary is non-injective. Reachability, approximate predictive utility, completeKV and all compression are separate claims. Raw tensor moments are not automatically central variance/skewness/kurtosis.','moment_boundary/result.json'],
      ['finite_probe_manifold_equivalence','correct','Small divergence under a finite declared query set gives an operational comparison only. It establishes neither all-future equivalence, manifold topology nor unique gears.','predictive_state/result.json'],
      ['format_entropy_is_nonsense','unverified_testable','High vocabulary entropy or low digit mass is not a demonstrated detector of unnecessary explanation. Test late logit bias without gold; report premature answers, content errors and future-history changes.','long_answers/result.json'],
      ['ordered_gate_up_pairs','new_testable','Keep branch-labeled g_s*u_r before symmetric marginalization; retain all sources/units with exact factors and explicit other/rounding. Reconstruct and test across conditions before causal naming.','native_paths/result.json'],
      ['query_transfer_and_injection','new_testable','Compare EN/ZH/Python conditional responses on group-heldout materials, then test only a validated bounded mapping. No presumption of cross-modal isomorphism or universal hallucination repair.','learning/forecast_audit.json'],
      ['new_math_required_or_impossible','correct','Current observations prove neither necessity nor impossibility of new mathematics. Known algebraic identities are not newly discovered universal language laws.','theory_snapshot.json']]
    p={'timestamp':stamp(),'common_question':'How do ordered, branch-specific native source interactions survive query changes, cross layers and generation steps, and form under actual learning?',
      'phases':[
       {'phase':2740,'question':'Joint ordered-event and native-calculation atlas','tasks':['Audit all attachment claims and the transpose boundary','Freeze 100 fixed query strings and measure feasibility before scaling','Natural English/Chinese document-disjoint prefix graph with identity/position metadata','Recover successful/failed prior generation event times; native full-coordinate and ordered gate/up factor atlas, Q/K-only RoPE and exact query accumulation checks']},
       {'phase':2741,'question':'Available-prefix query-rule competition and response transfer','tasks':['Use the same samples and targets for query-only, source summaries and ordered native-key/value query rules','Predict complete later-coordinate states and full-vocabulary response, with separate unseen source/query/combination splits','Cross-expression matched program response and identity controls','Keep every coordinate and factor; no PCA/TopK/eigenvector deletion; deployable inputs explicitly distinguished from observed future factors']},
       {'phase':2742,'question':'Actual formation and own-history tests','tasks':['Continue actual middle-MLP parameters on available training prefixes, two draw orders and matched targets','Evaluate at real answer-formation times rather than assuming prompt-final digit','Bounded gold-free late-logit/entropy intervention and fixed-time controls; same-history/current-KV versus own-history future-KV separated','Sequential originalBF16 Q4/Q14/GLM matched native replication, client and theory integration']},
       {'phase':2743,'question':'Information-bearing same-goal follow-up','tasks':['Choose and freeze independent natural/query confirmation based on resolved2740–2742 uncertainties','Complete remaining calibration and inferential audits, append all actual results and cumulative puzzle/formula ledger','Assess another whole-stage continuation against measured remaining resources; no indefinite or duplicate loop']}],
      'probe_scale_proposal':{'attachment_requested_prefixes':10000,'attachment_requested_probes':100,'initial_pilot_prefixes':3,'fixed_probe_strings':100,
        'main_candidate_natural_prefixes':576,'measurement_before_scaling':'Measured prefix/cache/suffix costs and bounded storage;10000x100 is a proposal, not assumed executed. Final scientific sample counts frozen after the independent pilot and before main outcomes.'},
      'safety_and_scope':['No original checkpoint writeback','Only one native CUDA model at a time','No change to user services/processes/system settings','All retained arrays registered with read-only client or documented evidence purpose','Parsing changes cannot count as ability gains','Fixed probes are deliberate query interventions, not natural free generation or100single tokens','Gold/true future QKV are scoring targets, not available-prefix predictor inputs','No universal manifold/AGI closure completion criterion is claimed achievable by this finite campaign']}
    immutable(BASE/'plan.json',p)
    result={'timestamp':stamp(),'source':snapshot(__file__),'attachments':attachments,'last_phase_before_campaign':last,
      'memo_prefix_bytes':len(raw),'memo_prefix_sha256':hashlib.sha256(raw).hexdigest(),'prior_required_artifacts_verified':checks,
      'prior_model_fingerprint_receipt_sha':sha(PRIOR/'verification/model_checkpoint_fingerprints.json'),
      'claim_audit':[{'claim':a,'status':b,'decision':c,'evidence':str(PRIOR/d)} for a,b,c,d in assertions],
      'prior_results_preserved':True,'source_versions':[snapshot(ROOT/'tests/glm5'/n) for n in ('rdc_query_common.py','rdc_update_graph.py','phase2738_rdc_update_native_paths.py','phase2739_rdc_update_predictive_state.py','rdc_update_scoring.py','rdc_update_terminal_audit.py')]}
    immutable(BASE/'contract.json',result);ledger('attachment_evidence_contract',time.monotonic()-start)
    print('QUERY_CONTRACT_PASS',last,'verified',len(checks),'claims',len(assertions),flush=True)

if __name__=='__main__':main()
