"""Audit the two new attachments against actual Phase2715--2718 artifacts, without rerunning them."""
from rdc_joint_common import *

ATTACHMENTS = [
    Path('C:/Users/Admin/.codex/attachments/02eadfd7-147b-4ae7-acc9-e8d937554018/pasted-text.txt'),
    Path('C:/Users/Admin/.codex/attachments/cb263397-365b-401f-8d54-f5ba47d1c1dc/pasted-text.txt'),
]


def main():
    guard(2 * 1024**2, preflight=True)
    if (BASE / 'review.json').exists():
        print('JOINT_REVIEW_ALREADY_EXISTS', flush=True)
        return
    memo = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'
    immutable(BASE / 'memo_prefix.json', {'timestamp': stamp(), 'bytes': memo.stat().st_size, 'sha256': sha(memo)})
    paths = [
        'material_audit.json', 'relation_atlas/parser_baseline_result.json', 'relation_atlas/result.json',
        'rules/current/result.json', 'rules/full_quadratic/result.json', 'dynamics/result.json',
        'confirmation/state_result.json', 'confirmation/result.json', 'native/result.json',
        'native_source_audit/result.json', 'generation/result.json', 'scalar_parameters/result.json',
        'boundary_compilation/result.json', 'boundary_compilation/constant_update_result.json',
        'source_normalization/result.json', 'boundary_relations/result.json',
        'output_geometry/result.json', 'surrogate_stability/result.json',
        'verification/final_integrity.json', 'verification/extension_integrity.json',
        'continuation_decision.json',
    ]
    evidence = []
    missing = []
    for rel in paths:
        p = PREVIOUS / rel
        if not p.exists():
            missing.append(rel)
        else:
            evidence.append({'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size})
    # Missing optional filenames are explicitly logged and resolved before claiming their numbers verified.
    checks = []
    material = read(PREVIOUS / 'material_audit.json')
    assert material['main_units'] == 512 and material['fresh_units'] == 128
    assert material['main_tokens'] == 19593 and material['fresh_tokens'] == 4454
    checks.append({'claim': 'Material count', 'verified': [512, 128, 19593, 4454]})
    constant = read(PREVIOUS / 'boundary_compilation/constant_update_result.json')
    assert abs(constant['common_increment_MSE'] - .01366774781405603) < 1e-12
    assert constant['common_increment_MSE'] < constant['affine_FP32_MSE']
    assert constant['fresh_sources'] == 128
    checks.append({'claim': 'Common boundary update is a stronger first-position null control, not a new native block run',
                   'MSE': constant['common_increment_MSE'], 'affine_MSE': constant['affine_FP32_MSE']})
    stability = read(PREVIOUS / 'surrogate_stability/result.json')
    cycles = stability['cycles']
    zero = next(c for c in cycles if c['incoming_token_cycle'] == [15])
    assert len(cycles) == 10 and len(zero['observed_suffix_sources']) == 39
    assert abs(zero['spectral_radius'] - .9598794795081012) < 1e-12
    assert zero['greedy_self_consistent'] and zero['locally_attracting_affine_cycle']
    checks.append({'claim': 'A token15 stable, greedy-consistent branch of the fitted surrogate',
                   'candidate_count': 10, 'actual_suffix_sources': 39, 'radius': zero['spectral_radius']})
    continuation = read(PREVIOUS / 'continuation_decision.json')
    assert continuation['completed_phases'] == [2715, 2716, 2717, 2718]
    assert continuation['next_phase']['number'] == 2719 and continuation['next_phase']['status'] == 'not executed'
    checks.append({'claim': 'Phase2719 is new execution, not a previous completed result', 'verified': True})

    corrections = [
        ['A numerical and methodological summary', 'Retain the stated scopes after source checks; independent arrays/model reruns were not performed by the attachment author.'],
        ['A full-source/all-layer distinction', 'Retain: H12/H23 all tokens; only six positions at H24/H36/postnorm. Full coordinate is not full-layer coverage.'],
        ['A RMS and common boundary update', 'Retain as posthoc findings. New materials must confirm frozen old rules before fitting new ones.'],
        ['A conditional profile uncertainty', 'Retain: strict matching changes source coverage; conditional bootstrap and weak cosine do not identify pure semantics.'],
        ['A broad historical puzzle table', 'Inherit only scoped, traceable entries; this audit does not independently rerun all listed historical phases.'],
        ['A formal atlas tuple', 'Useful proposed organization, not an already proved new theorem or a replacement of the actual historical RDC formula.'],
        ['A need for new mathematics', 'Retain openness: a new theorem can use existing objects. No requirement to first prove all existing mathematics incapable.'],
        ['B external syntax invalid / probes dead', 'Unsupported. Only specified soft binding/aggregation candidates lacked additional prediction value; external relation discrimination and weak conditional profiles remain.'],
        ['B all models lost to quadratic hence special nonlinear core', 'Capacity confounded: effective-df matching removed quadratic advantage. Joint state/input evidence concerns fitted predictors, not a unique native algorithm.'],
        ['B native block completely collapsed', 'Misleading: first-source predicted H23 MSE was enormous, but query H24 MSE was1.640595, not millions; native arithmetic oracle was bitwise correct.'],
        ['B infinite zero loop', 'Observed at most16 generated tokens, plus local constant-token surrogate stability; no infinite-time global attractor claim.'],
        ['B constant offset fixed downstream to0.01', '0.01366775 is first-source H23 MSE. This constant control was not compiled through block23 in2718; H24 conclusions cannot inherit that number.'],
        ['B all overinterpretation eliminated / native gears precise', 'Neither absolute claim follows. Language mechanisms and remaining surrogate errors are still unresolved.'],
        ['B every Top-K or dimensional reduction must fail', 'Unsupported universal. Low-energy group contributes about21--22 percent absolute path attribution here; this is not a universal causal necessity claim. Full native coordinates remain primary by user instruction.'],
        ['B gate matrix changes with context', 'Weights are fixed. Gate and up activations use the same normalized post-attention residual input, including residual and attention. They are not direct functions of KV alone.'],
        ['B no KV cache inevitably causes long-range collapse', 'Cache is an implementation optimization; uncached native prefix recomputation still reads history. Limited extracted state may omit useful history, but necessity/unique cause is not shown.'],
        ['B eat queries future apple in I like eating apples', 'Causal ordering error: apple has not entered the prefix at eat. Future completion preference is not reading a future input token.'],
        ['B Up fruit/edibility coordinates and knowledge-chain KV deformation', 'No such semantic localization was measured; do not turn a teaching story into a native circuit discovery.'],
        ['B syntax high frequency / knowledge long-range rotation / geodesics', 'Undefined or unmeasured metric, frequency, manifold and geodesic. Knowledge may reside in weights; syntax may involve long distance.'],
        ['B proposed H[t+1] equation', 'Conflates layer and token indices, omits incoming embedding, normalizations and the post-attention residual; replace by separately indexed native block equations and explicit extracted-state update interfaces.'],
        ['B Tucker core is pure physical gear', 'Factorization is not semantic/causal identification; leading-subspace primary analysis conflicts with requested full-coordinate approach. Use exact factorized or kernel contractions without claiming recovered native factors.'],
        ['B projection will cure every multihop or hallucination', 'Unsupported intervention promise. No established manifold or causal sufficiency. Stabilization may reduce repetition while harming probability/content; test both if justified.'],
        ['B cross-domain isometry / sensory modality / AGI proof', 'Natural language, code and math text are domains in these models. Isometry, transferable logic and hallucination elimination require independent evidence; no AGI conclusion follows.'],
        ['B first-position0 and emitted character0', 'Separate systems and branches; no demonstrated causal connection.'],
    ]
    plan = {
        'timestamp': stamp(), 'status': 'preflight; execution pending storage allocation and measured pilot',
        'scientific_goal': 'Extract reusable conditional language relations in complete native coordinates and connect available source history, parameter computations and next-token competition.',
        'phases': [
            {'phase': 2719, 'question': 'Which prior conditional patterns survive independent natural material, and where do boundary/ordinary-position differences arise?',
             'work_packages': ['Audit attachments and freeze unified scope and numerical controls.',
                 'New bilingual natural treebank sources with source/document and exact skeleton exclusions; retrospective typed relations separated from prefix-available inputs.',
                 'Pilot then capture full-token selected layers and embedding-to-all-layer specified positions, retaining full native coordinate order and raw values.',
                 'Apply frozen2718 common-boundary/RMS rules and noninitial relation profiles before new fitting; locate all-layer boundary emergence and stratify by lexical identity, language, genre and relation condition.']},
            {'phase': 2720, 'question': 'Does preserving query-dependent source organization and optimizing output probability improve coordinate and temporal rule extraction?',
             'work_packages': ['Same partitions: current/common-update, raw/RMS/direction-plus-scale source rules and native-history-informed candidates; zero-history and source-permutation controls.',
                 'State-only, state plus known embedding, and selected-layer full-source K/V retrieval with explicit available inputs; exact multilinear kernels without Tucker truncation.',
                 'Match capacity summaries; compare state-loss versus full-vocabulary probability objectives using all coordinates and frozen validation selection.',
                 'Freeze before untouched confirmation; evaluate by declared language relation families, same-prefix probabilities and whole-source uncertainty.']},
            {'phase': 2721, 'question': 'What original parameter relations implement reproducible source-conditioned updates, and what survives unrefreshed continuation?',
             'work_packages': ['Follow reproducible whole-field patterns through actual attention, gate/up and down factors with complete-unit and scalar reconstruction.',
                 'Natural teacher prefixes, native generated prefixes and independently self-generated prefixes kept separate; longer continuation without true-state refresh in autonomous branches.',
                 'Diagnose repetition/fixed branches and probability/content tradeoffs; no assumed manifold projection or donor transport solution.',
                 'Sequential nonquantized Qwen4/Qwen14/GLM targeted confirmation, numerical execution floors, local client linkage, complete provenance and append-only MEMO.']},
        ],
        'initial_design_counts_subject_to_outcome_blind_material_and_cost_pilot': {'main_sources': 512, 'fresh_sources': 256, 'train_validation_test': [320, 96, 96], 'generation_sources': 64, 'generation_steps': 32},
        'full_coordinate_policy': 'All native coordinates and requested source positions; no Top-K/PCA/Tucker as the primary mechanism definition. Exact factorization/reconstruction saves storage without deleting coordinates.',
        'storage': 'Awaiting asynchronous user choice: new C-drive destination with project result junction, or explicitly revised D-drive reserve. Old640MiB campaign remains immutable.',
        'continuation': 'After integrated completion, automatically execute another bounded same-goal stage only with meaningful new evidence value and remaining resources. Do not report long-term AGI closure.',
        'not_promised': ['Universal semantic gear dictionary', 'Manifold or isometric-language theorem', 'Complete knowledge/reasoning/syntax recovery', 'Guaranteed generation repair or hallucination elimination'],
    }
    save(BASE / 'review.json', {'timestamp': stamp(), 'source': snapshot(Path(__file__)),
        'attachments': [{'path': str(p), 'sha256': sha(p), 'lines': len(p.read_text(encoding='utf-8').splitlines())} for p in ATTACHMENTS],
        'evidence': evidence, 'unresolved_optional_evidence_paths': missing, 'numeric_assertions': checks,
        'corrections': corrections,
        'naive_one_sample_full_H2560_E2560_KV_36layers32tokens8heads128_fp32_bytes': 2560**2 * (2*36*32*8*128) * 4,
        'formula_scope': 'Known native architecture, fitted-rule identities and pathwise readout attribution are distinct. No new universal language theorem.',
        'literature': [{'url': 'https://arxiv.org/abs/2402.17762', 'scope': 'Primary abstract: massive activations previously reported; local Qwen behavior not inferred from that paper.'},
                       {'url': 'https://arxiv.org/abs/2309.17453', 'scope': 'Primary abstract: initial-token attention sinks/streaming cache methods; native recomputation still uses history.'},
                       {'url': 'https://arxiv.org/abs/2503.22329', 'scope': 'Primary abstract: architecture/condition limits; mitigation is model-specific.'}]})
    save(BASE / 'plan.json', plan)
    print('JOINT_REVIEW_SAVED', len(evidence), 'evidence artifacts', len(corrections), 'scoped corrections',
          'optional_paths_to_resolve', missing, flush=True)


if __name__ == '__main__':
    main()
