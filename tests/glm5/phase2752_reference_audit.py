"""Trace the supplied 'universal linear heads' claims and save counterexamples.

Toy counterexamples establish mathematical non-implications, not new model tests.
The checkpoint-only norm analysis uses the actual unembedding distribution, not
the distribution of activations. No new language-model inference occurs here.
"""
import json
from pathlib import Path
import numpy as np
from phase2752_context_interaction import ROOT, OUT, now, sha, write, snapshot


def mathematical_checks():
    def softmax(x):
        e = np.exp(x-x.max(-1,keepdims=True))
        return e/e.sum(-1,keepdims=True)
    def attn(x):
        return softmax(x@x.T/np.sqrt(x.shape[-1]))@x
    x = np.array([[1.,0.],[0.,1.]])
    z = np.array([[.3,-1.],[.2,.4]])
    a = softmax(x@x.T/np.sqrt(2))
    b = softmax((x+z)@(x+z).T/np.sqrt(2))
    full_difference = attn(x+z)-attn(x)
    pieces = [a@z,(b-a)@x,(b-a)@z]
    assert np.allclose(sum(pieces),full_difference)
    sigma = np.array([[4.,1.8],[1.8,1.]])
    diag = np.diag(1/np.sqrt(np.diag(sigma)))
    r = np.array([2.,1.])
    head = np.array([.5,2.])
    norm = lambda v:v/np.sqrt(np.mean(v*v)+1e-6)
    xx = np.array([1.,2.,3.])
    gamma = np.array([1.,2.,.5])
    rms = np.sqrt(np.mean(xx*xx)+1e-6)
    jac = np.diag(gamma)@(np.eye(3)/rms-np.outer(xx,xx)/(3*rms**3))
    return dict(created_utc=now(),scope='Explicit mathematical toy examples; not measurements of Qwen or language ability',
        attention=dict(x=x.tolist(),z=z.tolist(),nonlinear_additivity_error=float(np.linalg.norm(attn(x+z)-attn(x)-attn(z))),
            fixed_attention_linearity_error=float(np.linalg.norm(a@(x+z)-a@x-a@z)),
            general_delta_decomposition_error=float(np.linalg.norm(sum(pieces)-full_difference)),
            routing_term_norm=float(np.linalg.norm(pieces[1]+pieces[2])),
            identity='Delta(AV)=A_base DeltaV + DeltaA V_base + DeltaA DeltaV'),
        whitening=dict(covariance=sigma.tolist(),diagonal_standardization=(diag@sigma@diag).tolist(),
            remaining_offdiagonal=float((diag@sigma@diag)[0,1]),
            meaning='Even exact inverse standard deviations do not whiten a correlated covariance.'),
        norm=dict(jacobian=jac.tolist(),max_offdiagonal=float(np.max(np.abs(jac-np.diag(np.diag(jac))))),
            fixed_gamma_head_rescale_output_change=float(np.linalg.norm(norm(r+2*head)-norm(r+head))),
            meaning='Shared RMS denominator couples coordinates; changing a head contribution affects normalized output with gamma fixed.'),
        rank_bound='rank(W_O_head W_V_group) <= head_dim = 128 in local Qwen3-4B; no infinite family of exactly orthogonal nonzero directions in a finite-dimensional head write space')


def readout_statistics():
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    mdir = ROOT/'models/hf/qwen3-4b'
    index = json.loads((mdir/'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    def read(name):
        with safe_open(mdir/index[name],framework='pt',device='cpu') as f:
            return f.get_tensor(name)
    gamma = read('model.norm.weight').double().numpy()
    # This checkpoint ties input and output embeddings; identify the actual source.
    wname = 'lm_head.weight' if 'lm_head.weight' in index else 'model.embed_tokens.weight'
    weight = read(wname)
    n,d = weight.shape
    sums = np.zeros(d)
    squares = np.zeros(d)
    selected = np.random.default_rng(2752003).choice(d,128,replace=False)
    gram = np.zeros((128,128))
    for start in range(0,n,4096):
        block = weight[start:start+4096].double().numpy()
        sums += block.sum(0)
        squares += (block*block).sum(0)
        sample = block[:,selected]
        gram += sample.T@sample
    mean = sums/n
    variance = squares/n-mean*mean
    sd = np.sqrt(np.maximum(variance,0))
    after = np.abs(gamma)*sd
    cov = gram/n-np.outer(mean[selected],mean[selected])
    cor = cov/np.outer(sd[selected],sd[selected])
    mask = ~np.eye(128,dtype=bool)
    write(OUT/'reference_readout_coordinates.json',dict(coordinates=list(range(d)),gamma=gamma.tolist(),column_std=sd.tolist(),
        reweighted_std=after.tolist(),sampled_covariance_coordinate_ids=selected.tolist()))
    return dict(created_utc=now(),checkpoint_source=wname,model_config_sha256=sha(mdir/'config.json'),
        checkpoint_files={name:sha(mdir/name) for name in sorted({index[wname],index['model.norm.weight']})},
        dtype=str(weight.dtype),shape=[n,d],scope='Vocabulary rows as observations, unembedding columns as variables. Not activation whitening.',
        gamma_std_correlation=float(np.corrcoef(gamma,sd)[0,1]),
        column_std_cv_before=float(sd.std()/sd.mean()),column_std_cv_after=float(after.std()/after.mean()),
        positive_gamma_count=int((gamma>0).sum()),nonpositive_gamma_count=int((gamma<=0).sum()),
        sampled128_mean_abs_correlation=float(np.mean(np.abs(cor[mask]))),sampled128_max_abs_correlation=float(np.max(np.abs(cor[mask]))),
        interpretation='Diagonal rescaling preserves absolute pairwise correlations for nonzero scales. These measurements do not establish whitening, all-gain localization, or independent training.')


def trace():
    base = ROOT/'tests/glm5/result/rdc_query_construction_20260913'
    records = []
    for phase in (3055,3056,3057,3058,3059,3072,3079,3080,3081,3091):
        path = next((base/f'phase{phase}').glob('*/result.json'))
        result = json.loads(path.read_text(encoding='utf-8'))
        sealpath = path.with_name('seal.json')
        oldseal = json.loads(sealpath.read_text(encoding='utf-8')) if sealpath.exists() else {}
        short = oldseal.get('result_sha256_8')
        records.append(dict(phase=phase,path=str(path.relative_to(ROOT)),sha256=sha(path),
            existing_seal_result_prefix=short,existing_seal_matches=(sha(path).startswith(short) if short else None),
            verdict=result.get('verdict'), stats=result.get('stats'), note='Read-only source trace; not a fresh independent replication or re-certification of all legacy statistics.'))
    write(OUT/'reference_source_trace.json',dict(created_utc=now(),sources=records,
        primary_external_sources=[dict(title='Attention Is All You Need, section3.2',url='https://arxiv.org/html/1706.03762v7'),
                                  dict(title='Root Mean Square Layer Normalization, section4',url='https://arxiv.org/html/1910.07467v1')]))
    claims = [
        dict(id='REF01',claim='Whole attention heads are pure linear, as proved by3072',status='incorrect_generalization',
             evidence='3072 script explicitly clamps V with identical Q/K/attention; lineage relative error0.00173658 is conditional. General attention also changes A(X); local Qwen implementation has q/k RMSNorm and softmax.',retained='Fixed attention is linear in values and output projection.'),
        dict(id='REF02',claim='Heads encode semantic families instead of concrete words, and all language abilities',status='unsupported',
             evidence='3072 OV top10 is exploratory direct W_U projection without final norm. h1 list includes Twe,Sche,Trom,Sle,Jeg alongside theatre variants; no global semantic purity or all-task coverage test.',retained='Some observed write directions have related vocabulary readouts.'),
        dict(id='REF03',claim='Writing and reading are absolutely decoupled; only gamma amplifies',status='incorrect',
             evidence='W_V/W_O and nonlinear downstream computation affect gain; shared norm denominator couples coordinates.3059 compares energy concentration0.5184 read-side vs0.1321 write-side on selected16 coordinates, not statistical or causal independence.',retained='Writing and final vocabulary readout are distinct operations; gamma assignment has causal effects under tested intervention.'),
        dict(id='REF04',claim='Gamma is an inverse-variance whitening pipe',status='misnamed_and_overstated',
             evidence='3057 loglog exponent3.3076,R2=.4158 is a fit, not exact inverse std. Diagonal scaling does not remove correlations. New checkpoint-only standard deviation and correlation diagnostics saved.',retained='Anisotropic diagonal readout scaling, negatively associated with column standard deviation in this checkpoint.'),
        dict(id='REF05',claim='3079-3080 measure resonance of target direction and a head subspace',status='wrong_measured_object',
             evidence='Measured cos(TT_f,TT_g) compares two output logit-change vectors, NOT a target-to-head-subspace angle. No frequency dynamics or resonance experiment. TT needs both output forwards and is not a pre-outcome language-only predictor.',retained='Conditional association between output-change similarity and intervention-response similarity on a small frozen family.'),
        dict(id='REF06',claim='Angle rule is independently confirmed and universal',status='not_supported_counterevidence_omitted',
             evidence='3080 revisits the SAME3079 data.3081 independent model yields1/6 positive significant comparisons and negative minimumrho. Earlier audit corrects correlated Stouffer aggregation/pseudoreplication.',retained='4B local hypothesis with explicit model/template limitations.'),
        dict(id='REF07',claim='A finite head has infinitely many interference-free orthogonal task spaces',status='mathematically_invalid_as_stated',
             evidence='Finite rank and dimension bound exactly orthogonal directions; approximate superposition permits interference and requires tolerance/capacity assumptions.',retained='Finite computations and shared parameters can be compositionally reused; a compositional rule still needs demonstration.'),
        dict(id='REF08',claim='32 heads with10 modes prove10^32 language abilities and complete theory',status='unsupported_counting_argument',
             evidence='Mode identification, independent accessibility, composability, and one-to-one ability mapping are untested. Counting assignments does not establish correctness or universality.',retained='Combinations can be numerous; this is a motivation, not a proof.'),
        dict(id='REF09',claim='Static weights and upstream state do not determine which head acts',status='false_exclusion',
             evidence='Inference is a function of weights, prefix state/history, positions and numerical configuration. Failure of limited linear probes cannot prove absence of information in upstream states.',retained='Head effect is context dependent; parameters alone without inputs do not specify the current effect.')]
    write(OUT/'reference_claims.json',dict(created_utc=now(),overall='Some local evidence is valid; the claimed complete universal explanation is not.',claims=claims))


if __name__=='__main__':
    write(OUT/'reference_audit_execution.json',dict(created_utc=now(),source=snapshot(Path(__file__))))
    write(OUT/'reference_mathematical_checks.json',mathematical_checks())
    trace()
    write(OUT/'reference_readout_statistics.json',readout_statistics())
    print(json.dumps(json.loads((OUT/'reference_readout_statistics.json').read_text(encoding='utf-8')),indent=2))
