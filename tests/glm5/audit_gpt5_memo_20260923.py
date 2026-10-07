"""Read-only source/result audit; writes only new audit artifacts. No model loads."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEMO = ROOT / 'research/gpt5/docs/AGI_GPT5_MEMO.md'
BASE = ROOT / 'tests/glm5/result/rdc_query_construction_20260913'
OUT = ROOT / 'tests/glm5/result/gpt5_memo_audit_20260923'
SNAPSHOT = OUT / 'memo_snapshot.md'
if SNAPSHOT.exists():
    MEMO = SNAPSHOT

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def average_rank(a):
    a = np.asarray(a, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - counts + ends - 1) / 2
    return ranks[inv]

def corr(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    a=a-a.mean(); b=b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None

def sp(a,b):
    return corr(average_rank(a),average_rank(b))

def legacy_sp(a,b):
    return corr(np.argsort(np.argsort(a)),np.argsort(np.argsort(b)))

def cluster_test(x,y):
    """Exact permutation of complete model blocks, preserving AB/AC/BC roles.
    Sensitivity test conditional on model exchangeability, not universal proof.
    """
    x=np.asarray(x); y=np.asarray(y)
    observed=sp(x.ravel(),y.ravel())
    values=[sp(x.ravel(),y[list(p)].ravel()) for p in itertools.permutations(range(len(x)))]
    count=sum(abs(v)>=abs(observed)-1e-12 for v in values)
    return dict(rho=observed, p_two_sided=count/len(values), permutations=len(values), extreme=count)

def load_phase(phase):
    paths=[p for p in (BASE/f'phase{phase}').glob('*/*.npz') if 'smoke' not in p.parts]
    return paths

def inventory():
    text=MEMO.read_text(encoding='utf-8-sig'); lines=text.splitlines()
    heads=list(re.finditer(r'^## Phase ([^:：\n]+)[:：].*$',text,re.M))
    records=[]
    for i,m in enumerate(heads):
        body=text[m.end():heads[i+1].start() if i+1<len(heads) else len(text)]
        refs=sorted(set(re.findall(r'`([^`\n]+)`',body)))
        paths=[]
        for ref in refs:
            if ref.startswith(('tests/','research/')) and not any(c in ref for c in '{}*|'):
                p=ROOT/ref
                paths.append(dict(reference=ref,exists=p.exists(),file=p.is_file()))
        records.append(dict(phase=m.group(1),line=text.count('\n',0,m.start())+1,
                            title=m.group(0),characters=len(body),paths=paths))
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),memo_sha256=sha(MEMO),
             bytes=MEMO.stat().st_size,lines=len(lines),sections=len(records),phases=records)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'inventory.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out

def continuum():
    p88=load_phase(3088)[0]
    with np.load(p88,allow_pickle=False) as z:
        assert np.isclose(float(z['RHO_T']), 0.9020979020979021)
    specs={
      '4B':(3079,'4B_',3082,'SP_R1_REPLAY'),
      'DS7B':(3081,'DS7B_',3082,'MIG'),
      '3B':(3085,'',3085,'MIG'),
      'GLM4':(3087,'',3087,'MIG'),
      '14B':(3093,'',3093,'MIG')}
    models={}; files={}
    for model,(phase,prefix,sphase,migkey) in specs.items():
        ps=load_phase(phase)
        p=next((p for p in ps if 'arbitration' in str(p)),ps[0])
        qs=load_phase(sphase)
        q=next((q for q in qs if 'arbitration' in str(q)),qs[0])
        with np.load(p,allow_pickle=False) as z, np.load(q,allow_pickle=False) as zz:
            top={f:float(zz['E3_TOP3_CS_'+prefix+f]) for f in 'ABC'}
            units=[]
            for a,b in ('AB','AC','BC'):
                units.append(dict(pair=a+b,s_lo=min(top[a],top[b]),T_med=float(np.median(z['T_'+a+b])),
                                  U_med=float(np.median(z['U_'+a+b]))))
            models[model]=units
        files[str(p.relative_to(ROOT))]=sha(p); files[str(q.relative_to(ROOT))]=sha(q)
    variants={}
    for name,names in [('n9',['4B','DS7B','3B']),('n12',['4B','DS7B','3B','GLM4']),('n15_14B',['4B','DS7B','3B','GLM4','14B'])]:
        x=np.array([[u['s_lo'] for u in models[m]] for m in names])
        y=np.array([[u['T_med'] for u in models[m]] for m in names])
        stats=cluster_test(x,y)
        stats['models']=names
        stats['legacy_rho']=legacy_sp(x.ravel(),y.ravel())
        stats['model_mean_test']=cluster_test(x.mean(1)[:,None],y.mean(1)[:,None])
        stats['model_centered_rank_correlation']=corr((average_rank(x.ravel()).reshape(x.shape)-average_rank(x.ravel()).reshape(x.shape).mean(1,keepdims=True)).ravel(),(average_rank(y.ravel()).reshape(y.shape)-average_rank(y.ravel()).reshape(y.shape).mean(1,keepdims=True)).ravel())
        stats['leave_one_model_out']={m:sp(np.delete(x,j,0).ravel(),np.delete(y,j,0).ravel()) for j,m in enumerate(names)}
        variants[name]=stats
    p89=load_phase(3089)[0]
    with np.load(p89,allow_pickle=False) as z:
        top={f:float(z['E3_TOP3_CS_'+f]) for f in 'ABC'}
        units=[dict(pair=a+b,s_lo=min(top[a],top[b]),T_med=float(np.median(z['T_'+a+b]))) for a,b in ('AB','AC','BC')]
    files[str(p89.relative_to(ROOT))]=sha(p89)
    models['GLM4_L38']=units
    for name,names in [('n12_L38',['4B','DS7B','3B','GLM4_L38']),('n15_14B_L38',['4B','DS7B','3B','GLM4_L38','14B'])]:
        x=np.array([[u['s_lo'] for u in models[m]] for m in names]);y=np.array([[u['T_med'] for u in models[m]] for m in names])
        variants[name]=cluster_test(x,y)
    out=dict(models=models,analyses=variants,input_sha256=files,
             caution='Block permutations assume model blocks exchangeable. Tiny, related model sample; descriptive sensitivity only. Fresh semantic/text holdout not added.')
    (OUT/'continuum_reanalysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(variants,ensure_ascii=False,indent=2))

def mathematical_checks():
    # Unit checks for audit implementation, independent of original code.
    assert np.array_equal(average_rank([0,2,3,2]),[0,1.5,3,1.5])
    assert sp([1,1,1],[2,3,4]) is None
    aa=np.array([0,2,3,2]);bb=np.array([3,2,1,0]);order=[2,0,3,1]
    assert abs(sp(aa,bb)-sp(aa[order],bb[order]))<1e-14
    assert cluster_test([[1],[2],[3]],[[1],[2],[3]])['p_two_sided']==2/6
    # Counterexample: nonempty-set coverage is submodular with positive mu_3.
    f=np.array([0.]+[1.]*7)
    mu=f.copy()
    for bit in range(3):
        for mask in range(8):
            if mask&(1<<bit): mu[mask]-=mu[mask^(1<<bit)]
    sms=[]
    for i,j in itertools.combinations(range(3),2):
        for mask in range(8):
            if not(mask & ((1<<i)|(1<<j))):
                sms.append(f[mask|(1<<i)|(1<<j)]-f[mask|(1<<i)]-f[mask|(1<<j)]+f[mask])
    assert max(sms)<=0 and mu[7]==1
    # SwiGLU directional derivative: synthetic numerical check, not model test.
    g=1.;u=2.;dg=.3;du=.4;eps=1e-6
    silu=lambda x: x/(1+np.exp(-x))
    sig=1/(1+np.exp(-g));der=sig+g*sig*(1-sig)
    fd=(silu(g+eps*dg)*(u+eps*du)-silu(g-eps*dg)*(u-eps*du))/(2*eps)
    correct=u*der*dg+silu(g)*du
    wrong=der*(u*dg+g*du)
    assert abs(fd-correct)<1e-8 and abs(fd-wrong)>.01
    # Recheck the empirical second differences independently of the false theorem.
    p75=load_phase(3075)[0]
    with np.load(p75,allow_pickle=False) as z:
        values=np.r_[0.,z['A_S']]
        measured_mu=z['MU'].copy()
    empirical_mu=values.copy()
    for bit in range(8):
        for mask in range(256):
            if mask&(1<<bit): empirical_mu[mask]-=empirical_mu[mask^(1<<bit)]
    empirical_seconds=[]; identity_error=0.
    for i,j in itertools.combinations(range(8),2):
        ij=(1<<i)|(1<<j)
        for base in range(256):
            if base&ij: continue
            d=values[base|ij]-values[base|(1<<i)]-values[base|(1<<j)]+values[base]
            empirical_seconds.append(d)
            reconstructed=sum(empirical_mu[u|ij] for u in range(256) if u&base==u)
            identity_error=max(identity_error,abs(d-reconstructed))
    assert identity_error<1e-12
    empirical_submod=dict(npz_sha256=sha(p75),mu_reconstruction_max_error=float(np.max(np.abs(empirical_mu-measured_mu))),
                         correct_identity_max_error=float(identity_error),conditional_second_difference_count=len(empirical_seconds),
                         positive_over_002=int(np.sum(np.array(empirical_seconds)>.02)),
                         maximum=float(max(empirical_seconds)),
                         caution='Finite frozen set-function violations remain real numerically. This does not establish biological/linguistic synergy or uncertainty significance.')
    p100=load_phase(3100)[0]
    with np.load(p100,allow_pickle=False) as z:
        names=[k for k in z.files if 'REL' in k or 'RES' in k]
        shares={}
        for side in ['4B','14B']:
            for family in 'ABC':
                a=z[f'ASHARE_{side}_{family}'];h=z[f'HSHARE_{side}_{family}']
                shares[f'{side}_{family}']=dict(median_a=float(np.median(a)),median_h=float(np.median(h)),
                    median_sum=float(np.median(a+h)),min_sum=float(np.min(a+h)),max_sum=float(np.max(a+h)))
        residuals={k:dict(median=float(np.median(z[k])),min=float(np.min(z[k])),max=float(np.max(z[k]))) for k in names if z[k].dtype.kind in 'fi' and z[k].size>1}
    out=dict(audit_self_checks='passed',
             mobius_counterexample=dict(f=f.tolist(),mu=mu.tolist(),conditional_second_differences=sms,submodular=True,mu3_positive=True),
             phase3075_empirical_submodularity=empirical_submod,
             swiglu_synthetic=dict(g=g,u=u,dg=dg,du=du,epsilon=eps,finite_difference=fd,correct_jvp=correct,phase3100_jvp=wrong,absolute_error=abs(fd-wrong)),
             phase3099_gate=dict(control_jaccard=.5647,required=3*.5647,maximum_possible=1.,feasible_at_observed_control=False),
             phase3100_saved_shares=shares,phase3100_saved_residuals=residuals,phase3100_npz_sha256=sha(p100))
    (OUT/'mathematical_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

def coverage():
    selected=[2750,2751,2752,2753,2755,2761,2778,2780,2786,2787,2802,2803,2807,2809,2814,2836,2853,2854,2860,2861,2913,2917,2929,2931,2961,2987,2996,3004,3035,3038,3039,3045,3054,3057,3067,3074,3075,3076,3086,3088,3091,3092,3093,3098,3099,3100]
    rows=[]
    for phase in selected:
        folder=BASE/f'phase{phase}'
        found=sorted(folder.glob('*/result.json'))
        for p in found:
            d=json.loads(p.read_text(encoding='utf-8-sig'))
            row=dict(phase=phase,path=str(p.relative_to(ROOT)),sha256=sha(p),verdict=d.get('verdict',d.get('status')))
            seal=p.parent/'seal.json'
            if seal.exists():
                sd=json.loads(seal.read_text(encoding='utf-8-sig'))
                expected=sd.get('result_sha256_8',sd.get('result_sha8'))
                row['seal_expected_result_sha8']=expected
                row['seal_result_matches']=(expected==row['sha256'][:8]) if expected else None
            rows.append(row)
    scripts={str(p.relative_to(ROOT)):sha(p) for n in selected for p in (ROOT/'tests/glm5').glob(f'phase{n}_*.py')}
    out=dict(selected_phases=selected,result_files=rows,script_sha256=scripts,note='Presence/hash check is not scientific validation. Not all raw model arrays or checkpoints audited.')
    (OUT/'artifact_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(dict(selected_phases=len(selected),results=len(rows),scripts=len(scripts),seal_matches=sum(r.get('seal_result_matches') is True for r in rows),seal_mismatches=[r for r in rows if r.get('seal_result_matches') is False]))

def digest(start,end):
    lines=MEMO.read_text(encoding='utf-8-sig').splitlines()
    active=False
    for n,line in enumerate(lines,1):
        m=re.match(r'^## Phase (\d+)',line)
        if m: active=start<=int(m.group(1))<=end
        if active and (line.startswith('## Phase') or any(w in line for w in ('硬伤','不足','局限','伪影','恒等','oracle','推广失败','判决：','判决**','未测','未验证','不支持','撤回','推翻'))):
            print(f'{n}: {line}')

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--digest',nargs=2,type=int);parser.add_argument('--continuum',action='store_true');parser.add_argument('--checks',action='store_true');parser.add_argument('--coverage',action='store_true')
    args=parser.parse_args()
    if args.digest: digest(*args.digest)
    else:
        result=inventory();print({k:v for k,v in result.items() if k!='phases'})
        if args.continuum: continuum()
        if args.checks: mathematical_checks()
        if args.coverage: coverage()
