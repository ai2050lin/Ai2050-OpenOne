"""Independent cache-exclusion, full Gram and per-head probability checks for N."""
from rdc_conditional_common import *
OUT=CAMPAIGN/'n_cached_attention';M=CAMPAIGN/'m_order'


def main():
    rows=read(OUT/'selected_rows.json');checks=[];positions=[]
    with np.load(OUT/'input_grams.npz') as z:tr=z['train'];grams={name:z[name] for name in ('past_K','past_V')};scales={name:float(z[name+'_scale']) for name in ('K','V')}
    pairs=[(int(tr[0]),int(tr[-1])),(0,len(rows)-1),(len(rows)//2,len(rows)//2)]
    for i,j in pairs:
        aa=[]
        for index in (i,j):
            r=rows[index];assert r['generation_step']>=1
            with np.load(M/f'fields/{r["sample_id"]}.npz') as z:aa.append({name:unbits(z['L23_'+name]) for name in ('k','v')})
        for name in ('k','v'):
            left,right=aa[0][name][:,:-1].astype(np.float64),aa[1][name][:,:-1].astype(np.float64);length=min(left.shape[1],right.shape[1])
            expected=float(np.sum(left[:,:length]*right[:,:length]))/scales[name.upper()]**2;observed=float(grams['past_'+name.upper()][i,j])
            assert abs(expected-observed)<1e-10
            checks.append({'rows':[i,j],'field':name,'full_past_dot':expected,'saved_gram':observed,'current_token_excluded':True})
    for prefix in read(M/'prefixes.json'):
        b=read(M/f'behavior/{prefix["sample_id"]}.json');candidates=b['boundary_candidates']
        for old,new in zip(candidates,candidates[1:]):
            ro,rn=read(M/f'steps/{old}.json'),read(M/f'steps/{new}.json')
            if rn['generation_step']!=ro['generation_step']+1:continue
            with np.load(M/f'fields/{old}.npz') as a,np.load(M/f'fields/{new}.npz') as z:
                for l in (11,23,35):
                    for part in ('k','v'):
                        before=a[f'L{l}_{part}'];after=z[f'L{l}_{part}'][:,:before.shape[1]];assert np.array_equal(before,after)
            positions.append({'old':old,'new':new,'all_old_cache_KV_bitwise_unchanged_at_three_layers':True})
    probabilities=[]
    for path in (OUT/'predictions').glob('*.npz'):
        with np.load(path) as z:
            if 'probabilities' not in z:continue
            p=z['probabilities'];te=z['test'];assert np.all(p>=0)
            error=float(np.max(np.abs(p.sum(-1,dtype=np.float64)-1)));assert error<1e-6
            for k,i in enumerate(te):assert np.all(p[k,:,rows[i]['query_position']+1:]==0)
            probabilities.append({'file':path.name,'states':len(te),'all32heads_all_sources':True,'max_abs_sum_error':error})
    save(OUT/'cache_input_audit.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),'passed':True,'independent_full_coordinate_dot_checks':checks,
      'consecutive_candidate_cache_pairs':positions,'forecast_probabilities':probabilities,'limits':'Read-only provenance and arithmetic checks, not a causal or whole-program proof.'})
    print('N_CACHE_INPUT_AUDIT',len(checks),len(positions),len(probabilities),'PASS',flush=True)


if __name__=='__main__':main()
