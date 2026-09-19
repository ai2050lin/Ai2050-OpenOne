"""All million retained native endpoints, with window/document weighting."""
from collections import Counter
from rdc_construction_common import *
from phase2745_rdc_construction_analysis import preflight

OUT=BASE/'phase2746/natural_interactions'


def freeze():
    path=OUT/'protocol.json'
    if path.exists():return read(path)
    value={'timestamp':stamp(),'source':snapshot(__file__),'phase':2746,
        'question':'Does the observed non-additive prefix/query organization persist on the entire existing natural-language grid, beyond five constructed families?',
        'material':'Reuse all10000windows/2777source documents and all100queries from2740. No new native endpoint or independent confirmation claimed.',
        'fields':'All2560postnorm coordinates for1million endpoints; detailed576windows additionally allnative H12/H24/H36 endpoints. Other query-layer coverage is not implied.',
        'views':['raw','whole_vector_RMS'],'weights':['equal_window','equal_document'],
        'strata':'Pooled and every registered source cohort; source-side labels are descriptive metadata, not internal semantic modules.',
        'math':'For each weighted balanced grid, mu=sum_p w_p mean_q Y_pq; A_p=mean_qY_pq-mu; B_q=sum_pw_pY_pq-mu; C_pq=Y_pq-mu-A_p-B_q. All-coordinate energies retain both main effects and interaction.',
        'numerical_checks':'Independent synthetic identity; verify all source archive digests; exact explicit residual energy on predeclared first/last8 rows per grid and full-coordinate weighted centering.',
        'scope':'Descriptive after previous grid exposure; no new prediction, causal identification, model-size claim or new mathematical theorem.',
        'retention':'Reuse original endpoint arrays; save every row mean, all weighted query/grand means, every coordinate energy and the complete source index. No duplicate raw grid, no coordinate pruning.',
        'material_sha256':sha(OLD/'material/natural.json.gz'),'query_sha256':sha(OLD/'probes/protocol.json')}
    immutable(path,value);return value


def analyze(rows,probes,layer,out):
    n,d,q=len(rows),2560,100
    cohorts=['all']+sorted({r['cohort'] for r in rows})
    strata=[]
    for cohort in cohorts:
        selected=[i for i,r in enumerate(rows) if cohort=='all' or r['cohort']==cohort]
        counts=Counter(rows[i]['source_group'] for i in selected)
        for weighting in ['equal_window','equal_document']:
            weights=np.zeros(n)
            weights[selected]=1 if weighting=='equal_window' else [1/counts[rows[i]['source_group']] for i in selected]
            weights/=weights.sum()
            strata.append({'cohort':cohort,'weighting':weighting,'indices':selected,'weights':weights,'documents':len(counts)})
    means=np.zeros((2,n,d))
    query_sums=np.zeros((len(strata),2,q,d))
    square_sums=np.zeros((len(strata),2,d))
    row_squares=np.zeros_like(square_sums)
    fixtures={}
    for p,row in enumerate(rows):
        file=OLD/'capture/fields'/(row['sample_id']+'.npz')
        assert sha(file)==read(OLD/'capture/commits'/(row['sample_id']+'.json'))['array_sha256']
        with np.load(file) as z:
            y=unbits(z['postnorm'] if layer=='postnorm' else z['query_H12_H24_rawH36'][:,[12,24,36].index(layer)]).astype(float)
        assert y.shape==(q,d) and np.isfinite(y).all()
        values=[y,y/np.sqrt(np.mean(y*y,-1,keepdims=True)).clip(1e-12)]
        if p<8 or p>=n-8:fixtures[p]=np.stack(values)
        for v,y in enumerate(values):
            mean=y.mean(0);square=(y*y).mean(0)
            means[v,p]=mean
            for s,stratum in enumerate(strata):
                w=stratum['weights'][p]
                if w:
                    query_sums[s,v]+=w*y
                    square_sums[s,v]+=w*square
                    row_squares[s,v]+=w*mean*mean
        if (p+1)%200==0:
            save(out/'progress.json',{'timestamp':stamp(),'rows':p+1,'total':n,'layer':layer})
            print('NATURAL_INTERACTIONS',layer,p+1,n,flush=True)
    grand=query_sums.mean(2)
    energy=np.empty((len(strata),2,4,d))
    reports=[];fixture_checks=[]
    for s,stratum in enumerate(strata):
        for v,view in enumerate(['raw','whole_vector_RMS']):
            mu=grand[s,v];weights=stratum['weights']
            total=square_sums[s,v]-mu*mu
            prefix=row_squares[s,v]-mu*mu
            query=((query_sums[s,v]-mu)**2).mean(0)
            interaction=total-prefix-query
            values=np.stack([total,prefix,query,interaction])
            tolerance=max(1e-10,float(abs(total).max())*1e-8)
            assert values.min()>-tolerance,(layer,stratum['cohort'],view,values.min(),tolerance)
            energy[s,v]=values
            centered=weights@(means[v]-mu)
            assert abs(centered).max()<1e-9
            reports.append({'layer':layer,'cohort':stratum['cohort'],'weighting':stratum['weighting'],'view':view,
                'windows':len(stratum['indices']),'documents':stratum['documents'],
                'total':float(total.mean()),'prefix':float(prefix.mean()),'query':float(query.mean()),
                'interaction':float(interaction.mean()),'interaction_fraction':float(interaction.mean()/total.mean()),
                'minimum_energy_roundoff':float(values.min()),'prefix_centering_max_error':float(abs(centered).max())})
            for p,y in fixtures.items():
                if weights[p]:
                    c=y[v]-means[v,p]-query_sums[s,v]+mu
                    direct=(c*c).mean(0)
                    expanded=(y[v]*y[v]).mean(0)-means[v,p]**2+((query_sums[s,v]-mu)**2).mean(0)-2*(y[v]*(query_sums[s,v]-mu)).mean(0)
                    error=float(abs(direct-expanded).max())
                    assert error<max(1e-9,float(direct.max())*1e-9)
                    fixture_checks.append({'row':p,'stratum':s,'view':v,'all_coordinate_energy_expansion_max_error':error})
    path=out/('layer_'+str(layer)+'.npz')
    guard(means.nbytes+query_sums.nbytes+energy.nbytes)
    npz(path,row_means=means,weighted_query_means=query_sums,grand_means=grand,
        all_coordinate_energies=energy,weights=np.stack([s['weights'] for s in strata]))
    save(out/('layer_'+str(layer)+'.json'),{'timestamp':stamp(),'all_passed':True,'layer':layer,'archive_sha256':sha(path),
        'row_index':[{'sample_id':r['sample_id'],'source_group':r['source_group'],'cohort':r['cohort'],'split':r['split']} for r in rows],
        'strata':[{'cohort':s['cohort'],'weighting':s['weighting'],'documents':s['documents']} for s in strata],
        'reports':reports,'fixture_checks':fixture_checks,'scalars_in_original_grid':n*q*d})
    return reports


def main():
    protocol=freeze();start=time.monotonic();checks=preflight()
    if (OUT/'result.json').exists():return
    rows=gzread(OLD/'material/natural.json.gz');probes=read(OLD/'probes/protocol.json')['probes']
    assert len(rows)==10000 and len({r['source_group'] for r in rows})==2777
    detail=set(read(OLD/'material/protocol.json')['detailed_prefix_ids'])
    reports=[]
    for layer,selected in [('postnorm',rows)]+[(layer,[r for r in rows if r['sample_id'] in detail]) for layer in [12,24,36]]:
        path=OUT/('layer_'+str(layer)+'.json')
        if path.exists():
            result=read(path);assert sha(path.with_suffix('.npz'))==result['archive_sha256'];reports+=result['reports']
        else:reports+=analyze(selected,probes,layer,OUT)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'protocol':protocol,'reports':reports,
        'numerical_preflight':checks,'seconds':time.monotonic()-start,'new_native_endpoints':0,
        'coverage':'1millionpostnormendpoints;57600endpoints at each ofH12/H24/H36; allnative coordinates.',
        'scope':protocol['scope']}
    save(OUT/'result.json',result);ledger('phase2746_natural_interactions',result['seconds'])
    print('NATURAL_INTERACTIONS_DONE',result['seconds'],flush=True)


if __name__=='__main__':main()
