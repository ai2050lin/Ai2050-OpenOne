"""Full-coordinate atlas aggregation and prospectively bounded native-state pairs."""
from collections import defaultdict,Counter
from rdc_query_common import *

OUT=BASE/'atlas'

def main():
    if (OUT/'result.json').exists():return
    for b in range(0,10000,2000):assert read(BASE/'capture/chunks'/f'{b:05d}_{b+2000:05d}.json')['all_passed']
    start=time.monotonic();rows=gzread(BASE/'material/natural.json.gz');probes=read(BASE/'probes/protocol.json')['probes']
    protocol={'source_ids':[r['sample_id'] for c in ['gum','ewt','cmrc'] for r in sorted([x for x in rows if x['cohort']==c and x['split']=='test'],key=lambda r:rank('pairs/'+r['sample_id']))[:32]],
      'pair_rule':'For each frozen center, samecohort, samecurrenttokenID, different source document, length difference<=max(16,25%centerlength). Need>=2 candidates. Near=min full2560coordinate cosine distance of actual unqueried postnorm; far=90thpercentile rank. Unavailable centers reported without replacement.',
      'distance':'No future query output enters pair selection. Centered cosine similarity is not exact state equality; finite test equivalence threshold mean symmetricKL<0.05 is descriptive only.',
      'full_coordinates':'All2560coordinates used, noPCA/TopK. Per-group raw/RMS fields use fixed coordinate order; aggregate profiles are descriptive, not compressed sufficient states.'}
    immutable(OUT/'pair_protocol.json',protocol)
    prefix=np.zeros((len(rows),2560),np.uint16);stats=np.zeros((len(rows),100,6),np.float64)
    groups=defaultdict(lambda:{'count':0,'sum':np.zeros((37,2560)),'square':np.zeros((37,2560)),'rms_sum':np.zeros((37,2560))})
    qgroups=defaultdict(lambda:{'count':0,'sum':np.zeros((100,2560)),'square':np.zeros((100,2560))})
    digest=hashlib.sha256();bits_checks=0;scalars_scanned=0
    for i,row in enumerate(rows):
        sid=row['sample_id'];file=BASE/'capture/fields'/f'{sid}.npz';rec=read(BASE/'capture/commits'/f'{sid}.json')
        assert sha(file)==rec['array_sha256'];digest.update(rec['array_sha256'].encode())
        with np.load(file) as z:
            prefix[i]=z['prefix_postnorm'];stats[i]=z['full_vocabulary_statistics'];h=unbits(z['prefix_layers']).astype(float);post=unbits(z['postnorm']).astype(float)
        assert np.isfinite(h).all() and np.isfinite(post).all() and np.isfinite(stats[i]).all()
        group=row['cohort']+'/'+row['split'];g=groups[group];g['count']+=1;g['sum']+=h;g['square']+=h*h;g['rms_sum']+=h/rms(h)
        qg=qgroups[group];qg['count']+=1;qg['sum']+=post;qg['square']+=post*post
        scalars_scanned+=len(row['prompt_ids'])*37*2560;bits_checks+=rec['source_cache_bits_checked']
        if (i+1)%500==0:print('QUERY_ATLAS_AGGREGATE',i+1,10000,round(time.monotonic()-start,1),flush=True)
    arrays={}
    for key,g in groups.items():
        k=key.replace('/','__');mean=g['sum']/g['count'];arrays[k+'_layer_mean']=mean;arrays[k+'_layer_std']=np.sqrt(np.maximum(g['square']/g['count']-mean*mean,0));arrays[k+'_layer_RMS_normalized_mean']=g['rms_sum']/g['count']
    for key,g in qgroups.items():
        k=key.replace('/','__');mean=g['sum']/g['count'];arrays[k+'_query_mean']=mean;arrays[k+'_query_std']=np.sqrt(np.maximum(g['square']/g['count']-mean*mean,0))
    npz(OUT/'all_coordinate_group_profiles.npz',**arrays)
    npz(OUT/'million_endpoint_statistics.npz',statistics=stats,unqueried_postnorm=prefix)
    summary=[]
    for cohort in ['all','gum','ewt','cmrc']:
      ix=[i for i,r in enumerate(rows) if cohort=='all' or r['cohort']==cohort]
      for lang in ['en','zh']:
       for family in sorted({p['family'] for p in probes}):
        qq=[i for i,p in enumerate(probes) if p['language']==lang and p['family']==family];value=stats[np.ix_(ix,qq,[1])][:,:,0].mean(1)
        summary.append({'cohort':cohort,'query_language':lang,'family':family,'prefixes':len(ix),'mean_KL_to_query_only':clustered(value,[rows[i]['source_group'] for i in ix])})
    mapping={r['sample_id']:i for i,r in enumerate(rows)};p=unbits(prefix).astype(float);p/=np.linalg.norm(p,axis=1,keepdims=True).clip(1e-12);pairs=[];missing=[]
    for sid in protocol['source_ids']:
        i=mapping[sid];r=rows[i];n=len(r['prompt_ids']);eligible=[j for j,s in enumerate(rows) if s['cohort']==r['cohort'] and s['prompt_ids'][-1]==r['prompt_ids'][-1]
          and s['source_group']!=r['source_group'] and abs(len(s['prompt_ids'])-n)<=max(16,.25*n)]
        if len(eligible)<2:missing.append({'center':sid,'eligible_count':len(eligible)});continue
        dist=1-p[eligible]@p[i];order=np.argsort(dist);near=int(order[0]);far=int(order[min(len(order)-1,int(.9*(len(order)-1)))])
        for name,j in [('near',near),('far',far)]:
            other=eligible[j];pairs.append({'center':sid,'other':rows[other]['sample_id'],'center_group':r['source_group'],'other_group':rows[other]['source_group'],
              'cohort':r['cohort'],'kind':name,'current_cosine_distance':float(dist[j]),'eligible_count':len(eligible),'same_current_token':True,
              'current_token_id':r['prompt_ids'][-1],'lengths':[n,len(rows[other]['prompt_ids'])]})
    save(OUT/'state_pairs.json',{'timestamp':stamp(),'pairs':pairs,'missing_centers':missing,'selection_only_current_state':True})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'natural_prefixes':10000,'query_endpoints':1000000,
      'source_documents':len({r['source_group'] for r in rows}),'full_vocabulary':151936,'all_layer_all_token_scalars_scanned_in_native_capture':scalars_scanned,
      'every_endpoint_coordinates_retained':2560,'whole_cache_bit_verified_prefixes':bits_checks,'summary':summary,
      'pair_centers':len(pairs)//2,'missing_pair_centers':len(missing),'ordered_capture_digest':digest.hexdigest(),'seconds':time.monotonic()-start,
      'scope':'One million fixed suffix endpoints, not one million independent sentences or successful natural continuations. Fullcoordinate persistence does not assert sufficient-state compression.'}
    save(OUT/'result.json',result);ledger('million_endpoint_full_coordinate_atlas',result['seconds']);print('QUERY_ATLAS_DONE',result['seconds'],flush=True)

if __name__=='__main__':main()
