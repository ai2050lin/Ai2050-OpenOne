"""All-coordinate observational relation spectra; no low-rank or top-k definition of structure."""
from collections import Counter,defaultdict
from rdc_prefix_common import *

OUT=CAMPAIGN/'atlas'
LAYERS=(0,12,24,36)


def covariance(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    am=a.mean(0);bm=b.mean(0)
    c=(a-am).T@(b-bm)/len(a)
    sa=a.std(0);sb=b.std(0)
    return c.astype(np.float32),am.astype(np.float32),bm.astype(np.float32),sa.astype(np.float32),sb.astype(np.float32)


def span_token(row,span):
    if span is None:return None
    candidates=[i for i,(a,b) in enumerate(row['token_offsets']) if a<span[1] and b>span[0]]
    return candidates[-1] if candidates else None


def main():
    rows=read(CAMPAIGN/'material_stratified.json');source=CAMPAIGN/'qwen4'
    assert read(source/'status.json')['state']=='captured'
    immutable(OUT/'protocol.json',{'source_sha':sha(Path(__file__)),'material_sha':sha(CAMPAIGN/'material_stratified.json'),
      'question':'Shared natural-prefix full-coordinate response atlas, not a unique semantic mechanism.',
      'covariances':'Every coordinate pair, no truncation. Adjacent observed token states at H0/H12/H24/H36 on all train anchors; UD nsubj/obj role pairs at H12/H24 on 16 complete panels only.',
      'conditioning':'All 21 prefix-cue features with counts/means/effects. Gold UD classes are retrospective descriptions, not predictive graph inputs.',
      'replication':'Train/test condition-difference profiles compared without coordinate selection; small rare groups explicitly reported.',
      'identity':'H0 equality for repeated token IDs versus context-conditioned H12/H24/H36 spread. Ordered token/word spans remain in material.',
      'limits':['Natural corpus is modest and observational, not controlled semantic variation.','Token states within a sentence are correlated; sentences/documents are the grouping units.',
        'A covariance matrix is a second-moment statistic, not geometric adjacency, a manifold, or a causal path.','Full field panel has only 16 sentences and is not independent of main corpus.']})
    n=len(rows)*2;h=np.empty((n,37,2560),np.float32);nexts={l:np.empty((n,2560),np.float32) for l in LAYERS}
    metadata=[];features=[];byid=defaultdict(list);read_identity={};identity_mismatch=[];panel_pairs=defaultdict(list)
    for i,r in enumerate(rows):
        with np.load(source/f'fields/{r["sample_id"]}.npz') as z:
            values=unbits(z['h']);h[2*i:2*i+2]=values[:,[0,3]].transpose(1,0,2)
            for l in LAYERS:nexts[l][2*i:2*i+2]=values[l,[1,4]]
            for j,p in enumerate(r['positions']):
                tid=r['prompt_ids'][p];current=z['h'][0,j]
                if tid in read_identity:
                    if not np.array_equal(current,read_identity[tid]):identity_mismatch.append([r['sample_id'],p,tid])
                else:read_identity[tid]=current.copy()
        for j,k in enumerate((0,3)):
            p=r['positions'][k];graph=r['anchor_graphs'][k];tid=r['prompt_ids'][p]
            word=next((w for w in r['retrospective_ud'] if w['char_span'] and w['char_span'][0]<r['token_offsets'][p][1]<=w['char_span'][1]),None)
            meta={k:r[k] for k in ('sample_id','source_group','source_sentence_id','split','language','genre')}
            meta.update(anchor=j,position=p,token_id=tid,token=r['tokens'][p],text=r['text'],
              char_span=r['token_offsets'][p],retrospective_upos=word['upos'] if word else 'UNALIGNED',graph=graph)
            metadata.append(meta);features.append(graph['features']);byid[tid].append(2*i+j)
        if r['full_panel']:
            with np.load(source/f'full_panels/{r["sample_id"]}.npz') as z:panel=unbits(z['h'])
            words={w['id']:w for w in r['retrospective_ud']}
            for w in words.values():
                relation=w['relation'].split(':')[0]
                if relation not in ('nsubj','obj') or w['head'] not in words:continue
                a=span_token(r,w['char_span']);b=span_token(r,words[w['head']]['char_span'])
                if a is None or b is None:continue
                pair_id={'sample_id':r['sample_id'],'dependent_word':w['id'],'head_word':w['head'],'token_positions':[a,b],
                  'label':relation,'annotation_scope':'full-sentence retrospective UD; no causal role assignment claim'}
                for l in (12,24):panel_pairs[(relation,l)].append((panel[l,a].copy(),panel[l,b].copy(),pair_id))
        if i%64==63:print('ATLAS_LOAD',i+1,len(rows),flush=True)
    assert not identity_mismatch
    g=np.asarray(features,np.float64);train=np.array([i for i,r in enumerate(metadata) if r['split']=='train'])
    test=np.array([i for i,r in enumerate(metadata) if r['split']=='test'])
    npz(OUT/'coordinate_profiles.npz',layer_mean=h[train].mean(0),layer_std=h[train].std(0),graph_features=g.astype(np.float32))
    matrix_records=[]
    for l in LAYERS:
        c,am,bm,sa,sb=covariance(h[train,l],nexts[l][train]);fp=OUT/f'matrices/adjacent_H{l}.npz'
        npz(fp,covariance=c,source_mean=am,target_mean=bm,source_std=sa,target_std=sb)
        matrix_records.append({'id':f'adjacent_H{l}','kind':'next_observed_token_state','layer':l,'n':len(train),'source_indices':train.tolist(),
          'path':str(fp.relative_to(CAMPAIGN)),'shape':list(c.shape),'full_coordinate_pairs':True,'split':'train'})
        print('MATRIX',l,len(train),flush=True)
    for (relation,l),pairs in panel_pairs.items():
        a=np.stack([p[0] for p in pairs]);b=np.stack([p[1] for p in pairs])
        c,am,bm,sa,sb=covariance(a,b);mid=f'ud_{relation}_H{l}';fp=OUT/f'matrices/{mid}.npz'
        npz(fp,covariance=c,source_mean=am,target_mean=bm,source_std=sa,target_std=sb)
        matrix_records.append({'id':mid,'kind':'retrospective_UD_word_role','layer':l,'n':len(a),
          'path':str(fp.relative_to(CAMPAIGN)),'shape':list(c.shape),'pairs':[p[2] for p in pairs],'full_coordinate_pairs':True,'scope':'16 full panels only'})
    effects=[];effect_arrays={}
    for j,name in enumerate(GRAPH_NAMES[:18]):
        masks={s:(g[ii,j]>0) for s,ii in [('train',train),('test',test)]}
        record={'feature':name,'counts':{s:{'present':int(m.sum()),'absent':int((~m).sum())} for s,m in masks.items()}}
        deltas={}
        for s,ii in [('train',train),('test',test)]:
            mask=masks[s]
            if mask.any() and (~mask).any():
                deltas[s]=h[ii[mask]].mean(0,dtype=np.float64)-h[ii[~mask]].mean(0,dtype=np.float64)
                effect_arrays[f'{name}_{s}']=deltas[s].astype(np.float32)
        if len(deltas)==2:
            a=deltas['train'];b=deltas['test'];den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
            record['all_coordinate_profile_cosine_by_layer']=np.divide((a*b).sum(1),den,out=np.zeros(37),where=den>0).tolist()
            record['same_sign_fraction_by_layer']=((a>0)==(b>0)).mean(1).tolist()
        effects.append(record)
    npz(OUT/'conditional_profiles.npz',**effect_arrays)
    repeat_ids=[tid for tid,ii in byid.items() if len(ii)>=2];identity=[]
    for tid in repeat_ids:
        ii=byid[tid]
        identity.append({'token_id':tid,'token':metadata[ii[0]]['token'],'observations':len(ii),'indices':ii,
          'distinct_source_units':len({metadata[k]['sample_id'] for k in ii}),
          'coordinate_mse_about_identity_mean':{f'H{l}':float(np.var(h[ii,l].astype(np.float64),axis=0).mean()) for l in LAYERS}})
    save(OUT/'identity_groups.json',identity);save(OUT/'anchors.json',metadata);save(OUT/'matrix_index.json',matrix_records)
    # Complete alphabet of node/edge types with evidence levels, not an assertion that all semantic types are known.
    graph_schema={'nodes':['source_unit','token_id','word_span','observed_cue_event','prefix','layer_state','native_coordinate','native_mlp_unit','scalar_parameter','output_distribution'],
      'edges':[{'type':t,'evidence':e} for t,e in [('tokenizes','tokenizer offsets'),('precedes','observed text order'),('annotated_dependency','retrospective UD gold'),
        ('cue_present_in','prefix-only lexical match'),('responds_at','observed full coordinates'),('adjacent_state_covariance','observational second moment'),
        ('native_parameter_connection','checkpoint and architecture'),('predicts','held-out model required; pending phase2712')]],
      'unknown':['full word senses','resolved semantic role bindings','universal invariants','brain correspondence'],
      'row_links':'anchors.json -> qwen4/rows and fields -> material_stratified.json and source sentence IDs; legacy_index.json retains previous controlled references.'}
    save(OUT/'graph_schema.json',graph_schema)
    raw_commits=[read(p) for p in (source/'commits').glob('*.json')]
    result={'timestamp':stamp(),'status':'completed_observational_atlas','natural_units':len(rows),'anchors':n,
      'all_observed_token_positions':sum(c['observed_token_count'] for c in raw_commits),
      'all_observed_H_scalars':sum(np.prod(c['observed_allH_shape']).item() for c in raw_commits),
      'retained_anchor_states':len(rows)*6,'full_panels':sum(r['full_panel'] for r in rows),'coordinate_width':2560,'H_checkpoints':37,
      'matrix_count':len(matrix_records),'all_coordinate_pairs_per_matrix':2560**2,'matrices':matrix_records,
      'prefix_feature_names':GRAPH_NAMES,'condition_replication':effects,'repeated_anchor_token_ids':len(repeat_ids),
      'all_six_anchor_unique_token_ids':len(read_identity),'embedding_identity_mismatches':len(identity_mismatch),
      'representative_repeated_tokens':sorted(identity,key=lambda x:-x['observations'])[:12],
      'retrospective_pos_counts':dict(Counter(r['retrospective_upos'] for r in metadata)),
      'interpretation':'Shared conditional response profiles and coordinate-pair statistics are candidate observations. Context dependence does not imply the native coordinate system is arbitrary, and H0 identity does not explain lexical semantics.',
      'new_mathematical_theorem':False,'mechanism_closed':False,'artifacts':['matrix_index.json','anchors.json','coordinate_profiles.npz','conditional_profiles.npz','identity_groups.json','graph_schema.json']}
    save(OUT/'result.json',result);status('atlas',state='complete',natural_units=len(rows),anchors=n)
    guard();print('ATLAS_COMPLETE',len(rows),n,len(matrix_records),flush=True)


if __name__=='__main__':main()
