"""Exact full cross-coordinate relational contrasts; no per-example outer-product archive."""
from collections import Counter,defaultdict
from rdc_relation_common import *
from phase2715_rdc_prefix_relations import annotation,PrefixRelations,DISTANCE_CUTS,LABELS

OUT=BASE/'relation_atlas'
CONTROLS=('distance','distance_pos','distance_pos_same_dependent_id')
PAIRS=((12,12),(12,23),(23,23))


def sentence_root(words,wid):
    seen=set()
    while words[wid]['head'] in words and wid not in seen:
        seen.add(wid);wid=words[wid]['head']
    return wid


def make_index(material):
    index=[];coverage=Counter()
    for row in material:
        _,_,at=annotation(row);words={w['id']:w for w in row['retrospective_ud']};roots={i:sentence_root(words,i) for i in at}
        linked={frozenset((w['id'],w['head'])) for w in words.values() if w['head'] in words}
        for w in words.values():
            rel=w['relation'].split(':')[0];d,h=w['id'],w['head']
            if rel not in RELATIONS or d not in at or h not in at:continue
            coverage[(rel,'aligned_edges')]+=1;distance=at[h]-at[d]
            if not distance:coverage[(rel,'same_native_token_excluded')]+=1;continue
            control=[]
            for a in at:
                for b in at:
                    if roots[a]!=roots[d] or roots[b]!=roots[d] or a==b or at[b]-at[a]!=distance or frozenset((a,b)) in linked:continue
                    control.append((a,b))
            groups={'distance':control,'distance_pos':[(a,b) for a,b in control if words[a]['upos']==w['upos'] and words[b]['upos']==words[h]['upos']]}
            groups['distance_pos_same_dependent_id']=[(a,b) for a,b in groups['distance_pos'] if row['prompt_ids'][at[a]]==row['prompt_ids'][at[d]]]
            item={'sample_id':row['sample_id'],'split':row['split'],'language':row['language'],'relation':rel,'dependent':at[d],'head':at[h],
              'signed_native_distance':distance,'dependent_word_id':d,'head_word_id':h,'controls':{k:[[at[a],at[b]] for a,b in v] for k,v in groups.items()}}
            index.append(item)
            for k,v in groups.items():coverage[(rel,k)]+=int(bool(v))
    save(OUT/'pair_index.json',index);save(OUT/'coverage.json',{r:{k:v for (rel,k),v in coverage.items() if rel==r} for r in RELATIONS})
    return index


def scales(material):
    with np.load(BASE/'main/all_token_moments.npz') as z:
        n=z['counts'][:2].sum();mu=z['sums'][:2].sum(0)/n;std=np.sqrt(np.maximum(z['squares'][:2].sum(0)/n-mu**2,0));std=np.maximum(std,1e-6)
    npz(OUT/'training_scales.npz',mean=mu,standard_deviation=std,training_tokens=np.array(n))
    return mu,std


def arrays(index,field,relation,control,split,layerpair,view,mu,std):
    entries=[e for e in index if e['relation']==relation and e['split']==split and e['controls'][control]];counts=Counter(e['sample_id'] for e in entries)
    a=[];b=[];weight=[];source=[];languages={}
    for e in entries:
        sid=e['sample_id'];x,y=(field[sid][l] for l in layerpair);languages[sid]=e['language']
        links=[(e['dependent'],e['head'],1.)]+[(i,j,-1/len(e['controls'][control])) for i,j in e['controls'][control]]
        for i,j,w in links:
            a.append(x[i]);b.append(y[j]);weight.append(w/counts[sid]);source.append(sid)
    if not a:return None
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    if view=='train_z':
        i,j=[(12,23).index(l) for l in layerpair];a=(a-mu[i])/std[i];b=(b-mu[j])/std[j]
    return a.astype(np.float32),b.astype(np.float32),np.array(weight,np.float32),np.array(source),languages


def matrix(arr):
    a,b,w,s,_=arr;g=len(set(s));result=np.empty((a.shape[1],b.shape[1]),np.float32)
    for start in range(0,len(result),128):result[start:start+128]=(a[:,start:start+128]*w[:,None]).T@b/g
    return result


def cosine(a,b):return float(np.sum(a.astype(float)*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def projection_interval(c,arr):
    a,b,w,s,languages=arr;norm=max(float(np.linalg.norm(c)),1e-30);score=np.zeros(len(a))
    for start in range(0,c.shape[0],128):score+=np.sum((a[:,start:start+128]@c[start:start+128])*b,axis=1,dtype=np.float64)
    ids=sorted(set(s));values=np.array([np.sum(score[s==sid]*w[s==sid])/norm for sid in ids]);rng=np.random.default_rng(2715)
    boots=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(1000)])
    return {'source_units':len(ids),'mean':float(values.mean()),'conditional_train_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
      'by_language':{lang:{'sources':sum(languages[sid]==lang for sid in ids),'mean':float(np.mean([v for sid,v in zip(ids,values) if languages[sid]==lang]))} for lang in sorted(set(languages.values()))},
      'small_source_warning':len(ids)<30},ids,values


def baseline_parser():
    """Separate artifact: frozen parser is never changed after seeing its test score."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    counts=np.zeros((2,8,17,17,len(LABELS)),np.float64)
    for r in read(PREVIOUS/'material_stratified.json'):
        if r['split']!='train':continue
        tags,edges,_=annotation(r);lang=int(r['language']=='zh')
        for t in range(1,len(tags)):
            for s in range(t):counts[lang,np.searchsorted(DISTANCE_CUTS,t-s),tags[s],tags[t],edges.get((s,t),0)]+=1
    distance=counts.sum((2,3))+.1;distance/=distance.sum(-1,keepdims=True)
    pos=counts.sum(1)+.1;pos/=pos.sum(-1,keepdims=True)
    npz(OUT/'parser_baselines.npz',distance=distance.astype(np.float32),pos=pos.astype(np.float32))
    parser=PrefixRelations();truth=[];scores={k:[] for k in ('distance_only','predicted_POS_only','distance_predicted_POS','distance_gold_POS_extra_information')};loss={k:[] for k in scores}
    for r in rows():
        if r['split']!='test':continue
        tags,edges,_=annotation(r);lp=parser.tags(r['prompt_ids'],r['language']);lang=int(r['language']=='zh')
        for t in range(1,len(tags)):
            bins=np.searchsorted(DISTANCE_CUTS,t-np.arange(t));y=np.array([edges.get((s,t),0) for s in range(t)]);truth.extend((y>0).tolist())
            prob={'distance_only':distance[lang,bins], 'predicted_POS_only':np.einsum('si,ijr,j->sr',lp[:t],pos[lang],lp[t]),
              'distance_predicted_POS':parser.weights(r['prompt_ids'][:t+1],r['language'])[0],
              'distance_gold_POS_extra_information':parser.tables[lang,bins,tags[:t],tags[t]]}
            for k,p in prob.items():scores[k].extend((1-p[:,0]).tolist());loss[k].extend((-np.log(np.maximum(p[np.arange(t),y],1e-30))).tolist())
    save(OUT/'parser_baseline_result.json',{'timestamp':stamp(),'test_sources':96,'pairs':len(truth),'baselines':{k:{'ROC_AUC':float(roc_auc_score(truth,v)),'average_precision':float(average_precision_score(truth,v)),'categorical_NLL':float(np.mean(loss[k]))} for k,v in scores.items()},
      'limits':'Gold POS variant uses full sentence labels and is explicitly extra information. Distance alone can create high AUC under severe class imbalance.'})


def main():
    OUT.mkdir(parents=True,exist_ok=True);protocol=OUT/'protocol.json'
    if not protocol.exists():save(protocol,{'timestamp':stamp(),'code':snapshot(Path(__file__)),'primary_relations':['conj','compound','nmod'],'overview_relations':list(RELATIONS),'layer_pairs':PAIRS,
      'views':['raw','train_z'],'controls':CONTROLS,'word_alignment':'Last overlapping native piece; dependent-to-head signed exact native distance; controls in same gold dependency-root sentence, excluding every gold linked word pair.',
      'statistic':'Equal-source mean of equal matched-edge means of dependent outer head minus mean matched unlinked control outer products; raw contrast and train-only z contrast, not a covariance or a causal path.',
      'full_coordinates':2560,'matrix_entries_each':2560**2,'storage':'All exact pair/control indices + bit-preserved H12/H23 + training scales. Any matrix or tile reconstructible without rank reduction; no per-example tensor archive.',
      'uncertainty':'Frozen train full matrix projected on each held-out source full contrast; 1000 source bootstrap conditional on training matrix. Exploratory multiple relations are not familywise significance tests.',
      'strict_control_policy':'No fallback. Report coverage including zero; only compute statistics where train and test have matched units.',
      'predeclared_before_first_new_hiddenstate_analysis':True})
    material=rows();index=make_index(material);mu,std=scales(material);field={}
    for r in material:
        if r['split'] not in ('train','test'):continue
        z=load_field(r);field[r['sample_id']]={l:unbits(z[f'h{l}']) for l in (12,23)}
    output=[];vectors={};start=time.monotonic()
    for rel in RELATIONS:
      for control in CONTROLS:
       for pair in PAIRS:
        for view in ('raw','train_z'):
            key=f'{rel}/{control}/H{pair[0]}_H{pair[1]}/{view}';cp=OUT/'commits'/f'{key.replace("/","_")}.json'
            if cp.exists():output.append(read(cp));continue
            train=arrays(index,field,rel,control,'train',pair,view,mu,std);test=arrays(index,field,rel,control,'test',pair,view,mu,std)
            if train is None or test is None:
                result={'key':key,'status':'no_matched_train_or_test','train_sources':0 if train is None else len(set(train[3])),'test_sources':0 if test is None else len(set(test[3]))}
            else:
                ct=matrix(train);ce=matrix(test);projection,ids,values=projection_interval(ct,test);diag=np.diag(ct);energy=float(np.sum(ct.astype(float)**2))
                result={'key':key,'status':'computed','train_sources':len(set(train[3])),'test_sources':len(set(test[3])),
                  'all_coordinate_train_test_cosine':cosine(ct,ce),'diagonal_train_test_cosine':cosine(diag,np.diag(ce)),
                  'train_Frobenius_norm':float(np.sqrt(energy)),'test_Frobenius_norm':float(np.linalg.norm(ce)),
                  'off_diagonal_squared_energy_fraction':1-float(np.sum(diag.astype(float)**2))/max(energy,1e-30),'test_frozen_train_projection':projection}
                npz(OUT/'profiles'/f'{key.replace("/","_")}.npz',train_diagonal=diag,test_diagonal=np.diag(ce),train_row_energy=np.sum(ct.astype(float)**2,1).astype(np.float32),test_row_energy=np.sum(ce.astype(float)**2,1).astype(np.float32),test_source_projection=values.astype(np.float32),source_ids=np.array(ids))
                del ct,ce
            save(cp,result);output.append(result)
        print('RELATION_ATLAS',rel,control,'completed',round(time.monotonic()-start,1),flush=True)
        assert time.monotonic()-start<3600;guard(16*1024**2)
    baseline_parser();save(OUT/'result.json',{'timestamp':stamp(),'entries':output,'computed':sum(r['status']=='computed' for r in output),'protocol_sha':sha(protocol),'pair_index_sha':sha(OUT/'pair_index.json'),'mechanism_claim':False})
    print('RELATION_ATLAS_COMPLETE',len(output),flush=True)


if __name__=='__main__':main()
