"""Typed full cross-coordinate contrasts with explicit coverage and document weights."""
from collections import Counter, defaultdict
from rdc_joint_common import *
from phase2719_rdc_joint_material import endpoint
from rdc_joint_prior_rules import rms_sources
from rdc_relation_common import RELATIONS, load_field as old_field, rows as prior_rows
from phase2715_rdc_relation_atlas import arrays as old_arrays, matrix, cosine

CONTROL = 'exact_distance_POS_noninitial'


def relation_type(value):
    if value.startswith('ud:'):
        return ':'.join(value.split(':')[:2])
    if value.startswith('discourse:'):
        return value.split('-')[0]
    return value


def make_index(material):
    index, coverage = [], Counter()
    for r in material:
        word_at = {}
        component = {s:i for i,s in enumerate(r['component_ids'])}
        for w in r['retrospective_ud']:
            p = endpoint(r,w)
            if p is not None:
                word_at[p] = w
        linked = {frozenset((g['dependent_token'],g['head_token'])) for g in r['retrospective_graph']}
        candidates = defaultdict(list)
        banded = defaultdict(list)
        for a,wa in word_at.items():
            for b,wb in word_at.items():
                if a == 0 or b == 0 or a == b or frozenset((a,b)) in linked:
                    continue
                cd = component[wb['sentence_id']]-component[wa['sentence_id']]
                key = (b-a,wa['upos'],wb['upos'],cd)
                candidates[key].append([a,b])
                banded[(int(np.sign(b-a)),int(np.searchsorted([2,4,8,16,32,64],abs(b-a))),*key[1:])].append([a,b])
        for g in r['retrospective_graph']:
            kind = relation_type(g['type'])
            if kind.startswith('ud:') and kind[3:] not in RELATIONS:
                continue
            a,b = g['dependent_token'],g['head_token']
            coverage[(kind,'annotated_edges')] += 1
            if a <= 0 or b <= 0 or a not in word_at or b not in word_at:
                continue
            wa,wb = word_at[a],word_at[b]
            key = (b-a,wa['upos'],wb['upos'],component[wb['sentence_id']]-component[wa['sentence_id']])
            exact = candidates[key]
            band = banded[(int(np.sign(b-a)),int(np.searchsorted([2,4,8,16,32,64],abs(b-a))),*key[1:])]
            same_id = [p for p in exact if r['prompt_ids'][p[0]] == r['prompt_ids'][a]]
            controls = {CONTROL:exact,'distance_band_POS_noninitial':band,'same_dependent_ID':same_id}
            for c,ps in controls.items():
                coverage[(kind,c)] += int(bool(ps))
            index.append({k:r[k] for k in ('sample_id','source_group','split','language')}|
                         {'relation':kind,'original_type':g['type'],'dependent':a,'head':b,'controls':controls})
    compressed_json(BASE/'relation_atlas/pair_index.json.gz',index)
    save(BASE/'relation_atlas/coverage.json',{k:{c:n for (r,c),n in coverage.items() if r==k} for k in sorted({r for r,c in coverage})})
    return index


def scales(store):
    sums, squares = np.zeros((2,2560),float),np.zeros((2,2560),float)
    n = 0
    for r in store.material:
        if r['split'] != 'train':
            continue
        for li,l in enumerate((12,23)):
            h = unbits(store[r]['h'+str(l)]).astype(float)
            sums[li] += h.sum(0); squares[li] += (h*h).sum(0)
        n += len(h)
    mu = sums/n
    std = np.maximum(np.sqrt(np.maximum(squares/n-mu**2,0)),1e-6)
    npz(BASE/'relation_atlas/training_scales.npz',mean=mu,standard_deviation=std,training_tokens=np.array(n))
    return mu,std


def arrays(index,store,kind,split,control,view,mu,std):
    entries = [e for e in index if e['relation']==kind and (split=='all' or e['split']==split) and e['controls'][control]]
    if not entries:
        return None
    counts = Counter(e['sample_id'] for e in entries)
    groups = defaultdict(set)
    for e in entries:
        groups[e['source_group']].add(e['sample_id'])
    a,b,w,s,langs = [],[],[],[],{}
    cached_sid = None
    for e in entries:
        sid = e['sample_id']; group = e['source_group']; langs[group] = e['language']
        if sid != cached_sid:
            x,y = (unbits(store[sid]['h'+str(l)]) for l in (12,23))
            if view == 'train_z':
                x,y = ((h-mu[li])/std[li] for li,h in enumerate((x,y)))
            elif view == 'source_RMS':
                x,y = rms_sources(x),rms_sources(y)
            cached_sid = sid
        ps = e['controls'][control]
        for i,j,v in [(e['dependent'],e['head'],1.)]+[(i,j,-1/len(ps)) for i,j in ps]:
            a.append(x[i]); b.append(y[j]); w.append(v/counts[sid]/len(groups[group])); s.append(group)
    return np.asarray(a,np.float32),np.asarray(b,np.float32),np.asarray(w,np.float32),np.array(s),langs


def projection(c,arr):
    a,b,w,groups,langs = arr
    score = np.zeros(len(a),float)
    for i in range(0,2560,128):
        score += np.sum((a[:,i:i+128]@c[i:i+128])*b,axis=1,dtype=float)
    norm = max(float(np.linalg.norm(c)),1e-30)
    ids = sorted(set(groups))
    values = np.array([np.sum(score[groups==g]*w[groups==g])/norm for g in ids])
    report = paired_summary(values,ids)
    report['by_language'] = {l:paired_summary([v for g,v in zip(ids,values) if langs[g]==l], [g for g in ids if langs[g]==l]) for l in sorted(set(langs.values()))}
    return report,ids,values


def old_confirmation(index,store):
    oldindex = []
    for e in read(PREVIOUS/'relation_atlas/pair_index.json'):
        ps = [[i,j] for i,j in e['controls']['distance_pos'] if i>0 and j>0]
        if e['dependent']>0 and e['head']>0 and ps:
            oldindex.append({**e,'controls':{'distance_pos_noninitial':ps}})
    fields = {}
    for r in prior_rows():
        if r['split']=='train':
            z = old_field(r)
            fields[r['sample_id']] = {l:unbits(z['h'+str(l)]) for l in (12,23)}
    with np.load(PREVIOUS/'relation_atlas/training_scales.npz') as z:
        mu,std = z['mean'],z['standard_deviation']
    result = []
    for rel in ('nmod','conj','compound'):
        tr = old_arrays(oldindex,fields,rel,'distance_pos_noninitial','train',(12,23),'train_z',mu,std)
        ct = matrix(tr)
        with np.load(PREVIOUS/'boundary_relations'/f'{rel}_profiles.npz') as z:
            assert np.array_equal(np.diag(ct),z['train_diagonal']), 'Old training matrix reconstruction differs'
        te = arrays(index,store,'ud:'+rel,'all',CONTROL,'train_z',mu,std)
        if te is None:
            result.append({'relation':rel,'status':'no_exact_match'})
            continue
        ce = matrix(te)
        p,ids,values = projection(ct,te)
        result.append({'relation':rel,'status':'computed','new_groups':len(ids),'cosine_old_train_new':cosine(ct,ce),'frozen_old_train_projection':p})
        npz(BASE/'relation_atlas/prior'/f'{rel}.npz',old_train_diagonal=np.diag(ct),new_diagonal=np.diag(ce),new_row_energy=np.sum(ce.astype(float)**2,1).astype(np.float32),projection=values.astype(np.float32),group_ids=np.array(ids))
        print('JOINT_OLD_RELATION',result[-1],flush=True)
    save(BASE/'relation_atlas/prior_confirmation.json',{'timestamp':stamp(),'old_diagonal_replay_all_equal':True,'entries':result,
        'scope':'Original train matrix/scales unchanged, independent new main512 evaluated together with equal document then window then edge weights. New complement excludes all annotated typed pairs, a stricter set than old UD-only complement. Not an exact-identical-control distribution or semantic causal test.'})


def relation_atlas(store):
    start = time.monotonic()
    out = BASE/'relation_atlas'
    snapshot(Path(__file__))
    index = make_index(store.material)
    mu,std = scales(store)
    old_confirmation(index,store)
    result = []
    for kind in sorted({e['relation'] for e in index}):
        controls = [CONTROL] if kind.startswith('ud:') else [CONTROL,'distance_band_POS_noninitial']
        for control in controls:
            units = {s:len({e['source_group'] for e in index if e['relation']==kind and e['split']==s and e['controls'][control]}) for s in ('train','test')}
            # Coverage-only choice, prior to any matrix cosine/projection. Sparse cases remain disclosed.
            if units['train']<20 or units['test']<10:
                result.append({'relation':kind,'control':control,'status':'insufficient_prespecified_group_coverage','groups':units})
                continue
            for view in ('raw','train_z','source_RMS'):
                key = kind.replace(':','_')+'_'+control+'_'+view
                tr = arrays(index,store,kind,'train',control,view,mu,std)
                te = arrays(index,store,kind,'test',control,view,mu,std)
                ct,ce = matrix(tr),matrix(te)
                p,ids,values = projection(ct,te)
                energy = float(np.sum(ct.astype(float)**2))
                report = {'relation':kind,'control':control,'view':view,'status':'computed','groups':units,
                    'all_coordinate_train_test_cosine':cosine(ct,ce),'train_Frobenius_norm':float(np.sqrt(energy)),
                    'offdiagonal_squared_energy_fraction':1-float(np.sum(np.diag(ct).astype(float)**2))/max(energy,1e-30),
                    'frozen_train_projection':p,'profile':f'profiles/{key}.npz'}
                npz(out/'profiles'/f'{key}.npz',train_diagonal=np.diag(ct),test_diagonal=np.diag(ce),
                    train_row_energy=np.sum(ct.astype(float)**2,1).astype(np.float32),test_row_energy=np.sum(ce.astype(float)**2,1).astype(np.float32),
                    test_group_projection=values.astype(np.float32),group_ids=np.array(ids))
                result.append(report)
                del ct,ce,tr,te
            print('JOINT_NEW_RELATION',kind,control,units,'elapsed',round(time.monotonic()-start,1),flush=True)
            guard(1024**2)
            assert time.monotonic()-start<3600
    save(out/'result.json',{'timestamp':stamp(),'entries':result,'coordinate_pairs_per_matrix':2560**2,
        'layers':[12,23],'primary':CONTROL,'views':['raw','train_z','source_RMS'],
        'equation':'C=mean_document mean_matched_window mean_matched_edge (x_dep outer y_head - mean_control x_a outer y_b). All2560 squared pairs.',
        'weighting':'Equal declared document/content group; equal eligible window within document; equal matched edge within window. GSD sentence IDs are not verified original article IDs.',
        'controls':'Both endpoints noninitial; equal POS pair and signed native-token distance and signed sentence-component distance. Band controls separately declared: distance bins2,4,8,16,32,64, never fallback. Exclude every direct annotated graph pair.',
        'uncertainty':'2000 document cluster bootstrap conditional on training matrix; exploratory views/relations without familywise significance control.',
        'retention':'Full coordinate diagonal and row-energy profiles retained. Complete matrix needs native field replay for nonfixture sources, stored SHA verification, exact pair index and train scales; not presently a fully stored matrix.',
        'limits':'Unannotated complement is not semantically unrelated. Endpoint lexical identity control has separate coverage only; train/test topic/genre confounds remain. GUM entities/discourse are retrospective labels, not proved reasoning mechanism.'})
    print('JOINT_RELATION_COMPLETE',len(result),'computed',sum(r['status']=='computed' for r in result),flush=True)
