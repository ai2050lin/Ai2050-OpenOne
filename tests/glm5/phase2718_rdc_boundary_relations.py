"""Exact noninitial-position stratum of the three previously primary full relation matrices."""
from rdc_relation_common import *
from phase2715_rdc_relation_atlas import arrays,matrix,cosine,projection_interval


def main():
    out=BASE/'boundary_relations';guard(400*1024)
    if (out/'result.json').exists():return
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
      'question':'Are earlier distance/POS relation contrasts wholly dependent on edges touching the exceptional first-token state?',
      'scope':'Posthoc position-conditioned reanalysis. nmod/conj/compound were original primary types; retain all2560x2560 H12/H23 coordinate pairs and frozen training z scales.',
      'filter':'Original edge dependent>0 and head>0; among original distance_pos controls retain only i>0 and j>0; omit edge if none remain. Reweight edges and source units after this explicit filter, never fall back.',
      'recipe':'Original pair_index.json plus this deterministic positional condition and original fields/scales completely reconstruct each matrix. No duplicate per-example tensor archive.',
      'uncertainty':'Same conditional-training source bootstrap as2715. Reused test data; no new independent semantic confirmation or multiple-comparison correction.',
      'original_index_sha':sha(BASE/'relation_atlas/pair_index.json')})
    index=[]
    for e in read(BASE/'relation_atlas/pair_index.json'):
        pairs=[(i,j) for i,j in e['controls']['distance_pos'] if i>0 and j>0]
        if e['dependent']>0 and e['head']>0 and pairs:index.append({**e,'controls':{'distance_pos_noninitial':pairs}})
    field={}
    for r in rows():
        if r['split'] in ('train','test'):
            z=load_field(r);field[r['sample_id']]={l:unbits(z[f'h{l}']) for l in (12,23)}
    with np.load(BASE/'relation_atlas/training_scales.npz') as z:mu=z['mean'];std=z['standard_deviation']
    result=[]
    for rel in ('nmod','conj','compound'):
        tr=arrays(index,field,rel,'distance_pos_noninitial','train',(12,23),'train_z',mu,std);te=arrays(index,field,rel,'distance_pos_noninitial','test',(12,23),'train_z',mu,std)
        assert tr is not None and te is not None;ct=matrix(tr);ce=matrix(te);projection,ids,values=projection_interval(ct,te)
        r={'key':rel+'/distance_pos_noninitial/H12_H23/train_z','relation':rel,'train_sources':len(set(tr[3])),'test_sources':len(set(te[3])),
          'all_coordinate_train_test_cosine':cosine(ct,ce),'test_frozen_train_projection':projection,'condition':'No endpoint at token0; exact distance/POS control remains'}
        npz(out/f'{rel}_profiles.npz',train_diagonal=np.diag(ct),test_diagonal=np.diag(ce),train_row_energy=np.sum(ct.astype(float)**2,1).astype(np.float32),test_row_energy=np.sum(ce.astype(float)**2,1).astype(np.float32),test_source_projection=values,source_ids=np.array(ids));result.append(r);print('NONINITIAL_RELATION',r,flush=True)
    save(out/'result.json',{'timestamp':stamp(),'entries':result,'coordinate_pairs_per_matrix':2560**2,'full_matrix_reconstruction':True,'limits':'Conditional patterns are not causal semantic gears; removing a special position does not remove lexical, topic, sentence or annotation confounds.'})


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
