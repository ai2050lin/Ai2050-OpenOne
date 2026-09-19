"""Retrospective endpoint-visible scope audit of frozen combination predictions.

The material holdout was defined for complete natural windows.  This audit must
not equate an early anchor in such a window with a prefix that already contains
both relation endpoints.  No fit, generation decision or data split is changed.
"""
from rdc_law_common import *


def main():
    start=time.monotonic();out=BASE/'confirmation/combination_visibility'
    rows={r['sample_id']:r for r in gzread(BASE/'confirmation_material.json.gz')}
    meta=gzread(BASE/'confirmation/query_catalog.json.gz');catalog=[]
    pairs=('obj+advcl','nsubj:pass+obl')
    for q in meta:
        row=rows[q['sample_id']];p=q['position'];graph=row.get('retrospective_graph',[])
        types=sorted({edge['type'][3:] for edge in graph if edge.get('type','').startswith('ud:') and edge.get('available_after_token',p+1)<=p})
        visible=[pair for pair in pairs if all(t in types for t in pair.split('+'))]
        assert set(visible)<=set(row['held_relation_combinations'])
        catalog.append({k:q[k] for k in ('sample_id','source_group','cohort','anchor_index','position')}|
            {'window_declared_pairs':row['held_relation_combinations'],'endpoint_visible_pairs':visible,'endpoint_visible_UD_types':types,
             'scope':'Both annotated head/dependent endpoints are in the availableprefix. Gold labels may still depend on complete sentence; no onlineparser or semanticoperator correctness is asserted.'})
    masks={'window_held_any':np.array([bool(r['window_declared_pairs']) for r in catalog]),
           'prefix_endpoint_visible_any':np.array([bool(r['endpoint_visible_pairs']) for r in catalog]),
           'declared_but_not_yet_endpoint_visible':np.array([bool(r['window_declared_pairs']) and not r['endpoint_visible_pairs'] for r in catalog])}
    masks.update({pair:np.array([pair in r['endpoint_visible_pairs'] for r in catalog]) for pair in pairs})
    frozen=read(BASE/'prediction/frozen.json');gains=[];counts=[];training=[]
    for name,mask in masks.items():
        ix=np.flatnonzero(mask)
        counts.append({'scope':name,'queries':len(ix),'samples':len({catalog[i]['sample_id'] for i in ix}),
            'sources':len({catalog[i]['source_group'] for i in ix}),'anchor_counts':{str(a):sum(catalog[i]['anchor_index']==a for i in ix) for a in (0,1,2)}})
        for b in (16,35):
            decoder=frozen['winners'][str(b)]['decoder']
            with np.load(BASE/'confirmation/predictions'/f'L{b}_fixed_early.npz') as base,np.load(BASE/'confirmation/predictions'/f'L{b}_{decoder}.npz') as winner:
                for metric in ('relative_MSE','KL'):
                    if metric not in winner:continue
                    gains.append({'group':name,'block':b,'metric':metric,'gain_over_frozen_early':clustered(base[metric][ix]-winner[metric][ix],[catalog[i]['source_group'] for i in ix])})
        with np.load(BASE/'confirmation/training/initial_full_panel.npz') as z:initial=z['loss']
        for run in read(BASE/'formation/trajectories/result.json')['runs']:
            with np.load(BASE/'confirmation/training'/f"{run['name']}_full_panel.npz") as z:delta=z['loss']-initial
            training.append({'group':name,'run':run['name'],'loss_change':clustered(delta[ix],[catalog[i]['source_group'] for i in ix])})
    compressed(out/'anchor_catalog.json.gz',catalog)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'query_count':len(catalog),'counts':counts,'frozen_prediction_gains':gains,'training':training,
        'original_prediction_files_unchanged':True,'refit_or_selection_performed':False,
        'scope_refinement':'Posthoc material/interpretation audit after originalofflineoutcomes were seen, beforelivegeneration; endpoint rule is not tuned to outcomes. Originalwindowholdout results retained, now explicitly distinguished from prefixendpointvisible subset.',
        'limits':['Endpointvisibility does not make full-sentence goldannotations an onlineavailable interpretation.',
            'Two relationtypes can coexist in differentparts of a window; this is not a demonstrated compositionaloperation or deeperreasoning test.',
            'Subsets overlap, bootstrap is conditionalonexistingfit/material, and no independent extraexperiment was performed.'],
        'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('held_combination_endpoint_visibility_audit',result['seconds'])
    print('LAW_COMBINATION_VISIBILITY_AUDIT',counts,flush=True)


if __name__=='__main__':main()
