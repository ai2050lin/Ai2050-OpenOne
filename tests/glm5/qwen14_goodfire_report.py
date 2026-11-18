"""Audit the entire referenced atlas, summarize limitations, and append memo."""
import argparse
from datetime import datetime
import gc
import hashlib
import re
import time
import numpy as np
import qwen14_goodfire_extension as E
C=E.C


def build():
    from qwen14_mlp_publish import effects_summary
    start=time.monotonic();layers=[];checks=[]
    for binding in C.read(E.OUT/'source-binding.json')['files']:
        assert C.file_sha(binding['path'])==binding['sha256'], binding['path']
    bound=C.read(E.OUT/'analysis-extension-binding.json')
    assert C.file_sha(bound['source'])==bound['source_sha256']
    assert C.file_sha(E.ROOT/'tests/glm5/qwen14_goodfire_scaled_prediction.py')==C.read(E.OUT/'scaled-prediction-plan.json')['source_sha256']
    assert np.array_equal(np.load(E.OLD/'final-hidden.npy',mmap_mode='r'),np.load(E.OUT/'final-hidden.npy',mmap_mode='r'))
    assert C.read(E.OUT/'historical-replay.json')['counts']=={'G':1024,'V':1024,'A':1024}
    for b in E.ALL:
        root=E.source(b);p=C.read(root/f'B{b}-profile.json');pred=C.read(root/f'B{b}-prediction.json')['fields']
        cues=C.read(root/f'B{b}-cues.json')['units'];ro=C.read(root/f'B{b}-readout.json')
        assert len(cues)==C.WIDTH and [x['unit'] for x in cues]==list(range(C.WIDTH))
        for f,w in [('G',C.WIDTH),('V',C.WIDTH),('A',C.WIDTH),('input',C.DIM),('output',C.DIM)]:
            a=np.load(root/f'B{b}-{f}.npy',mmap_mode='r');assert a.shape==(1024,8,w) and a.dtype==np.uint16
            for lo in range(0,1024,64):assert np.isfinite(C.unbits(a[lo:lo+64])).all()
        for f in ['G','V','A']:
            for k in ['mean','std','min','max']:assert len(p['fields'][f][k])==C.WIDTH
            assert len(pred[f]['candidate_predictable'])==C.WIDTH
        for k in ['positive_ids','negative_ids','positive_values','negative_values']:assert np.asarray(ro[k]).shape==(C.WIDTH,5)
        for f,shape in [('gate',(C.WIDTH,C.DIM)),('up',(C.WIDTH,C.DIM)),('down',(C.DIM,C.WIDTH))]:
            a=np.load(root/f'B{b}-W-{f}.npy',mmap_mode='r');assert a.shape==shape
        with np.load(E.OUT/f'B{b}-regimes.npz') as z:
            for s in C.SPLITS:
                assert np.allclose(z[s+'_quadrant_fraction'].sum(1),1)
                nonzero=(z[s+'_A_mean']!=0)|(z[s+'_A_std']>0)
                assert np.allclose(z[s+'_quadrant_abs_mass_fraction'].sum(1),nonzero.astype(float),atol=1e-6)
                assert np.allclose(z[s+'_quadrant_energy_fraction'].sum(1),nonzero.astype(float),atol=1e-6)
                assert all(np.isfinite(z[k]).all() for k in z.files)
        with np.load(E.OUT/f'B{b}-scaled-prediction.npz') as z:
            assert all(np.asarray(z[k]).shape==(C.WIDTH,) and np.isfinite(z[k]).all() for k in z.files)
        layers.append({'block':b,'units':C.WIDTH,'source_directory':str(root),'historical_reuse':b<36,
            'prediction_candidates':{f:pred[f]['count'] for f in pred},
            'zero_holdout_keyword_hits':sum(r['repeat']['holdout']['hits']==0 for r in cues)})
        checks.append({'block':b,'full_width_finite_and_complete':True,'quadrants_partition_and_mass_conserve':True})
        E.log('audit_layers',b-27,12,start);del p,pred,cues,ro;gc.collect()
    natural=effects_summary('natural_exact');cap=effects_summary('capability_exact')
    assert natural['status']==cap['status']=='complete'
    assert natural['replay_passed']==natural['samples']==256
    assert cap['replay_passed']==cap['samples']==512
    for summary in [natural,cap]:
        for row in summary['rows']:
            if 'statistical_scope' in row:row['statistical_scope']='Descriptive paired test; 8 new units, conservative correction planned for 16; no semantic promotion.'
    interventions={'natural':natural,'capability':cap,'selected_units':C.read(E.OUT/'intervention-selection.json')['units']}
    E.save(E.OUT/'intervention-summary.json',interventions)
    collection=C.read(E.OUT/'collection.json');replay=C.read(E.OUT/'historical-replay.json')
    assert len(collection['replay'])==16 and all(r['product_bitwise_equal'] and r['output_relative_error']<=.02 for r in collection['replay'])
    neighbors=C.read(E.OUT/'neighbor-summary.json')
    neighbor_metrics=[]
    for b in range(28,39):
        r=E.OLD if b<35 else E.OUT
        raw=C.read(r/f'B{b}-B{b+1}-prediction.json')['splits']['holdout']
        denominator=np.asarray(raw['mean_mse']);numerator=np.asarray(raw['mse'])
        neighbor_metrics.append({'source':b,'target':b+1,
            'unit_mean_MSE_ratio':float(np.mean(raw['ratio'])),'unit_median_MSE_ratio':float(np.median(raw['ratio'])),
            'pooled_MSE_ratio':float(numerator.sum()/denominator.sum()),
            'pooled_same_index_ratio':float(np.sum(raw['same_index_mse'])/denominator.sum()),
            'pooled_random_parent_ratio':float(np.sum(raw['random_parent_mse'])/denominator.sum())})
    report={'schema':'qwen14_goodfire_complete_coverage.v2','title':'Qwen3-14B B28–B39原生神经元全量分析',
        'blocks':E.ALL,'width':C.WIDTH,'native_units':len(E.ALL)*C.WIDTH,'new_native_units':4*C.WIDTH,
        'parameters_per_unit':3*C.DIM,'native_parameter_vectors':12*C.WIDTH*3,
        'documents':1024,'tokens_per_document':1024,'positions':C.POSITIONS,
        'extrema_scope':'Each G/V/A pole over positions512..1023; full vectors retained at 8 declared readouts.',
        'splits':C.SPLITS,'layers':layers,'regimes':C.read(E.OUT/'regime-summary.json')['layers'],
        'secondary_standardized_prediction':C.read(E.OUT/'scaled-prediction-summary.json'),
        'neighbors':{**neighbors,'rollout':[{k:v for k,v in r.items() if k!='unit_ratios'} for r in neighbors['rollout']]},
        'neighbor_additional_descriptive_metrics':neighbor_metrics,
        'new_interventions':interventions,'historical_intervention_units':16,'total_intervention_units':24,
        'collection_seconds':collection['seconds'],'historical_replay_max_L2':replay['max_relative_L2'],
        'historical_replay_bitwise':replay['all_bitwise_equal'],'semantic_confirmed_units':0,
        'formal_mechanism_promotion':False,
        'completed':'All requested native units covered by observational profiles, signed regimes, parameters, readout proxies, context retrieval and predictive screening. Selected units have finite-dose interventions.',
        'not_completed':'Per-unit semantic explanations, heldout semantic minimal pairs, natural generation steering and independent causal confirmation for 208896 units.',
        'limitations':['Same 1024 historical licensed documents; split-specific fitting, not globally unexposed corpus.',
            'B28-B35 historical initial provenance limitation retained; reuse does not repair it.',
            'Eight retained readout positions; whole second-half scan retains extrema rather than every activation.',
            'Keyword clues and unembedding proxies are not neuron names or semantic evidence.',
            'Equal-dimensional hash ridge and one-parent predictors are restricted baselines; failure does not establish absence of structure.',
            'No SAE/transcoder/VPD comparison, no independent node, no new model.',
            'Final-token finite-column writes are not weight edits; norm controls with >2% BF16 mismatch excluded from matched analyses.',
            '512 capability questions were previously exposed; no universal capability preservation.',
            'Original UI/remote server is not changed by this local research result.'],
        'references':['https://already-clean.goodfire.com/'],
        'query_entry':'tests/glm5/qwen14_goodfire_query.py --block B --unit U'}
    E.save(E.OUT/'report.json',report)
    E.save(E.OUT/'audit.json',{'passed':True,'checks':checks,'new_native_replay':collection['replay'],
        'historical_replay':{k:v for k,v in replay.items() if k!='rows'},
        'scope':'Same-node numerical and artifact audit; not independent scientific replication.'})
    # Compact coverage index retains exact model, layer, boundary and native index.
    E.save(E.OUT/'atlas-index.json',{'schema':'native_neuron_atlas_index.v2','model_snapshot':E.SNAP,
        'boundary':'MLP gate/up preactivations and down_proj.input','readout_positions':C.POSITIONS,
        'layers':[{'block':b,'first_unit':0,'last_unit':C.WIDTH-1,'units':C.WIDTH,
            'profile':str(E.source(b)/f'B{b}-profile.json'),'regimes':str(E.OUT/f'B{b}-regimes.npz'),
            'parameters':str(E.source(b))} for b in E.ALL],
        'query':'python -B tests/glm5/qwen14_goodfire_query.py --block 39 --unit 0',
        'semantic_confirmed':False})
    print('report complete',flush=True)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    std=[];dn=[]
    for b in E.ALL:
        with np.load(E.OUT/f'B{b}-regimes.npz') as z:
            std.append(z['holdout_A_std']);dn.append(z['holdout_quadrant_abs_mass_fraction'][:,3])
    std=np.array(std);dn=np.array(dn)
    np.savez(E.OUT/'heatmap-native-arrays.npz',A_std=std,double_negative_abs_mass_fraction=dn,blocks=E.ALL)
    fig,axes=plt.subplots(2,1,figsize=(15,6.5),constrained_layout=True)
    for ax,arr,title,cmap in [(axes[0],np.log1p(std),'log(1 + A std), holdout','viridis'),
                              (axes[1],dn,'Double-negative share of absolute A mass, holdout','magma')]:
        view=arr.reshape(12,-1,16).mean(2)
        im=ax.imshow(view,aspect='auto',origin='upper',cmap=cmap,extent=(-.5,C.WIDTH-.5,39.5,27.5),interpolation='nearest')
        ax.set_yticks(E.ALL);ax.set_ylabel('Native block');ax.set_title(title);fig.colorbar(im,ax=ax,pad=.01)
    axes[-1].set_xlabel('Native unit index; overview averages consecutive groups of 16; no sorting or top-k')
    fig.savefig(E.OUT/'all-neuron-overview.png',dpi=160);plt.close(fig)
    for name,arr,cmap in [('A-std-native-pixels',np.log1p(std),'viridis'),('double-negative-native-pixels',dn,'magma')]:
        plt.imsave(E.OUT/(name+'.png'),np.repeat(arr,24,axis=0),cmap=cmap,vmin=float(arr.min()),vmax=float(arr.max()))
    E.save(E.OUT/'plot-specification.json',{'all_unit_arrays':'heatmap-native-arrays.npz','overview_grouping':16,
        'native_png':'17408 columns, one per native neuron; 24 identical pixel rows per block.',
        'sorting':'none','top_k_filter':'none','standardization':'log1p for A std only; double-negative mass is an unstandardized fraction.',
        'color_ranges':{name:{'native_min':float(a.min()),'native_max':float(a.max()),
            'overview_min':float(a.reshape(12,-1,16).mean(2).min()),'overview_max':float(a.reshape(12,-1,16).mean(2).max())}
            for name,a in [('log1p_A_std',np.log1p(std)),('double_negative_mass',dn)]},
        'display_limit':'Overview means can hide isolated peaks. Full native-pixel images and unaggregated NPZ retain every neuron.'})


def manifest():
    start=time.monotonic();rows=[]
    for file in sorted(E.OUT.iterdir()):
        if file.is_file() and file.name not in ['manifest.json'] and not file.name.endswith('.log'):
            rows.append({'path':file.name,'bytes':file.stat().st_size,'sha256':C.file_sha(file)})
    E.save(E.OUT/'manifest.json',{'schema':'native_atlas_extension_manifest.v2','files':rows,
        'reused_manifest_sha256':C.file_sha(E.OUT/'reused-artifacts.json'),
        'retention':'All native fields, parameters, coefficients and frozen materials retained for queries and reanalysis; review 2026-12-13.',
        'total_new_bytes':sum(r['bytes'] for r in rows),'seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','plots','manifest']);args=p.parse_args()
    E.configure();globals()[args.stage]()
