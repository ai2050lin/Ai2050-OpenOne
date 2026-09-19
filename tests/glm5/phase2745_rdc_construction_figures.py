"""Scientific figures from committed full-coordinate records, without axis pruning."""
from rdc_construction_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

OUT = BASE / 'figures'
MODELS_ORDER = ['qwen4', 'qwen14', 'glm4']
COLORS = ['#176b87', '#cb7032', '#7952a0']


def save_figure(fig, name, caption, inputs, entries):
    path = OUT / (name+'.png')
    fig.savefig(path, dpi=170, facecolor='white')
    plt.close(fig)
    entries.append({'name': name, 'path': str(path.relative_to(BASE)), 'sha256': sha(path),
        'caption': caption, 'inputs': [str(p.relative_to(BASE)) for p in inputs],
        'rendering': 'Original numeric axes. Rasterization can aggregate display pixels; all source arrays and numeric pages remain available. No data-selected coordinate subset.'})


def intervals(ax, labels, values, color='#176b87', marker='o'):
    y = np.array([v['mean'] for v in values])
    lo = np.array([v['interval95'][0] for v in values])
    hi = np.array([v['interval95'][1] for v in values])
    positions = np.arange(len(values))
    assert np.all(lo <= hi) and np.isfinite([y, lo, hi]).all()
    ax.vlines(positions, lo, hi, color=color)
    ax.plot(positions, y, marker, color=color, linestyle='none')
    ax.plot(positions, lo, '_', color=color)
    ax.plot(positions, hi, '_', color=color)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.grid(alpha=.18, axis='y')


def main():
    if (OUT/'index.json').exists() and read(OUT/'index.json')['source']['sha256']==sha(__file__):
        assert all(sha(BASE/r['path'])==r['sha256'] for r in read(OUT/'index.json')['figures'])
        return
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    assert all(read(BASE/'compilation'/m/'result.json')['all_passed'] for m in MODELS_ORDER)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    entries = []
    analyses = {m:read(BASE/'analysis'/m/'result.json') for m in MODELS_ORDER}
    material = gzread(BASE/'material.json.gz')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for vi, view in enumerate(['raw','per_vector_RMS']):
        for model, color in zip(MODELS_ORDER, COLORS):
            rows = [r for r in analyses[model]['anova_summaries'] if r['query_language']=='all' and r['view']==view and isinstance(r['boundary'],int)]
            depth = read(BASE/'capture'/model/'result.json')['depth']
            axes[vi].plot([r['boundary']/depth for r in rows], [100*r['interaction_fraction'] for r in rows], '.--', color=color, label=model)
        axes[vi].set(title='Raw state values' if vi==0 else 'Whole-vector RMS view', xlabel='Native boundary / model depth', ylabel='Statistical interaction / total variance (%)')
        axes[vi].grid(alpha=.2)
        axes[vi].legend()
    fig.suptitle('Prefix x query interaction: all native coordinates, same320x100panel')
    save_figure(fig,'interaction_depth','Balanced-grid descriptive ANOVA, not semantic or causal variance. Dotted lines connect retained boundaries; intervening values are not observed here. Fractional depth is not cross-model functional alignment.',[BASE/'analysis'/m/'result.json' for m in MODELS_ORDER],entries)

    families = sorted({r['family'] for r in material['models']['qwen4']['rows']})
    fig, axes = plt.subplots(len(families),3,figsize=(16,14),constrained_layout=True)
    pair_inputs = []
    for mi, model in enumerate(MODELS_ORDER):
        fields = []
        for family in families:
            r = next(r for r in material['models'][model]['rows'] if r['family']==family and r['case']==0 and r['language']=='en' and r['world']==0)
            path = BASE/'capture'/model/'pairs'/('pair_'+rank(r['pair_id'])[:20]+'.npz')
            pair_inputs.append(path)
            with np.load(path) as z:
                fields.append(z['all_coordinate_query_mean_squared_difference'])
        vmax = max(float(a.max()) for a in fields)
        norm = SymLogNorm(linthresh=max(vmax*1e-5,1e-12),vmin=0,vmax=vmax,base=10)
        for fi, (family, a) in enumerate(zip(families,fields)):
            im=axes[fi,mi].imshow(a,aspect='auto',interpolation='nearest',origin='lower',cmap='magma',norm=norm,
                extent=[-.5,a.shape[1]-.5,-.5,a.shape[0]-.5])
            axes[fi,mi].set(title=model+' / '+family,xlabel='All native residual coordinates',ylabel='Native H boundary')
        fig.colorbar(im,ax=axes[:,mi].tolist(),shrink=.35,label='Raw mean squared pair change; shared column scale')
    fig.suptitle('Predeclared English case0 in every family: full-layer/all-coordinate trajectories\nMean over all100queries; color uses sym-log, with no clipping or low-value deletion',fontsize=13)
    save_figure(fig,'paired_full_coordinate_trajectories','Each cell retains every coordinate and every native boundary. Case0/English/all five families were predeclared; these are representative observations, not selected best cases. Color scale shared within each model, not across different widths.',pair_inputs,entries)

    # A separate full-vector-normalized view exposes the early low-amplitude
    # background without clipping or independently reordering coordinates.
    fig, axes = plt.subplots(len(families),3,figsize=(16,14),constrained_layout=True)
    normalized_fields = {}
    for mi, model in enumerate(MODELS_ORDER):
        fields = []
        for family in families:
            pair = [r for r in material['models'][model]['rows'] if r['family']==family and r['case']==0 and r['language']=='en']
            pair.sort(key=lambda r:r['world'])
            with np.load(BASE/'capture'/model/'fields'/(pair[0]['sample_id']+'.npz')) as za, np.load(BASE/'capture'/model/'fields'/(pair[1]['sample_id']+'.npz')) as zb:
                assert np.array_equal(za['query_layer_indices'],zb['query_layer_indices'])
                left,right=za['query_selected_states'],zb['query_selected_states']
                layers=[]
                for a,b in zip(left,right):
                    a,b=unbits(a).astype(float),unbits(b).astype(float)
                    a/=np.sqrt(np.mean(a*a,-1,keepdims=True)).clip(1e-12)
                    b/=np.sqrt(np.mean(b*b,-1,keepdims=True)).clip(1e-12)
                    layers.append(np.mean((b-a)**2,0))
                value=np.stack(layers);fields.append(value)
                normalized_fields[model+'__'+family]=value
        vmax=max(float(a.max()) for a in fields)
        norm=SymLogNorm(linthresh=1e-7,vmin=0,vmax=vmax,base=10)
        for fi,(family,a) in enumerate(zip(families,fields)):
            im=axes[fi,mi].imshow(a,aspect='auto',origin='lower',interpolation='nearest',cmap='magma',norm=norm,
                extent=[-.5,a.shape[1]-.5,-.5,a.shape[0]-.5])
            axes[fi,mi].set(title=model+' / '+family,xlabel='All native residual coordinates',ylabel='Native H boundary')
        fig.colorbar(im,ax=axes[:,mi].tolist(),shrink=.35,label='Mean squared pair change after each full vector RMS')
    normalized_path=OUT/'paired_full_vector_RMS_arrays.npz'
    npz(normalized_path,**normalized_fields)
    fig.suptitle('Same predeclared case0/English pairs: separate whole-vector RMS view\nEvery coordinate at every native boundary; sym-log linear threshold 1e-7',fontsize=13)
    save_figure(fig,'paired_full_coordinate_RMS','RMS is performed separately on each original query state before pair subtraction. This view complements raw amplitude; no coordinate sorting, clipping or pruning. Full all-layer fixtures only.',[normalized_path],entries)

    fig, axes = plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for mi, model in enumerate(MODELS_ORDER):
        r=read(BASE/'compilation'/model/'result.json')
        for ti,item in enumerate(r['same_input_pair_prediction']):
            intervals(axes[ti,mi],[item['target']],[item['diagonal_minus_full_MSE']],COLORS[mi])
            axes[ti,mi].axhline(0,color='#555',lw=1)
            axes[ti,mi].set(title=model,ylabel='Diagonal MSE - full matrix MSE\nPositive favors cross-coordinate predictor')
    fig.suptitle('Same actualH1 input: heldout pair-change prediction,20semantic clusters\nSeparate native target scales to keep the early-layer intervals visible')
    save_figure(fig,'same_input_cross_coordinate_test','Frozen validation-selected paired maps. Both receive the same actualH1; full matrix has greater parameter capacity. Intervals concern20test semantic groups conditional on the fit, not coordinates or random training seeds.',[BASE/'compilation'/m/'result.json' for m in MODELS_ORDER],entries)

    fig, axes = plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    shown=[('standalone','Alone'),('H1_diagonal_paired__0','Paired diag'),('H1_diagonal_absolute__0','Abs diag'),('full__actual_query_H1__0','Full H1'),('full__actual_query_H1__1','Shuffled full')]
    for mi, model in enumerate(MODELS_ORDER):
        r=read(BASE/'compilation'/model/'result.json')
        rows={x['variant']:x for x in r['summaries'] if x['family']=='all'}
        for ai, metric in enumerate(['Q_MSE','KL_native_to_prediction']):
            intervals(axes[ai,mi],[label for _,label in shown],[rows[name]['metrics'][metric] for name,_ in shown],COLORS[mi])
            axes[ai,mi].set(title=model,ylabel='All pre-RoPEQ MSE' if ai==0 else 'Full-vocabulary KL(native || forecast)')
            axes[ai,mi].tick_params(axis='x',labelrotation=25)
            axes[ai,mi].set_yscale('symlog',linthresh=1e-4)
            if ai==0:
                axes[ai,mi].axhline(r['Q_native_last_token_projection_floor_MSE']['mean'],color='#999',ls=':',label='Native last-token shape floor')
                axes[ai,mi].legend(fontsize=8)
    fig.suptitle('Explicit standalone-anchored absolute prediction, then originalQ and originallm_head\n80test expressions x20unseen queries; no future queryK/V or targets supplied to forecast')
    save_figure(fig,'native_projection_and_probability','Pair-map absolute anchoring is an added assumption, not a consequence of pair prediction. NativeQ and complete-vocabulary compile results shown without selecting a winning metric. Same original head batch shapes are replay-verified.',[BASE/'compilation'/m/'result.json' for m in MODELS_ORDER],entries)

    norm=read(BASE/'norm_controls/analysis.json')
    fig, axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    metrics=[('natural_raw','Natural raw NLL'),('natural_joint_calibrated','Natural calibrated NLL'),('conditional_NLL','Relation conditional NLL'),('complete_success','Correct and stopped success rate')]
    for ai,(metric,title) in enumerate(metrics):
        ax=axes.flat[ai]
        for control,color,shift in [('permuted',COLORS[0],-.002),('coordinate_shuffle',COLORS[1],0),('reverse',COLORS[2],.002)]:
            rr=sorted([r for r in norm['direction_comparisons'] if r['control']==control and isinstance(r['seed'],str)],key=lambda r:r['radius'])
            vv=[r['natural_minus_control'][metric] for r in rr]
            means=np.array([v['mean'] for v in vv])
            low=np.array([v['interval95'][0] for v in vv]);high=np.array([v['interval95'][1] for v in vv])
            ax.errorbar(np.array([r['radius'] for r in rr])+shift,means,yerr=[means-low,high-means],fmt='o',color=color,capsize=3,label=control)
        ax.axhline(0,color='#555',lw=1);ax.axvline(.1,color='#aaa',ls=':',label='Predeclared primary radius')
        ax.set(title=title,xlabel='Actual BF16 global displacement radius',ylabel='Natural direction - control')
        ax.set_xticks([.05,.10,.18]);ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle('Direction is not magnitude: every tested radius, two fixed-seed paired average\nNLL negative favors natural direction; success-rate positive favors natural direction')
    save_figure(fig,'matched_direction_outcomes','All planned radius/control combinations shown. Tiny horizontal offsets separate overlapping intervals, not actual radius changes. Cluster intervals do not estimate the population of training seeds. Actual BF16 norm errors and per-matrix norms remain in parameter_matching.',[BASE/'norm_controls/analysis.json'],entries)

    fig,axes=plt.subplots(2,2,figsize=(15,7),constrained_layout=True)
    variants=[('matched_natural_target_2742_r0p10','Natural'),('matched_within_cohort_permuted_target_2742_r0p10','Permuted target'),('coordinate_shuffle_2742_r0p10','Coordinate shuffle'),('reverse_2742_r0p10','Reverse')]
    inputs=[]
    for ci,(variant,label) in enumerate(variants):
        path=BASE/'norm_controls/coordinate_changes'/(variant+'.npz');inputs.append(path)
        with np.load(path) as z:
            for li,layer in enumerate([16,35]):
                a=z[f'all__L{layer}_activation__MSE']
                axes.flat[ci].plot(np.arange(len(a)),a,lw=.5,label='Block'+str(layer))
        axes.flat[ci].set(title=label+' / seed2742 / r0.10',xlabel='Every native MLP unit',ylabel='Mean squared activation change')
        axes.flat[ci].set_yscale('symlog',linthresh=1e-6);axes.flat[ci].legend(fontsize=8);axes.flat[ci].grid(alpha=.15)
    fig.suptitle('All9728units at editedblock16 and downstreamblock35; no unit selection or smoothing')
    save_figure(fig,'all_unit_training_changes','Every MLP activation unit, averaging all320expressions. Predeclared first seed illustrated; both seeds/all conditions are retained and queryable. Dense overplotting is display only, not a claim of sparse semantic gears.',inputs,entries)
    save(OUT/'index.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'figures':entries,
        'visual_inspection':'pending_actual_image_review','seconds':time.monotonic()-start})
    print('CONSTRUCTION_FIGURES_DONE',len(entries),flush=True)


if __name__=='__main__':
    main()
