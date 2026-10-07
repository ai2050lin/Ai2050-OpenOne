"""Independent material/measurement audit and native-coordinate scientific figures."""
import argparse
import json
from pathlib import Path
import numpy as np
from phase2752_context_interaction import OUT, ROOT, now, sha, write, FAMILIES


def audit(side):
    material=json.loads((OUT/'material.json').read_text(encoding='utf-8'))
    byid={r['id']:r for r in material['rows']}
    worlds={w['id']:w for w in material['worlds']}
    dest=OUT/side
    rows=[]
    for p in sorted(dest.glob('chunk_*.json')):
        rows.extend(json.loads(p.read_text(encoding='utf-8')))
    expected={r['id'] for r in material['rows'] if side=='4B' or r['replication14B']}
    assert set(r['id'] for r in rows)==expected and len(rows)==len(expected)
    train_names={n for w in worlds.values() if w['cohort']=='train' for n in w['entities']}
    test_names={n for w in worlds.values() if w['cohort'] not in ('train','validation') for n in w['entities']}
    assert not train_names&test_names
    train_pairs={ (r['role'],r['order']) for r in material['rows'] if r['split']=='train'}
    assert (1,1) not in train_pairs
    assert {r['depth'] for r in material['rows'] if r['split']=='train'}=={1,2}
    groups={}
    for r in rows:
        groups.setdefault(r['group'],[]).append(r)
    assert all(len(rs)==4 and len({byid[r['id']]['tokenization'][side]['length'] for r in rs})==1 for rs in groups.values())
    anchors=json.loads((dest/'anchors.json').read_text(encoding='utf-8'))
    assert len(anchors)==len(rows)
    assert all(a['residual_max']==0 and a['norm_max']==0 for a in anchors)
    # Exact identity of the independently repeated pilot and formal capture.
    pilot=OUT/(side+'_pilot')
    prows=json.loads((pilot/'chunk_000.json').read_text(encoding='utf-8'))
    with np.load(pilot/'chunk_000.npz') as z:
        repeat={r['id']:z['hidden'][i].copy() for i,r in enumerate(prows)}
    diffs={}
    for path in sorted(dest.glob('chunk_*.npz')):
        rr=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if not any(r['id'] in repeat for r in rr):continue
        with np.load(path) as z:
            for i,r in enumerate(rr):
                if r['id'] in repeat:
                    diffs[r['id']]=float(np.max(np.abs(z['hidden'][i]-repeat[r['id']])))
    behavior={}
    for family in FAMILIES:
        rr=[r for r in rows if byid[r['id']]['family']==family]
        behavior[family]={str(neg):dict(n=sum(byid[r['id']]['negation']==neg for r in rr),
            accuracy=float(np.mean([r['prediction_text'].strip().lower()==byid[r['id']]['expected'] for r in rr if byid[r['id']]['negation']==neg]))) for neg in (0,1)}
    allcorrect=sum(all(r['prediction_text'].strip().lower()==byid[r['id']]['expected'] for r in rs) for rs in groups.values())
    write(dest/'quality_audit.json',dict(created_utc=now(),capture_count=len(rows),groups=len(groups),
        unique_worlds=len({r['world'] for r in rows}),train_new_entity_overlap=0,within_quartet_token_length_matched=True,
        training_role_order_pairs=sorted(train_pairs),training_depths=[1,2],depth_test=3,
        max_residual_anchor=0,max_norm_anchor=0,repeated_pilot_max_differences=diffs,
        behavior_by_family_and_negation=behavior,all_four_first_tokens_correct_groups=allcorrect,
        behavior_limit='Expected labels use propositional negation; natural negative yes/no questions can have pragmatic ambiguity. Low accuracy limits claims of correct reasoning. No free generation checked.',
        leakage_limit='Wordings of train worlds intentionally recur only in wording test. Unique synthetic entity strings do not imply disjoint subword vocabulary or unknown pretraining examples.'))
    return rows


def plots(side):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=OUT/side
    summary=json.loads((dest/'prediction_summary.json').read_text(encoding='utf-8'))
    methods=('zero','family_mean','surface','graph_only','source_product','hybrid')
    splits=('entity','wording','joint_wording','role_order','depth')
    fig,ax=plt.subplots(figsize=(11,5))
    for m in methods:
        ax.plot(range(5),[summary['splits'][s]['methods'][m]['interaction_mean'] for s in splits],marker='o',label=m)
    if (dest/'source_ablation_summary.json').exists():
        ab=json.loads((dest/'source_ablation_summary.json').read_text(encoding='utf-8'))
        ax.plot(range(5),[ab['splits'][s]['methods']['source_linear'] for s in splits],marker='s',linestyle='--',label='source_linear (diagnostic)')
    ax.set(xticks=range(5),xticklabels=['New entities','New wording','New both','New role/order','Depth 3'],ylabel='Mean interaction relative L2 error (lower is better)',title=f'{side}: fixed held-out tests; winner selected on validation only')
    ax.legend(ncol=2,fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(dest/'heldout_errors.png',dpi=170)
    plt.close(fig)
    with np.load(dest/'full_coordinate_fields.npz') as z:
        observed=z['observed'];pred=z[summary['selected_method']]
    # Raw and row-RMS displays retain coordinate order, and show all coordinates.
    for standardized in (False,True,'symlog'):
        arrays=[observed,pred,observed-pred]
        if standardized:
            scale=np.maximum(np.sqrt(np.mean(observed**2,axis=1,keepdims=True)),1e-12)
            arrays=[a/scale for a in arrays]
        bound=max(float(np.abs(a).max()) for a in arrays)
        fig,axs=plt.subplots(3,1,figsize=(14,7),sharex=True)
        for ax,a,title in zip(axs,arrays,('Observed interaction','Validation-selected prediction','Observed minus predicted')):
            color=dict(vmin=-bound,vmax=bound)
            if standardized=='symlog':
                from matplotlib.colors import SymLogNorm
                color=dict(norm=SymLogNorm(linthresh=.1,vmin=-bound,vmax=bound))
            im=ax.imshow(a,aspect='auto',cmap='coolwarm',interpolation='nearest',**color)
            ax.set_ylabel('Boundary index')
            ax.set_title(title,fontsize=10)
        axs[-1].set_xlabel('Native coordinate index (no sorting or dimensionality reduction)')
        fig.subplots_adjust(right=.90,hspace=.4)
        cb=fig.add_axes([.92,.15,.015,.68])
        fig.colorbar(im,cax=cb,label='Observed row RMS units' if standardized else 'Raw activation units')
        fig.suptitle(f'{side}: mean over test groups; full arrays retained; last row is raw final residual')
        filename='interaction_field_row_rms_symlog.png' if standardized=='symlog' else 'interaction_field_row_rms.png' if standardized else 'interaction_field_raw.png'
        fig.savefig(dest/filename,dpi=160)
        plt.close(fig)
    write(dest/'plot_metadata.json',dict(created_utc=now(),coordinate_order='native',compression='Raster display averages pixels when viewer downsamples; full numeric matrices are retained',
        raw_colorbar='Shared symmetric full min/max over observed, prediction, residual',
        row_rms='Divide every displayed row by observed row RMS; same denominator for observed/predicted/residual; shared color limit',
        symlog='Additional row-RMS view uses symmetric log color normalization with linear threshold0.1. All coordinates/extremes retained; no clipping.',
        aggregation='All test groups equally weighted; no claim of per-example equality from mean heatmap'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='4B',choices=['4B','14B'])
    a=p.parse_args()
    audit(a.model)
    plots(a.model)
    print('Quality audit and figures saved',a.model)
