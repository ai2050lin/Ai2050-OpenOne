"""Final numeric audits and unsorted native-coordinate scientific figures."""
import json,math
from pathlib import Path
import numpy as np
from phase2754_relation_stability import OUT,ROOT,FAMILIES,decode,write,sha,now,snapshot

def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    material=read('material.json');worlds=material['worlds'];rows=material['rows']
    assert len({w['id'] for w in worlds})==320 and len({r['id'] for r in rows})==5120
    names=[n for w in worlds for n in w['names']];assert len(names)==len(set(names))
    assert read('selection.json')['created_utc']<read('4B/confirmation/execution.json')['created_utc']
    reference={r['id']:r for r in rows};checks={}
    for side in ('4B','14B'):
        checks[side]={}
        for stage in (('discovery','confirmation') if side=='4B' else ('confirmation',)):
            seen=[];finite=True;residual=0
            for p in sorted((OUT/side/stage).glob('chunk_*.npz')):
                meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));seen.extend(r['id'] for r in meta)
                residual=max(residual,max(r['residual_anchor_max'] for r in meta))
                with np.load(p) as z:
                    for key in ('hidden','attention','mlp','query_slots'):
                        assert z[key].dtype==np.uint16;finite=finite and bool(np.isfinite(decode(z[key])).all())
            done=read(f'{side}/{stage}/done.json');assert len(seen)==done['count']==len(set(seen)) and finite and residual==0
            checks[side][stage]=dict(count=len(seen),all_finite=finite,native_residual_max=residual)
        # Exact state repeat across pilot and main (every pilot sample is in confirmation).
        pmeta=read(f'{side}/pilot/chunk_000.json')
        with np.load(OUT/side/'pilot/chunk_000.npz') as z:pilot={r['id']:z['hidden'][i] for i,r in enumerate(pmeta)}
        repeat=0;found=0
        for p in sorted((OUT/side/'confirmation').glob('chunk_*.npz')):
            meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'));selected=[(i,r) for i,r in enumerate(meta) if r['id'] in pilot]
            if selected:
                with np.load(p) as z:
                    for i,r in selected:
                        repeat=max(repeat,float(np.max(abs(decode(z['hidden'][i])-decode(pilot[r['id']])))));found+=1
        assert found==8 and repeat==0;checks[side]['pilot_repeat_max']=repeat
    # An all-success bootstrap is degenerate: provide a finite-n descriptive bound.
    success={}
    for side in ('4B','14B'):
        groups=read(f'{side}/regularity_groups.json')
        for split in ('entity','surface','depth'):
            ww=sorted({r['world'] for r in groups if r['split']==split});n=len(ww);k=sum(all(r['truth_coefficient']>0 for r in groups if r['world']==w) for w in ww)
            z=1.959963984540054;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
            success[f'{side}_{split}']=dict(worlds=n,all_templates_positive_worlds=k,wilson95=[mid-half,mid+half],note='Descriptive binomial bound for fixed templates; not a universal-language probability or replacement for stratified paired inference.')
    probes=read('probe_confirmation.json');reg=read('4B/regularity_summary.json')['splits']
    fig,axs=plt.subplots(1,2,figsize=(12,4));splits=('entity','surface','depth');xx=np.arange(3)
    for j,(method,title) in enumerate([('surface','Surface control'),('end16','End state at16'),('slots16','End + role slots at16'),('finalnorm','Final state decoder')]):
        vals=[probes['splits'][s][method]['truth_accuracy'] for s in splits];axs[0].bar(xx+(j-1.5)*.18,vals,width=.18,label=title)
    axs[0].set_xticks(xx,splits);axs[0].set_ylim(0,1.05);axs[0].axhline(.5,color='black',ls='--',lw=1);axs[0].set_ylabel('Frozen truth decoding accuracy');axs[0].legend(fontsize=7,loc='lower left')
    for j,(metric,title) in enumerate([('native_accuracy','Native first-token accuracy'),('paired_ranking','Matched true/false ranking')]):
        vals=[reg[s][metric]['mean'] for s in splits];ci=np.array([reg[s][metric]['ci95'] for s in splits]);axs[1].bar(xx+(j-.5)*.32,vals,width=.32,label=title,yerr=np.stack([np.array(vals)-ci[:,0],ci[:,1]-np.array(vals)]),capsize=3)
    axs[1].set_xticks(xx,splits);axs[1].set_ylim(0,1.05);axs[1].legend(fontsize=8,loc='lower left');axs[1].set_ylabel('Measured behavior,64worlds per axis')
    fig.tight_layout();fig.savefig(OUT/'decoding_and_behavior.png',dpi=150);plt.close(fig)
    for side in ('4B','14B'):
        profile=read(f'{side}/source_profile_rows.json')
        with np.load(OUT/side/'source_profiles.npz') as z:
            fig,axs=plt.subplots(1,3,figsize=(15,4),sharey=True)
            for ax,split in zip(axs,splits):
                ix=[i for i,r in enumerate(profile) if r['split']==split]
                for k in ('attention','mlp','rounding'):ax.plot(np.arange(z[k].shape[1])+1,z[k][ix].mean(0),label=k)
                ax.axhline(0,color='grey',lw=.5);ax.set_title(side+' / '+split);ax.set_xlabel('Block (1-based)');ax.grid(alpha=.2)
            axs[0].set_ylabel('Truth-aligned logit contribution');axs[0].legend();fig.tight_layout();fig.savefig(OUT/side/'source_layers.png',dpi=150);plt.close(fig)
        fm=read(f'{side}/native_factor_metadata.json');norm_layer=36 if side=='4B' else 40
        ix=[i for i,r in enumerate(fm['rows']) if r['split'] in splits]
        with np.load(OUT/side/'native_factor_means.npz') as z:values=z['means'][ix,7,norm_layer]
        names=[fm['rows'][i]['split']+'/'+fm['rows'][i]['family'] for i in ix]
        rms=np.sqrt(np.mean(values**2,axis=1,keepdims=True));vmax=float(abs(values).max())
        fig,axs=plt.subplots(2,1,figsize=(16,7),sharex=True)
        for ax,v,title,scale in [(axs[0],values,'Raw truth factorial field',vmax),(axs[1],values/np.maximum(rms,1e-10),'Row RMS view (display clipped at +/-6; raw retained)',6)]:
            im=ax.imshow(v,aspect='auto',interpolation='nearest',cmap='RdBu_r',vmin=-scale,vmax=scale);ax.set_title(side+' '+title);ax.set_yticks(range(len(names)),names,fontsize=8);fig.colorbar(im,ax=ax,pad=.01)
        axs[1].set_xlabel('Every native coordinate, original order');fig.tight_layout();fig.savefig(OUT/side/'native_truth_field.png',dpi=150);plt.close(fig)
        write(OUT/side/'plot_scale.json',dict(raw_symmetric_limit=vmax,normalized='Per observed row RMS,display clip +/-6',rows=names,coordinates='All,unsorted',aggregation='Mean factorial truth response,not single-neuron semantic identity'))
    mm=read('mechanism_summary.json')
    with np.load(OUT/'all_unit_truth_contributions.npz') as z:
        fig,axs=plt.subplots(len(z.files),1,figsize=(17,7),sharex=True)
        for ax,k in zip(np.atleast_1d(axs),z.files):
            values=z[k];rms=np.sqrt(np.mean(values**2,axis=1,keepdims=True));im=ax.imshow(values/np.maximum(rms,1e-12),aspect='auto',cmap='RdBu_r',interpolation='nearest',vmin=-6,vmax=6)
            ax.set_title('Block '+str(int(k)+1)+' all9728 SwiGLU units / row RMS; display +/-6');ax.set_yticks(range(12),[r['split']+'/'+r['family'] for r in mm['unit_rows']],fontsize=7);fig.colorbar(im,ax=ax,pad=.01)
        np.atleast_1d(axs)[-1].set_xlabel('MLP unit index, unsorted, noTopK');fig.tight_layout();fig.savefig(OUT/'all_mlp_units.png',dpi=150);plt.close(fig)
        fig,axs=plt.subplots(len(z.files),1,figsize=(17,7),sharex=True)
        for ax,k in zip(np.atleast_1d(axs),z.files):
            values=z[k];limit=float(abs(values).max());im=ax.imshow(values,aspect='auto',cmap='RdBu_r',interpolation='nearest',vmin=-limit,vmax=limit)
            ax.set_title('Block '+str(int(k)+1)+' all9728 units, raw truth-aligned logit contributions; per-panel scale')
            ax.set_yticks(range(12),[r['split']+'/'+r['family'] for r in mm['unit_rows']],fontsize=7);fig.colorbar(im,ax=ax,pad=.01)
        np.atleast_1d(axs)[-1].set_xlabel('MLP unit index, unsorted, noTopK');fig.tight_layout();fig.savefig(OUT/'all_mlp_units_raw.png',dpi=150);plt.close(fig)
    probe_rows=read('probe_rows.json');ix={r['id']:i for i,r in enumerate(probe_rows)}
    with np.load(OUT/'probe_predictions.npz') as z:probe_pred=z[read('selection.json')['selected']['semantic']][:,1]
    baselines=[r for r in read('mechanism_rows.json') if r['mode']=='baseline']
    probe_match=max(abs(r['edit']['probe_before']-float(probe_pred[ix[r['id']]])) for r in baselines)
    assert probe_match<1e-4
    write(OUT/'quality_audit.json',dict(created_utc=now(),source=snapshot(Path(__file__)),checks=checks,primary_worlds=320,primary_4B_prompts=5120,
        confirmation_worlds=192,replication14B_worlds=24,replication14B_prompts=192,unique_primary_names=len({n for w in worlds for n in w['names']}),
        early_selection_preceded_confirmation=True,finite_success_bounds=success,effective_intervention_probe_vs_frozen_prediction_max=probe_match,
        no_perfection_requirement='All behavior errors remain; positive group effects and decoder generalization retained with explicit limits.',
        limitation='Some bootstrap all-success intervals collapse; no claim of population perfection. Primitive graph families and role annotations remain supplied.'))

if __name__=='__main__':run()
