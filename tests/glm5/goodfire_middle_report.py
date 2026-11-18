"""Whole-atlas reports, native-resolution plots and content-addressed manifests."""
import argparse
import gc
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import goodfire_middle as M
C=M.C


def build():
    from qwen14_mlp_publish import effects_summary
    M.require_admission()
    assert C.read(M.OUT/'native-audit.json')['passed'] and C.read(M.OUT/'analysis-audit.json')['passed']
    natural=effects_summary('natural_exact');capability=effects_summary('capability_exact')
    assert natural['status']==capability['status']=='complete'
    assert natural['replay_passed']==256 and capability['replay_passed']==512
    assert natural['restore_all'] and capability['restore_all']
    for row in natural['rows']:
        row['statistical_scope']='Descriptive paired test; zero dose primary; six selected units within this model. No semantic or cross-model promotion.'
    selection=C.read(M.OUT/'intervention-selection.json')['units']
    intervention=dict(natural=natural,capability=capability,selected_units=selection)
    M.write(M.OUT/'intervention-summary.json',intervention)
    neighbors=C.read(M.OUT/'neighbor-summary.json')
    compact={**neighbors,'rollout':[{k:v for k,v in row.items() if k!='unit_ratios'} for row in neighbors['rollout']]}
    examples=[]
    # Examples are retrospective retrieval aids, never new blind confirmation.
    for b in M.BLOCKS:
        with np.load(M.OUT/f'B{b}-prediction.npz') as z:
            good=z['A_candidate'];den=z['A_holdout_exact_token_background_mse']
            ratio=z['A_holdout_conditional_mse']/np.maximum(den,1e-20)
            u=int(np.argmin(np.where(good,ratio,np.inf))) if good.any() else int(np.argmin(ratio))
            examples.append(dict(block=b,unit=u,passes_frozen_screen=bool(good[u]),
                conditional_over_exact_token=float(ratio[u]),selection='Retrospective best A prediction example; no independent confirmation.'))
    report=dict(schema='native_mlp_atlas_report.v1',title=f'{M.SPEC["name"]} · B18–B28全量原生神经元',
        model_name=M.SPEC['name'],config=C.read(C.MODEL/'config.json'),precision='BF16',
        blocks=M.BLOCKS,width=C.WIDTH,hidden_size=C.DIM,model_layers=M.SPEC['layers'],
        n_units=11*C.WIDTH,documents=1024,context_tokens=1024,positions=C.POSITIONS,
        coverage=dict(native_profiles=11*C.WIDTH,native_parameter_vectors=33*C.WIDTH,
            semantic_confirmed=0,intervened_native_units=len(selection),adjacent_boundaries=10,
            hidden_layers=M.SPEC['layers']+1),
        regimes=C.read(M.OUT/'regime-summary.json')['layers'],prediction=C.read(M.OUT/'prediction-summary.json'),
        neighbor_summary=compact,intervention=intervention,total_intervention_units=len(selection),
        illustrative_prediction_units=examples,collection_seconds=C.read(M.OUT/'collection.json')['seconds'],
        numerical_reference=C.read(M.OUT/'native-audit.json'),
        rotary_initialization='CPU float32 frequencies, then CUDA; matches HF loader. Legacy CUDA-initialized RoPE is a separate execution version.',
        scope='Whole native-unit observation, family-heldout response screening, and selected finite final-token write interventions.',
        formal_evidence_promotion=False,semantic_confirmed_units=0,
        completed='Every requested unit has G/V/A fields, both-pole contexts, parameters, signed regimes, readout proxy and frozen prediction scores.',
        not_completed='Per-unit semantic explanations, full-circuit causal identification and text-to-full-vocabulary generative closure.',
        limitations=['Same historical licensed documents; candidate-specific document holdouts, not globally new material.',
            'Eight saved positions per document; full second-half scanning retains extremes rather than every activation.',
            'Keywords are retrieval cues; no verified language-function label for every unit.',
            'Exact-token plus background is stronger than hashes but not a strong contextual language baseline.',
            'No matched-budget SAE/transcoder comparison or independent node replication in this phase.',
            'Adjacent correlations and parameter alignment do not establish actual causal paths.',
            'Finite column writes are not weight edits. Capability items were previously exposed; free generation not tested.',
            'The two models share materials, not independent sample sets; unit indices across models are not equivalent.',
            'v1 was superseded before response fitting due to RoPE initialization drift; thresholds were not relaxed.'],
        references=['https://already-clean.goodfire.com/'],read_api='/api/research/native-mlp?run='+M.RUN)
    M.write(M.OUT/'report.json',report)
    M.write(M.OUT/'audit.json',dict(passed=True,native_audit_sha256=C.file_sha(M.OUT/'native-audit.json'),
        analysis_audit_sha256=C.file_sha(M.OUT/'analysis-audit.json'),
        replay_passed=dict(natural=natural['replay_passed'],capability=capability['replay_passed']),
        report_source_sha256=C.file_sha(__file__),scope='Execution and numerical audit, not independent scientific replication.'))
    plan=C.read(M.OUT/'plan.json')
    source_names=set(plan['source_hashes'])|{'tests/glm5/goodfire_middle_report.py','tests/glm5/goodfire_middle_pipeline.py',
        'tests/glm5/qwen14_mlp_publish.py','tests/glm5/qwen14_mlp_provenance.py',
        'tools/provenance.py','tools/power_analysis.py','deploy/storage_policy.py','deploy/storage_policy.json'}
    with zipfile.ZipFile(M.OUT/'experiment-sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(source_names):
            if name in plan['source_hashes']:assert C.file_sha(M.ROOT/name)==plan['source_hashes'][name]
            archive.write(M.ROOT/name,name)
    shutil.copyfile(C.MODEL/'tokenizer.json',M.OUT/'tokenizer.json')
    license_source=M.ROOT/'tests/glm5/result/mlp_tail_goodfire_v1/QWEN-LICENSE.txt'
    shutil.copyfile(license_source,M.OUT/'QWEN-LICENSE.txt')
    (M.OUT/'WEIGHT-NOTICE.txt').write_text(f'{M.SPEC["name"]} by Qwen/Alibaba, Apache-2.0. '
        'Original BF16 B18-B28 gate/up/down vectors; no training or weight changes. '
        'Snapshot hashes in model-identity.json. Natural text licenses and attribution in documents.json.\n',encoding='utf-8')
    print('Complete report built',M.MODEL_KEY,flush=True)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    std=[];dn=[]
    for b in M.BLOCKS:
        with np.load(M.OUT/f'B{b}-regimes.npz') as z:
            std.append(z['holdout_A_std']);dn.append(z['holdout_quadrant_abs_mass_fraction'][:,3])
    std=np.asarray(std);dn=np.asarray(dn)
    np.savez(M.OUT/'heatmap-native-arrays.npz',A_std=std,double_negative_abs_mass_fraction=dn,blocks=M.BLOCKS)
    fig,axes=plt.subplots(2,1,figsize=(15,6.2),constrained_layout=True)
    for ax,values,title,cmap in [(axes[0],np.log1p(std),'log(1 + A std), holdout','viridis'),
                                (axes[1],dn,'Double-negative share of |A|, holdout','magma')]:
        view=values.reshape(11,-1,16).mean(2)
        im=ax.imshow(view,aspect='auto',interpolation='nearest',cmap=cmap,extent=(-.5,C.WIDTH-.5,28.5,17.5))
        ax.set_yticks(M.BLOCKS);ax.set_ylabel('Native block');ax.set_title(M.SPEC['name']+' - '+title)
        fig.colorbar(im,ax=ax,pad=.01)
    axes[-1].set_xlabel('Native unit index; overview averages consecutive 16 units; no sorting or top-k')
    fig.savefig(M.OUT/'all-neuron-overview.png',dpi=160);plt.close(fig)
    for name,a,cmap in [('A-std-native-pixels',np.log1p(std),'viridis'),('double-negative-native-pixels',dn,'magma')]:
        plt.imsave(M.OUT/(name+'.png'),np.repeat(a,24,axis=0),cmap=cmap,vmin=float(a.min()),vmax=float(a.max()))
    M.write(M.OUT/'plot-specification.json',dict(grouping=16,native_png_columns=C.WIDTH,
        sorting='none',top_k_filter='none',color_ranges=dict(log1p_A_std=[float(np.log1p(std).min()),float(np.log1p(std).max())],
        double_negative=[float(dn.min()),float(dn.max())]),scope='Every native coordinate retained in NPZ and full-width PNG. Overview can hide narrow peaks.'))


def manifest():
    assert C.read(M.OUT/'audit.json')['passed'];start=time.monotonic();files=[]
    names=[p for p in sorted(M.OUT.iterdir()) if p.is_file() and p.name!='manifest.json'
        and not p.name.endswith('.log') and '-progress.' not in p.name]
    for i,p in enumerate(names):
        files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=C.file_sha(p)))
        if i%12==0:M.log('manifest',i+1,len(names),start)
    M.write(M.OUT/'manifest.json',dict(schema='native_mlp_manifest.v1',state='complete',blocks=M.BLOCKS,
        width=C.WIDTH,hidden_size=C.DIM,files=files,bytes=sum(p['bytes'] for p in files),
        retention='Native all-unit query and future conditional-mechanism research. Review 2026-12-13.',
        source_hashes=C.read(M.OUT/'plan.json')['source_hashes']))


def main():
    p=argparse.ArgumentParser();p.add_argument('model',choices=M.SPECS);p.add_argument('stage',choices=['build','plots','manifest'])
    a=p.parse_args();M.configure(a.model);M.require_admission();globals()[a.stage]()


if __name__=='__main__':main()
