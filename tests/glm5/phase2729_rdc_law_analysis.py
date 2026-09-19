"""Source-clustered fit comparisons and complete trained-state change summaries."""
from rdc_law_common import *


def main():
    start=time.monotonic();out=BASE/'analysis';meta=gzread(BASE/'prediction/query_catalog.json.gz')
    frozen=read(BASE/'prediction/frozen.json');comparisons=[]
    for b in (16,35):
        winner=frozen['winners'][str(b)]['decoder']
        with np.load(BASE/'prediction/predictions'/f'L{b}_fixed_early.npz') as z:base={k:z[k] for k in z.files if k!='mlp'}
        for name in ('direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product'):
            with np.load(BASE/'prediction/predictions'/f'L{b}_{name}.npz') as z:data={k:z[k] for k in z.files if k!='mlp'}
            for split in ('validation','test'):
                for cohort in ('all_source_equal','gum','ewt','cmrc','squad_qa','cmrc_qa','hotpot_qa'):
                    ix=np.array([i for i,r in enumerate(meta) if r['split']==split and (cohort=='all_source_equal' or r['cohort']==cohort)])
                    record={'block':b,'decoder':name,'frozen_winner':name==winner,'split':split,'cohort':cohort,'queries':len(ix)}
                    for k in ('relative_MSE','KL'):
                        if k not in data:continue
                        record[k]={'query_mean':float(data[k][ix].mean()),'baseline_query_mean':float(base[k][ix].mean()),
                            'gain_source_cluster':clustered(base[k][ix]-data[k][ix],[meta[i]['source_group'] for i in ix])}
                    comparisons.append(record)
    p=read(BASE/'formation/protocol.json');training=[]
    with np.load(BASE/'formation/trajectories/initial_panel.npz') as z:
        baseline={k:z[k] for k in ('m','gradient_gram_total','condition_unit_moments','loss')}
    for seed in p['multistep_seeds']:
        for condition in ('coherent','prefix_order_control'):
            name=f'{condition}_seed{seed}';r=read(BASE/'formation/trajectories'/name/'result.json')
            for ck in r['checkpoints']:
                with np.load(BASE/ck['panel_file']) as z:
                    g=z['gradient_gram_total'];bg=baseline['gradient_gram_total']
                    norms=np.sqrt(np.maximum(np.diag(g),1e-30));oldnorm=np.sqrt(np.maximum(np.diag(bg),1e-30))
                    diff=g/(norms[:,None]*norms[None,:])-bg/(oldnorm[:,None]*oldnorm[None,:])
                    diag=np.arange(len(g));diff[diag,diag]=0
                    training.append({'run':name,'step':ck['step'],'relative_parameter_displacement':ck['relative_parameter_displacement'],
                        'mean_abs_full_gradient_cosine_change':float(np.mean(abs(diff))),
                        'complete_unit_moment_change_norm':float(np.linalg.norm(z['condition_unit_moments']-baseline['condition_unit_moments'])),
                        'test_loss_delta':float(np.mean((z['loss']-baseline['loss'])[[i for i,r in enumerate(p['panel']) if r['split']=='test']])),
                        'scope':'Descriptive all-unit/full-gradient changes, not independent-coordinate significance or proof of semantic reorganization.'})
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'comparison':comparisons,'training_changes':training,
        'interpretation':'Model selection solely validation. Source-cluster intervals conditional on the fitted model and available sources; per-source weighting differs from six-cohort-equal selection. No population significance across all language families.',
        'seconds':time.monotonic()-start}
    save(out/'result.json',result);ledger('phase2729_prediction_training_analysis',result['seconds'])
    print('LAW_ANALYSIS_COMPLETE', [r for r in comparisons if r['cohort']=='all_source_equal' and r['split']=='test' and r['frozen_winner']],flush=True)


if __name__=='__main__':main()
