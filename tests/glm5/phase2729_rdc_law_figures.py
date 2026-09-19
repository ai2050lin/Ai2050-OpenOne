"""Frozen prediction comparisons and actual training trajectories on all native coordinates."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_law_common import *


def main():
    start=time.monotonic();out=BASE/'figures';index=read(out/'index.json');new=[]
    plt.rcParams.update({'font.size':9,'figure.dpi':130})
    def keep(fig,name,title,scope):
        fig.tight_layout();fig.savefig(out/name);plt.close(fig)
        new.append({'path':name,'title':title,'scope':scope,'sha256':sha(out/name)})
    source=read(BASE/'prediction/result.json')['per_kernel_MSE_selected_records']
    names=('early_linear','additive_history','multiplicative_history','routed_history','task_conditioned')
    decoders=('direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product')
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,b,metric in zip(axes,(16,35,35),('relative_MSE','relative_MSE','KL')):
        for d in decoders:
            values=[]
            for name in names:
                r=next(r for r in source if r['block']==b and r['kernel']==name and r['decoder']==d)
                values.append(np.mean([m[metric] for m in r['metrics'] if m['split']=='test']))
            ax.plot(range(len(names)),values,'o-',label=d,lw=1)
        ax.set(xticks=range(len(names)),xticklabels=['early','+history','x history','routed','task'],ylabel=metric,title=f'block{b}: six-cohort-equal main test')
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    keep(fig,'earlier_state_native_prediction_competition.png','192 frozen candidates: early states, histories and native compilers',
        'Full2560coordinates/9728units. Main-test means weight six cohorts equally, not query counts. This plot uses eachkernel MSE-selected df; deployment has a separately declared validationKLselection atblock35. DirectMLP and joint-product are a linear-commutation calibration pair, not independent mechanism gains.')
    p=read(BASE/'formation/protocol.json');fig,axes=plt.subplots(1,3,figsize=(17,5))
    for seed in (2728,2729):
        for condition in ('coherent','prefix_order_control'):
            run=condition+'_seed'+str(seed);r=read(BASE/'formation/trajectories'/run/'result.json')
            for ax,cohort in zip(axes,('gum','ewt','cmrc')):
                xx=[0];yy=[0]
                for ck in r['checkpoints']:
                    st=next(s for s in ck['strata'] if s['split']=='test' and s['cohort']==cohort)
                    xx.append(ck['step']);yy.append(st['mean_loss_delta'])
                ax.plot(xx,yy,'o-',label=run,lw=1)
                ax.axhline(0,c='gray',lw=.7);ax.set(title=cohort+' : held-out actual CE',xlabel='Native-parameter SGD step',ylabel='Loss change from original FP32 replica');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    keep(fig,'native_parameter_training_trajectories.png','Four true75M-parameter trajectories, with unchanged token targets',
        'OnlyfinalMLP updated, nativecheckpointfiles read-only. Positive losschange is worse. Same seed usesidenticaltarget/source draws acrosscoherent/ordercontrol. Two sample-orderseeds do not establish alltrainingformations; ordercontrol also changesdifficulty. Each plottedcohort32testqueries.')
    r=read(BASE/'confirmation/result.json');fig,axes=plt.subplots(1,3,figsize=(17,5))
    groups=('gum','ewt','cmrc','held_relation_pair')
    for ax,b,metric in zip(axes,(16,35,35),('relative_MSE','relative_MSE','KL')):
        rows=[next(x for x in r['frozen_winner_gains'] if x['block']==b and x['metric']==metric and x['group']==g) for g in groups]
        means=np.array([x['gain']['mean'] for x in rows]);limits=np.array([x['gain']['interval95'] for x in rows]);err=np.stack([means-limits[:,0],limits[:,1]-means])
        ax.errorbar(range(4),means,yerr=err,fmt='o',capsize=4,color='#226f84');ax.axhline(0,c='gray',lw=1)
        ax.set(xticks=range(4),xticklabels=['GUM','EWT','CMRC','heldpair'],title=f'block{b}: {metric}',ylabel='Gain over frozen early baseline');ax.grid(alpha=.2)
    keep(fig,'frozen_new_combination_source_gains.png','Prospective192-window confirmation: state gain versus probability limits',
        '576anchors,143sources. Frozenwinner-minus-frozenbaseline measuredafterfreeze, source-cluster95%bootstrapconditionalonthisfit/sample. Heldpair includes64Englishwindows/192anchors,58sources andoverlapsGUM/EWT; columnsnotindependent. Positive gain isbetter. FinalKLintervals includezero; no robust outputgain claim.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    runs=[x['name'] for x in read(BASE/'formation/trajectories/result.json')['runs']]
    for ax,group in zip(axes,('all','held_relation_pair')):
        rows=[next(x for x in r['training'] if x['run']==run and x['group']==group) for run in runs]
        mean=np.array([x['loss_delta']['mean'] for x in rows]);ci=np.array([x['loss_delta']['interval95'] for x in rows])
        ax.errorbar(range(4),mean,yerr=np.stack([mean-ci[:,0],ci[:,1]-mean]),fmt='o',capsize=4)
        ax.axhline(0,c='gray',lw=1);ax.set(xticks=range(4),xticklabels=['coherent28','order28','coherent29','order29'],title=group+' : new sources',ylabel='Source-equal loss change (positive=worse)');ax.grid(alpha=.2)
    keep(fig,'training_formation_unseen_combination_limits.png','Actual trained parameters on new sources and held relation-type combinations',
        'Allfourfixed64stepFP32states, no confirmation selection. Originalprefixstates/finalnorm/head fixed, fullCE observed. Sourceequalmeans can differ fromqueryweightedmeans. In heldpairgroup allfourmeanchangespositive; this isnot improved combinatorial languageability.')
    newnames={r['path'] for r in new};index['figures']=[r for r in index['figures'] if r['path'] not in newnames]+new
    index['timestamp']=stamp();index['phase2729_source']=snapshot(Path(__file__));save(out/'index.json',index)
    ledger('phase2729_confirmation_figures',time.monotonic()-start,figures=len(new));print('LAW_TRAINING_PREDICTION_FIGURES',len(new),flush=True)


if __name__=='__main__':main()
