"""Validation calibration, full-vocabulary transfer and format-prefix boundaries."""
from rdc_formation_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.monotonic();folder=OUT/'figures/probability_v2';finish=folder/'index.json'
    if finish.exists():
        assert all(sha(BASE/r['path'])==r['sha256'] for r in read(finish)['figures'])
        return
    folder.mkdir(parents=True,exist_ok=True);figures=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def keep(fig,name,caption,paths):
        path=folder/(name+'.png');fig.savefig(path,dpi=160,facecolor='white',bbox_inches='tight');plt.close(fig)
        figures.append({'name':name,'path':path.relative_to(BASE).as_posix(),'sha256':sha(path),'caption':caption,
            'sources':[{'path':p.relative_to(BASE).as_posix(),'sha256':sha(p)} for p in paths],
            'coordinate_policy':'Scalar complete-vocabulary/source-group summaries, not a compressed state representation. All candidate directions and both heldout splits retained.'})
    calpath=OUT/'calibration_analysis/result.json';radpath=OUT/'radius_analysis/result.json'
    cal=read(calpath)['paired'];rad=read(radpath)['records']
    names=[('true_token','identity','True target'),('within_surface_permuted','identity','Class-matched shuffle'),
        ('surface_class_only','identity','Surface-class mass'),('true_token','reverse','Reversed true'),
        ('true_token','coordinate_shuffle','Index-shuffled true')]
    actual_conditions=read(OUT/'training/protocol.json')['conditions']
    names[1]=(actual_conditions[1],'identity',names[1][2]);names[2]=(actual_conditions[2],'identity',names[2][2])
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained',sharey=True)
    for ax,calibrated in zip(axes,[False,True]):
        for ni,(condition,transform,label) in enumerate(names):
            for si,seed in enumerate([2747,2748]):
                matching=lambda v:v['precision']=='native_BF16' and v['seed']==seed and v['radius_factor']==1 and v['condition']==condition and v['transform']==transform
                if calibrated:
                    r=next(r for r in cal if matching(r['variant']) and r['split']=='prospective_source_confirmation')
                    value=r['calibrated_minus_own_precision_baseline_NLL']
                else:
                    r=next(r for r in rad if matching(r['variant']))
                    value=next(r for r in r['endpoint_reports'] if r['split']=='prospective_source_confirmation' and r['family']=='all')['NLL_minus_baseline']
                y=ni+(si-.5)*.16;color=['#216485','#b46727'][si]
                ax.hlines(y,*value['interval95'],color=color);ax.plot(value['mean'],y,'o' if si==0 else '^',color=color,
                    label='Seed '+str(seed) if ni==0 else None)
        ax.axvline(0,color='gray',linestyle='--');ax.grid(axis='x',alpha=.2)
        ax.set(title='Independent validation calibration' if calibrated else 'Uncalibrated original readout',
            xlabel='192 new-source NLL change from the same treatment of native baseline')
        ax.legend(loc='lower right',fontsize=9)
    axes[0].set_yticks(range(5),[r[2] for r in names]);axes[0].invert_yaxis()
    fig.suptitle('Matched actual BF16 parameter radius: probability gains depend on calibration\n95% source intervals are conditional on each fixed seed and validation choice; not a unique semantic residual',fontsize=13)
    keep(fig,'calibrated_matched_radius','Every predeclared direction at the full matched nativeBF16 radius; two fixed seeds. Each variant and native reference is separately validation-calibrated. Lower NLL does not imply changed token ranking.',[calpath,radpath])
    rdpath=OUT/'transfer/readout_result.json';rd=read(rdpath)
    directions=['python_to_en','en_to_python','zh_to_en','en_reordered_to_en']
    labels={'python_to_en':'Python -> English','en_to_python':'English -> Python',
        'zh_to_en':'Chinese -> English','en_reordered_to_en':'Reordered -> English'}
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained',sharey=True)
    ys=[];yl=[]
    for di,direction in enumerate(directions):
        for si,split in enumerate(['test','mixed_holdout']):
            y=di*2+si;ys.append(y);yl.append(labels[direction]+' / '+split.replace('_',' '))
            for ax,control in zip(axes,['identity','shuffled_pair']):
                entry=next(r for r in rd['paired_mapping_comparisons'] if r['direction']==direction and r['split']==split
                    and r['query_split']=='unseen_query' and r['control']==control)
                v=entry['control_minus_mapped_KL'];color='#216485' if si==0 else '#b46727'
                ax.hlines(y,*v['interval95'],color=color);ax.plot(v['mean'],y,'o',color=color)
    for ax,control in zip(axes,['Unmapped source response','Wrong-pair fitted mapping']):
        ax.axvline(0,color='gray',linestyle='--');ax.grid(axis='x',alpha=.2)
        ax.set(title=control+' as comparator',xlabel='Comparator KL minus query-conditioned mapping KL\nPositive favors the mapping')
    axes[0].set_yticks(ys,yl);axes[0].invert_yaxis()
    fig.suptitle('Unseen-query full-vocabulary prediction: mapping gain is not pairing-specific everywhere\n32 semantic groups per split; all original 151,936 output tokens. Fitted mapping remains frozen.',fontsize=13)
    keep(fig,'full_vocabulary_mapping_comparators','Every direction and heldout split; source-group paired intervals. Source response is already observed and supplied. These are text representations, not proof of cross-modal semantic isomorphism.',[rdpath])
    apath=OUT/'transfer/analysis/result.json';ans=read(apath)['summary']
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained',sharey=True)
    for di,direction in enumerate(directions):
        for si,split in enumerate(['test','mixed_holdout']):
            y=di*2+si
            for bi,bias in enumerate(['none','uniform_digit_plus8']):
                r=next(r for r in ans if r['direction']==direction and r['split']==split and r['candidate']=='query_conditioned' and r['bias']==bias)
                color=['#216485','#b46727'][bi];mark=['o','^'][bi]
                for ax,metric in zip(axes,['gold_digit_NLL','digit_rank_correct']):
                    v=r['metrics'][metric];offset=(bi-.5)*.16
                    ax.hlines(y+offset,*v['interval95'],color=color);ax.plot(v['mean'],y+offset,mark,color=color,
                        label=['No bias','All 1..8 digits +8'][bi] if y==0 else None)
    axes[0].set(xlabel='Correct literal digit NLL at the first next-token position',title='Absolute probability can change substantially')
    axes[1].set(xlabel='Correct digit wins among 1..8 (same before/after bias)',title='Conditional digit order is unchanged by construction',xlim=(-.02,.55))
    for ax in axes:ax.grid(axis='x',alpha=.2)
    axes[1].legend(loc='lower right',fontsize=9)
    axes[0].set_yticks(ys,yl);axes[0].invert_yaxis()
    fig.suptitle('Fixed Answer query: literal-digit scoring is not complete-answer correctness\nThese prefixes can emit Markdown before a number; full-vocabulary correct-digit argmax is zero in all displayed cases.',fontsize=13)
    keep(fig,'digit_probability_order_boundary','All8direction/split pairs and both bias controls. First literal-digit NLL excludes later Markdown/whitespace handling; own-history complete-answer scoring is separate. Intervals are semantic-group resamples.',[apath,rdpath])
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'figures':figures,
        'visual_review':'Pending actual main-agent inspection; rendering success alone is not visual QA.',
        'seconds':time.monotonic()-start,'phase_complete':False}
    save(finish,value);ledger('phase2747_probability_figures',value['seconds'])
    print('FORMATION_PROBABILITY_FIGURES',len(figures),flush=True)


if __name__=='__main__':main()
