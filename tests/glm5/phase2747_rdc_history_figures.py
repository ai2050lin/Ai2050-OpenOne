"""Complete own-history outcomes; no probability score substituted for answers."""
from rdc_formation_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.monotonic();folder=OUT/'figures/history_v1';finish=folder/'index.json'
    if finish.exists():
        assert all(sha(BASE/r['path'])==r['sha256'] for r in read(finish)['figures']);return
    opath=OUT/'own_history/analysis/result.json';ppath=OUT/'program_own_history/analysis/result.json'
    own=read(opath);program=read(ppath);tpath=OUT/'program_own_history/terminal_review/result.json';terminal=read(tpath)
    lpath=OUT/'own_history/terminal_review/result.json';language_review=read(lpath);assert language_review['all_passed']
    assert terminal['all_passed']
    assert own['all_passed'] and own['complete_runs']==9 and not own['partial'] and program['all_passed']
    folder.mkdir(parents=True,exist_ok=True);figures=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def keep(fig,name,caption,paths):
        path=folder/(name+'.png');fig.savefig(path,dpi=160,facecolor='white',bbox_inches='tight');plt.close(fig)
        figures.append({'name':name,'path':path.relative_to(BASE).as_posix(),'sha256':sha(path),'caption':caption,
            'sources':[{'path':p.relative_to(BASE).as_posix(),'sha256':sha(p)} for p in paths],
            'coordinate_policy':'Complete declared sample counts, not a compressed state model or discovered coordinate backbone. Full collected native arrays remain independently queryable.'})
    labels=[];records=[]
    variants={'true_token':'True target','within_surface_class_permuted_token':'Class-matched shuffle','surface_class_mass':'Class mass'}
    ordered=sorted(own['reports'],key=lambda r:(r['variant']!='native',{'qwen4':0,'qwen14':1,'glm4':2}[r['model']],r['variant']))
    for r in ordered:
        label=r['model']+' native' if r['variant']=='native' else variants[r['variant'].rsplit('_',1)[0]]+' '+r['variant'].rsplit('_',1)[1]
        labels.append(label);summary=r['summary'];allrow=next(v for v in summary if v['family']=='all')
        natural=[v for v in summary if v['family'].startswith('natural_')]
        assert allrow['controlled_expressions']==320 and allrow['pairs']==160 and sum(v['expressions'] for v in natural)==192
        records.append({'label':label,'correct':allrow['correct_and_stopped'],'pairs':allrow['both_worlds_correct_and_stopped'],
            'strict':allrow['strict_answer_only'],'natural_EOS':sum(v['EOS'] for v in natural),
            'natural_censored':sum(v['censored'] for v in natural),'shape_mismatches':allrow['first_B1_B8_argmax_mismatches']})
    save(folder/'native_trained_counts.json',{'source_sha256':sha(opath),'records':records})
    fig,axes=plt.subplots(1,3,figsize=(17,7),layout='constrained',sharey=True);y=np.arange(9)
    for i,(field,denom,color,label) in enumerate([('correct',320,'#216485','Correct and EOS / 320 expressions'),
        ('pairs',160,'#b46727','Both worlds correct and EOS / 160 pairs')]):
        yy=y+(i-.5)*.28;values=[r[field]/denom for r in records]
        axes[0].barh(yy,values,height=.25,color=color,label=label)
        for iy,v,r in zip(yy,values,records):axes[0].text(v+.01,iy,f'{r[field]}/{denom}',va='center',fontsize=8)
    axes[0].set(xlim=(0,1.18),title='Controlled content plus stopping',xlabel='Fraction of the frozen evaluation set')
    axes[0].legend(loc='lower right',fontsize=8)
    for field,color,label in [('natural_EOS','#216485','EOS'),('natural_censored','#bbbbbb','Reached 96-token cap')]:
        values=np.array([r[field]/192 for r in records]);left=np.zeros(9) if field=='natural_EOS' else np.array([r['natural_EOS']/192 for r in records])
        axes[1].barh(y,values,left=left,color=color,height=.6,label=label)
    for iy,r in enumerate(records):axes[1].text(.5,iy,f"EOS {r['natural_EOS']}/192; cap {r['natural_censored']}/192",ha='center',va='center',fontsize=8)
    axes[1].set(xlim=(0,1),title='Natural continuation: no unique gold',xlabel='Own-token steps; tokenizer units differ by model')
    axes[1].legend(loc='lower right',fontsize=8)
    axes[2].barh(y,[r['shape_mismatches'] for r in records],height=.6,color='#7b608e')
    for iy,r in enumerate(records):axes[2].text(r['shape_mismatches']+.3,iy,str(r['shape_mismatches']),va='center',fontsize=9)
    axes[2].set(xlim=(0,max(r['shape_mismatches'] for r in records)*1.25+2),title='Execution-shape control',xlabel='B1 / original B8 first-token mismatches out of 512')
    axes[0].set_yticks(y,labels);axes[0].set_ylim(10,-.7)
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Actual own histories: native models and six final Q4 parameter deployments\nPlots retain frozen parsing; trained endpoints are NOT radius-matched or temperature-calibrated.\nGLM separate unblinded terminal review: 233/320 and81/160 pairs (primary230/320 and79/160). Not whole-chain grading.',fontsize=12)
    keep(fig,'native_trained_own_history_counts','Nine full512expression runs,320controlledexpressions/160paired worlds and192natural prefixes each. Frozen primary scores remain; five GLM stopped-unparsed outputs have a separate unblinded terminal review, not a whole-reasoning grade. Own-history outcomes are distinct from fixed-prefix probability improvements. B1/B8 execution effects stay visible.',[opath,lpath])
    names={'native':'Native target','code_identity':'Observed code directly','mapped_code':'Query-conditioned map',
        'shuffled_map':'Wrong-pair map','mapped_digit_bias':'Map + all digits +8','mapped_letter_bias':'Map + all letters +8'}
    rows=[r for r in program['summary'] if r['depth']=='all'];assert len(rows)==6 and all(r['groups']==32 for r in rows)
    yy=np.arange(6);labels=[names[r['branch']] for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained',sharey=True)
    for i,(key,color,label) in enumerate([('correct_and_stopped','#216485','Correct and EOS'),('EOS','#b46727','Any EOS')]):
        axes[0].barh(yy+(i-.5)*.3,[r[key] for r in rows],height=.27,color=color,label=label)
        for yv,r in zip(yy+(i-.5)*.3,rows):axes[0].text(r[key]+.3,yv,str(r[key]),va='center',fontsize=8)
    axes[0].set(xlim=(0,35),title='Conservative complete-answer parser',xlabel='Count / 32 semantic groups');axes[0].legend(loc='lower right',fontsize=8)
    for yv,r in zip(yy,rows):
        v=r['paired_correct_change'];axes[1].hlines(yv,*v['interval95'],color='#216485');axes[1].plot(v['mean'],yv,'o',color='#216485')
    axes[1].axvline(0,color='gray',linestyle='--');axes[1].set(title='Paired change from native',xlabel='Correct-and-EOS fraction change\n95% semantic-group resampling interval')
    axes[2].barh(yy,[r['same_complete_tokens_as_native'] for r in rows],height=.6,color='#7b608e')
    for yv,r in zip(yy,rows):axes[2].text(r['same_complete_tokens_as_native']+.2,yv,str(r['same_complete_tokens_as_native']),va='center',fontsize=9)
    axes[2].set(xlim=(0,35),title='Whole generated token sequence unchanged',xlabel='Count / 32 pairs of native and branch trajectories')
    axes[0].set_yticks(yy,labels);axes[0].set_ylim(6.9,-.7)
    for ax in axes:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    supplement=next(r for r in terminal['summary'] if r['branch']=='code_identity')['supplemental_terminal_correct_and_stopped']
    fig.suptitle('Only the initial readout changes; later generation uses native parameters and each own history\nObserved code response is extra input. Plots retain the frozen parser and 256-token cap.\nSupplemental unblinded terminal review: direct code '+str(supplement)+'/32; other branches unchanged. Original scores are not rewritten.',fontsize=12)
    keep(fig,'program_one_shot_own_history_counts','32heldout semanticgroups, sixpaired readout branches. Equal first chosen token with untouched initialKV and laterparameters implies equal latergreedy computation, verified on every such case. Capped output is not accepted as completed; separate post-outcome terminal review accounts for the single unparsed EOS case.',[ppath,tpath])
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'figures':figures,
        'visual_review':'Pending actual main-agent inspection. Rendering alone is not visual QA.','seconds':time.monotonic()-start}
    save(finish,result);ledger('phase2747_history_figures',result['seconds']);print('FORMATION_HISTORY_FIGURES',len(figures),flush=True)


if __name__=='__main__':main()
