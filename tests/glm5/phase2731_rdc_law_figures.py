"""Native behavior, model replication and independent own-history evidence plots."""
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
    analysis=read(BASE/'deployment/paired_analysis.json');p=read(BASE/'deployment/protocol.json')
    names=p['rollout_branches'];short=['native','early16','early35','coherent28','order28','coherent29','order29']
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,cohort in zip(axes,('squad_qa','cmrc_qa','hotpot_qa')):
        rows=[next(r for r in analysis['strata'] if r['branch']==b and r['group']==cohort) for b in names]
        for metric in ('EM','F1','EOS_fraction'):ax.plot(range(7),[r[metric] for r in rows],'o-',label=metric)
        ax.set(xticks=range(7),xticklabels=short,ylim=(-.03,1.03),title=cohort+' :16 originalquestions',ylabel='Fullanswer/stop score');ax.tick_params(axis='x',labelrotation=30);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    keep(fig,'native_seven_branch_full_answers.png','Seven native-model branches: complete answers and stop behavior',
        'Same48 humanQAquestions,16/cohort, greedy48newtokencap. Eachbranch owns its history/KV. Nativeanswers and errors retained. EM=fullnormalizedstringmatch, F1=answer token/character overlap, EOS=actualnative stop. No firsttoken substitution for fullanswer quality; naturalcontinuation has no uniquecorrectanswer.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,metric in zip(axes,('mean_new_tokens','repeated4gram')):
        for cohort in ('gum','ewt','cmrc'):
            rows=[next(r for r in analysis['strata'] if r['branch']==b and r['group']==cohort) for b in names]
            ax.plot(range(7),[r[metric] for r in rows],'o-',label=cohort)
        ax.set(xticks=range(7),xticklabels=short,title='48naturalprefixes: '+metric,ylabel=metric);ax.tick_params(axis='x',labelrotation=30);ax.grid(alpha=.2)
    axes[-1].legend()
    keep(fig,'natural_own_history_behavior.png','Natural continuation: complete trajectories, lengths and repetitions',
        '16naturalprefixes/cohort from frozenconfirmation, English includesheldrelationcooccurrences. Repetitionandlength are diagnostics, not standalone semanticquality or proof of collapse. Native48tokenlengthcap stated; alltokens/histories retained.')
    fig,axes=plt.subplots(1,3,figsize=(17,5));models=('qwen4','qwen14','glm4');decoders=('direct_mlp','predicted_x_native','product_of_predicted_factors','predicted_joint_product')
    for ax,model in zip(axes,models):
        result=read(BASE/'scale'/model/'result.json')
        for decoder in decoders:
            yy=[next(r for r in result['confirmation'] if r['decoder']==decoder and r['group']==g)['relative_MSE'] for g in ('gum','ewt','cmrc','held_relation_pair')]
            ax.plot(range(4),yy,'o-',label=decoder)
        ax.set(xticks=range(4),xticklabels=['GUM','EWT','CMRC','window-held'],title=f"{model}: D={result['native_width']}, M={result['native_units']}",ylabel='NativeMLP relativeMSE');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    keep(fig,'three_models_native_full_coordinate_predictions.png','Matched-source three-model native-coordinate confirmation',
        '144naturalwindows/model, own72train24validation48confirmation, threecharacteralignedanchors. Ownvalidationfreezes4decoderchoices; allnativecoordinates/units used. Q4matchedsubset is reanalysis. Window-held overlapsEnglishcohorts and includes earlyprefixes before bothrelationendpoints; own-tokenizer endpoint-visible audit is separately recorded in paired_analysis.json. LocalMLPerror is not fullvocabKL or languageability; unequalarchitectures/tokenization preclude a causal modelsize interpretation.')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,cohort in zip(axes,('squad_qa','cmrc_qa','hotpot_qa')):
        for metric in ('EM','F1','EOS'):
            ax.plot(range(3),[read(BASE/'scale'/m/'result.json')['QA_summary'][cohort][metric] for m in models],'o-',label=metric)
        ax.set(xticks=range(3),xticklabels=models,ylim=(-.03,1.03),title=cohort+' :8 matchedquestions',ylabel='Native fullanswer/stop score');ax.grid(alpha=.2)
    axes[-1].legend()
    keep(fig,'three_models_matched_native_answers.png','Original nonquantized local models:24 identical human questions',
        'Ownnativechattemplates, common32newtokencap, greedyEOS. Differentfrommain4B48token experiment.8questions/cohort is boundedcrossmodelreplication, not a broadbenchmark. Actualquestions/outputs/answers preserved, no exclusion for failednativebehavior.')
    own=read(BASE/'own_history/result.json');bs=read(BASE/'own_history/protocol.json')['branches'];labels=['early16','early35','coherent28','order28']
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for group in ('natural','QA'):
        rr=[next(r for r in own['summaries'] if r['branch']==b and r['kind']==group) for b in bs]
        axes[0].plot(range(4),[r['source_mean_KL']['mean'] for r in rr],'o-',label=group)
        axes[1].plot(range(4),[r['source_argmax_agreement']['mean'] for r in rr],'o-',label=group)
        axes[2].plot(range(4),[r['all_KV_equal_checks']/r['total_KV_checks'] for r in rr],'o-',label=group)
    for ax,title in zip(axes,('Fullvocab KL on same OWN history','Samehistory nexttoken agreement','All-layerKV exact equality fraction')):
        ax.set(xticks=range(4),xticklabels=labels,title=title);ax.grid(alpha=.2)
    axes[2].set_ylim(-.03,1.03);axes[-1].legend()
    keep(fig,'same_own_history_output_vs_native_KV.png','No oracle feedback: separating output-rule error from direct cache effects',
        '24existingprefixes x4fixedbranches, independentnativecompanion fedONLYthe samealreadychosenbranchIDs. Allvocabularyprobabilities eachstep, everyKVentry atfirst/every8/finalstep. Nativecache neverinjected. EqualKV can coexistwith outputerror afterlastMLPchanges; differentselectedhistories are not used to assert cachecorruption. This is diagnosticreanalysis, not newconfirmation.')
    names={r['path'] for r in new};index['figures']=[r for r in index['figures'] if r['path'] not in names]+new
    index.update(timestamp=stamp(),phase2731_source=snapshot(Path(__file__)));save(out/'index.json',index)
    ledger('law_native_history_scale_figures',time.monotonic()-start,figures=len(new));print('LAW_NATIVE_SCALE_FIGURES_COMPLETE',len(new),flush=True)


if __name__=='__main__':main()
