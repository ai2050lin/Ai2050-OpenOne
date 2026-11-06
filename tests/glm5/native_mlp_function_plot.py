"""Publication figure from the immutable real measurements; no fitted embellishment."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    a=P.read(P.OUT/'analysis.json');d=P.read(P.OUT/'dose-result.json');c=P.read(P.OUT/'capability-result.json');pol=P.read(P.OUT/'place-polarity/polarity-result.json')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8.5),layout='constrained')
    x=np.arange(3)
    for j,(split,color) in enumerate([('confirm','#327db7'),('lockbox','#d47d45')]):
        vals=[f['results'][split]['A']['accuracy'] for f in a['findings']];ax[0,0].bar(x+(j-.5)*.3,vals,.28,label=split,color=color)
    ax[0,0].set_xticks(x,[f['family']+(' *' if not f['behavior_gate'] else '') for f in a['findings']]);ax[0,0].set_ylim(0,1.1);ax[0,0].axhline(.65,c='gray',ls=':');ax[0,0].legend();ax[0,0].set_title('A. Preselected native units: paired response order\n* held-out behavior gate failed')
    rows=[f for f in pol['findings'] if f['stage']==0];x=np.arange(len(rows))
    for i,(key,color) in enumerate([('fact_order','#a4acb6'),('answer_order','#327db7')]):
        vals=[min(r['results'][q][key]['accuracy'] for q in ['0','1']) for r in rows];ax[0,1].bar(x+(i-.5)*.35,vals,.32,color=color,label=key)
    ax[0,1].set_xticks(x,[f'B{r["layer"]}\nU{r["unit"]}' for r in rows]);ax[0,1].set_ylim(0,1.13);ax[0,1].legend(loc='lower left');ax[0,1].set_title('B. Fresh, name-balanced question inversion\nWorst order accuracy across both question polarities')
    for group,color in [('native','#d47d45'),('coordinate','#327db7')]:
        cond=['base',group+'_half',group+'_full'];vals=[d['effects'][k]['pair_both_correct'] for k in cond];ax[1,0].plot([0,.5,1],vals,'o-',color=color,label=group)
    ax[1,0].set_ylim(.78,.95);ax[1,0].set_xlabel('Last-answer-query ablation dose');ax[1,0].set_ylabel('Both variants correct');ax[1,0].legend();ax[1,0].set_title('C. Joint ablation: correctness unchanged\n256 unique background pairs; loss interval +/-2.23 pp')
    for i,(group,r) in enumerate(c['groups'].items()):
        v=r['net_loss']*100;lo,hi=np.array(r['ci95'])*100;ax[1,1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color='#327db7')
    ax[1,1].set_yticks(range(2),list(c['groups']));ax[1,1].axvline(0,c='gray',ls=':');ax[1,1].axvline(2,c='#d47d45',ls='--',label='NI margin 2 pp');ax[1,1].set_xlabel('Net accuracy loss (percentage points)');ax[1,1].legend();ax[1,1].set_title('D. Capability: 2200 previously exposed questions\nConservative four-tail paired 95% intervals')
    fig.suptitle('Qwen3-4B / BF16 / NMLP-FUNCTION-20261104-v4\nScoped response evidence, polarity counterexamples, negative operations',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(P.OUT/f'research-results.{ext}',dpi=180)
    print(json.dumps({'figure':str(P.OUT/'research-results.png')}))
if __name__=='__main__':main()
