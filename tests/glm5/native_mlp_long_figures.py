"""Export real frozen-output diagnostics, never illustrative model results."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_long_materials as P
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    a=P.read(P.OUT/'analysis.json');d=P.read(P.OUT/'dose-result.json');c=P.read(P.OUT/'capability-result.json');z=P.read(P.OUT/'diagnostics.json')
    cards=[r for r in a['cards'] if r['selection']=='stratified_random'];fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),layout='constrained')
    axes[0].scatter([100*r['confirm']['delta'] for r in cards],[100*r['lockbox']['delta'] for r in cards],s=10,c='#2979bf',alpha=.5)
    axes[0].axvline(15,color='#dd7843',ls='--');axes[0].axhline(15,color='#dd7843',ls='--');axes[0].set(xlabel='Confirm accuracy gain (pp)',ylabel='Lockbox accuracy gain (pp)',title='0 / 432 pass both preregistered gates')
    axes[0].text(.02,.97,'Matched token + absolute position\nAdditional Bonferroni p and tie gates',va='top',transform=axes[0].transAxes,fontsize=8)
    effects=d['effects'];xs=[0,.5,1];ys=[effects[s]['accuracy'] for s in ['base','target_half','target_full']]
    axes[1].plot(xs,ys,'o-',c='#2979bf',label='Selected16');axes[1].scatter([1],[effects['random_full']['accuracy']],color='#dd7843',marker='x',s=70,label='Matched random16');axes[1].set(ylim=(.38,.48),xlabel='Query-only ablation strength',ylabel='Natural next-token top1 accuracy',title='Joint effect gate not supported');axes[1].legend(fontsize=8)
    lo,hi=np.asarray(c['loss_interval'])*100;loss=c['net_loss']*100;axes[2].errorbar([loss],[0],xerr=[[loss-lo],[hi-loss]],fmt='o',color='#2979bf',capsize=6);axes[2].axvline(2,color='#dd7843',ls='--',label='NI margin: 2 pp');axes[2].set(xlabel='Keyed capability accuracy loss (pp)',yticks=[],xlim=(-.6,2.4),ylim=(-1,1),title='2,200-item repeated-pool NI gate passed');axes[2].legend(fontsize=8)
    fig.suptitle('Qwen3-4B | 2,048 natural documents x 1,024 tokens | finite operational context model',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(P.OUT/'research-results.png',dpi=200);fig.savefig(P.OUT/'research-results.pdf');plt.close(fig)
if __name__=='__main__':main()
