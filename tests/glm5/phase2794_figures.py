"""Standalone scientific figure, derived only from frozen real observations."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'));a=read(P.OUT/'analysis.json');s=read(P.OUT/'strong-baselines.json')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    ds=[r for r in a['dose_curve'] if r['condition']=='mlp'];base=ds[-1]['accuracy']+ds[-1]['loss']['mean']
    axes[0].plot([0]+[r['alpha'] for r in ds],[base]+[r['accuracy'] for r in ds],'o-',color='#2979bf',label='Selected 32 MLP units')
    rnd=next(r for r in a['dose_curve'] if r['condition']=='random_mlp');axes[0].scatter([1],[rnd['accuracy']],color='#dd7843',marker='x',s=70,label='Matched random 32')
    axes[0].set(xlabel='Ablation strength',ylabel='Complete-integer accuracy',ylim=(0,1.08),title='No preregistered accuracy effect');axes[0].legend(fontsize=8)
    axes[1].bar(['Scope-blind','Explicit rule*','Mixture'],[a['output_prediction']['scope_blind_KL'],s['rule_KL'],s['graph_KL']],color=['#a7b6c7','#dd7843','#2979bf'])
    axes[1].set(ylabel='Full-vocabulary next-token KL',title='Not superior to the stronger rule*');axes[1].text(.01,.97,'* Posthoc audit; interval includes zero',transform=axes[1].transAxes,va='top',fontsize=8)
    for i,r in enumerate(a['internal_prediction']):
        axes[2].bar(i-.18,r['entity_scope_baseline_mse'],width=.35,color='#a7b6c7',label='Entity x scope mean' if i==0 else None)
        axes[2].bar(i+.18,r['mse'],width=.35,color='#2979bf',label='Conditional linear prediction' if i==0 else None)
    axes[2].set(xticks=[0,1],xticklabels=['MLP: 9,728 units','Hidden: 2,560 dims'],ylabel='Training-standardized MSE',title='B34 heldout state prediction');axes[2].legend(fontsize=8)
    fig.suptitle('Qwen3-4B: reference-assisted source arbitration | 64 familiar entities, new combinations',fontsize=11)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(P.OUT/'research-results.png',dpi=220);fig.savefig(P.OUT/'research-results.pdf');plt.close(fig)
if __name__=='__main__':main()
