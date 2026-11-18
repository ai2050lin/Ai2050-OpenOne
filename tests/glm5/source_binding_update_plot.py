"""Plots from saved update diagnostics; no fitting or selection."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
os.environ['MPLCONFIGDIR']=str(D.ROOT/'tests/glm5_temp/source_binding_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def run(model):
    root=D.OUT/model;panels=[s for s in ['confirm','lockbox_frame'] if (root/f'{s}-conditional-output.json').exists()]
    if not panels:return
    fig,axes=plt.subplots(3,len(panels),figsize=(7.5*len(panels),10),layout='constrained',sharey='row',squeeze=False)
    names=['layer_mean','h32','h16_prefix16'];labels=['Layer mean','Last state32','Last state16 + prefix16']
    for col,split in enumerate(panels):
        result=D.read(root/f'{split}-conditional-update.json');xs=[x['block']+1 for x in result['steps']]
        for name,label in zip(names,labels):
            steps=result['steps'];axes[0,col].plot(xs,[x['methods'][name]['one_step_error_vs_mean'] for x in steps],label=label)
            axes[1,col].plot(xs,[x['methods'][name]['rollout_squared_error']/x['methods']['layer_mean']['rollout_squared_error'] for x in steps],label=label)
        axes[0,col].set_title(split);axes[0,col].set_ylabel('One-step MSE / mean baseline');axes[1,col].set_ylabel('Recursive state MSE / mean baseline')
        for row in [0,1]:axes[row,col].set_xlabel('Predicted raw boundary');axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
        output=D.read(root/f'{split}-conditional-output.json')['summary'];vals=[output[n]['KL_mean'] for n in names]
        axes[2,col].bar(np.arange(3),vals);axes[2,col].set_xticks(np.arange(3),labels,rotation=15);axes[2,col].set_ylabel('Final KL(true model || prediction)')
        for i,n in enumerate(names):axes[2,col].text(i,vals[i],f"top1 match {output[n]['top1_agreement']:.1%}",ha='center',va='bottom',fontsize=8)
        axes[2,col].set_ylim(0,max(vals)*1.15)
    fig.suptitle(f'{model}: conditional last-token rollout; original past-token KV remains available\nQwen exploratory; GLM recipe fixed before its execution',fontsize=12)
    fig.savefig(root/'conditional-update-diagnostic.png',dpi=160);plt.close(fig)

if __name__=='__main__':run(sys.argv[1])
