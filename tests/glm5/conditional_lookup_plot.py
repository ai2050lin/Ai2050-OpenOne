"""Descriptive figure from sealed CLT results; no feature selection or tuning."""
import os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'tests/glm5/result/conditional_lookup_native_v1'
os.environ['MPLCONFIGDIR']=str(ROOT/'tests/glm5_temp/conditional_lookup_discovery/mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=json.loads((OUT/'discovery-analysis.json').read_text(encoding='utf-8'))
c=json.loads((OUT/'confirm-analysis.json').read_text(encoding='utf-8'))
fig,ax=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
for k,color in [(1,'#8573b5'),(32,'#087e8b')]:
    rows=[r for r in d['grid'] if r['key'].startswith('h') and r['k']==k]
    ax[0].plot([int(r['key'][1:]) for r in rows],[r['joint'] for r in rows],label=f'{k} native coordinates',color=color)
ax[0].axhline(.25,color='gray',linestyle=':',label='Balanced joint chance')
ax[0].set(title='Discovery: selected OOF information readout',xlabel='Residual boundary after block count',ylabel='Both row bits correct',ylim=(0,1.05))
ax[0].legend(fontsize=8);ax[0].grid(alpha=.15)
labels=['Selected']+list(c['comparisons']);values=[c['encoding']['joint']]+[s['joint'] for s in c['comparisons'].values()]
ax[1].bar(labels,values,color=['#087e8b']+['#9da7b1']*(len(labels)-1))
for i,v in enumerate(values):ax[1].text(i,v+.015,f'{v:.1%}',ha='center',fontsize=9)
ax[1].set(title='Confirmation: frozen readouts, 128 families',ylabel='Both row bits correct',ylim=(0,1.12))
ax[1].tick_params(axis='x',labelsize=8);ax[1].grid(axis='y',alpha=.15)
fig.suptitle('Qwen3-14B independent two-row lookup | Decoding is not causal use',fontsize=12)
fig.savefig(OUT/'native-readout-summary.png',dpi=170)
print(OUT/'native-readout-summary.png')

# Every frozen coordinate is displayed in frozen order. No post-confirm picking.
import numpy as np
seal=json.loads((OUT/'candidate.json').read_text(encoding='utf-8'))
model=np.load(OUT/'readout-models.npz');idx=model['row_idx'];mu=model['row_mu'];sd=model['row_sd']
cards={};grids=[]
for split in ('discovery','confirm'):
    rows=json.loads((OUT/f'{split}-materials.json').read_text(encoding='utf-8'))
    chunks=[]
    for p in sorted(OUT.glob(split+'-??.npz')):
        with np.load(p) as z:chunks.append(z[seal['row_key']][:,idx])
    x=(np.concatenate(chunks).astype(np.uint32)<<16).view(np.float32)
    means=[]
    for r0,r1 in ((0,0),(0,1),(1,0),(1,1)):
        keep=[i for i,r in enumerate(rows) if r['late'] and r['b']==0 and (r['r0'],r['r1'])==(r0,r1)]
        means.append(x[keep].mean(0))
    means=np.array(means);cards[split]={'raw_means_4_by_32':means.tolist(),'n_families_per_cell':len(keep)}
    grids.append(((means-mu)/sd).T)
cards.update(provenance=seal['provenance'],key=seal['row_key'],coordinates=idx.tolist(),cell_order=['00','01','10','11'],standardization='frozen discovery coordinate mean/std; display only; no new selected features',candidate_sha=c['candidate_sha'])
(OUT/'native-coordinate-card.json').write_text(json.dumps(cards,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(8,9),layout='constrained');lim=max(abs(g).max() for g in grids)
for ax,g,split in zip(axes,grids,['Discovery','Confirmation']):
    im=ax.imshow(g,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),['00','01','10','11']);ax.set_yticks(range(len(idx)),idx,fontsize=7)
    ax.set(title=split,xlabel='Independent row bits r0 r1',ylabel='Native coordinate (zero based)')
fig.colorbar(im,ax=axes,label='Mean activation / frozen discovery standard deviation',shrink=.7)
fig.suptitle('Frozen 32 coordinates at h18 | Information patterns, not causal units')
fig.savefig(OUT/'native-coordinate-card.png',dpi=160)
print(OUT/'native-coordinate-card.png')
