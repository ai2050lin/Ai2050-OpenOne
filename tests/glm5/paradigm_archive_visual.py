"""Render a layer-by-token map from the archived discovery field."""

import os

import numpy as np

import paradigm_archive_field as A
from deploy.storage_policy import check_write


def render():
    os.environ['MPLCONFIGDIR']=str(A.TMP/'matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from transformers import AutoTokenizer

    tok=AutoTokenizer.from_pretrained(A.R.MODEL,local_files_only=True)
    rows=A.M.material(tok,'discovery')[1][::8]
    data=A.load('discovery-cross')
    assert [r['id'] for r in rows]==data['ids'][::8].tolist()
    positions=np.arange(A.M.F.WIDTH)
    valid=positions[None,:]<=np.array([r['query'][0] for r in rows])[:,None]
    end=int(np.max([r['query'][0] for r in rows]))+1
    n=valid.sum(0).clip(1)
    response=data['trace'].reshape(len(rows),4,40,-1,8).astype(np.float32)
    residual=(np.abs(response[...,3]).mean(1)*valid[:,None]).sum(0)/n
    gate=(np.abs(response[...,5]).mean(1)*valid[:,None]).sum(0)/n
    read=np.abs(data['map'].astype(np.float32)).max(-1)
    source=(read*valid[:,None]).sum(0)/n
    plan=A.R.read('candidate.json')
    arc=np.array([r['spans']['archive'] for r in rows])
    pick=[int(np.median([r['query'][0] for r in rows])),
          int(np.median([r['spans']['grammar'][-1] for r in rows]))]
    fig,axes=plt.subplots(3,1,figsize=(10.2,8.2),sharex=True,layout='constrained')
    channels=((residual,'Absolute residual read difference','viridis'),
              (source,'Largest head archive write difference','magma'),
              (gate,'Mean absolute MLP gate difference','cividis'))
    for ax,(matrix,title,color) in zip(axes,channels,strict=True):
        field=np.log10(np.maximum(matrix[:,:end],1e-5))
        im=ax.imshow(field,origin='lower',aspect='auto',interpolation='nearest',cmap=color,
                     norm=Normalize(float(np.quantile(field,.08)),float(np.quantile(field,.99))))
        fig.colorbar(im,ax=ax,label='log10 difference',pad=.01)
        ax.axvspan(int(arc.min())-.5,int(arc.max())+.5,color='white',alpha=.1)
        for r in plan['routes']:
            ax.scatter(pick[r['site_index']],r['layer'],facecolor='none',edgecolor='white',s=42,linewidth=1.25)
        ax.set_title(title,loc='left')
        ax.set_ylabel('Qwen block')
    axes[-1].set_xlabel('Prompt token position  ·  shaded = archive, circles = frozen sites')
    fig.suptitle('Discovery pairs with equal-length archive replacements',fontsize=14)
    path=check_write(A.OUT/'full-field.png',5*2**20,operation='绘制档案全场')
    fig.savefig(path,dpi=170)
    plt.close(fig)


if __name__=='__main__':
    A.setup()
    render()
