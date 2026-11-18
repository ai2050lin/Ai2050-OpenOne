"""All-head native write maps and full-coordinate MLP response view for the held-out panel."""

from pathlib import Path
import os

import numpy as np

import paradigm_routes as X
from deploy.storage_policy import checked_dir,check_write


def run():
    os.environ['MPLCONFIGDIR']=str(checked_dir(X.TEMP/'plot-cache',2**20,operation='范式图临时缓存'))
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    output=X.OUT
    with np.load(output/'confirm-maps.npz',allow_pickle=False) as z:
        writes=z['head_effect'].mean(0)
        unit=z['unit_axis_mean']
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
    for axis,index,title in zip(axes,(0,1,3),('Requested format','Archived example format','Format interaction')):
        panel=writes[index]
        limit=max(abs(panel).max(),1e-5)
        picture=axis.imshow(panel,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        axis.set_title(title)
        axis.set_xlabel('Native head index')
        axis.set_ylabel('Native block index')
        fig.colorbar(picture,ax=axis,label='Local print minus The write')
    fig.savefig(check_write(output/'attention-write-map.png',4*2**20,operation='原生注意力响应图'),dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,6),constrained_layout=True)
    for axis,index,title in zip(axes,(0,3),('Requested-format effect, all native MLP units','Request/example interaction, all native MLP units')):
        panel=unit[index]
        limit=max(abs(panel).max(),1e-6)
        image=axis.imshow(panel,origin='lower',aspect='auto',cmap='RdBu_r',interpolation='none',
                          norm=SymLogNorm(linthresh=limit*.001,vmin=-limit,vmax=limit))
        axis.set_title(title)
        axis.set_xlabel('Native unit index within block')
        axis.set_ylabel('Block')
        ticks=np.asarray([-1,-.1,-.01,0,.01,.1,1])*limit
        bar=fig.colorbar(image,ax=axis,ticks=ticks,label='Local write (symlog)')
        bar.ax.set_yticklabels([f'{value:.2g}' for value in ticks])
    fig.savefig(check_write(output/'mlp-all-unit-map.png',4*2**20,operation='全坐标原生 MLP 响应图'),dpi=180,bbox_inches='tight')
    plt.close(fig)
    print('Saved both native response maps',flush=True)


if __name__=='__main__':
    run()
