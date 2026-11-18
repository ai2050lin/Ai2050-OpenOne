"""Descriptive discovery-only full-coordinate map; never changes selection."""
from collections import defaultdict
import time
import numpy as np
import recursive_branch14 as R


def run():
    R.configure();R.admission()
    from deploy.storage_policy import check_write
    check_write(R.OUT,150*2**20,operation='Discovery-only full-coordinate descriptive maps')
    allrows=R.materials();meta=R.read(R.OUT/'discovery/collection.json')
    rows=[allrows[i] for i in meta['row_indices']]
    assert meta['behavior_passed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries=[];start=time.monotonic()
    for h in R.CANDIDATES:
        families=defaultdict(dict)
        for i,r in enumerate(rows):
            if r['hypothesis']==h:families[r['family']][r['role'],r['lexical']]=i
        ordered=[families[k] for k in sorted(families)]
        assert len(ordered)==128 and all(len(f)==6 for f in ordered)
        indices={key:np.array([f[key] for f in ordered]) for key in ordered[0]}
        raw=np.zeros((3,40,R.WIDTH),np.float32);sd=np.zeros_like(raw)
        means=np.zeros((3,3,40,R.WIDTH),np.float32)
        for b in range(39,-1,-1):
            for fi,field in enumerate(['G','V','A']):
                data=np.load(R.OUT/'discovery'/f'B{b}-{field}.npy',mmap_mode='r')
                rolevals={role:(R.unbits(data[indices[role,0],0])+R.unbits(data[indices[role,1],0]))/2
                          for role in ['operative','quoted','inventory']}
                delta=(rolevals['operative']+rolevals['quoted'])/2-rolevals['inventory']
                raw[fi,b]=delta.mean(0);sd[fi,b]=delta.std(0,ddof=1)
                for ri,role in enumerate(['operative','quoted','inventory']):means[fi,ri,b]=rolevals[role].mean(0)
                del data,delta,rolevals
            R.log('full_coordinate_map_'+h,40-b,40,start,block=b)
        dz=np.divide(raw,sd,out=np.zeros_like(raw),where=sd>0)
        out=R.OUT/f'full-field-{h}.npz'
        np.savez_compressed(out,raw_relation_mean=raw,family_difference_sd=sd,paired_dz=dz,role_means=means)
        specs=[]
        for field_type,values in [('raw',raw),('standardized',dz)]:
            fig,axes=plt.subplots(3,1,figsize=(15,10),layout='constrained')
            for fi,(field,ax) in enumerate(zip(['G','V','A'],axes)):
                # A single field-specific symmetric range across every block.
                limit=max(float(np.quantile(np.abs(values[fi]),.995)),1e-6)
                im=ax.imshow(values[fi,::-1],aspect='auto',interpolation='nearest',cmap='RdBu_r',
                             vmin=-limit,vmax=limit,extent=(-.5,R.WIDTH-.5,-.5,39.5),origin='upper')
                ax.set_yticks([0,9,19,29,39],['B0','B9','B19','B29','B39'])
                ax.set_ylabel('Native block');ax.set_xlabel('Native unit index (unchanged order)')
                ax.set_title(f'{field}: mean relation contrast' if field_type=='raw' else f'{field}: paired-family dz')
                fig.colorbar(im,ax=ax,pad=.01,label='Raw native units' if field_type=='raw' else 'Paired dz')
                specs.append(dict(view=field_type,field=field,symmetric_limit=limit,
                                  clipped_coordinates=int(np.sum(np.abs(values[fi])>limit))))
            fig.suptitle(f'{h}: discovery only, content-end readout; all 40 x 17,408 native units\n'
                         'Display is pixel-limited; exact unsorted arrays are retained. No causal or semantic promotion.',fontsize=12)
            fig.savefig(R.OUT/f'full-field-{h}-{field_type}.png',dpi=170);plt.close(fig)
        summaries.append(dict(hypothesis=h,families=128,shape=list(raw.shape),archive=out.name,sha256=R.file_sha(out),views=specs))
    R.save('full-field-maps.json',dict(kind='descriptive_discovery_observation',maps=summaries,
        selection_unchanged=True,discovery_collection_sha256=R.file_sha(R.OUT/'discovery/collection.json'),
        source_sha256=R.file_sha(__file__),readout='content_end_before_probe',
        definitions=dict(raw='family mean of (instruction+reference)/2-inventory, titles averaged first',
                         standardized='mean paired difference divided by sample SD of family differences',
                         role_means='field, role [instruction, reference, inventory], block, native unit'),
        visualization='All native coordinates retained without sorting or top-K. Block39 at top. Each field/view has a fixed symmetric 99.5% absolute range across all blocks; clipping counts saved. Native arrays are authoritative at screen-limited resolution.',
        limitations=['Discovery map only; no multiple-comparison semantic claims','Same coordinate index across blocks is not a shared function','This observed contrast includes wording and length differences; not a pure relation mechanism']))


if __name__=='__main__':run()
