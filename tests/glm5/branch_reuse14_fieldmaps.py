"""Unsorted complete native MLP coordinate views from discovery only."""
import time
import numpy as np
import branch_reuse14 as B
import branch_reuse14_analysis as A


def run():
    B.configure(200*2**20)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image,ImageDraw,ImageFont
    from matplotlib import font_manager
    rows,_=A.load('discovery');start=time.monotonic();matrices={}
    for kind in 'GVA':
        for branch in ['pooled','number','coordinate']:
            for statistic in ['difference','dz','rule_mean','record_mean']:
                matrices[f'{kind}_{branch}_{statistic}']=np.zeros((40,17408),np.float32)
    for block in range(40):
      for kind in 'GVA':
        values=A.field('discovery',block,kind)
        for branch in ['pooled','number','coordinate']:
          ix=[i for i,r in enumerate(rows) if branch=='pooled' or r['branch']==branch]
          rr=[rows[i] for i in ix];y=values[ix];delta=A.paired(rr,y)
          matrices[f'{kind}_{branch}_difference'][block]=delta.mean(0)
          matrices[f'{kind}_{branch}_dz'][block]=delta.mean(0)/np.maximum(delta.std(0,ddof=1),1e-7)
          for role in ['rule','record']:matrices[f'{kind}_{branch}_{role}_mean'][block]=y[[i for i,r in enumerate(rr) if r['role']==role]].mean(0)
      if block%5==4:B.log('full_native_discovery_maps',block+1,40,start)
    np.savez_compressed(B.OUT/'full-native-fieldmaps.npz',**matrices)
    plots=[];cmap=plt.get_cmap('coolwarm');font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),18)
    for kind in 'GVA':
      for statistic in ['difference','dz']:
        key=f'{kind}_pooled_{statistic}';values=matrices[key];limit=float(np.quantile(np.abs(values),.995));limit=max(limit,1e-7)
        colors=(cmap(np.clip((values/limit+1)/2,0,1))[:,:,:3]*255).astype(np.uint8)
        native=Image.new('RGB',(17568,480),'white');native.paste(Image.fromarray(np.repeat(colors,10,axis=0)),(160,40));draw=ImageDraw.Draw(native)
        for block in range(0,40,2):draw.text((8,40+10*block),f'B{block}',fill='black',font=font)
        for unit in range(0,17408,1024):draw.text((160+unit,5),str(unit),fill='black',font=font)
        draw.text((160,450),f'{kind} {statistic}; full native order; one unit per horizontal pixel; symmetric limits +/-{limit:.7g}',fill='black',font=font)
        native_name=f'field-{kind}-{statistic}-native.png';native.save(B.OUT/native_name)
        # Explicit 16-coordinate mean for the overview, never used for selection.
        overview=values.reshape(40,1088,16).mean(-1)
        fig,ax=plt.subplots(figsize=(15,4.8));im=ax.imshow(overview,aspect='auto',origin='upper',cmap=cmap,vmin=-limit,vmax=limit,
            extent=[0,17408,39.5,-.5],interpolation='nearest')
        ax.set_xlabel('Native MLP unit index (16-unit mean per displayed column)');ax.set_ylabel('Zero-based block')
        ax.set_title(f'Discovery only: {kind}, {statistic}; rule minus individual records; unsorted')
        fig.colorbar(im,ax=ax,label=statistic);fig.tight_layout();name=f'field-{kind}-{statistic}.png';fig.savefig(B.OUT/name,dpi=110);plt.close(fig)
        plots.append(dict(field=kind,statistic=statistic,overview=name,native=native_name,color_limits=[-limit,limit],
            clipped_fraction=float(np.mean(np.abs(values)>limit)),coordinate_order='Native 0..17407, blocks0..39; no sort, PCA, or selected-unit truncation',
            overview_processing='Mean of each consecutive16 coordinates; opposite effects may cancel in this overview. Native PNG and float32 arrays retain every coordinate.'))
    B.save('fieldmap-manifest.json',dict(kind='discovery_descriptive_only',readout='content end before query/options',plots=plots,
        arrays_sha256=B.file_sha(B.OUT/'full-native-fieldmaps.npz'),shape=[40,17408],fields=list(matrices),source_sha256=B.file_sha(__file__),
        limitations='Same coordinate index across blocks is not assumed same function. Contrasts still mix wording/length and applicability; display is not a mechanism or a causal edge.'))


if __name__=='__main__':run()
