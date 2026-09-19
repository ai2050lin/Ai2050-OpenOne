"""Display all relation-matrix nodes, preserving undefined entries explicitly."""
import argparse
from rdc_question_common import *


def label(name):
    if name=='question_token_frequency':return'queryTF'
    if name=='question_token_length':return'queryLen'
    return name.replace('block','B').replace('_product',':prod').replace('_gate',':gate').replace('_up',':up')


def main(key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'relation_geometry'/key
    source=read(folder/'result.json');assert source['all_passed']
    ref=source['field'];assert sha(ROOT/ref['path'])==ref['sha256']
    with np.load(ROOT/ref['path'])as z:arrays={k:z[k].copy()for k in z.files if k.startswith('diagnostic__')}
    outputs=[];labels=[label(x)for x in source['labels']]
    for name in ['similarity','excess']:
        values=[np.ma.array(arrays[f'diagnostic__{c}__{name}_mean'],mask=arrays[f'diagnostic__{c}__valid_contexts']==0)for c in ['drop','quoref']]
        maximum=max(float(np.max(np.abs(v.compressed())))for v in values)if name=='excess'else 1.
        minimum=-maximum if name=='excess'else 0.
        fig,axes=plt.subplots(1,2,figsize=(21,11.5),constrained_layout=True)
        palette=plt.get_cmap('coolwarm'if name=='excess'else'viridis').copy();palette.set_bad('#c7c7c7')
        for ax,values_,cohort in zip(axes,values,['DROP','Quoref']):
            im=ax.imshow(values_,origin='lower',interpolation='nearest',aspect='equal',vmin=minimum,vmax=maximum,cmap=palette)
            ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=7)
            ax.set_yticks(range(len(labels)),labels,fontsize=7)
            ax.set_title(f'{key} / {cohort}: 48 diagnostic contexts, 4 questions each',fontsize=11)
            ax.set_xlabel(f'Observed space (all{len(labels)}spaces; NOT native hidden coordinate)',fontsize=9)
            ax.set_ylabel('Observed space in identical order',fontsize=9)
        bar=fig.colorbar(im,ax=axes,fraction=.025,pad=.025,shrink=.82)
        bar.set_label('Centered Gram cosine minus exact24permutationmean'if name=='excess'else'Centered Gram cosine (uncorrected)',fontsize=10)
        fig.suptitle(('Relation preservation beyond shuffled question identity'if name=='excess'else'Raw relation similarity: isotropic geometry may already give high values')+
            '\nAll coordinates contribute to each4questionGram; gray = zero-energy / undefined, not zero similarity.\nB#:gate/up/prod = completeMLPchannels; queryTF/Len = external lexical/length controls. No semantic or causal identification.',fontsize=12)
        path=folder/('all_spaces_'+name+'.png');assert not path.exists()
        fig.savefig(path,dpi=200);plt.close(fig)
        outputs.append({'name':name,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'color_limits':[minimum,maximum],
            'spaces':len(labels),'rows_and_columns':source['labels'],'normalization':'Native full-coordinate centeredGram Frobenius cosine'+(' minusexactpermutationmean'if name=='excess'else''),
            'undefined':'Gray where validcontextcountzero; same ordered nodes, no clustering/reordering','no_percentile_clipping':True})
    immutable(folder/'display.json',{'timestamp':stamp(),'source':snapshot(__file__),'result_sha256':sha(folder/'result.json'),
        'figures':outputs,'visual_QA':'Pending actual image inspection'})
    print('NATURAL_RELATION_GEOMETRY_DISPLAY',key,len(outputs),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True);main(p.parse_args().model)
