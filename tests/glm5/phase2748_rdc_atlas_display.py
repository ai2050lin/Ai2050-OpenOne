"""Display-only revision: separate final residual and postnorm axis labels."""
import argparse
from rdc_question_common import *


def main(key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'atlas'/key
    native=read(folder/'result.json')
    ref=native['fields']
    assert sha(ROOT/ref['path'])==ref['sha256']
    figures=[]
    with np.load(ROOT/ref['path'])as z:
        for name in ['raw_RMS','train_standardized_RMS']:
            values=[np.sqrt(z['diagnostic__'+c+'__H_and_postnorm__within_variance']) if name=='raw_RMS'
                else z['diagnostic__'+c+'__H_and_postnorm__within_RMS_train_standardized'] for c in ['drop','quoref']]
            maximum=max(float(x.max())for x in values)
            fig,axes=plt.subplots(2,1,figsize=(15,9),constrained_layout=True)
            for ax,value,cohort in zip(axes,values,['DROP','Quoref']):
                im=ax.imshow(value,aspect='auto',origin='lower',interpolation='nearest',vmin=0,vmax=maximum,cmap='viridis')
                depth=value.shape[0]-2
                positions=[p for p in range(0,depth+1,4)if depth+1-p>=3]+[depth+1]
                ax.set_yticks(positions,['H'+str(p)if p<=depth else 'postnorm'for p in positions])
                ax.set_xlabel('Native residual coordinate (unchanged order)')
                ax.set_ylabel(f'H0 to H{depth}, then separate postnorm')
                ax.set_title(f'{key} | {cohort} | 48 held contexts x 4 questions | {name}')
                fig.colorbar(im,ax=ax,label='Question-within-context RMS'+(' / training coordinate SD'if name.startswith('train')else ' (native units)'))
            fig.suptitle('Natural same-context question response atlas: all declared coordinates; no semantic-module assignment',fontsize=12)
            path=folder/(name+'_labels_v2.png')
            assert not path.exists(),'Display output already exists; preserve it'
            fig.savefig(path,dpi=220);plt.close(fig)
            figures.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'name':name,
                'color_limits':[0,maximum],'coordinates':values[0].shape[1],
                'change':'Tick labels spaced to avoid H_last/postnorm overlap. Every underlying row/coordinate, color scale and normalization unchanged.'})
    immutable(folder/'display_v2.json',{'timestamp':stamp(),'source':snapshot(__file__),
        'source_atlas_sha256':sha(folder/'result.json'),'source_field_sha256':ref['sha256'],
        'figures':figures,'previous_figures_retained':True,'visual_QA_pending':True})
    print('NATURAL_ATLAS_DISPLAY_V2',key,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    main(parser.parse_args().model)
