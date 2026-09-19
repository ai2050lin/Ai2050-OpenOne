"""Common-scale, every-coordinate comparison for N and independent-material O."""
from rdc_conditional_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=CAMPAIGN/'n_cached_attention/figures';out.mkdir(exist_ok=True);values={};sources={}
    fig,axes=plt.subplots(2,1,figsize=(19,8),dpi=220,sharey=True,constrained_layout=True)
    models=[('native_arithmetic_floor','Native arithmetic floor','#333333'),('H12_linear_factors_validation','H12 encoder + pastKV operator','#147c8a'),
      ('H12_fullpastKV_linear_factors_validation','Full-input factor operator','#bc6438'),('H12_fullpastKV_linear_direct_head_equal6144_validation','Same full-input direct head','#8c549f')]
    for ax,run,label in zip(axes,('n_cached_attention','o_generalization'),('N: entity-heldout, same Result-field domain','O: frozen N models, new eight-operation material')):
        result=read(CAMPAIGN/run/'result.json');assert result
        sources[run]=sha(CAMPAIGN/run/'result.json')
        for mid,title,color in models:
            with np.load(CAMPAIGN/run/f'predictions/{mid}.npz') as z:error=z['coordinate_mse']
            assert len(error)==2560;values[run+'__'+mid]=error;ax.plot(np.arange(2560),error,lw=.55,label=title,color=color)
        ax.set_yscale('symlog',linthresh=1e-6);ax.set_xlabel('All native L23 attention-output coordinates 0..2559; fixed unsorted order')
        ax.set_ylabel('Held-out coordinate MSE (shared symlog scale)');ax.set_title(label+'; numerical prediction, not task accuracy');ax.legend(fontsize=7)
    fig.canvas.draw();pixels=[float(ax.get_window_extent().width) for ax in axes];assert min(pixels)>=2560
    fig.savefig(out/'full_coordinate_attention_forecast_errors.png');plt.close(fig)
    npz(out/'plotted_values.npz',**values)
    save(out/'display_contract.json',{'source_sha':sha(Path(__file__)),'result_shas':sources,'all_coordinates':2560,'axis_pixel_width':pixels,
      'order':'Unchanged native0..2559; no TopK/PCA/sorting','scale':'Shared Y range, symlog linthresh1e-6; no clipping or omitted coordinates',
      'scope':'N and O use same native L23 coordinate basis but different states and target energy. H12 label is fitted encoder only; factor composition also uses actual prior cached KV. No whole-answer correctness or closed-decoder claim.'})
    print('ATTENTION_FULL_COORDINATE_FIGURES_COMPLETE',flush=True)


if __name__=='__main__':main()
