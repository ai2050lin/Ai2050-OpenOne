"""Every native coordinate, fixed unsorted basis, exploratory P scope only."""
from rdc_conditional_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=CAMPAIGN/'p_token_conditioned';out=root/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,1,figsize=(19,8),dpi=220,sharey=True,constrained_layout=True);values={}
    panels=[('Prediction route under identical full-context quadratic kernel',[
      ('native_H23_compiler_floor','Native H23 arithmetic floor'),('full_quadratic_factors_validation','Q/K/V factor predictor'),
      ('full_quadratic_hidden23_validation','Predicted H23 + real native projections'),('full_quadratic_direct_head_validation','Direct head predictor')]),
      ('Current embedding conditions did not improve these selected kernels',[
      ('H0_linear_factors_validation','Current H0 alone'),('full_linear_factors_validation','Full-context linear'),
      ('full_quadratic_factors_validation','Full-context quadratic'),('token_add_factors_validation','Add current H0'),('token_product_factors_validation','Current H0 product interaction')])]
    for ax,(title,models) in zip(axes,panels):
        for mid,label in models:
            with np.load(root/f'predictions/{mid}.npz') as z:v=z['coordinate_mse']
            assert len(v)==2560;values[mid]=v;ax.plot(np.arange(2560),v,lw=.55,label=label)
        ax.set_title(title);ax.set_yscale('symlog',linthresh=1e-6);ax.legend(fontsize=7)
        ax.set_xlabel('All native L23 attention-output coordinates 0..2559, unsorted');ax.set_ylabel('Test-coordinate MSE (shared symlog scale)')
    fig.suptitle('P: exploratory repartition after inspecting O; 384 states / four test entity groups; NOT independent confirmation',fontsize=11)
    fig.canvas.draw();pixels=[float(ax.get_window_extent().width) for ax in axes];assert min(pixels)>2560
    fig.savefig(out/'all_coordinate_token_conditioned_errors.png');plt.close(fig);npz(out/'plotted_values.npz',**values)
    save(out/'display_contract.json',{'source_sha':sha(Path(__file__)),'result_sha':sha(root/'result.json'),'all_coordinates':2560,'axis_pixel_width':pixels,
      'scope':'Post-O-result exploratory repartition, not independent confirmation; both factor and H23 compiler use already-available past KV.',
      'order':'Native0..2559, no sorting/PCA/TopK','scale':'Shared Y symlog1e-6; no coordinate clipping; raw plotted MSE retained'})
    print('P_FIGURES_COMPLETE',flush=True)


if __name__=='__main__':main()
