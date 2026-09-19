"""Static scientific figures retain every native coordinate/source/unit, with explicit rendering transforms."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_operator_common import *


def heat(ax, values, title, xlabel, ylabel, transform='asinh'):
    if transform=='asinh':
        display=np.arcsinh(values)
    else:
        display=values
    lim=max(float(np.max(np.abs(display))),1e-12)
    im=ax.imshow(display,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest',origin='lower')
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    plt.colorbar(im,ax=ax,fraction=.025,pad=.02,label='asinh(value), unit scale1' if transform=='asinh' else 'value')


def main():
    out=BASE/'figures';out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();index=[]
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.dpi':120})
    def savefig(fig,name,title,scope):
        fig.tight_layout();fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
        index.append({'path':name,'title':title,'scope':scope,'sha256':sha(out/name)})
    sid='train-en-o0000'
    with np.load(BASE/'capture/main/full_fields'/f'{sid}.npz') as z:
        h=unbits(z['H'])
    row=next(r for r in rows() if r['sample_id']==sid)
    fig,ax=plt.subplots(2,2,figsize=(15,9))
    heat(ax[0,0],h[:,row['anchors'][0]],'All37 layer boundaries at one ordinary anchor','native residual coordinate0..2559','H boundary0..36')
    for a,l in zip([ax[0,1],ax[1,0],ax[1,1]],(12,23,36)):
        heat(a,h[l],f'H{l}: every token, every coordinate','native residual coordinate0..2559','native token position')
    savefig(fig,'native_full_fields.png','一个自然来源的全层／全 token／全坐标原场',f'{sid}; original native order, no coordinate removal. asinh is display only; H0 is embedding, H36 precedes final RMSNorm.')
    with np.load(BASE/'observation/full_coordinate_condition_profiles.npz') as z:
        means={k:z[k] for k in z.files}
    fig,ax=plt.subplots(2,3,figsize=(17,7))
    for i,lang in enumerate(('en','zh')):
      for j,l in enumerate((7,17,35)):
        heat(ax[i,j],means[f'train_{lang}_train_z'][l,:7],f'{lang} H{l}: seven token-piece conditions','native residual coordinate0..2559','initial/blank/punct/number/latin/CJK/other',transform='asinh')
    savefig(fig,'ordinary_condition_coordinates.png','训练条件差异的完整原生坐标图', 'Full2560 columns; class means standardized using TRAIN ordinary-token scales. Piece labels are observed tokenizer properties, not gold semantic classes.')
    reports=read(BASE/'operators/result.json')['reports']
    names=['constant_global','coordinate_affine','frozen_gate_global','frozen_gate_piece','frozen_gate_cue','frozen_gate_token','frozen_gate_position','tangent_global','tangent_piece','tangent_cue','quadratic_global']
    fig,ax=plt.subplots(1,2,figsize=(16,7))
    colors=['#256c95','#d07835','#46967a']
    for j,b in enumerate((6,16,34)):
        vals=[next(r for r in reports if r['name']==n and r['block']==b and r['split']=='test' and r['stratum']=='all') for n in names]
        y=np.arange(len(names))+(j-1)*.23
        ax[0].barh(y,[r['relative_MSE'] for r in vals],height=.22,color=colors[j],label=f'block{b}')
        ax[1].barh(y,[r['raw_MSE'] for r in vals],height=.22,color=colors[j],label=f'block{b}')
    for a in ax:
        a.set_yticks(range(len(names)),names);a.legend();a.grid(axis='x',alpha=.2)
    ax[0].set(title='TEST: mean per-token relative coordinate MSE',xlabel='MSE / actual MLP squared amplitude')
    ax[1].set(title='TEST: raw coordinate MSE (log x scale)',xlabel='Raw MSE');ax[1].set_xscale('log')
    savefig(fig,'native_operator_comparison.png','三个 block 的全坐标局部预测比较','All512 test anchors per block; candidate choices were made on validation, not this plot. Raw and per-token-relative MSE answer different questions; native formula oracle is omitted from bars, not confused with an extracted rule.')
    fig,ax=plt.subplots(3,2,figsize=(16,10))
    evmeta=[r for r in gzread(BASE/'operators/row_index.json.gz') if r['split']!='train']
    ordinary=np.array([not r['event'] for r in evmeta])
    for i,b in enumerate((6,16,34)):
        with np.load(BASE/'operators'/f'L{b}_predictions.npz') as z:
            decomp=z['decomposition'][ordinary].astype(float).mean(0)
        with np.load(BASE/'operators'/f'L{b}_full_unit_decomposition.npz') as z:
            units=z['mean']
        heat(ax[i,0],decomp,f'block{b}: ordinary finite product terms','native write coordinate0..2559','shared / value / gate / interaction')
        heat(ax[i,1],units,f'block{b}: every native unit, all heldout anchors','native MLP unit0..9727','shared / value / gate / interaction')
    savefig(fig,'native_four_term_all_coordinates_units.png','共享、读值、门控和交互项的全坐标／全单元分解','Exact finite product identity at FP32 same-valued weights, trained global centers. Left excludes labeled numerical events; right uses all heldout anchors. No scalar is removed; term magnitude alone is not semantic necessity.')
    hyper=gzread(BASE/'qa_atlas/hyperedges.json.gz')
    example=next(r for r in hyper if r['question_type']=='bridge' and r['normalized_full_EM'] and r['hyperedge']['support_char_spans'])
    qid=example['question_id']
    with np.load(BASE/'qa/qwen4/main/fields'/f'{qid}.npz') as z:
        all_attention={b:unbits(z[f'L{b}_attention_sources']) for b in (6,16,34)}
    fig,ax=plt.subplots(3,1,figsize=(15,8))
    for a,b in zip(ax,(6,16,34)):
        heat(a,all_attention[b],f'block{b}: all32 heads, final prompt query','every actual prompt source position','head0..31',transform='raw')
        for p in example['hyperedge']['token_groups']['gold_support_sentences']:
            a.plot(p,-2,'|',color='#398144',markersize=3,clip_on=False)
    fig.suptitle(example['question'][:145],fontsize=10)
    savefig(fig,'native_QA_all_source_attention.png','真实多跳问答的全部来源 attention',f'{qid}; green marks are retrospective gold supporting-sentence tokens, not online instructions. Attention mass does not establish causal reasoning; all sources and heads retained.')
    save(out/'example_ids.json',{'natural':sid,'QA':qid,'QA_actual_question':example['question'],'QA_actual_generation':example['generated_text']})
    save(out/'index.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'figures':index,
        'display_contract':'Native order fixed, all requested rows/columns retained. Pixel downsampling is display-only. Raw BF16 and floating diagnostic arrays remain separately queryable; no PCA/Top-K or inferred geometric axes.'})
    ledger('static_native_operator_figures',time.monotonic()-start,figures=len(index))
    print('OPERATOR_FIGURES_COMPLETE',len(index),flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
