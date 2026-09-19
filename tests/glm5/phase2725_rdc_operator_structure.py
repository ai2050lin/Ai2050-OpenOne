"""Complete signed product accounting, ordinary-background figures and precision-audit targets."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_operator_common import *


def main():
    out=BASE/'structure';start=time.monotonic();reports=[];units={};gram_fields={};worst=[]
    meta=[r for r in gzread(BASE/'operators/row_index.json.gz') if r['split']!='train']
    for b in (6,16,34):
        with np.load(BASE/'operators'/f'L{b}_predictions.npz') as z:
            terms=z['decomposition'].astype(np.float64);native=z['FP32_native_oracle'].astype(np.float64)
            quad=z['quadratic_global'];target=z['native']
        g=np.einsum('nkd,njd->nkj',terms,terms)/terms.shape[-1]
        e=np.mean(native*native,-1)
        for group in ('all','ordinary','event'):
            ii=np.array([i for i,r in enumerate(meta) if group=='all' or group=='ordinary' and not r['event'] or group=='event' and r['event']])
            if not len(ii):continue
            actual=np.mean(native[ii]*native[ii],-1)
            closure=np.max(np.abs(g[ii].sum((1,2))-actual)/np.maximum(actual,1e-20))
            reports.append({'block':b,'stratum':group,'anchors':len(ii),'mean_signed_Gram':g[ii].mean(0).tolist(),
                'mean_energy_normalized_signed_Gram':(g[ii]/np.maximum(e[ii,None,None],1e-20)).mean(0).tolist(),
                'mean_abs_coordinate_by_term':np.mean(np.abs(terms[ii]),(0,2)).tolist(),
                'full_sum_energy_closure_relative_max':float(closure),
                'scope':'All four terms and every one of16Gram entries included. Diagonal term energies are not additive contributions; both symmetric cross entries are required.'})
            gram_fields[f'L{b}_{group}_coordinate_RMS']=np.sqrt((terms[ii]**2).mean(0)).astype(np.float32)
            gram_fields[f'L{b}_{group}_coordinate_mean']=terms[ii].mean(0).astype(np.float32)
        v=np.array([r['split']=='validation' for r in meta]);loss=np.mean((quad.astype(float)-target)**2,1)/np.maximum(np.mean(target.astype(float)**2,1),1e-20)
        ix=np.flatnonzero(v)[np.argmax(loss[v])]
        worst.append({'block':b,'sample_id':meta[ix]['sample_id'],'anchor':meta[ix]['anchor'],'event':meta[ix]['event'],'position':meta[ix]['position'],
            'worst_validation_quadratic_relative_MSE':float(loss[ix]),'validation_mean':float(loss[v].mean()),
            'validation_mean_excluding_one_diagnostic_only':float(np.delete(loss[v],np.argmax(loss[v])).mean()),
            'scope':'Diagnostic only; no exclusion applied and original validation choice unchanged.'})
        with np.load(BASE/'operators'/f'L{b}_full_unit_decomposition.npz') as z:
            units[b]=np.sqrt(z['mean_square'])
            gram_fields[f'L{b}_all_unit_RMS']=units[b].astype(np.float32)
    npz(out/'complete_product_statistics.npz',**gram_fields)
    maxerr=(-1,None)
    for r in rows():
        scope='confirmation' if r['split']=='confirmation' else 'main'
        with np.load(BASE/'capture'/scope/'energies'/f'{r["sample_id"]}.npz') as z:
            b=z['block_terms'].astype(float)
            err=np.abs(b[:,:6].sum(1)-b[:,6])/np.maximum(b[:,6],1e-20)
            ix=np.unravel_index(err.argmax(),err.shape)
            if err[ix]>maxerr[0]:maxerr=(float(err[ix]),{'sample_id':r['sample_id'],'scope':scope,'block':int(ix[0]),'position':int(ix[1]),'energy_terms':b[ix[0],:,ix[1]].tolist()})
    precision={'max_saved_FP32_aggregate_relative_discrepancy':maxerr[0],'target':maxerr[1],
        'status':'Audit target located across all2048sources; native BF16 sequential residual and FP64 replay still required. Do not interpret this aggregate cancellation discrepancy as a new physical law.'}
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'signed_Gram_reports':reports,'quadratic_validation_outliers':worst,'residual_precision_target':precision})
    index=read(BASE/'figures/index.json');figures=[];plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':120})
    def heat(ax,x,title,labels):
        scale=max(float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2))),1e-12)
        v=np.arcsinh(x/scale);lim=max(float(np.max(np.abs(v))),1e-12)
        im=ax.imshow(v,aspect='auto',origin='lower',interpolation='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim)
        ax.set(title=title,xlabel='unchanged native last-axis coordinate',ylabel=labels)
        plt.colorbar(im,ax=ax,fraction=.024,pad=.02,label=f'asinh(value / {scale:.3g}); no clipping')
    def figure(fig,name,title,scope):
        fig.tight_layout();fig.savefig(BASE/'figures'/name,bbox_inches='tight');plt.close(fig)
        figures.append({'path':name,'title':title,'scope':scope,'sha256':sha(BASE/'figures'/name)})
    with np.load(BASE/'observation/full_coordinate_condition_profiles.npz') as z:
        fig,ax=plt.subplots(2,3,figsize=(17,7))
        for i,lang in enumerate(('en','zh')):
            for j,l in enumerate((7,17,35)):
                heat(ax[i,j],z[f'train_{lang}_train_z'][l,1:7],f'{lang} H{l}: six NONINITIAL piece means','blank/punct/number/latin/CJK/other')
        figure(fig,'noninitial_condition_background.png','非初始位置的完整条件背景（与含首位置图并列）',
            'Every2560coordinate kept; only declared first-position stratum shown separately in existing companion. TRAINz mean values scaled by each full-panel RMS for display, with actual scale on colorbar; numerical event positions remain included, no winsorization.')
    fig,ax=plt.subplots(3,2,figsize=(16,10))
    for i,b in enumerate((6,16,34)):
        heat(ax[i,0],gram_fields[f'L{b}_ordinary_coordinate_RMS'],f'block{b}: ordinary coordinate term RMS','shared/value/gate/interaction')
        heat(ax[i,1],units[b],f'block{b}: all-unit term RMS (all heldout)','shared/value/gate/interaction')
    figure(fig,'native_four_term_RMS_background.png','四项分解的完整坐标／单元均方根，避免符号平均抵消',
        'RMS is a nonnegative magnitude diagnostic, not a signed or additive contribution. Exact signed16-entry Gram accounting is in structure/result.json; original signed-mean companion remains. All9728 units and2560coordinates retained; population distinction explicit.')
    qid=read(BASE/'figures/example_ids.json')['QA']
    with np.load(BASE/'qa/qwen4/main/fields'/f'{qid}.npz') as z:
        fig,ax=plt.subplots(3,1,figsize=(15,8))
        for a,b in zip(ax,(6,16,34)):
            value=unbits(z[f'L{b}_attention_sources']).astype(float)
            display=np.log10(value+1e-8)
            im=a.imshow(display,aspect='auto',origin='lower',interpolation='nearest',cmap='viridis',vmin=-8,vmax=0)
            a.set(title=f'block{b}: complete source attention, logarithmic companion',xlabel='every original prompt source position',ylabel='all32heads')
            plt.colorbar(im,ax=a,fraction=.02,pad=.02,label='log10(attention + 1e-8)')
    figure(fig,'native_QA_all_source_attention_log.png','全部来源 attention 的对数背景图（原概率图保留）',
        f'{qid}; positive probability shown by log10(p+1e-8), color range[-8,0], all sources/heads; offset only for display, original BF16 probabilities remain. Attention does not prove causation or recovered reasoning.')
    old={r['path']:r for r in index['figures']}
    old.update({r['path']:r for r in figures});index['figures']=list(old.values());index['companion_source']=snapshot(Path(__file__))
    save(BASE/'figures/index.json',index);ledger('complete_signed_structure_and_background',time.monotonic()-start);guard()
    print('STRUCTURE_COMPLETE',worst,precision,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
