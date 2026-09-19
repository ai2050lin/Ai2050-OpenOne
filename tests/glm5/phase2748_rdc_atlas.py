"""Full native coordinate maps of absolute and same-context question variation."""
import argparse
from collections import defaultdict
from rdc_question_common import *
import rdc_question_data as data


def main(key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    start=time.monotonic()
    folder=Path('atlas')/key
    finalpath=OUT/folder/'result.json'
    if finalpath.exists():
        assert read(finalpath)['source']['sha256']==sha(__file__)
        print('NATURAL_ATLAS_ALREADY_COMPLETE',key,flush=True)
        return
    assert read(OUT/'fit'/key/'result.json')['all_passed']
    contract=effective_contract()
    blocks=contract['capture']['selected_MLP_blocks'][key]
    arrays={}
    identities={}
    for split in ['train','validation','diagnostic']:
        rows,groups,questions=data.index(key,{split})
        identities[split]=[{k:r[k]for k in ['question_id','group_id','cohort','within_context_index']}for r in rows]
        group_rows=defaultdict(list)
        for row in rows:group_rows[row['group_id']].append(row)
        totals={}
        for gid,rr in group_rows.items():
            cohort=rr[0]['cohort']
            hidden,units=[],[]
            for row in rr:
                fields=data.field(questions[row['question_id']]['field'],['hidden_BF16','postnorm_BF16']+[f'block{b}_{name}_BF16'for b in blocks for name in ['gate','up','product']])
                hidden.append(np.concatenate([fields['hidden_BF16'],fields['postnorm_BF16'][None]],axis=0))
                units.append(np.stack([np.stack([fields[f'block{b}_{name}_BF16']for name in ['gate','up','product']])for b in blocks]))
            for name,values in [('H_and_postnorm',np.stack(hidden)),('selected_MLP_gate_up_product',np.stack(units))]:
                ident=(cohort,name)
                if ident not in totals:
                    totals[ident]={'sum':np.zeros_like(values[0]),'square':np.zeros_like(values[0]),
                        'within_square':np.zeros_like(values[0]),'n':0}
                t=totals[ident]
                t['sum']+=values.sum(0);t['square']+=(values**2).sum(0)
                t['within_square']+=((values-values.mean(0))**2).sum(0);t['n']+=len(values)
        for (cohort,name),t in totals.items():
            stem=split+'__'+cohort+'__'+name
            mean=t['sum']/t['n']; variance=t['square']/t['n']-mean**2
            assert variance.min()>=-1e-8*max(1.,float(np.max(t['square']/t['n'])))
            arrays[stem+'__mean']=mean
            arrays[stem+'__variance']=np.maximum(variance,0.)
            arrays[stem+'__within_variance']=t['within_square']/t['n']
        print('NATURAL_ATLAS_AGGREGATE',key,split,len(rows),round(time.monotonic()-start,1),flush=True)
    for split in ['validation','diagnostic']:
        for cohort in ['drop','quoref']:
            for name in ['H_and_postnorm','selected_MLP_gate_up_product']:
                prefix=split+'__'+cohort+'__'+name
                trainvar=arrays['train__'+cohort+'__'+name+'__variance']
                arrays[prefix+'__within_RMS_train_standardized']=np.sqrt(arrays[prefix+'__within_variance'])/np.maximum(np.sqrt(trainvar),1e-8)
    reference=commit_arrays(folder,'full_native_coordinate_aggregates',arrays)
    # Each pixel column retains native coordinate order. No top-coordinate list,
    # clipping percentile, PCA, reordered clustering or interpolation is used.
    figures=[]
    raw=[np.sqrt(arrays['diagnostic__'+c+'__H_and_postnorm__within_variance'])for c in ['drop','quoref']]
    normalized=[arrays['diagnostic__'+c+'__H_and_postnorm__within_RMS_train_standardized']for c in ['drop','quoref']]
    for name,values in [('raw_RMS',raw),('train_standardized_RMS',normalized)]:
        fig,axes=plt.subplots(2,1,figsize=(15,9),constrained_layout=True)
        maximum=max(float(x.max())for x in values)
        for ax,value,cohort in zip(axes,values,['DROP','Quoref']):
            im=ax.imshow(value,aspect='auto',origin='lower',interpolation='nearest',vmin=0,vmax=maximum,cmap='viridis')
            depth=value.shape[0]-2
            positions=list(range(0,depth+1,4))+[depth+1]
            labels=['H'+str(p)if p<=depth else 'postnorm'for p in positions]
            ax.set_yticks(positions,labels);ax.set_xlabel('Native residual coordinate (unchanged order)')
            ax.set_ylabel('Residual boundary / separate postnorm')
            ax.set_title(f'{key} | {cohort} | 48 held contexts x 4 questions | {name}')
            fig.colorbar(im,ax=ax,label='Question-within-context RMS'+(' / training coordinate SD'if name.startswith('train')else ' (native units)'))
        fig.suptitle('Natural same-context question response atlas: all declared coordinates; no semantic-module assignment',fontsize=12)
        path=OUT/folder/(name+'.png')
        path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=220);plt.close(fig)
        figures.append({'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),
            'name':name,'native_coordinate_columns':values[0].shape[1],
            'color_limits':[0,maximum],'same_color_limits_across_cohorts':True,
            'coordinate_sorting':'Original index, no reordering','normalization':'None'if name=='raw_RMS'else 'Divide coordinatewise by own-cohort unshuffled training total standard deviation with1e-8floor; no diagnostic fitting.'})
    summaries=[]
    for cohort in ['drop','quoref']:
        for split in ['train','validation','diagnostic']:
            stem=split+'__'+cohort+'__H_and_postnorm'
            total=arrays[stem+'__variance'].mean(-1)
            within=arrays[stem+'__within_variance'].mean(-1)
            summaries.append({'cohort':cohort,'split':split,'all_coordinate_total_variance_by_boundary':total.tolist(),
                'all_coordinate_within_question_variance_by_boundary':within.tolist(),
                'within_fraction_by_boundary':np.divide(within,total,out=np.zeros_like(within),where=total>0).tolist()})
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,
        'native_result_sha256':sha(OUT/f'native/{key}/nonconfirmation/result.json'),
        'fit_selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
        'fields':reference,'figures':figures,'identities':identities,'boundary_variances':summaries,
        'MLP_blocks':blocks,'MLP_channel_order':['gate','up','product'],
        'estimand':'Population moments over frozen questions; subtract each four-question context mean only for retrospective description. Context mean is not an available predictor input.',
        'scope':'All coordinates at every first-prefix residual boundary and separate final postnorm; all units of4predeclaredMLPblocks. Not every token/layer/unit/timeaxis. Low-amplitude coordinates retained in numerical arrays and both raw/normalized views.',
        'limits':'These variations include wording, positions, lexical identity and semantics; a bright coordinate is not a concept, causal gate or cross-layer functional identity. No cross-model coordinate correspondence is implied.',
        'visual_QA':'Figures generated; must be separately viewed before final delivery.',
        'seconds':time.monotonic()-start}
    immutable(finalpath,result)
    print('NATURAL_ATLAS_COMPLETE',key,round(result['seconds'],1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    main(parser.parse_args().model)
