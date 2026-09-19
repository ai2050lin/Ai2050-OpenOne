"""Matched-source native-model uncertainty and character-endpoint audit; no model loading."""
from rdc_operator_common import *
from phase2726_rdc_operator_scale import selected


def main():
    start=time.monotonic();material=selected();confirmation=[r for r in material if r['split']=='confirmation']
    comparisons=[];alignments={};oracle=[]
    for model in ('qwen4','qwen14','glm4'):
        folder=BASE/'scale'/model;result=read(folder/'result.json')
        assert result['sources']==128
        saved_protocol=read(folder/'protocol.json')
        assert saved_protocol['source_ids']==[r['sample_id'] for r in material]
        meta=[m for r in confirmation for m in read(folder/'rows'/f'{r["sample_id"]}.json')['anchor_meta']]
        assert len(meta)==64
        for block in result['own_blocks']:
            name=result['choices'][str(block)]
            with np.load(folder/'operators'/f'confirmation_L{block}_predictions.npz') as z:
                target=z['native'].astype(float);den=np.maximum(np.mean(target*target,1),1e-20)
                loss={k:np.mean((z[k].astype(float)-target)**2,1)/den for k in (name,'frozen_gate_global','native32_oracle')}
            for key,v in loss.items():
                original=next(r for r in result['reports'] if r['stage']=='confirmation' and r['block']==block and r['name']==key)
                assert abs(float(v.mean())-original['relative_MSE'])<1e-10
            gain=loss['frozen_gate_global']-loss[name]
            comparisons.append({'model':model,'block':block,'selected':name,'anchors':64,
                'global_relative_MSE':float(loss['frozen_gate_global'].mean()),'selected_relative_MSE':float(loss[name].mean()),
                'anchor_mean_gain':float(gain.mean()),'fraction_anchors_improved':float(np.mean(gain>0)),
                'article_cluster_gain':clustered(gain,[r['source_group'] for r in meta])})
            oracle.append({'model':model,'block':block,'native32_mean_relative_MSE':float(loss['native32_oracle'].mean()),
                'native32_max_token_relative_MSE':float(loss['native32_oracle'].max())})
            assert float(loss['native32_oracle'].mean())<1e-4
        endpoint_rows=[]
        for row in material:
            native=read(folder/'rows'/f'{row["sample_id"]}.json')
            for anchor,q4pos in enumerate(row['anchors']):
                target_end=row['token_offsets'][q4pos][1]
                if model=='qwen4':native_end=target_end;native_pos=q4pos
                else:
                    native_pos=native['positions'][anchor]
                    native_end=native['token_offsets'][native_pos][1]
                    assert native_end<=target_end
                endpoint_rows.append({'sample_id':row['sample_id'],'anchor':anchor,'split':row['split'],
                    'qwen4_position':q4pos,'native_position':native_pos,'qwen4_char_end':target_end,
                    'native_char_end':native_end,'missing_characters':target_end-native_end})
        mismatches=[r for r in endpoint_rows if r['missing_characters']]
        alignments[model]={'all_anchors':len(endpoint_rows),'exact_endpoint_anchors':len(endpoint_rows)-len(mismatches),
            'max_missing_characters':max(r['missing_characters'] for r in endpoint_rows),
            'nonexact_anchors':mismatches,'all_sources_identical_by_ID':True}
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'comparisons':comparisons,'alignment':alignments,
        'numeric_oracle':oracle,'scope':'Nine paired comparisons using complete native-coordinate predictions and the fixed validation-selected candidates. Article bootstrap is conditional on each fit, without multiplicity correction. Identical raw sources do not imply identical token information; every nonexact endpoint is retained. This is not a controlled parameter-scaling or coordinate-isomorphism theorem.'}
    save(BASE/'scale/paired_audit.json',result);ledger('matched_native_models_paired_and_endpoint_audit',time.monotonic()-start)
    print('MATCHED_NATIVE_SCALE_AUDIT_PASS',[(r['model'],r['block'],r['anchor_mean_gain']) for r in comparisons],flush=True)


if __name__=='__main__':main()
