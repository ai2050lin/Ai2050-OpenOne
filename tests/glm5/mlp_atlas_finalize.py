"""Append audit and roadmap to the existing all-neuron ledger without erasing history."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
import mlp_atlas_parameter_audit as Q
from mlp_atlas_publish_final import Archive

def main():
    result=P.read(Q.OUT/'audit-result.json');A=Archive(Q.PUBLIC,Q.RUN,result);layerrefs={}
    for row in result['layers']:
        b=row['block'];native=A.native(Q.OUT/f'B{b:02d}.npz',row['sha256'])
        with np.load(Q.OUT/f'B{b:02d}.npz') as z:values={k:z[k].tolist() for k in ['Gate_term_RMS','Value_term_RMS','rounding_residual_RMS','G_relative_RMS','V_relative_RMS','A_relative_RMS']}
        layerrefs[str(b)]=A.put(f'layer-B{b:02d}',{'schema':'mlp_parameter_audit_layer.v1','block':b,'d_ff':9728,'native_index_order':True,'full_tensor':native,**values})
    refs={'layers':layerrefs,'sources':A.sources(['tests/glm5/mlp_atlas_parameter_audit.py','research/mlp_atlas/MLP_REVERSE_RESEARCH_PLAN.md']),'source_bank':P.read(P.OUT/'final-registration.json')['report_sha256']}
    rr=A.put('report-final',{**result,'artifact_refs':refs,'limitations':['CPU BF16 SiLU is applied to recordedG for allocation; residual includes rounding and arithmetic implementation differences.','This is exhaustive arithmetic verification onregistered inputs, not exhaustive testing of allpossiblecontexts.','A symmetric product identity does not identify unique causal or semantic modules.','Originalweights unchanged; this observer audit does not claim a newsteering orcapability improvement.']})
    cards=[]
    for b,u in [(24,8443),(28,4371),(32,5610),(35,9727)]:
        cards.append({'id':f'{A.snap}:B{b}:U{u}:{Q.RUN}','layer':b,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':u},'method':'all-unit native parameter projection audit; symmetric product identity with numeric residual','run_id':Q.RUN,'label':f'B{b}/U{u} · 真实参数算术来源已核验','evidence':{'level':'parameter_source','artifact_sha256':layerrefs[str(b)]},'counterexamples':[{'reason':'no semantic orunique causal interpretation; finiteinputscope'}],'scope':{'source_bank':P.RUN,'all_unit_arrays_ref':layerrefs[str(b)]}})
    gr=A.put('graph-final',{'schema':'research_graph.v1','feature_cards':cards,'graph_edges':[],'note':'Four navigation anchors; all350208native unit audit records retained. Parameter source is not semantic closure.'})
    A.bind(rr,gr,{'study_type':'mlp_parameter_audit','native_bank':True,'layer_refs':layerrefs},'全部MLP真实参数审计 · Gate/Value响应来源',{'parameter_audit':{'status':'done','inputs':9984,'units':350208},'semantic_causal_closure':{'status':'not_complete'}})
    P.save(Q.OUT/'final-registration.json',{'report_sha256':rr,'graph_sha256':gr,'layer_refs':layerrefs,'run_id':A.run,'model_snapshot':A.snap,'provenance':A.prov},False)
    import mlp_atlas_natural_collect as D
    dr=D.read(D.OUT/'final-registration.json');DA=Archive(D.PUBLIC,D.RUN,dr);dd=DA.lib.artifact(dr['report_sha256']);fvu=np.asarray([[r['results']['lockbox']['FVU'][k] for k in ['dictionary','native32','discovery_mean']] for r in dd['layers']])
    dd['conclusion']=f"36层全量评估完成：字典在lockbox优于均值36/36层，优于原生32贡献0/36层；字典FVU中位数{np.median(fvu[:,0]):.4f}。本字典保真度不足，不能当作已破解特征。"
    dd['critical_analysis']={'lockbox_FVU_median':dict(zip(['dictionary','native32','discovery_mean'],np.median(fvu,0).tolist())),'native_dictionary_association_repeated':sum(r['association_repeated_units'] for r in dd['layers']),'no_feature_win_claim':True,'generalization_counterexample':'B35 dictionaryFVU discovery.1961, confirm.3176, lockbox.5922; native32lockbox.4700. A favorableconfirmdidnottransfer.'}
    drr=DA.put('report-interpretation',dd);DA.bind(drr,dr['graph_sha256'],{'study_type':'mlp_dictionary_atlas','native_bank':True,'layer_refs':dr['layer_refs']},'全36层自然字典 · 保真度不足，完整负结果',{'collection':{'status':'done','docs':2048},'dictionary':{'status':'done','features':36864},'fidelity':{'status':'not_supported','lockbox_better_than_native32':0,'semantic_causal_closure':False}})
    D.save(D.OUT/'interpretation-registration-v2.json',{**dr,'report_sha256':drr},False)
    old=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v3.json');I=Archive(P.ROOT/'research/mlp_atlas/inventory_v1',old['run_id'],old);report=I.lib.artifact(old['report_sha256']);inventoryrefs={}
    for b in range(36):
        layer=I.lib.artifact(old['layer_refs'][str(b)]);layer['flags']['parameter_projection_checked']=[True]*9728;layer['research_refs']['parameter_audit']=layerrefs[str(b)];inventoryrefs[str(b)]=I.put(f'layer-final-B{b:02d}',layer)
    report.update(layer_refs=inventoryrefs);report['research_refs']['dictionary_report']=drr;report['research_refs']['parameter_audit_report']=rr;report['evidence_counts']['parameter_projection_checked']=350208
    ir=I.put('report-final',report);ig=I.put('graph-final',{'schema':'research_graph.v1','feature_cards':[],'graph_edges':[],'all_native_unit_records':350208,'semantic_causal_closed':0})
    I.bind(ir,ig,{'study_type':'mlp_reverse_inventory','layer_refs':inventoryrefs,'all_units':350208,'code_report':report['research_refs']['code_report'],'dictionary_report':report['research_refs']['dictionary_report'],'parameter_audit_report':rr},'350208个原生MLP单元 · 全量计算与研究证据账本',{'inventory':{'status':'done','units':350208},'parameter_audit':{'status':'done'},'controlled_behavior':{'status':'behavior_failed'},'dictionary':{'status':'done','layers':36},'semantic_causal_closure':{'status':'not_complete','individual_mechanisms_closed':0}})
    P.save(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v4.json',{'report_sha256':ir,'graph_sha256':ig,'layer_refs':inventoryrefs,'run_id':I.run,'model_snapshot':I.snap,'provenance':I.prov,'counts':report['evidence_counts']},False);print(json.dumps({'inventory_report':ir,'parameter_audit_report':rr}))
if __name__=='__main__':main()
