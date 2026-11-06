"""Append immutable whole-unit ledger revision from actual controlled/dictionary results."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from mlp_atlas_publish_final import Archive
import mlp_atlas_natural_collect as D

def main():
    old=P.read(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v2.json');A=Archive(P.ROOT/'research/mlp_atlas/inventory_v1',old['run_id'],old);lib=A.lib;report=lib.artifact(old['report_sha256']);layerrefs={};counts={}
    code=P.read(P.OUT/'final-registration.json');dictionary=D.read(D.OUT/'final-registration.json');layers=[]
    for b in range(36):
        d=lib.artifact(old['layer_refs'][str(b)]);flags=d['flags']
        with np.load(P.OUT/f'analysis/B{b:02d}.npz') as z:
            flags.update(code_response_question=z['response_replicated'][0].tolist(),code_response_assistant=z['response_replicated'][1].tolist(),code_function_eligible=z['eligible_supported'].any(0).tolist())
        with np.load(D.OUT/f'dictionaries/B{b:02d}-associations.npz') as z:
            flags['natural_dictionary_association_attempted']=[True]*9728
            flags['natural_dictionary_association_repeated']=((np.abs(z['discovery_partner_correlation'])>=.2)&(z['confirm_partner_correlation']*z['discovery_partner_correlation']>0)&(z['lockbox_partner_correlation']*z['discovery_partner_correlation']>0)&(np.abs(z['confirm_partner_correlation'])>=.2)&(np.abs(z['lockbox_partner_correlation'])>=.2)).tolist()
        d['research_refs']={'code':code['layer_refs'][str(b)],'dictionary':dictionary['layer_refs'][str(b)]};d['scope']+=' New code-response flags are descriptive when behavior gate fails. Dictionary correlations use a fixed descriptive .2 threshold, not a semantic or multiplicity-corrected causal test.'
        layerrefs[str(b)]=A.put(f'layer-revision-B{b:02d}',d);row={'block':b,'n_units':9728,'counts':{k:sum(v) for k,v in flags.items()}};layers.append(row)
        for k,v in row['counts'].items():counts[k]=counts.get(k,0)+v
    report.update(layer_refs=layerrefs,layers=layers,evidence_counts=counts,research_refs={'code_report':code['report_sha256'],'dictionary_report':dictionary['report_sha256']},next_run_id='MLP-ATLAS-CAUSAL-CLOSURE-v1')
    report['limitations']+=['Allcodeunitstested butglobalbehaviorgatefailed, soresponseflags arenotfunctionalpromotion.','Naturalunit/dictionaryassociation iscorrelationonly; no semanticlabel orcausalproof.']
    rr=A.put('report-revision',report);gr=A.put('graph-revision',{'schema':'research_graph.v1','feature_cards':[],'graph_edges':[],'full_native_unit_entries':350208,'note':'Fullnative arrays, not350208claimedsemanticfeatures.'})
    A.bind(rr,gr,{'study_type':'mlp_reverse_inventory','layer_refs':layerrefs,'all_units':350208,'code_report':code['report_sha256'],'dictionary_report':dictionary['report_sha256']},'350208个原生MLP单元 · 累积逆向证据账本',{'inventory':{'status':'done','units':350208,'natural_tokens':report['raw_census_tokens']},'controlled':{'status':'done','function_gate':'failed'},'dictionary':{'status':'done','layers':36},'semantic_causal_closure':{'status':'not_complete','individual_mechanisms_closed':0}})
    P.save(P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v3.json',{'report_sha256':rr,'graph_sha256':gr,'layer_refs':layerrefs,'run_id':A.run,'model_snapshot':A.snap,'provenance':A.prov,'counts':counts},False);print(json.dumps({'inventory_report':rr,'counts':counts}))
if __name__=='__main__':main()
