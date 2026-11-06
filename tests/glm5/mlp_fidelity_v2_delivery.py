"""Publish source/provenance and connect the new run to the durable all-unit ledger."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from mlp_atlas_publish_final import Archive
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    final=P.read(P.OUT/'registration-final.json');A=Archive(P.PUBLIC,P.RUN,final);report=A.lib.artifact(final['report_sha256']);refs=report['artifact_refs'];cap=P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json';nextp=P.PUBLIC/'NEXT_STAGE.md'
    refs['capability_inputs']=A.put('capability-inputs',{'schema':'research_keyed_capability_inputs.v1','source_file_sha256':file_sha(cap),'scope':'previously exposed monitor; samequestions, 2200 gateeligible and140reference','rows':P.read(cap)})
    refs['storage_audit']=A.put('storage-audit',P.read(P.OUT/'storage-audit.json'))
    refs['next_stage']=A.put('next-stage',{'schema':'research_plan.v1','status':'planned','sha256':file_sha(nextp),'markdown':nextp.read_text(encoding='utf-8')})
    refs['execution_corrections']={p.stem:A.put(p.stem,P.read(p)) for p in P.PUBLIC.glob('*correction.research.json')}
    rr=A.put('report-delivery',report);A.bind(rr,final['graph_sha256'],{'study_type':'mlp_fidelity_v2','native_bank':True,'dictionary_layers':refs['dictionary_layers'],'semantic_layers':refs['semantic_layers'],'case_index':refs['case_index'],'next_stage':refs['next_stage']},'字典保真度与自然属性 · 生成、干预及能力闭环',{'dictionary':{'status':'done','layers':36,'fidelity_passed_layers':[35]},'controlled_generation':{'status':'done','inputs':4096,'all_behavior_cells_passed':True},'causal':{'status':'not_supported','selected_features':16},'capability':{'status':'noninferior','n':2200},'semantic_mechanism':{'status':'not_complete'}})
    P.save(P.OUT/'registration-delivery.json',{**final,'report_sha256':rr,'artifact_refs':refs},False)
    previous=P.ROOT/'tests/glm5/result/mlp_atlas_inventory_v1/registration-v5.json';old=P.read(previous);I=Archive(P.ROOT/'research/mlp_atlas/inventory_v1',old['run_id'],old);d=I.lib.artifact(old['report_sha256']);d['research_refs']['fidelity_v2_report']=rr;d['next_run_id']='MLP-FUNCTIONAL-FIDELITY-AND-UPSTREAM-v1';d['research_plan_sha256']=refs['next_stage'];d['next_stages']=[{'id':'behavior_metric','title':'面向输出分布的字典','acceptance':'新留出同时比较FVU、KL、真实next-token损失恢复、同活跃预算与成本'},{'id':'upstream','title':'问题前与进入末层的真实来源','acceptance':'全隐态候选、真实QKV/MLP/RMS来源与未见实体/关系组合预测'},{'id':'intervention','title':'跨答案余量的范围干预','acceptance':'冻结因果位置、剂量、随机/救援对照与新能力非劣效门'},{'id':'graph','title':'图谱独立预测','acceptance':'保留误差通道，预测未见激活和next-token，超过词汇/位置/n-gram基线'}]
    ir=I.put('report-fidelity-v2-continuation',d);I.bind(ir,old['graph_sha256'],{'study_type':'mlp_reverse_inventory','all_units':350208,'layer_refs':old['layer_refs'],'continuation':rr,'next_stage':refs['next_stage']},'350208个原生MLP单元 · 全量计算与持续证据账本',{'inventory':{'status':'done','units':350208},'fidelity_v2':{'status':'done','report_sha256':rr,'causal_effect_supported':False},'semantic_causal_closure':{'status':'not_complete'}})
    P.save(previous.with_name('registration-v6.json'),{**old,'report_sha256':ir,'fidelity_v2_report':rr,'research_plan_sha256':refs['next_stage']},False);print({'report':rr,'inventory':ir,'next_stage':refs['next_stage']},flush=True)
if __name__=='__main__':main()
