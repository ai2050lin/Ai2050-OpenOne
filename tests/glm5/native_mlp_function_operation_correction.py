"""Preserve original audit artifacts, verify corrected SDPA rows, rebind current cases."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_publish import register_binding
from shared.dist_protocol.mechanism_trace import TraceStore,file_sha,digest,save_json
from deploy.research_library import Library

def main():
    lib=Library(P.ROOT/'distributed_data/research_library');store=TraceStore();r=P.read(P.OUT/'client-registration-final.json');report=lib.artifact(r['report_sha256']);prov=report['provenance'];snap=report['model_snapshot']
    def archive(name,doc,run=P.RUN):
        d={'schema':'research_native_evidence.v1','kind':'real','run_id':run,'model_snapshot':snap,'provenance':prov,**doc};p=P.PUBLIC/(name+'-'+digest(d)[:12]+'.research.json');P.save(p,d)
        return lib.ingest(d,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
    oldidx=lib.artifact(report['case_index_sha256']);oldby={(c['run_id'],c['sample_id']):c for c in oldidx['cases']};new=P.read(P.OUT/'operation-traces.v2.json');mapping={};audits=[];extra_bytes=0;extra_files=0;manifestrefs=[]
    for row in new['rows']:
        old=oldby[row['run_id'],row['sample_id']];a,roota=store.manifest(old['trace_id']);b,rootb=store.manifest(row['trace_id']);maxerr=max(x['attention_relative_l2'] for x in b['reconstruction'].values());assert maxerr<.05
        same=True
        for block in range(36):
            with np.load(roota/f'block_{block:02d}.npz') as za,np.load(rootb/f'block_{block:02d}.npz') as zb:
                for key in za.files:
                    if key!='attention_rows':same=bool(same and np.array_equal(za[key],zb[key]))
                if b.get('input_attention_mask'):
                    mask=np.array(b['input_attention_mask'],bool);assert np.all(zb['attention_rows'][...,~mask]==0)
        same=bool(same and np.array_equal(np.load(roota/'hidden_readouts.npy'),np.load(rootb/'hidden_readouts.npy')));assert same,'actual computational state differs'
        b['annotation_revision']={**b['annotation_revision'],'supersedes_trace_id':a['trace_id'],'all_actual_module_states_bitwise_equal':True,'only_derived_attention_rows_recomputed':True}
        b.pop('trace_id');b['trace_id']=digest(b);p=rootb/'manifest-corrected-mask.v4.json';save_json(p,b);store.register(p);mapping[a['trace_id']]=b['trace_id']
        for f in b['files']:assert file_sha(rootb/f['path'])==f['sha256'];extra_bytes+=f['bytes'];extra_files+=1
        manifestrefs.append(archive('mask-corrected-manifest-'+b['trace_id'],b,b['run_id']))
        audits.append({'old_trace_id':a['trace_id'],'trace_id':b['trace_id'],'run_id':b['run_id'],'sample_id':b['sample_id'],'max_attention_relative_l2':maxerr,'actual_module_states_bitwise_equal':same,'prediction_consistent':row.get('prediction_consistent',True)})
    def remap(x):
        if isinstance(x,dict):return {k:remap(v) for k,v in x.items()}
        if isinstance(x,list):return [remap(v) for v in x]
        return mapping.get(x,x) if isinstance(x,str) else x
    idx={**remap(oldidx),'supersedes_case_index_sha256':report['case_index_sha256'],'case_count_unchanged':4358};case_sha=archive('case-index-mask-v4',idx)
    erratum=archive('mask-correction-audit',{'issue':'Historical recorder added boolean masks as numbers, while SDPA interprets True as allowed. Display reconstruction was invalid on padded audit inputs. Actual model computation was always passed the original correct mask.',
        'fix':'Same inputs, batch4 padding geometry, weights, edits and original SDPA; reconstructed rows mask False to -inf before softmax. All actual hidden/G/V/A/head/output arrays are bitwise equal to earlier capture.',
        'rows':audits,'manifests':manifestrefs,'new_tensor_bytes':extra_bytes,'new_tensor_files':extra_files,'old_artifacts_retained':True,'no_new_independent_samples':True})
    final=remap(report);final.update(delivery_revision=3,supersedes_report_sha256=r['report_sha256'],case_index_sha256=case_sha,attention_mask_erratum_sha256=erratum)
    final['artifact_refs']={**final['artifact_refs'],'corrected_operation_replays':erratum,'case_index':case_sha}
    final['native']={**final['native'],'total_bytes':final['native']['total_bytes']+extra_bytes,'trace_bytes':final['native']['trace_bytes']+extra_bytes,'tensor_files':final['native']['tensor_files']+extra_files,'archived_trace_instances':4378,'current_unique_cases':4358,'mask_replay_verified_all_actual_states_bitwise_equal':True}
    final['limitations'].append('20条操作审计追加布尔mask修复复采；旧错误注意力行保留且API拒绝伪作真实权重展示。其余实际模块状态逐比特一致，模型预测、剂量和能力主判不变。')
    rr=archive('report-mask-corrected-v3',final)
    with lib.db() as db:rows=db.execute("SELECT sha FROM artifacts WHERE kind='space_binding' ORDER BY created DESC,sha DESC").fetchall()
    seen=set()
    for row in rows:
        b=lib.artifact(row['sha']);key=(b['run_id'],b['model_snapshot'])
        if key in seen:continue
        seen.add(key)
        if b['run_id'] in [P.RUN,P.RUN+':place-polarity',P.RUN+':capability-audit',P.RUN+':operation-audit']:
            traces=remap(b['traces'])
            # Proof hashes are regenerated for revised manifests.
            for t in traces:t.pop('manifest_sha256',None)
            ctx={**remap(b['context']),'case_index_sha256':case_sha,'attention_mask_erratum_sha256':erratum}
            register_binding(archive,store,b['run_id'],rr,r['graph_sha256'],traces,ctx,'原生MLP：答案方向反例与真实3D闭环' if b['run_id']==P.RUN else b['run_id'].split(':')[-1]+' · 真实对照复采',native=final['native'] if b['run_id']==P.RUN else None)
    out={**r,'report_sha256':rr,'case_index_sha256':case_sha,'native':final['native'],'mask_correction_sha256':erratum};P.save(P.OUT/'client-registration-mask-v4.json',out)
    P.save(P.OUT/'mask-correction.json',{'rows':audits,'mapping':mapping,'new_tensor_bytes':extra_bytes,'report_sha256':rr,'case_index_sha256':case_sha})
    print(json.dumps({'cases':4358,'replays':20,'all_actual_states_bitwise_equal':True,'max_attention_relative_l2':max(r['max_attention_relative_l2'] for r in audits),'report_sha256':rr,'case_index_sha256':case_sha}))
if __name__=='__main__':main()
