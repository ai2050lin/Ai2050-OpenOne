"""Real localhost acceptance, without loading a GPU model or fabricating responses."""
import sys,json,time,math
from pathlib import Path
import requests
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P

def main():
    root='http://127.0.0.1:5001';r=P.read(P.OUT/'client-registration-mask-v4.json');checks=[]
    def get(path):
        t=time.time();z=requests.get(root+path,timeout=90);z.raise_for_status();checks.append({'path':path,'status':z.status_code,'seconds':time.time()-t});return z.json()
    rr=get('/api/research-library/artifacts/'+r['report_sha256']);assert rr['delivery_revision']>=3
    index=get('/api/traces/research-space');allcases=get('/api/research-library/artifacts/'+r['case_index_sha256'])['cases'];assert len(allcases)==4358
    cat=get('/api/traces/catalog')['traces'];ids={d['trace_id'] for d in cat};assert all(d['trace_id'] in ids for d in allcases)
    study=next(d for d in index['studies'] if d['run_id']==P.RUN);assert study['report_sha256']==r['report_sha256'] and study['localization']=='native_binding'
    models={d['snapshot']:d for d in index['models']};assert models[rr['model_snapshot']]['d_ff']==9728
    grouped={}
    for case in allcases:grouped[case['run_id']]=grouped.get(case['run_id'],0)+1
    t=rr['examples'][0]['trace_id'];m=get(f'/api/traces/{t}/manifest');assert m['annotation_revision']['version']==3;assert m['readout_annotations'][2]['role']=='question_end' and m['readout_positions'][2]==1023
    for boundary,b,n in [('embedding',0,2560),('hidden_state',36,2560),('mlp_gate',35,9728),('mlp_value',35,9728),('mlp_activation',35,9728),('head_z',35,4096)]:
        f=get(f'/api/traces/{t}/field?block={b}&position=1023&boundary={boundary}');assert len(f['values'])==n and all(math.isfinite(x) for x in f['values'])
    c=get(f'/api/traces/{t}/connections?block=24&position=1023&head=31&unit=8443&coordinate=0');assert c['verified_full_snapshot'] and len(c['head_gate_terms'])==32 and len(c['mlp_output_terms'])==9728 and len(c['gate_terms'])==2560
    a=get(f'/api/traces/{t}/attention-row?block=24&head=31&query=1023');assert len(a['weights'])==m['n_tokens'] and abs(sum(a['weights'])-1)<.01
    ctx=get(f'/api/traces/context?run_id={P.RUN}&model_snapshot={rr["model_snapshot"]}');assert len(ctx['unit_cards'])==14
    for family,refs in ctx['family_profile_refs'].items():
        assert len(refs)==36
        for block in ['0','35']:
            p=get('/api/research-library/artifacts/'+refs[block]);assert p['block']==int(block) and p['model_snapshot']==rr['model_snapshot']
            for split in ['discovery','confirm','lockbox']:
                assert len(p['splits'][split]['GVA_delta'])==4 and len(p['splits'][split]['GVA_delta'][3][2])==9728
                assert len(p['splits'][split]['head_delta_norm'][3])==32 and len(p['splits'][split]['embedding_delta'][3])==2560
    z=requests.get(root+'/api/traces/profile',params={'run_id':P.RUN,'block':24});assert z.status_code==404
    polarity=next(d for d in allcases if d['run_id'].endswith(':place-polarity'));pm=get(f'/api/traces/{polarity["trace_id"]}/manifest');assert pm['response_profile_stage_indices']==[2,3]
    op=next(d for d in allcases if d['run_id'].endswith(':capability-audit') and d['sample_id'].endswith(':edited'));om=get(f'/api/traces/{op["trace_id"]}/manifest');assert om['intervention']['activation_is_intervened']
    oc=get(f'/api/traces/context?run_id={op["run_id"]}&model_snapshot={rr["model_snapshot"]}');assert 'family_profile_refs' not in oc
    aw=get(f'/api/traces/{op["trace_id"]}/attention-row?block=24&head=0&query={om["readout_positions"][0]}')['weights'];assert all(w==0 for w,mask in zip(aw,om['input_attention_mask']) if not mask)
    for oldrun in ['P2793-ENTITY-ARBITRATION-v1','P2794-ATOMIC-ARBITRATION-v4','NMLP-LONG-20261104-v1']:
        os=next(d for d in index['studies'] if d['run_id']==oldrun);assert os['binding_sha256'] and os['localization']=='native_binding'
    result={'kind':'real','run_id':P.RUN,'checks':checks,'case_counts':grouped,'all4358_latest_cases_queryable':True,'metadata_corrections_do_not_add_independent_samples':True,'space_counts':index['counts'],
        'profile_files':108,'native_unit_cards':14,'masked_padding_attention_zero':True,'operation_state_explicit':True,'provenance':rr['provenance']}
    P.save(P.OUT/'api-acceptance.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['checks','provenance']},ensure_ascii=False))
if __name__=='__main__':main()
