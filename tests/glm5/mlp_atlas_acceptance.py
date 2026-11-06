"""Actual localhost integration checks; not synthetic research evidence."""
import sys,json,time,urllib.request,urllib.error
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    root='http://127.0.0.1:5001';records=[]
    def get(path,expected=200):
        try:
            with urllib.request.urlopen(root+path,timeout=60) as r:code=r.status;d=json.load(r)
        except urllib.error.HTTPError as e:code=e.code;d=json.load(e)
        assert code==expected,(path,code,d);records.append({'path':path,'status':code});return d
    run=P.RUN;base='/api/traces/mlp-bank/'+run;man=get(base+'/manifest');assert man['state']=='complete' and man['completed']==9984
    for index in [0,9983]:
        s=get(base+f'/samples/{index}');assert s['model_snapshot']==man['model_snapshot']
        for stage in [0,1]:
            f=get(base+f'/samples/{index}/field?block=35&stage={stage}&boundary=mlp_activation');assert f['shape']==[9728] and f['position']==s['analysis_readouts'][stage]
            z=get(base+f'/samples/{index}/field?block=35&stage={stage}&boundary=head_z');assert z['shape']==[32,128]
    get(base+'/samples/9984',422);get(base+'/samples/0/field?block=0&stage=0&boundary=attention_rows',404)
    source=get(base+'/samples/9983/connections?block=35&stage=1&unit=9727&head=31&coordinate=2559');assert source['verified_full_snapshot'] is True
    layer=get(base+'/analysis/layers/35');assert len(layer['response_replicated'][0])==9728 and not layer['behavior_gate'];u=get(base+'/analysis/units/35/9727');assert not any(u['eligible_supported'])
    dic=get('/api/traces/mlp-dictionary/manifest');assert len(dic['layers'])==36
    dl=get('/api/traces/mlp-dictionary/layers/35');assert len(dl['lockbox_partner_correlation'])==9728
    du=get('/api/traces/mlp-dictionary/units/35/9727');assert len(du['all_discovery_correlations'])==1024
    df=get('/api/traces/mlp-dictionary/features/35/1023');assert len(df['weights']['decoder'])==2560 and len(df['all_native_unit_discovery_correlations'])==9728
    examples=sum(df['natural_examples'].values(),[])
    if examples:
        ex=examples[0];ns=get(f"/api/traces/mlp-dictionary/samples/{ex['document_index']}/{ex['slot']}?block=35");assert len(ns['dictionary_activation'])==1024 and len(ns['output'])==2560
        assert abs(ns['dictionary_activation'][1023]-ex['activation'])<=.001*max(1,abs(ex['activation']))
    al=get('/api/traces/mlp-parameter-audit/layers/35');assert len(al['Gate_term_RMS'][1])==9728
    au=get('/api/traces/mlp-parameter-audit/units/35/9727');assert len(au['Gate_term_split_code_mapping_mean'])==3
    catalog=get('/api/traces/research-space');studies={s['run_id']:s for s in catalog['studies']}
    for rid in [P.RUN,'MLP-ATLAS-INVENTORY-20261104-v1','MLP-ATLAS-DICTIONARY-20261104-v1','MLP-ATLAS-PARAMETER-20261104-v1']:
        report=get('/api/research-library/artifacts/'+studies[rid]['report_sha256']);assert report['model_snapshot']==man['model_snapshot']
    # Independently rehash the completed source bank after the parameter audit.
    audited=[]
    for row in man['files']:
        assert file_sha(P.OUT/'bank'/row['path'])==row['sha256'];audited.append(row['sha256'])
    assert file_sha(P.OUT/'bank/samples.sqlite3')==man['sample_index_sha256']
    result={'schema':'mlp_atlas_real_acceptance.v1','kind':'real','run_id':P.RUN,'model_snapshot':man['model_snapshot'],'provenance':man['provenance'],'time':time.time(),'checks':records,'source_bank_rehashed_files':len(audited),'sample_index_rehashed':True,'source_arithmetic_example':source,'dictionary_feature_source':df['source_sha256'],'scope':'Software/API and provenance acceptance, not scientific promotion'}
    P.save(P.OUT/'real-acceptance.json',result);print(json.dumps({'API_checks':len(records),'source_files_rehashed':len(audited),'passed':True}),flush=True)
if __name__=='__main__':main()
