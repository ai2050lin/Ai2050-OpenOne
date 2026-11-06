"""Real center/node calibration pilot and external-AI grading preparation."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
import phase2773_w6_loop as W
import phase2790_materials as M
from shared.dist_protocol import natural_register_runner as N

ROOT=W.ROOT;OUT=ROOT/'tests/glm5/result/phase2790_register';PUBLIC=ROOT/'research/phase2790'
RUN='P2790-REGISTER-PIPELINE-v1';TM='NAT-REGISTER-P2790-v1';OLD=ROOT/'tests/glm5/result/phase2773_w6_v2'


def provenance():
    git=lambda *args:subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*args],cwd=ROOT,text=True).strip()
    return {'git_sha':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),'model_path':str(W.MODEL),
        'model_revision':W.read(OLD/'freeze.json')['model_snapshot']['sha256'],'seed':27732026,'created_at':time.time()}


def status(stage,state='done',**kw):
    from deploy.research_library import Library
    p=PUBLIC/'00-progress.research.json';d=W.read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real',
        'run_id':RUN,'title':'多领域长文本 · 双AI盲评与真实分布式任务闭环','stages':{},
        'model_snapshot':W.read(OLD/'freeze.json')['model_snapshot']['sha256']}
    d['stages'][stage]={'state':state,'time':time.time(),**kw};d['updated_at']=time.time()
    W.save(p,d);Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report')
    print(json.dumps({'stage':stage,'state':state,**kw},ensure_ascii=False),flush=True)


def prepare():
    import numpy as np
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    tok=AutoTokenizer.from_pretrained(W.MODEL,local_files_only=True);builder=M.Builder(tok);items=[]
    for i in range(5000,5008):
        for frame in [0,1]:
            for length in [512,1024,2048]:
                for r in builder.pair(i,frame,length,'discovery_calibration'):r['shard_index']=i-5000;items.append(r)
    assert len(items)==96
    z=np.load(OLD/'directions.npz');edit={'alpha':6,'seed':27732026,
        'directions':{str(l):z[f'u_{l}'].tolist() for l in range(18,28)},
        'columns':{str(l):z[f'order_{l}'][:512].tolist() for l in range(18,28)},
        'energy':{str(l):z[f'energy_{l}'].tolist() for l in range(18,28)}}
    c={'schema':N.SCHEMA,'job_kind':N.JOB_KIND,'run_id':RUN,'model_snapshot':W.read(OLD/'freeze.json')['model_snapshot']['sha256'],
        'precision':'bf16','conditions':N.CONDITIONS,'items':items,'shards':8,'max_new_tokens':256,'batch_size':2,
        'edit':edit,'source_edit_sha256':N.digest(edit),'canary_item':items[0],'timeout_seconds':3600,
        'evaluation_mode':'two_external_ai','provenance':provenance(),
        'resources':{'gpu':'CUDA, sequential model execution','model':'Qwen3-4B only','min_vram_gib':12,
            'min_free_disk_gib':12,'native':'all37 hidden layers; all9728 MLP inputs B18..27; five prefix positions and every actual cached generation step, base and target',
            'native_not_uploaded':True,'estimated_max_native_bytes_per_shard':4*(1<<30)}}
    c['canary_recipe_sha256']=N.digest(c['canary_item']);audit=N.validate_corpus(c)
    formal_power=check_phase(.20,128,p_d=.5,alpha=.025);assert formal_power['pass']
    design={'schema':'natural_calibration_prereg.v1','kind':'real','run_id':RUN,'registered_at':time.time(),
        'scope':'discovery calibration and executable protocol validation, not a confirmatory mechanism claim',
        'user_evaluator_change':'user requested other AI models instead of unavailable human reviewers; all evidence remains AI-rated',
        'raters':['gpt-6-astra','gpt-6.1-sol'],'blindness':'text and core facts only; no weight condition, requested register, source ID or other-rater labels',
        'ratings':'formal/spoken/mixed/uncertain; preserved/changed/uncertain meaning; naturalness1..5; reasons saved',
        'primary_consensus':'style accepted only if both agree formal/spoken; meaning accepted only if both preserved; minimum naturalness3; disagreements never promoted by majority after results',
        'pilot_gate':'each requested register>=.70 consensus whole-output style, pooled meaning>=.70 and no cell with both source families failing style; this gate selects whether to expand, not mechanism proof',
        'pilot_n':audit,'domains':M.DOMAINS,'frames':[0,1],'lengths':[512,1024,2048],
        'material_boundary':'authored controlled documents, with repeated discourse frameworks; not a downloaded natural corpus; lengths add background information, not a pure length intervention',
        'prior_exposure':'P2773/P2777 results exposed; new evaluator calibration84 exposed before this pilot, with authored reference labels and no human gold standard',
        'formal_next_design':{'not_run':True,'n_families':128,'expected_delta':.20,'per_hypothesis_alpha':.025,
            'power':formal_power,'heldout_records':[6000,6127],'heldout_frames':[2,3],
            'criteria':'freeze a new version before any formal outputs; bidirectional style effects, content loss and capability NI, own random-direction and energy-column controls'},
        'capability':'reuse previous2200 monitoring results only after exact edited BF16 block hashes match, same full model snapshot/dtype/MCQ recipe; otherwise new NI run required',
        'no_claims':['human validation','independent second physical node','new dictionary feature discovery','graph-only prediction','minimum edit for natural generation'],
        'runner_sha256':N.file_sha(N.__file__),'materials_sha256':W.sha(M.__file__),'source_edit_sha256':c['source_edit_sha256'],
        'provenance':c['provenance']}
    W.save(OUT/'corpus.json',c,True);W.save(PUBLIC/'prereg.research.json',design,True)
    W.save(OUT/'freeze.json',{'run_id':RUN,'corpus_sha256':N.digest(c),'corpus_file_sha256':W.sha(OUT/'corpus.json'),
        'runner_sha256':N.file_sha(N.__file__),'materials_sha256':W.sha(M.__file__),
        'style_request_source_sha256':W.sha(M.OLD.__file__),'design_sha256':W.sha(PUBLIC/'prereg.research.json'),
        'source_directions_sha256':W.sha(OLD/'directions.npz'),'model_snapshot':c['model_snapshot'],'provenance':c['provenance']},True)
    status('freeze',samples=96,families=8,lengths=[512,1024,2048],evaluation='two_external_ai',formal_power_plan=formal_power)


def run():
    sys.path.insert(0,str(ROOT/'deploy'));import node_agent as A
    c=W.read(OUT/'corpus.json');freeze=W.read(OUT/'freeze.json')
    assert N.digest(c)==freeze['corpus_sha256'] and N.file_sha(N.__file__)==freeze['runner_sha256']
    private_path=ROOT/'tests/glm5_temp/phase2790-service-private.json'
    secret=json.loads(private_path.read_text(encoding='utf-8-sig'));server=secret['server'];headers={'X-Admin-Token':secret['admin_token']}
    import urllib.request
    def request(path,body=None):
        raw=json.dumps(body,ensure_ascii=False).encode() if body is not None else None
        req=urllib.request.Request(server+path,data=raw,headers={'Content-Type':'application/json',**headers})
        with urllib.request.urlopen(req,timeout=90) as response:return json.load(response)
    registered=request('/api/templates',{'tm_id':TM,'version':1,'name':'多领域长文本自然生成 · 双AI评分校准','dim':'ST',
        'description':'真实CUDA探索校准；任务执行通过不表示机制成立；条件按整家族分片','metric_pin':'external-ai-consensus.v1',
        'model_pin':c['model_snapshot'],'corpus':c,'runner':Path(N.__file__).read_bytes().decode('utf-8')})
    W.save(OUT/'template-registration.json',registered,True)
    creds_path=ROOT/'tests/glm5_temp/phase2790-node-private.json'
    if creds_path.exists():creds=json.loads(creds_path.read_text(encoding='utf-8'))
    else:
        _,new=A.http_json(server+'/api/nodes/register','POST',{'name':'P2790-local-cuda','gpu':'RTX5080','model':'Qwen3-4B','dtype':'bf16'})
        creds={'server':server,**new};creds_path.write_text(json.dumps(creds),encoding='utf-8')
    os.environ['AI2050_NATIVE_DIR']=str(OUT/'native');(OUT/'native').mkdir(exist_ok=True)
    os.environ['PYTHONIOENCODING']='utf-8'
    args=argparse.Namespace(python=str(ROOT/'.venv/Scripts/python.exe'),run_timeout=3600.,model_path=str(W.MODEL),
        model_id='qwen3-4b',model_rev=c['model_snapshot'],tokenizer_sha256='',gpu='RTX5080',dtype='bf16',seed=27732026,
        tm_id=TM,server=server,job_kind='natural_register')
    status('collection','running',task_kind='natural_register',node_id=creds['node_id'])
    for index in range(8):
        did=A.run_one(args,creds,OUT/'node-tasks')
        _,queue=A.http_json(server+'/api/tasks','GET');counts=next((x for x in queue['by_template'] if x['tm_id']==TM),{})
        status('collection','running',iteration=index+1,counts=counts)
        if not did:break
        if counts.get('failed'):raise RuntimeError('Real node shard failed; checkpoints and frozen version retained')
    _,queue=A.http_json(server+'/api/tasks','GET');counts=next(x for x in queue['by_template'] if x['tm_id']==TM)
    assert counts['done']==8 and not counts['failed'],counts
    request(f'/api/templates/{TM}/status',{'status':'closed','note':'本机校准分片完整；等待外部AI盲评，不自动晋级机制。'})
    summaries=[W.read(p) for p in (OUT/'node-tasks').glob('*/outputs/summary.json')]
    assert len(summaries)==8
    W.save(OUT/'execution.json',{'schema':'natural_execution.v1','kind':'real','run_id':RUN,'template':registered,
        'counts':counts,'summaries':summaries,'physical_nodes':1,'provenance':c['provenance']},True)
    status('collection',shards=8,samples=96,outputs=384,physical_nodes=1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();{'prepare':prepare,'run':run}[a.stage]()
