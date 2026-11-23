"""Registered one-forward 3D acceptance using a byte-verified historical input."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

ROOT = (Path(os.environ['PYTHONPATH'].split(os.pathsep)[0]) if Path(__file__).name=='test.py'
        else Path(__file__).resolve().parents[2])
sys.path.insert(0,str(ROOT))
from deploy.parameter_reader import model_snapshot
from deploy.storage_policy import checked_dir, check_write
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, file_sha, validate_plan

TESTS = ROOT/'tests/glm5'
OUT = TESTS/'result/live_original_14b_v4'
ENTRY = TESTS/'live_original_14b.py'
STAGE = 'live-original'
BROWSER = TESTS/'live_original_browser.cjs'
MODEL = ROOT/'models/hf/Qwen3-14B'
SOURCE = TESTS/'result/clt_b26_three_cues_v3'
CASE = 'confirm-000-02'
NOVELTY = '同一历史题完整原文重跑一次，确认 14B BF16 本机实时接口和 3D 读取相同原值。新增三个 token 的整层 MLP 与 Head 原场、缺测灰格、色标锁定和同题回放的端到端核验。'
CRITERIA = ('One BF16 CUDA forward in the regular local live endpoint. Full original prompt bytes, 191 actual input token IDs and model snapshot equal the historical record. '
            'Gold first token 81914. All 128 native B26/H36 channels round-trip to the 3D view at positions 0/95/190, as do whole-layer B26 MLP vector identity and last unit. '
            'Original final Head vector relative L2 <= 0.04 and maximum absolute delta <= 0.125; current UI values exactly equal the current native trace. '
            'An uncollected module position must show missing; scales must remain locked across position changes. Reopen the new trace and the original bundled case independently.')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sources():
    paths = [ENTRY,BROWSER,ROOT/'server/live_only_app.py',ROOT/'server/live_analysis_service.py',
             ROOT/'server/mechanism_trace_service.py',ROOT/'shared/dist_protocol/mechanism_trace.py',
             ROOT/'deploy/parameter_reader.py',ROOT/'deploy/storage_policy.py',
             *(ROOT/'frontend/src/components/v8'/p for p in ('AssocWorkbench.jsx','AssocLayerScene.jsx','useSceneHeadField.js',
               'nativeTraceField.js','GraphWorkspaceEvidence.jsx','liveRun.js'))]
    return {p.relative_to(ROOT).as_posix():file_sha(p) for p in paths}


def prepare():
    checked_dir(OUT,64*2**20,operation='同题实时验收方案')
    plan_path=OUT/f'{STAGE}-plan.json'
    if plan_path.exists():raise FileExistsError(plan_path)
    receipt=read(SOURCE/'delivery.json')
    for name in ('confirm-materials.json','confirm-00.npz','confirm-00.json'):
        if file_sha(SOURCE/name)!=receipt['files'][name]:raise ValueError('mismatched archive '+name)
    row=read(SOURCE/'confirm-materials.json')[2]
    history=read(SOURCE/'confirm-00.json')['scores'][2]
    snapshot=receipt['provenance']['model_revision']
    bundled=read(ROOT/'frontend/src/components/v8/data/archiveHeadInputs.json')
    if not (row['id']==history['id']==CASE and row['prompt']==bundled['prompts'][CASE]
            and bundled['snapshot']==snapshot and row['gold_id']==history['top1']==81914
            and len(row['tokens'])==191 and row['tokens'][-1]==510):
        raise ValueError('the frozen case, prompt, tokenizer boundary or historical prediction differ')
    with np.load(SOURCE/'confirm-00.npz',allow_pickle=False) as archive:
        z=archive['head_actual'][2].tolist()
    questions=OUT/f'{STAGE}-questions.json'
    save(questions,[dict(id=CASE,snapshot=snapshot,**{k:row[k] for k in ['prompt','tokens','gold_id']},archived_z=z)])
    token_names=('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','chat_template.jinja')
    tokenizer=digest({k:file_sha(MODEL/k) for k in token_names if (MODEL/k).is_file()})
    code=digest(sources()); entry_sha=file_sha(ENTRY); qs=file_sha(questions)
    plan=dict(schema='research_test_plan.v1',kind='real',title='Qwen3-14B B26/H36 同一道原题实时前向与 3D 回放验收',
        objective=NOVELTY,principle='在本机正常实时入口原样提交已审计历史输入，对比实际张量、界面原值和归档。',
        hypothesis='固定快照和字节输入能复现原答与 B26/H36 向量，并通过同轨迹逐位置验收。',
        alternatives='模板或 token 变化、BF16 批形舍入、原场缺测、同名快照或缓存串值。',criteria=CRITERIA,
        domains=['KN','RS'],family_ids=['CLT-ROLE-REREAD','SOURCE-BINDING-20261110','LIVE-3D-REPLAY'],
        entrypoint_sha256=entry_sha,execution_args=[],
        dataset=dict(id='phase2983-confirm-000-02-live',revision='v4',content_sha256=qs,questions_file_sha256=qs,
            split_manifest_sha256=digest([CASE,receipt['files']['confirm-materials.json']]),case_count=1,
            exposure='Previously opened and published benchmark case, no independent research sample.'),
        acquisition=dict(model_id='Qwen3-14B',model_snapshot=snapshot,tokenizer_sha256=tokenizer,inputs_sha256=digest([row['tokens']]),
            code_sha256=code,entrypoint_sha256=entry_sha,questions_file_sha256=qs,environment_sha256=digest(runtime_identity()),
            precision='BF16 original safetensors, CUDA layered SDPA',seed=3074,
            generation=dict(script_args=[],sample=False,prompt_format='rendered',max_new_tokens=1),
            capture=dict(layers=list(range(40)),readouts=[0,95,190],boundary='pre-O Z and down_proj input A; full token residual and Q/K/V; no KV cache',
                         targets=['B26/H36','B26/U17407'])),
        analysis=dict(method_version='live_original_14b.v4',code_sha256=code,scoring_sha256=digest(CRITERIA),
            selection=dict(stage=STAGE,source='clt_b26_three_cues_v3',case=CASE),required_artifact_roles=sorted(CORE_ROLES|{'native_field'})))
    validate_plan(plan); save(plan_path,plan)
    print(json.dumps({'plan':str(plan_path),'sha256':digest(plan),'case':CASE,'token_count':len(row['tokens'])}),flush=True)


def execute():
    from datetime import datetime
    import faulthandler
    from source_binding_resources import resources
    import uvicorn
    from server.live_only_app import app
    from server.live_analysis_service import safe_live_unload
    from shared.dist_protocol.mechanism_trace import unbits

    faulthandler.enable()

    work=Path(os.environ['AI2050_TEST_WORK_DIR'])
    question_file=Path(os.environ['AI2050_TEST_QUESTIONS'])
    plan, preflight, review=[read(question_file.parent/name) for name in ('protocol.json','preflight.json','review.json')]
    run_id=review['server_run_id']
    if (not run_id or not preflight.get('preflight_id') or digest(plan)!=preflight['plan_sha256'] or
        plan['entrypoint_sha256']!=file_sha(Path(__file__)) or plan['acquisition']['code_sha256']!=digest(sources()) or
        file_sha(question_file)!=plan['dataset']['questions_file_sha256']):
        raise ValueError('registered plan, preflight, script or original input mismatch')
    [question]=read(question_file)
    print(json.dumps({'stage':'verify_model','completed':0,'total':3,'start':datetime.now().astimezone().isoformat(),**resources()}),flush=True)
    if model_snapshot(MODEL)['sha256']!=question['snapshot']:raise ValueError('local Qwen3-14B snapshot differs')
    dist=Path(os.environ['AI2050_LIVE_ACCEPT_DIST']).resolve(strict=True)
    if not (dist/'index.html').is_file():raise FileNotFoundError('Vite build must precede browser acceptance')
    os.environ.update(AI2050_LIVE_MODEL_DIR=str(MODEL),AI2050_LIVE_TRACE_DIR=str(work/'trace'))
    checked_dir(work/'trace',GIB:=2**30,operation='本机同题原生采集')
    with ZipFile(work/'research-scripts.zip','x',compression=ZIP_DEFLATED) as package:
        for rel in sources(): package.write(ROOT/rel,arcname=rel)
    entries=[dict(path='research-scripts.zip',role='script',retention='core')]
    save(work/'artifacts.json',dict(schema='research_test_files.v1',files=entries))
    listener=socket.socket()
    listener.bind(('127.0.0.1',0))
    port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,log_level='info',lifespan='off',loop='asyncio'))
    thread=threading.Thread(target=server.run,kwargs={'sockets':[listener]},daemon=True);thread.start()
    started=time.monotonic()
    while not server.started and thread.is_alive() and time.monotonic()-started<12:time.sleep(.05)
    if not server.started:
        listener.close()
        raise RuntimeError('local live service did not start')
    proof=None
    try:
        print(json.dumps({'stage':'live_browser','completed':1,'total':3,'elapsed_seconds':time.monotonic()-started,**resources()}),flush=True)
        with subprocess.Popen(['node',str(BROWSER),f'http://127.0.0.1:{port}',str(dist),str(work),str(question_file)],
                              cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)) as process:
            for _ in range(8):
                try:code=process.wait(timeout=60);break
                except subprocess.TimeoutExpired:
                    check_write(work,GIB,operation='本机同题原生采集')
                    print(json.dumps({'stage':'live_browser','completed':1,'total':3,'elapsed_seconds':time.monotonic()-started,**resources()}),flush=True)
            else:process.terminate();code=process.wait(timeout=10)
        if code==0:proof=read(work/'browser-acceptance.json')
    finally:
        safe_live_unload();server.should_exit=True;thread.join(timeout=5)
        print(json.dumps({'stage':'unloaded','completed':2,'total':3,'elapsed_seconds':time.monotonic()-started,**resources()}),flush=True)
    manifests=list((work/'trace').glob('*/manifest.json'))
    if len(manifests)!=1:
        raise RuntimeError(f'one complete, durable native trace required; found {len(manifests)}')
    manifest=read(manifests[0])
    assert manifest['prompt']==question['prompt'] and manifest['model_snapshot']==question['snapshot']
    assert manifest['token_ids']==question['tokens'] and manifest['readout_positions']==[0,95,190]
    fields=manifests[0].parent/'block_26.npz'
    if file_sha(fields)!=next(f['sha256'] for f in manifest['files'] if f['path']==fields.name):
        raise ValueError('actual shard hash differs from capture manifest')
    with np.load(fields,allow_pickle=False) as data:
        z, a=data['head_z'],data['activation']
        assert z.shape==(3,40,128) and a.shape==(3,17408) and z.dtype==a.dtype==np.uint16
        new_z=unbits(z[-1,36]); old_z=np.asarray(question['archived_z'],dtype=np.float32)
        delta=new_z-old_z
        max_abs=float(np.max(np.abs(delta)));relative=float(np.linalg.norm(delta)/np.linalg.norm(old_z))
        np.savez_compressed(work/'native-0000.npz',head_z=z,mlp_activation=a,readout_positions=manifest['readout_positions'],
                            token_ids=manifest['token_ids'],manifest=json.dumps(manifest,ensure_ascii=False),legacy_z=old_z)
    entries.append(dict(path='native-0000.npz',role='native_field',retention='important'))
    save(work/'artifacts.json',dict(schema='research_test_files.v1',files=entries))
    passed=bool(proof and proof['passed'] and proof['trace_id']==manifest['trace_id'] and
                manifest['output_top'][0]['token_id']==question['gold_id'] and max_abs<=.125 and relative<=.04)
    comparison=dict(old_d0=float(old_z[0]),live_d0=float(new_z[0]),max_abs=max_abs,relative_l2=relative,
                    exact_channels=int(np.count_nonzero(delta==0)))
    case=dict(case_id=CASE,source_run='clt_b26_three_cues_v3',run_id=run_id,trace_id=manifest['trace_id'],
              model_snapshot=manifest['model_snapshot'],output_top=manifest['output_top'][0],correct=manifest['output_top'][0]['token_id']==question['gold_id'],
              readout_positions=manifest['readout_positions'],head_comparison=comparison)
    check_write(work/'case_results.jsonl',4096,operation='同题逐题结果').write_text(json.dumps(case,ensure_ascii=False)+'\n',encoding='utf-8')
    save(work/'conclusion.json',dict(passed=passed,run_id=run_id,case=case,browser=proof,new_validation=NOVELTY,
         placement=manifest['provenance']['placement'],archive_bundle='B26 full Z and A at 3 positions; the 40-block deep trace retained locally.'))
    print(json.dumps({'stage':'complete','completed':3,'total':3,'passed':passed,'comparison':comparison,**resources()}),flush=True)
    if not passed:raise RuntimeError('same-question numeric or browser acceptance failed; actual trace retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--launch-desktop',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.launch_desktop:
        from clt_role_reread_desktop import launch
        raise SystemExit(launch(STAGE,entry=ENTRY,output=OUT))
    else:execute()
