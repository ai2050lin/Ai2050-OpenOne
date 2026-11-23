"""Run the fixed first-record archive workflow from the desktop's active admin session."""
from argparse import Namespace
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from zoneinfo import ZoneInfo

from run_routine_test import ROOT, run
from deploy.storage_policy import check_write, storage_dir
from deploy.test_archive_client import Client, save, cleanup_scratch
from source_binding_design import read

HERE = ROOT/'tests/glm5'
OUT = HERE/'result/clt_first_record_path_v1'
STATE = OUT/'desktop-archive.json'
PLAN = OUT/'discovery-plan-v2.json'
QUESTIONS = HERE/'result/clt_b26_competition_v1/confirm-materials.json'
SCRIPT = HERE/'clt_first_record_discovery.py'
BLOCK_DUPLICATE = {
    'already_running': '已有同一采集在运行。',
    'reuse_results': '中心已有可复用的结果。',
    'reuse_acquisition': '中心已有可复用的采集。',
}


def execute(client):
    prior = read(STATE) if STATE.is_file() else {}
    pending = None
    if prior.get('stage') in {'failed', 'registered'} and prior.get('preflightId'):
        candidates = []
        for receipt in (HERE/'result/routine').glob('run-*/preflight.json'):
            if read(receipt).get('preflight_id') != prior['preflightId']:
                continue
            review = read(receipt.with_name('review.json'))
            if review['script'] == str(SCRIPT.relative_to(ROOT)) and not (Path(review['scratch'])/'output.log').exists():
                candidates.append(receipt.parent)
        if len(candidates) == 1:
            pending = candidates[0]
    state = {'stage': 'checking'}

    def update(stage, **extra):
        state.update(stage=stage, **extra)
        save(STATE, state)

    def checked(receipt):
        if receipt.get('schema') != 'research_test_preflight.v1' or not isinstance(receipt.get('matches'), list):
            raise ValueError('中心返回的预检格式不符')
        if not re.fullmatch(r'check-[0-9a-f]{32}', receipt.get('preflight_id', '')):
            raise ValueError('中心返回的预检 ID 不符')
        save(OUT/'discovery-preflight-v2.json', receipt)
        matches = receipt['matches']
        update('preflight', preflightId=receipt['preflight_id'], matchCount=len(matches))
        duplicate = next((m for m in matches if m.get('suggestion') in BLOCK_DUPLICATE), None)
        if duplicate:
            update('duplicate', matchedRunId=duplicate['run_id'], message=BLOCK_DUPLICATE[duplicate['suggestion']])
            raise DuplicateAcquisition

    def registered(registration, record):
        if (not registration.get('id') or registration.get('preflight_id') != state.get('preflightId')
                or registration.get('state') != 'running'):
            raise ValueError('中心返回的运行登记与预检不一致')
        update('registered', runId=registration['id'], record=str(record))

    update('checking')
    try:
        client.request('/runs?family=CLT-FIRST-RECORD-PATH&limit=20')
        args = Namespace(script=SCRIPT, questions=QUESTIONS, peak_gib=12, model_snapshot=None,
                         tokenizer_sha256=None, seed=None, archive_plan=PLAN, archive_decision='new_acquisition',
                         archive_reason='首轮绝对重建容差失败；改用原生量纲归一及独立 SDPA 复核，采集已暴露首行题的写入与 MLP。',
                         archive_source_run_id='', script_args=[])
        code = run(args, archive_client=client, on_preflight=checked, on_registered=registered, registered_record=pending)
        review = read(Path(state['record'])/'review.json')
        result = review.get('conclusion') or {}
        update('discovery_archived' if code == 0 else 'incomplete',
               archiveState=review.get('archive_state', 'unknown'), completedCases=result.get('cases', 0),
               message=review.get('archive_error') or review.get('stop_reason') or
               ('发现采集已归档。留出预测与因果干预尚无结果。' if code == 0 else '采集未完成，请查看本机日志。'))
        return code
    except DuplicateAcquisition:
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        msg = str(error)
        if 'HTTP 401' in msg or 'HTTP 403' in msg:
            msg = '当前登录会话未通过正式中心的管理员校验。'
        update('failed', message=msg)
        return 1


class DuplicateAcquisition(Exception):
    pass


def continue_prediction(client, previous):
    import clt_first_record_predict as experiment

    discovery = Path(previous['record'])
    state = {'stage':'prediction_checking', 'discoveryRunId':previous['runId']}

    def update(stage, **details):
        state.update(stage=stage, **details)
        save(STATE,state)

    def checked(receipt):
        save(OUT/'prediction-preflight.json', receipt)
        matches = receipt['matches']
        update('prediction_preflight', preflightId=receipt['preflight_id'], matchCount=len(matches))
        duplicate = next((m for m in matches if m.get('suggestion') in BLOCK_DUPLICATE), None)
        if duplicate:
            update('duplicate', matchedRunId=duplicate['run_id'], message=BLOCK_DUPLICATE[duplicate['suggestion']])
            raise DuplicateAcquisition

    def registered(registration, record):
        if (not registration.get('id') or registration.get('preflight_id') != state.get('preflightId')
                or registration.get('state') != 'running'):
            raise ValueError('留出登记与预检身份不符')
        update('prediction_registered', runId=registration['id'], record=str(record))

    update('prediction_checking')
    try:
        experiment.build_plan(discovery)
        args = Namespace(script=experiment.SCRIPT, questions=experiment.QUESTIONS, peak_gib=4,
                         model_snapshot=None, tokenizer_sha256=None, seed=None, archive_plan=experiment.PLAN,
                         archive_decision='new_acquisition', archive_source_run_id='',
                         archive_reason='在独立校准与确认家族上检验从已核验发现采集建立的路径。', script_args=[])
        code = run(args, archive_client=client, on_preflight=checked, on_registered=registered)
        review = read(Path(state['record'])/'review.json')
        conclusion = review.get('conclusion') or {}
        update('prediction_archived' if code == 0 else 'incomplete',
               archiveState=review.get('archive_state','unknown'), conclusion=conclusion,
               message=review.get('archive_error') or review.get('stop_reason') or
               ('留出预测已归档。' if code == 0 else '留出预测或归档待完成。'))
        if code == 0 and conclusion.get('prediction',{}).get('gate_passed'):
            return continue_causal(client,state)
        return code
    except DuplicateAcquisition:
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        update('failed',message=str(error))
        return 1


def continue_causal(client, previous):
    import clt_first_record_causal as experiment

    prediction = Path(previous['record'])
    state = {'stage':'causal_checking','predictionRunId':previous['runId'],
             'prediction':previous['conclusion']['prediction'], 'readout':previous['conclusion'].get('readout')}

    def update(stage, **details):
        state.update(stage=stage, **details)
        save(STATE,state)

    def checked(receipt):
        save(OUT/'causal-preflight.json', receipt)
        matches = receipt['matches']
        update('causal_preflight',preflightId=receipt['preflight_id'],matchCount=len(matches))
        duplicate = next((m for m in matches if m.get('suggestion') in BLOCK_DUPLICATE), None)
        if duplicate:
            update('duplicate',matchedRunId=duplicate['run_id'],message=BLOCK_DUPLICATE[duplicate['suggestion']])
            raise DuplicateAcquisition

    def registered(registration, record):
        if registration.get('preflight_id') != state.get('preflightId') or registration.get('state') != 'running':
            raise ValueError('干预登记与预检身份不符')
        update('causal_registered',runId=registration['id'],record=str(record))

    update('causal_checking')
    try:
        experiment.build_plan(prediction)
        args = Namespace(script=experiment.SCRIPT, questions=experiment.QUESTIONS, peak_gib=4,
                         model_snapshot=None, tokenizer_sha256=None, seed=None, archive_plan=experiment.PLAN,
                         archive_decision='new_acquisition', archive_source_run_id='',
                         archive_reason='用独立冻结家族检验已过留出门的路径，按等剂量对照及同输入恢复。',script_args=[])
        code = run(args,archive_client=client,on_preflight=checked,on_registered=registered)
        review = read(Path(state['record'])/'review.json')
        update('causal_archived' if code==0 else 'incomplete',archiveState=review.get('archive_state','unknown'),
               conclusion=review.get('conclusion') or {}, message=review.get('archive_error') or review.get('stop_reason') or
               ('干预结果已归档。' if code==0 else '干预采集或归档待完成。'))
        return code
    except DuplicateAcquisition:
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        update('failed',message=str(error))
        return 1


def resume_archive(client, prior):
    record = Path(prior['record']).resolve(strict=True)
    if not record.is_relative_to(HERE/'result/routine'):
        raise ValueError('归档恢复记录超出本机研究目录')
    review = read(record/'review.json')
    work = Path(review['scratch']).resolve(strict=True)
    conclusion = review.get('conclusion') or {}
    script = Path(review['script']).name
    stages = {'clt_first_record_discovery.py':'discovery_archived',
              'clt_first_record_predict.py':'prediction_archived',
              'clt_first_record_causal.py':'causal_archived'}
    if (script not in stages or review['archive_state']!='pending_upload' or review['returncode']!=0
            or prior.get('runId')!=review['server_run_id'] or conclusion.get('run_id')!=prior['runId']
            or not work.is_relative_to(storage_dir('scratch','routine').resolve())):
        raise ValueError('没有可以在原运行下继续的采集归档')
    marker = work/'archive-files.json'
    save(STATE,{**prior,'stage':'archive_retry','message':'正在恢复原运行的上传。'})
    receipt=client.submit(prior['runId'],marker,'completed',json.dumps(conclusion,ensure_ascii=False))
    save(record/'server-receipt.json',receipt)
    if receipt.get('archive_state')!='server_verified' or not receipt.get('readback_verified'):
        raise ValueError('恢复后中心回执尚未核验')
    review.update(archive_state='server_verified',status='finished',
                  cleanup=cleanup_scratch(marker,receipt))
    review.pop('archive_error',None)
    save(record/'review.json',review)
    updated={**prior,'stage':stages[script],'archiveState':'server_verified',
             'conclusion':conclusion,'message':'原运行已恢复归档并读回核验。'}
    save(STATE,updated)
    return updated


def record_terminal():
    """Append only server-verified terminal results, once per run."""
    status = read(STATE)
    if status.get('archiveState')!='server_verified' or status.get('stage') not in {'prediction_archived','causal_archived'}:
        return
    run_id = status['runId']
    receipt = read(Path(status['record'])/'server-receipt.json')
    if receipt.get('run_id')!=run_id or not receipt.get('readback_verified'):
        raise ValueError('研究备忘录需要该运行的真实服务器回执')
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    report=status['conclusion']
    pred=status.get('prediction') or report.get('prediction')
    readable=status.get('readout') or report.get('readout')
    causal=report.get('causal')
    site=f"B{report.get('block','?')}/H{report.get('head','?')}"
    if pred:
        sec=pred['scores']['selected']
        prediction_text=(f"五记录一致率 {sec['agreement']:.2%}，"
                         f"中心化记录对数质量 MSE {sec['source_log_mass']:.6g}，O 写入 MSE {sec['vo_write']:.6g}，"
                         f"MLP 单元 MSE {sec['mlp_units']:.6g}。预测门{'通过' if pred['gate_passed'] else '未通过'}。")
    else:
        prediction_text='新校准未达门槛，未开启确认集。'
    if causal:
        causal_text=(f"剂量匹配 {causal['dose_match']:.2%}，随机对照减目标阻断的 margin 差 "
                     f"{causal['margin_drop_over_random']:.6g}（区间 {causal['interval']}），"
                     f"sham 与完全恢复 {causal['exact_sham_and_restore']}，"
                     f"因果门{'通过' if causal['causal_gate_passed'] else '未通过'}。")
    else:
        causal_text='本次终端记录无原模型匹配干预证据。'
    readable_text=(f"原头在新首行题的五记录真实首选率为 {readable['2']:.2%}。"
                   if readable else '未建立新家族的可读出证据。')
    trace=f'**正式运行 ID** {run_id}'
    body=f"""
{trace}。中心回执已读回校验，客户端按证据类型显示。候选 {site}。此前 1,536 道已暴露题只供发现。

**方法。** 从原生 Q、K、V 投影输入各保留 512 列，在固定其他头和原残差的边界上用原始 O、同层 RMSNorm 及 SwiGLU 重算 128 个候选 MLP 单元。匹配干预采用原 pre-O 头输出的半幅阻断、同层等剂量随机控制、sham 与同输入恢复。读出、重建和干预是不同测量。

**可读出。** {readable_text}

**可预测。** {prediction_text}

**因果作用。** {causal_text}

**边界。** 同层路径预测以其他头和真实残差为条件。不能据此扩大到全模型解释、所有语境或新模型。原生坐标、逐题结果、失败对照、方案及读回回执在 {status['record']}。后续以跨家族条件选择及新组合检验为决策依据。

通俗地说，记录一个头实际指向哪条记录、在新题中提前算出同样的指向、把路径改动后模型的答案是否改变，是三项分别过门的任务。
"""
    for relative, content in [
        ('research/glm5/docs/AGI_GLM5_MEMO.md', body),
        ('research/claude/docs/AGI_CLAUDE_MEMO.md',f'\n{trace}。{site}。{prediction_text}{causal_text}\n'),
    ]:
        path = ROOT/relative
        prior = path.read_bytes()
        if trace.encode('utf-8') in prior:
            continue
        phase=1+max(map(int,re.findall(rb'^## Phase (\d+):',prior,re.M)))
        extra=f'\n\n## Phase {phase}: 首行并行路径正式归档结果 [{stamp}]\n{content}'.encode('utf-8')
        check_write(path,len(extra),operation='追加真实研究结果')
        with path.open('ab') as stream:
            stream.write(extra)
        if path.read_bytes()!=prior+extra:
            raise ValueError('备忘录前缀或追加记录不符')


def main():
    # The bearer is received once over a private pipe and stays in this process.
    session = sys.stdin.buffer.readline(4097).strip()
    if not re.fullmatch(rb'[-A-Za-z0-9_.~+/=]{16,4096}', session):
        save(STATE, {'stage': 'failed', 'message': '没有收到可用的客户端登录会话。'})
        return 1
    client = Client(session=session.decode('ascii'))
    prior = read(STATE) if STATE.is_file() else {}
    if prior.get('stage') in {'incomplete','archive_retry'} and prior.get('archiveState')=='pending_upload':
        try:
            prior = resume_archive(client,prior)
        except (ValueError, RuntimeError, OSError) as error:
            save(STATE,{**prior,'stage':'incomplete','message':str(error)})
            return 1
    if prior.get('stage') == 'discovery_archived':
        return continue_prediction(client,prior)
    if prior.get('stage') == 'prediction_archived':
        return continue_causal(client,prior) if prior['conclusion'].get('prediction',{}).get('gate_passed') else 0
    if prior.get('stage') == 'causal_archived':
        return 0
    code = execute(client)
    current = read(STATE)
    if code == 0 and current.get('stage') == 'discovery_archived':
        return continue_prediction(client,current)
    return code


if __name__ == '__main__':
    code = main()
    if code == 0:
        record_terminal()
    raise SystemExit(code)
