"""Real zero-update precision receipt/UI checks, never synthetic app data."""
import argparse
import sys
import time
from urllib.error import HTTPError
import numpy as np
from phase2748_rdc_client_test import get, ROOT
from phase2748_rdc_source_coupling_client_test import cells
from rdc_question_common import OUT, read, sha, stamp, immutable, unbits


def raw_audit(result):
    """Independently check all192questions at both0and24trained checkpoints."""
    ref=result['field'];assert sha(ROOT/ref['path'])==ref['sha256']
    with np.load(ROOT/ref['path']) as z:saved={k:z[k].copy() for k in z.files}
    identities=result['identities'];assert len(identities)==192
    panels={}
    for ref in result['validation_sources']:
        assert sha(ROOT/ref['path'])==ref['sha256']
        r=read(ROOT/ref['path']);records={x['question_id']:x for x in r['records']}
        assert len(records)==192 and set(records)=={x['question_id'] for x in identities}
        packets=[]
        for row in identities:
            field=records[row['question_id']]['field'];assert sha(ROOT/field['path'])==field['sha256']
            with np.load(ROOT/field['path']) as z:
                nll=z['NLL'].copy();ids=z['teacher_ids'].copy();post=unbits(z['postnorm_BF16']).astype(float)
            packets.append((ids,np.array([np.mean(nll),nll[0],np.mean(nll[1:])]),post))
        panels[(r['run'],r['step'])]=packets
    native=panels[('untrained_original_BF16_reference',0)]
    bridge=panels[('untrained_FP32_bridge_reference',0)]
    np.testing.assert_array_equal(saved['native_NLL_metrics'],np.stack([x[1] for x in native]))
    np.testing.assert_array_equal(saved['untrained_bridge_NLL_metrics'],np.stack([x[1] for x in bridge]))
    np.testing.assert_array_equal(saved['bridge_minus_native_teacher_state_MSE'],[np.mean((a[2]-b[2])**2) for a,b in zip(native,bridge)])
    for row in result['checkpoints']:
        packets=panels[(row['run'],row['step'])];prefix=row['run']+'__step'+str(row['step'])
        np.testing.assert_array_equal(saved[prefix+'__NLL_metrics'],np.stack([x[1] for x in packets]))
        for label,base in [('native',native),('untrained_bridge',bridge)]:
            assert all(np.array_equal(a[0],b[0]) for a,b in zip(packets,base))
            np.testing.assert_array_equal(saved[prefix+'__teacher_MSE_vs_'+label],[np.mean((a[2]-b[2])**2) for a,b in zip(packets,base)])
    assert len(panels)==26 and len(result['checkpoints'])==24
    return {'actual_teacher_packets_independently_recomputed':26*192,'trained_checkpoints':24,'questions_per_checkpoint':192}


def table_rows(data,cohort):
    b=data['baseline_summary']
    baseline=[['原生 BF16',*b['native'][cohort],0.],['零更新 FP32 桥',*b['untrained_bridge'][cohort],b['teacher_state_MSE'][cohort]]]
    checkpoints=[]
    for r in data['checkpoints']:
        p=r['paired_vs_untrained_bridge']['answer_mean_NLL'][cohort]
        checkpoints.append([r['run'],str(r['step']),*r['summary'][cohort],p['mean_left_minus_right'],
            *p['paired_context_bootstrap_95_percent_interval'],r['teacher_state_MSE_vs_untrained_bridge'][cohort]])
    return baseline,checkpoints


def main(require_complete):
    start=time.monotonic();source=OUT/'training_bridge_baseline/result.json'
    response=get('/analysis',model='qwen4',kind='training_bridge',scope='nonconfirmation')
    assert response['complete']==source.exists()
    if require_complete:assert response['complete'],'Actual zero-update result not completed'
    numerical={};compared=0
    if source.exists():
        assert response['result']==read(source) and response['receipt']['sha256']==sha(source)
        numerical=raw_audit(response['result'])
    else:assert 'result' not in response
    for model in ['qwen14','glm4']:
        try:get('/analysis',model=model,kind='training_bridge')
        except HTTPError as exc:assert exc.code==422
        else:raise AssertionError('Unregistered training model admitted')
    try:get('/analysis',model='qwen4',kind='training_bridge',scope='confirmation')
    except HTTPError as exc:assert exc.code in [409,422]
    else:raise AssertionError('Validation-only diagnostic admitted as confirmation')
    folder=OUT/'client'/('training_bridge_'+str(time.time_ns()));folder.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
    from playwright.sync_api import sync_playwright,expect
    images=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1080});page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            study=page.locator('#construction-natural-questions');panel=page.locator('#natural-complete-analyses')
            panel.get_by_label('完整证据类型',exact=True).select_option('training_bridge')
            if response['complete']:
                expect(panel.locator('pre')).to_contain_text(response['receipt']['sha256'],timeout=60000)
                for cohort in ['equal_cohort','drop','quoref']:
                    panel.get_by_label('完整证据语料',exact=True).select_option(cohort)
                    baseline,checkpoints=table_rows(response['result'],cohort)
                    compared+=cells(page,'training-bridge-baselines',baseline)+cells(page,'training-bridge-checkpoints',checkpoints)
                panel.get_by_label('完整证据语料',exact=True).select_option('equal_cohort')
                baseline,checkpoints=table_rows(response['result'],'equal_cohort')
                cells(page,'training-bridge-baselines',baseline);cells(page,'training-bridge-checkpoints',checkpoints)
            else:
                expect(panel.get_by_role('status')).to_contain_text('待完成',timeout=60000);expect(panel.locator('table')).to_have_count(0)
            panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            path=folder/'training_bridge_desktop.png';page.screenshot(path=str(path));images.append(path)
            for model in ['qwen14','glm4']:
                study.get_by_label('自然研究模型',exact=True).select_option(model)
                expect(panel.get_by_role('status')).to_contain_text('当前模型没有这项训练证据');expect(panel.locator('table')).to_have_count(0)
            study.get_by_label('自然研究模型',exact=True).select_option('qwen4')
            panel.get_by_label('完整证据范围',exact=True).select_option('confirmation')
            expect(panel.get_by_role('status')).to_contain_text('零更新桥仅覆盖原验证集');expect(panel.locator('table')).to_have_count(0)
            panel.get_by_label('完整证据范围',exact=True).select_option('nonconfirmation')
            if response['complete']:
                expect(panel.locator('pre')).to_contain_text(response['receipt']['sha256'],timeout=60000)
                cells(page,'training-bridge-baselines',baseline);cells(page,'training-bridge-checkpoints',checkpoints)
            else:expect(panel.get_by_role('status')).to_contain_text('待完成',timeout=60000)
            page.set_viewport_size({'width':390,'height':844});panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            path=folder/'training_bridge_mobile.png';page.screenshot(path=str(path));images.append(path)
            assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1') and not errors,errors
        finally:context.close();browser.close()
    result={'timestamp':stamp(),'all_passed':True,'actual_result_complete':response['complete'],
        'independent_raw_audit':numerical,'displayed_cells_compared_to_actual_sources':compared,
        'source_result_sha256':sha(source) if source.exists() else None,'source_sha256':sha(__file__),
        'UI_sources':{name:sha(ROOT/name) for name in ['server/rdc_question_service.py','frontend/src/components/app/RdcNaturalAnalyses.jsx']},
        'screenshots':[p.relative_to(ROOT).as_posix() for p in images],'visual_QA_pending':True,'seconds':time.monotonic()-start,
        'scope':'Actual completed values or explicitly pending empty state only. No synthetic results inserted, no confirmation results queried.'}
    immutable(folder/'result.json',result);print('NATURAL_TRAINING_BRIDGE_CLIENT_PASS',str(folder),response['complete'],compared,round(result['seconds'],2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-complete',action='store_true');main(p.parse_args().require_complete)
