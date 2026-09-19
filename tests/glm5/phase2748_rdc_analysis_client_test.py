"""Actual read-only analysis API/UI checks; pending is not scientific completion."""
import sys
import time
import urllib.error
from phase2748_rdc_client_test import ROOT, get
from rdc_question_common import OUT, read, sha, stamp, immutable


def main():
    start=time.monotonic();folder=OUT/'client'/('analysis_'+str(time.time_ns()))
    checks=[];responses={}
    for model in ['qwen4','qwen14','glm4']:
        for kind in (['readout','prospective','learning','parameter'] if model=='qwen4' else ['readout','prospective']):
            path={'readout':OUT/'readout_analysis'/model/'nonconfirmation/result.json',
                  'prospective':OUT/'prospective_analysis'/model/'diagnostic/result.json',
                  'learning':OUT/'learning_analysis/nonconfirmation/result.json',
                  'parameter':OUT/'parameter_formation/result.json'}[kind]
            response=get('/analysis',model=model,kind=kind)
            assert response['kind']==kind and response['model']==model and response['scope']=='nonconfirmation'
            assert response['complete']==path.is_file()
            if path.is_file():
                assert response['result']==read(path) and read(path)['all_passed']
                assert response['receipt']['sha256']==sha(path)
            else:
                assert 'result' not in response and 'not a result' in response['status']
            responses[(model,kind)]=response
            checks.append({'model':model,'kind':kind,'completed_result_observed':response['complete'],
                           'actual_source_or_pending_state_exact':True})
    for params,status in [({'kind':'unknown'},422),({'scope':'train'},422),
                          ({'kind':'learning','model':'qwen14'},422),({'kind':'parameter','model':'glm4'},422)]:
        try:get('/analysis',**params);raise AssertionError('Invalid query admitted')
        except urllib.error.HTTPError as e:assert e.code==status
    opened=get('/overview')['confirmation_open']
    if not opened:
        for kind in ['readout','prospective','learning','parameter']:
            try:get('/analysis',kind=kind,scope='confirmation');raise AssertionError('Sealed query admitted')
            except urllib.error.HTTPError as e:assert e.code==409
    sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
    from playwright.sync_api import sync_playwright,expect
    folder.mkdir(parents=True);images=[];errors=[]
    with sync_playwright()as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1080})
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            study=page.locator('#construction-natural-questions');panel=page.locator('#natural-complete-analyses')
            for kind in ['readout','prospective','learning','parameter']:
                panel.get_by_label('完整证据类型',exact=True).select_option(kind)
                response=responses[('qwen4',kind)]
                if not response['complete']:
                    expect(panel.get_by_role('status')).to_contain_text('待完成',timeout=60000)
                    expect(panel.locator('table')).to_have_count(0)
                else:
                    data=response['result'];expected={'readout':9,'prospective':2,'learning':6,'parameter':6}[kind]
                    expect(panel.locator('table').first.locator('tbody tr')).to_have_count(expected,timeout=60000)
                    panel.locator('summary').filter(has_text='完整分析、配对区间与来源身份').click()
                    expect(panel.locator('pre')).to_contain_text(response['receipt']['sha256'])
            panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            image=folder/'actual_analysis_status.png';page.screenshot(path=str(image));images.append(image)
            study.get_by_label('自然研究模型',exact=True).select_option('qwen14')
            expect(panel.get_by_role('status')).to_contain_text('当前模型没有这项训练证据')
            expect(panel.locator('table')).to_have_count(0)
            study.get_by_label('自然研究模型',exact=True).select_option('qwen4')
            panel.get_by_label('完整证据类型',exact=True).select_option('prospective')
            panel.get_by_label('完整证据范围',exact=True).select_option('confirmation')
            if not opened:
                expect(panel.get_by_role('status')).to_contain_text('确认集尚未解封')
                expect(panel.locator('table')).to_have_count(0)
            image=folder/'confirmation_scope.png';page.screenshot(path=str(image));images.append(image)
            page.set_viewport_size({'width':390,'height':844})
            panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            image=folder/'analysis_mobile.png';page.screenshot(path=str(image));images.append(image)
            assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1') and not errors,errors
        finally:context.close();browser.close()
    immutable(folder/'result.json',{'timestamp':stamp(),'all_passed':True,'checks':checks,
        'confirmation_open_at_test':opened,'complete_analyses_observed':sum(r['complete']for r in responses.values()),
        'pending_analyses_observed':sum(not r['complete']for r in responses.values()),
        'UI_scope':'Only actual existing results or actual pending states; no synthetic scientific results were inserted. Rerun after GPU/CPU analyses complete.',
        'source_files':{name:sha(ROOT/name)for name in ['server/rdc_question_service.py',
            'frontend/src/components/app/RdcNaturalAnalyses.jsx','frontend/src/components/app/RdcNaturalQuestions.jsx']},
        'screenshots':[p.relative_to(ROOT).as_posix()for p in images],'visual_QA_pending':True,'seconds':time.monotonic()-start})
    print('NATURAL_ANALYSIS_CLIENT_TEST_PASS',str(folder),round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':main()
