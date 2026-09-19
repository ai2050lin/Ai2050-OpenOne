"""Read-only API and isolated authored-app regression; no user browser control."""
import urllib.request, urllib.parse, urllib.error
from rdc_formation_common import *
import sys
sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
from playwright.sync_api import sync_playwright, expect

API='http://127.0.0.1:5004/api/rdc-construction'


def get(route, **params):
    with urllib.request.urlopen(API+route+'?'+urllib.parse.urlencode(params), timeout=60) as response:
        import json
        return json.load(response)


def main():
    start=time.monotonic()
    out=OUT/'client'/('regression_'+str(time.time_ns()))
    out.mkdir(parents=True,exist_ok=True)
    summary=get('/formation-progress')
    assert len(summary['runs'])==6 and summary['material']['counts']['train']==2048
    run=next(r for r in summary['runs'] if r['latest_committed_step']>=1)['run']
    samples=get('/formation-samples')
    assert len(samples)==44
    sid=samples[0]['sample_id']
    checks=[]
    assert len(summary.get('tasks',[]))==7 and summary['tasks'][0]['complete']
    assert sum(r['complete'] for r in summary['runs'])==6
    checks.append('All6training runs and7integrated task states reflect real committed artifacts, not planned completion')
    gradient=read(OUT/'gradient/commits/true_token.json')
    gradient_path=(ROOT/gradient['field_path']).relative_to(BASE).as_posix()
    assert get('/arrays',path=gradient_path)
    checks.append('Enlarged complete-parameter gradient archive is accessible through exact receipts')
    for mode,expected in [('hidden',(37,2560)),('gate',(36,9728)),('Q',(36,4096))]:
        value=get('/formation-field',sample=sid,run=run,checkpoint='1',field=mode)
        actual=np.array(value['values'])
        assert actual.shape==expected
        receipt=read(OUT/'training'/run/'commits/fields_001.json')
        with np.load(ROOT/receipt['field_path']) as z:
            reference=z[mode][0].reshape(expected)
            assert np.array_equal(actual,reference)
        archive=value['archive']
        assert get('/arrays',path=archive)
    checks.append('Actual complete37x2560H,36x9728gate and36x4096Q values equal committed original arrays')
    receipt=read(OUT/'training'/run/'commits/delta_001.json')
    path=(ROOT/receipt['field_path']).relative_to(BASE).as_posix()
    headers=get('/arrays',path=path)
    assert len(headers['arrays'] if isinstance(headers,dict) and 'arrays' in headers else headers)==6
    checks.append('Full delta and lossless reconstruction residual arrays accessible through exact registered category/commit')
    for route,params,expected in [('/formation-field',{'sample':'nonexistent'},404),
        ('/formation-field',{'sample':sid,'run':'invented_run'},422),
        ('/arrays',{'path':'phase2747/field_store/../training/native.npz'},404),
        ('/arrays',{'path':'phase2747/field_store/unregistered/file.npz'},404)]:
        try:
            get(route,**params)
            raise AssertionError('Unexpected request success')
        except urllib.error.HTTPError as e:
            assert e.code==expected,(route,e.code)
    checks.append('Unknown IDs, runs, traversal and undeclared categories reject without substituted data')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            section=page.locator('#construction-formation')
            expect(section.locator('table tbody tr')).to_have_count(6,timeout=60000)
            section.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'progress_desktop.png'))
            section.get_by_label('形成参数条件',exact=True).select_option(run)
            section.get_by_role('button',name='读取形成全坐标场',exact=True).click()
            expect(section.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
            expect(section.locator('canvas')).to_have_attribute('height','37')
            section.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'complete_hidden.png'))
            section.get_by_label('形成原生场',exact=True).select_option('gate')
            expect(section.locator('canvas')).to_have_count(0)
            section.get_by_role('button',name='读取形成全坐标场',exact=True).click()
            expect(section.locator('canvas')).to_have_attribute('width','9728',timeout=60000)
            expect(section.locator('canvas')).to_have_attribute('height','36')
            section.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'complete_gate.png'))
            section.get_by_label('形成原生场',exact=True).select_option('Q')
            expect(section.locator('canvas')).to_have_count(0)
            section.get_by_role('button',name='读取形成全坐标场',exact=True).click()
            expect(section.locator('canvas')).to_have_attribute('width','4096',timeout=60000)
            section.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'complete_query.png'))
            page.set_viewport_size({'width':390,'height':844})
            section.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'progress_mobile.png'))
            assert not errors,errors
            checks+=['Six actual run states rendered; changing axes clears stale canvas',
                'Complete native H/unit/Q axes render without browser exceptions; desktop/mobile screenshots saved']
        finally:
            context.close()
            browser.close()
    result={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'checks':checks,
        'observed_run':run,'sample':sid,'page_errors':errors,'seconds':time.monotonic()-start,
        'automation_scope':'Isolated software test of authored localhost app, not Windows/user browser automation.',
        'visual_review':'Pending actual screenshot inspection; passing assertions alone are not visual QA.'}
    save(out/'result.json',result)
    save(OUT/'client/current_regression.json',{'timestamp':stamp(),'result':str((out/'result.json').relative_to(ROOT)),
        'result_sha256':sha(out/'result.json'),'image_directory':str(out.relative_to(ROOT))})
    print('FORMATION_CLIENT_PASS',len(checks),flush=True)


if __name__=='__main__':
    main()
