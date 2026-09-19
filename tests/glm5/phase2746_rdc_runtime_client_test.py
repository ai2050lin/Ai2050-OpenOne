"""Exact new committed fields and isolated authored-page browser regression."""
from rdc_construction_common import *
from phase2745_rdc_construction_client_test import API
import json,urllib.request,urllib.parse,urllib.error


def get(endpoint,expected=200,**parameters):
    try:
        with urllib.request.urlopen(API+endpoint+'?'+urllib.parse.urlencode(parameters),timeout=60) as response:
            assert response.status==expected;return json.load(response)
    except urllib.error.HTTPError as error:
        assert error.code==expected,(endpoint,error.code,expected,error.read().decode())
        return {'status':error.code}


def main():
    from playwright.sync_api import sync_playwright,expect
    out=BASE/'phase2746/client/runtime';out.mkdir(parents=True,exist_ok=True);start=time.monotonic();checks=[]
    for cohort,total in [('discovery',896),('confirmation',512)]:
        rows=get('/runtime-samples',cohort=cohort);assert len(rows)==total
        sid=next(r['sample_id'] for r in rows if r['kind']=='natural')
        record=get('/runtime',cohort=cohort,sample=sid)['record']
        with np.load(BASE/record['field_path']) as z:
            for mode,field,key,expected in [('units','product','units',unbits(z['units'][0,:,2])),
                ('coordinates','MLP_write','coordinates',unbits(z['coordinates'][0,:,4])),
                ('hidden','product','hidden',unbits(z['hidden'][0])),('query','product','Q_before_RoPE',unbits(z['Q_before_RoPE'][0,12]))]:
                actual=get('/runtime-field',cohort=cohort,sample=sid,mode=mode,field=field)
                assert np.array_equal(actual['values'],expected),(cohort,mode,key)
            get('/runtime-field',cohort=cohort,sample=sid,mode='units',step=8,expected=409)
            actual=get('/runtime-field',cohort=cohort,sample=sid,mode='hidden',step=31)
            assert np.array_equal(actual['values'],unbits(z['hidden'][31]))
        checks.append(cohort+': complete original axes and late-retention boundaries exactly match persisted arrays')
    get('/runtime-samples',cohort='invented',expected=422)
    get('/runtime',sample='missing',cohort='confirmation',expected=404)
    get('/self-history-field',sample=sid,route='teacher_future_patch',expected=422)
    records=get('/self-history',sample=sid)
    for route,r in records['records'].items():
        assert r.get('field_path'),'Selected first natural self-fed main row should already be committed'
        actual=get('/self-history-field',sample=sid,route=route,field='postnorm')
        with np.load(BASE/r['field_path']) as z:
            value=z['postnorm'] if route=='native_B1_cache' else z['compiled_postnorm']
            assert np.array_equal(actual['values'],unbits(value) if value.dtype==np.uint16 else value)
        get('/arrays',path=r['field_path']);get('/arrays',path=r['shared_initialization']['field_path'])
    checks.append('Each actual self-fed main trajectory and shared past-only initialization is accessible exactly, not pilot substitution')
    for receipt in ['history_prediction/test/heldout.json','history_prediction/deployment/coefficients.json',
        'history_prediction/confirmation/prediction/result.json','differential/main/start_12.json','runtime_reuse/commits/natural_ewt.json']:
        r=read(BASE/'phase2746'/receipt);assert get('/arrays',path=r['field_path'])
    checks.append('Every added precisely registered overflow category has a matching committed metadata route')
    linked=get('/mechanism-case',cohort='confirmation',sample=sid,step=0)
    assert linked['point_id']==sid+'_t0' and len(linked['frozen_current_state_forecasts'])==20
    assert len(linked['native_parameter_addresses'])==9 and linked['full_remaining_network_differentials']==[]
    fixture=read(BASE/'phase2746/differential/main/start_12.json')['records'][0]
    linked=get('/mechanism-case',cohort='discovery',sample=fixture['sample_id'],step=fixture['step'])
    assert len(linked['full_remaining_network_differentials'])==3
    for fig in read(BASE/'phase2746/figures/index.json')['figures']:
        url=API+'/figure?'+urllib.parse.urlencode({'stage':'2746','name':Path(fig['path']).stem})
        with urllib.request.urlopen(url,timeout=60) as response:assert hashlib.sha256(response.read()).hexdigest()==fig['sha256']
    checks.append('Same-expression parameter/state/differential/prediction identities and every registered scientific image verified')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1100},device_scale_factor=1);page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            panel=page.locator('#construction-runtime')
            panel.get_by_role('button',name='读取自身历史全场',exact=True).click()
            expect(panel.locator('canvas')).to_have_attribute('width','9728',timeout=60000)
            expect(panel.locator('canvas')).to_have_attribute('height','36')
            panel.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'all_units.png'))
            panel.get_by_label('运行材料',exact=True).select_option('confirmation')
            expect(panel.locator('canvas')).to_have_count(0)
            panel.get_by_label('运行表达',exact=True).select_option(sid)
            panel.get_by_label('运行场类型',exact=True).select_option('hidden')
            panel.get_by_role('button',name='读取自身历史全场',exact=True).click()
            expect(panel.locator('canvas')).to_have_attribute('height','37',timeout=60000)
            expect(panel.locator('canvas')).to_have_attribute('width','2560')
            panel.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'confirmation_hidden.png'))
            panel.get_by_label('运行场类型',exact=True).select_option('units');panel.get_by_label('真实生成步',exact=True).fill('8')
            panel.get_by_role('button',name='读取自身历史全场',exact=True).click()
            expect(panel.get_by_role('alert')).to_contain_text('not retained',timeout=60000);expect(panel.locator('canvas')).to_have_count(0)
            findings=page.locator('#construction-history-results');expect(findings.locator('table')).to_have_count(4)
            expect(findings.locator('table').nth(2).locator('tbody tr')).to_have_count(3)
            findings.get_by_role('heading',name='冻结规则：真实早期状态 + 可用过去 KV → 当前末端',exact=True).evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'history_confirmation.png'))
            findings.get_by_label('预测检验材料',exact=True).select_option('discovery')
            expect(findings.locator('table').nth(2)).to_contain_text('3.572634')
            own=page.locator('#construction-self-history');own.get_by_label('自身历史表达',exact=True).select_option(sid)
            expect(own.locator('table tbody tr')).to_have_count(3,timeout=60000)
            expect(own.locator('table')).to_contain_text('0.534375')
            own.get_by_role('button',name='读取三个自身历史分支',exact=True).click()
            expect(own.locator('article')).to_have_count(3,timeout=60000)
            own.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'self_history_outputs.png'))
            own.get_by_role('button',name='读取全部逐步坐标',exact=True).click()
            expect(own.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
            expect(own.locator('canvas')).to_have_attribute('height','32')
            own.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'self_history_field.png'))
            own.get_by_label('自身历史场',exact=True).select_option('Q35');expect(own.locator('canvas')).to_have_count(0)
            own.get_by_role('button',name='读取全部逐步坐标',exact=True).click()
            expect(own.locator('canvas')).to_have_attribute('width','4096',timeout=60000)
            expect(own.locator('canvas')).to_have_attribute('height','32')
            own.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'self_history_query.png'))
            checks+=['Complete9728unit and37boundary canvases with change-driven stale-data clearing',
                'Missing later MLP fields display explicit409, not zero or prior data',
                'New/old frozen prediction tables exactly reference committed cohorts',
                'Three real own histories plus all2560postnorm/all4096Q coordinates render']
            assert not errors,errors
        finally:context.close();browser.close()
    save(out/'result.json',{'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'checks':checks,'page_errors':errors,
        'seconds':time.monotonic()-start,'scope':'Isolated local authored-app regression. Raw screenshot inspection is a separate required check.'})
    print('RUNTIME_CLIENT_PASS',len(checks),flush=True)


if __name__=='__main__':main()
