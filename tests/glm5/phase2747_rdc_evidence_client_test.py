"""Exact coordinate evidence routes and isolated authored-UI regression."""
from phase2747_rdc_client_test import *
from rdc_formation_readout import checked_arrays


def get(endpoint, /, **params):
    with urllib.request.urlopen(API+endpoint+'?'+urllib.parse.urlencode(params), timeout=60) as response:
        import json
        return json.load(response)


def main():
    start=time.monotonic(); out=OUT/'client'/('evidence_regression_'+str(time.time_ns()));out.mkdir(parents=True)
    checks=[]; summary=get('/formation-progress'); figures=summary['figures']
    history_review=OUT/'figures/history_v1/visual_review.json'
    expected_figures=17 if history_review.exists() and read(history_review)['all_passed'] else 15
    assert len(figures)==expected_figures
    for item in figures:
        with urllib.request.urlopen(API+'/figure?'+urllib.parse.urlencode({'stage':'2747','name':item['name']})) as response:
            assert hashlib.sha256(response.read()).hexdigest()==item['sha256']
    checks.append(f'All{expected_figures}visually reviewed formation/propagation/probability/history figures served from exact registered bytes')
    samples=get('/formation-propagation-samples');assert len(samples)==64;sid=samples[0]['sample_id']
    result=read(OUT/'parameter_propagation/smooth/result.json')
    refs={field:[] for field in ['hidden','Q','gate']}
    for receipt in result['complete_field_receipts']:
        file=ROOT/receipt['field_path'];assert sha(file)==receipt['field_sha256']
        with np.load(file) as z:
            for field in refs:refs[field].append(z['tangent_'+field][0,:2].astype(float))
    for field,values in refs.items():
        for route in ['full_prefix','last_position_only','difference']:
            answer=get('/formation-propagation-field',sample=sid,run='true_token_2747',field=field,route=route)
            expected=np.stack([v[0]-v[1] if route=='difference' else v[int(route=='last_position_only')] for v in values])
            assert np.array_equal(np.array(answer['values']),expected), (field,route,np.array(answer['values']).shape,expected.shape)
    checks.append('All20layers exact H2560/Q4096/gate9728 for both original tangent routes and their difference')
    own=get('/formation-own-samples',model='qwen4',variant='native');assert len(own)==512
    full=next(r['sample_id'] for r in own if r['full_hidden_collected'] and r['kind']=='natural')
    unavailable=next(r['sample_id'] for r in own if not r['full_hidden_collected'])
    record=read(OUT/'own_history/qwen4/native/records'/(full+'.json'));arrays=checked_arrays(record['field'])
    a=get('/formation-own-field',sample=full,field='postnorm')
    b=get('/formation-own-field',sample=full,field='all_hidden',step=17)
    assert np.array_equal(np.array(a['values']),unbits(arrays['postnorm_BF16']))
    assert np.array_equal(np.array(b['values']),unbits(arrays['all_hidden_BF16'][17]))
    checks.append('Actual all96ownstep postnorm and all37H at step17 match native committed BF16 values')
    for params,status in [({'sample':unavailable,'field':'all_hidden'},409),({'sample':full,'field':'all_hidden','step':200},422),({'sample':'missing'},404),({'model':'glm4','variant':'true_token_2747'},422)]:
        try:get('/formation-own-field',**params);raise AssertionError('Expected rejection')
        except urllib.error.HTTPError as e:assert e.code==status
    checks.append('Uncollected allH, out-of-range step, unknownsample and unsupported modeldeployment reject honestly')
    for receipt in [read(OUT/'figures/commits/parameter_history_complete_coordinates.json'),read(OUT/'figures/commits/training_full_coordinate_changes.json')]:
        assert get('/arrays',path=(ROOT/receipt['field_path']).relative_to(BASE).as_posix())
    checks.append('Both complete numeric scientific figure archives accessible without unregistered paths')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1100},device_scale_factor=1);page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            evidence=page.locator('#formation-evidence');prop=page.locator('#formation-propagation-view');ownview=page.locator('#formation-own-view')
            old_image=evidence.locator('figure img').element_handle()
            evidence.get_by_label('形成研究图',exact=True).select_option('propagation_precision_and_zero_baseline')
            assert old_image.evaluate('(e)=>!e.isConnected'), 'Old bitmap node retained after figure identity change'
            expect(evidence.locator('figure img')).to_be_visible()
            expect(evidence.locator('figure img')).to_have_attribute('alt','propagation_precision_and_zero_baseline')
            evidence.locator('figure img').evaluate('async (e)=>{await e.decode();if(!e.complete||e.naturalWidth===0)throw new Error("Selected figure not decoded");}')
            evidence.locator('figure img').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'evidence_figure.png'))
            prop.get_by_label('传播表达',exact=True).select_option(sid)
            prop.get_by_role('button',name='读取完整传播场',exact=True).click()
            expect(prop.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
            expect(prop.locator('canvas')).to_have_attribute('height','20')
            prop.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'propagation_hidden.png'))
            prop.get_by_label('传播原生场',exact=True).select_option('gate');expect(prop.locator('canvas')).to_have_count(0)
            prop.get_by_role('button',name='读取完整传播场',exact=True).click()
            expect(prop.locator('canvas')).to_have_attribute('width','9728',timeout=60000)
            prop.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'propagation_gate.png'))
            ownview.get_by_label('自身历史表达',exact=True).select_option(full)
            ownview.get_by_role('button',name='读取真实自身历史',exact=True).click()
            expect(ownview.locator('canvas')).to_have_attribute('height','96',timeout=60000)
            ownview.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'own_postnorm.png'))
            ownview.get_by_label('自身历史场',exact=True).select_option('all_hidden');expect(ownview.locator('canvas')).to_have_count(0)
            ownview.get_by_label('自身生成步',exact=True).fill('17')
            ownview.get_by_role('button',name='读取真实自身历史',exact=True).click()
            expect(ownview.locator('canvas')).to_have_attribute('height','37',timeout=60000)
            ownview.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'own_all_hidden.png'))
            page.set_viewport_size({'width':390,'height':844})
            ownview.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'own_mobile.png'))
            assert ownview.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1'), 'Own-history controls overflow the mobile panel'
            assert not errors,errors
            checks.append('Five actual figure/field desktop views and own-history mobile view rendered without page errors; axis changes clear stale arrays')
        finally:context.close();browser.close()
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,
        'sample':full,'propagation_sample':sid,'images':[p.name for p in sorted(out.glob('*.png'))],
        'seconds':time.monotonic()-start,'visual_review':'Pending actual main-agent viewing of six new screenshots.'})
    save(OUT/'client/current_evidence_regression.json',{'timestamp':stamp(),'result':str((out/'result.json').relative_to(ROOT)),
        'result_sha256':sha(out/'result.json'),'image_directory':str(out.relative_to(ROOT))})
    print('FORMATION_EVIDENCE_CLIENT_PASS',len(checks),out,flush=True)


if __name__=='__main__':main()
