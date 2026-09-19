"""Read-only API and isolated authored-app browser regression, never user profile."""
import argparse, json, sys, urllib.request, urllib.parse, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'glm5_temp/rdc_update_browser_packages'))
from rdc_construction_common import *

API = 'http://127.0.0.1:5004/api/rdc-construction'
OUT = BASE / 'client'


def get(route, expected=200, **parameters):
    url = API+route+'?'+urllib.parse.urlencode(parameters)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            assert r.status == expected
            return json.load(r)
    except urllib.error.HTTPError as e:
        assert e.code == expected, (route, e.code, expected, e.read().decode())
        return {'status': e.code}


def api(final=False):
    start, checks = time.monotonic(), []
    overview = get('/overview')
    assert overview['norm']['all_passed'] and overview['norm']['variants'] == 23
    checks.append('Committed23condition outcome analysis and explicit ongoing-status overview')
    material = gzread(BASE / 'material.json.gz')
    for key in ['qwen4', 'qwen14', 'glm4']:
        listed = get('/samples', model=key)
        assert len(listed) == 320
        if not overview['models'][key]['capture']:
            assert not final
            continue
        row = material['models'][key]['rows'][0]
        sid = row['sample_id']
        data = get('/field', model=key, sample=sid)
        with np.load(BASE/'capture'/key/'fields'/(sid+'.npz')) as z:
            truth = unbits(z['postnorm']).astype(float)
            assert np.array_equal(np.asarray(data['values']), truth)
            rms = get('/field', model=key, sample=sid, view='RMS')
            assert np.allclose(rms['values'], truth/np.sqrt(np.mean(truth*truth, -1, keepdims=True)), rtol=1e-14, atol=1e-14)
            q = get('/field', model=key, sample=sid, mode='q_before_rope', block=0, query=99)
            assert np.array_equal(q['values'], unbits(z['p99_L0_q_before_rope']))
            packet = get('/array', path=f'capture/{key}/fields/{sid}.npz', name='query_selected_states', row_start=2, row_count=3, start=17, count=7)
            native = unbits(z['query_selected_states']).reshape(-1, truth.shape[-1])
            assert np.array_equal(packet['values'], native[2:5, 17:24])
        pair = get('/pair', model=key, sample=sid, query=99)
        values = np.asarray(pair['field']['values'])
        assert np.array_equal(values[2], values[1]-values[0])
        checks.append(key+': full100query state, normalized state, actual head components,3Darchive exact C-order page and both-world identity')
        if overview['models'][key]['analysis']:
            interaction = get('/interaction', model=key, sample=sid)
            with np.load(BASE/'analysis'/key/'anova/boundary_postnorm.npz') as z:
                expected = truth-z['grand_mean'][0]-z['prefix_effect'][0, 0]-z['query_effect'][0]
                assert np.array_equal(interaction['interaction']['values'], expected)
            checks.append(key+': ANOVA interaction exactly reconstructed from original margins and native field')
        if overview['models'][key]['fit']:
            prediction = get('/prediction', model=key, sample=sid)
            assert len(prediction['values']) == 3 and prediction['source_split'] == row['split']
            pair_rows = sorted([r for r in material['models'][key]['rows'] if r['pair_id'] == row['pair_id']], key=lambda r:r['world'])
            xx, yy = [], []
            for paired in pair_rows:
                with np.load(BASE/'capture'/key/'fields'/(paired['sample_id']+'.npz')) as z:
                    xx.append(unbits(z['query_selected_states'][z['query_layer_indices'].tolist().index(1), 0]).astype(float))
                    yy.append(unbits(z['postnorm'][0]).astype(float))
            with np.load(BASE/'fit'/key/'operators/actual_query_H1__true_correspondence__postnorm.npz') as z:
                expected_prediction = (xx[1]-xx[0])@z['operator']
            assert np.allclose(prediction['values'][1], expected_prediction, rtol=1e-12, atol=1e-12)
            assert np.array_equal(prediction['values'][0], yy[1]-yy[0])
            checks.append(key+': frozen matrix prediction recomputed from original input and coefficients; full-coordinate truth checked')
        if final:
            assert overview['models'][key]['native_language']['all_passed']
            language = get('/native-language', model=key, sample=sid)
            assert language['record']['sample_id'] == sid
            if key != 'qwen4':
                state = get('/native-language-field', model=key, sample=sid)
                with np.load(BASE/'native_language'/key/'fields'/(sid+'.npz')) as z:
                    assert np.array_equal(state['values'], unbits(z['selected_hidden_states'][:, -1]))
            else:
                get('/native-language-field', expected=409, model=key, sample=sid)
            checks.append(key+': actual own-history output and retained step fields, with explicit Q4 reuse boundary')
    q4sid = material['models']['qwen4']['rows'][0]['sample_id']
    from rdc_construction_parameters import catalog,mlp_factors
    for key in ['qwen4','qwen14','glm4']:
        config=catalog(key)['config'];width=config['hidden_size'];unit=config['intermediate_size']-1
        parameters = get('/parameter', model=key, block=16, unit=unit, input=width-1, input_r=0, output=width-1)
        assert parameters['Gamma_k_j_i_r'] == parameters['W_gate_k_i']*parameters['W_up_k_r']*parameters['W_down_j_k']
        assert np.array_equal(parameters['factors']['values'],np.stack(mlp_factors(key,16,unit)))
        checks.append(key+': last native MLP unit and coordinates, all three exact vectors; native fused/separate storage respected')
    variants = get('/norm-variants')
    assert len(variants) == 23
    for v in variants:
        response = get('/norm-behavior', variant=v['variant']['name'], sample=q4sid)
        assert response['record']['sample_id'] == q4sid and response['record']['generated_ids']
    checks.append('All23parameter conditions own-history outputs reachable without inference or mutation')
    for route, params, code in [('/sample', {'model':'unknown','sample':q4sid},422),
        ('/sample', {'model':'qwen4','sample':'not-a-source'},404),
        ('/array', {'path':'../../AGENTS.md','name':'x'},404),
        ('/parameter', {'model':'qwen4','block':100},422),
        ('/interaction', {'model':'qwen4','sample':q4sid,'boundary':'../../x'},422),
        ('/field', {'model':'qwen4','sample':q4sid,'query':100},422),
        ('/norm-behavior', {'variant':'../../x','sample':q4sid},404)]:
        get(route, expected=code, **params)
    request = urllib.request.Request(API+'/overview', method='POST')
    try:
        urllib.request.urlopen(request, timeout=30)
        raise AssertionError('Write method unexpectedly enabled')
    except urllib.error.HTTPError as e:
        assert e.code == 405
    checks.append('UnknownIDs, path traversal, invalid original-axis addresses and write methods rejected')
    save(OUT/('api_final.json' if final else 'api_preliminary.json'), {'timestamp':stamp(),'source':snapshot(__file__),
        'all_passed':True,'final':final,'checks':checks,'seconds':time.monotonic()-start})
    print('CONSTRUCTION_API_PASS',len(checks),flush=True)


def browser(final=False):
    from playwright.sync_api import sync_playwright, expect
    import importlib.metadata
    start, errors, checks = time.monotonic(), [], []
    mode = 'final' if final else 'preliminary'
    with sync_playwright() as p:
      browser = p.chromium.launch(channel='msedge', headless=True, chromium_sandbox=True)
      context = browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1)
      page = context.new_page()
      page.on('pageerror',lambda e:errors.append(str(e)))
      try:
        page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
        page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
        expect(page.get_by_role('heading',name='语境怎样构造查询',exact=True)).to_be_visible()
        expect(page.locator('#construction-findings').get_by_text('76/160',exact=True)).to_be_visible()
        page.screenshot(path=str(OUT/f'headless_{mode}_desktop.png'))
        checks.append('New route, actual Q4 native counts and committed direction analysis rendered')
        source = page.locator('#construction-fields')
        source.get_by_role('button',name='读取原生场',exact=True).click()
        expect(source.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
        expect(source.locator('canvas')).to_have_attribute('height','100')
        source.locator('.prefix-field').first.scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/f'headless_{mode}_field.png'))
        source.get_by_label('数值视图',exact=True).select_option('RMS')
        expect(source.locator('canvas')).to_have_count(0)
        source.get_by_role('button',name='读取配对对照',exact=True).click()
        expect(source.locator('canvas')).to_have_attribute('height','3')
        source.get_by_label('数值视图',exact=True).select_option('raw')
        expect(source.locator('canvas')).to_have_count(0)
        source.get_by_role('button',name='读取条件交互',exact=True).click()
        expect(source.locator('canvas')).to_have_count(2)
        source.locator('.prefix-field').first.scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/f'headless_{mode}_interaction.png'))
        checks.append('All100x2560native coordinates, paired worlds and actual ANOVA with stale-view clearing')
        source.get_by_label('原生字段',exact=True).select_option('q_before_rope')
        expect(source.locator('canvas')).to_have_count(0)
        source.get_by_role('button',name='读取原生场',exact=True).click()
        expect(source.locator('canvas')).to_have_attribute('width','128')
        checks.append('Pre-RoPEQ displayed by native head components, not residual axes')
        source.get_by_label('原生字段',exact=True).select_option('queries')
        for key, width in [('qwen14','5120')]+([('glm4','4096')] if final else []):
            source.get_by_label('原生模型',exact=True).select_option(key)
            expect(source.locator('canvas')).to_have_count(0)
            expect(source.get_by_role('button',name='读取原生场',exact=True)).to_be_enabled(timeout=30000)
            source.get_by_role('button',name='读取原生场',exact=True).click()
            expect(source.locator('canvas')).to_have_attribute('width',width)
        checks.append('Separate original model widths and no stale Q4 fallback')
        if final:
            source.get_by_role('button',name='读取冻结关系预测',exact=True).click()
            expect(source.locator('canvas')).to_have_count(2)
            checks.append('Actual frozen full operator prediction is visible after its commit')
        params = page.locator('#construction-parameters')
        params.get_by_label('MLP 单元',exact=True).fill('9727')
        params.get_by_label('输入坐标 i',exact=True).fill('2559')
        params.get_by_label('输出坐标 j',exact=True).fill('2559')
        params.get_by_role('button',name='查询原生参数因子',exact=True).click()
        expect(params.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
        expect(params.locator('canvas')).to_have_attribute('height','3')
        params.locator('.prefix-field').scroll_into_view_if_needed()
        page.screenshot(path=str(OUT/f'headless_{mode}_parameters.png'))
        checks.append('Last native scalar addresses and full read/write-factor vectors')
        training = page.locator('#construction-training')
        expect(training.locator('table').first.locator('tbody tr')).to_have_count(23)
        training.get_by_role('button',name='读取真实生成对照',exact=True).click()
        expect(training.locator('blockquote')).to_have_count(3,timeout=30000)
        training.get_by_label('训练参数条件',exact=True).select_option('reverse_2743_r0p10')
        expect(training.locator('blockquote')).to_have_count(0)
        training.get_by_role('button',name='读取真实生成对照',exact=True).click()
        expect(training.locator('blockquote')).to_have_count(3)
        training.get_by_role('heading',name='逐条原生输出回查',exact=True).evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
        page.screenshot(path=str(OUT/f'headless_{mode}_generation.png'))
        checks.append('All23outcome rows and actual same-sample own-history reference/variant with stale-ID clearing')
        if final:
            language = page.locator('#construction-language')
            expect(language.locator('table tbody tr')).to_have_count(6)
            language.get_by_role('button',name='读取原模型完整输出',exact=True).click()
            expect(language.locator('blockquote')).to_have_count(2,timeout=30000)
            language.get_by_role('button',name='读取逐步全部坐标',exact=True).click()
            expect(language.locator('canvas')).to_have_attribute('width','4096')
            language.locator('.prefix-field').scroll_into_view_if_needed()
            page.screenshot(path=str(OUT/f'headless_{mode}_native_language.png'))
            language.get_by_label('生成模型',exact=True).select_option('qwen4')
            expect(language.locator('canvas')).to_have_count(0)
            expect(language.get_by_role('button',name='读取逐步全部坐标',exact=True)).to_be_disabled()
            checks.append('Native full answer distinct from first formatting token; real every-step field and Q4 missing-field boundary')
        archive = page.locator('#construction-archives')
        archive.get_by_role('button',name='读取原始索引窗口',exact=True).click()
        expect(archive.locator('canvas')).to_have_count(1,timeout=60000)
        checks.append('Original-array page loaded from lossless archive')
        page.set_viewport_size({'width':390,'height':844})
        page.evaluate('document.documentElement.style.scrollBehavior="auto";window.scrollTo(0,0)')
        page.locator('main').evaluate('(e)=>{e.style.scrollBehavior="auto";e.scrollTop=0}')
        page.screenshot(path=str(OUT/f'headless_{mode}_mobile.png'))
        layout = page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
        assert layout['scroll'] <= layout['width']+1, layout
        assert not errors, errors
        checks.append('390px mobile no document-level horizontal overflow; no page exceptions')
        save(OUT/f'browser_{mode}.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'final':final,
            'checks':checks,'page_errors':errors,'mobile_layout':layout,'browser':browser.version,
            'playwright':importlib.metadata.version('playwright'),'seconds':time.monotonic()-start,
            'mode':'Ephemeral headless Edge for authored-app regression, not attached to user browser/profile. Earlier app-control kernel initialization failures remain distinct.'})
        print('CONSTRUCTION_BROWSER_PASS',len(checks),flush=True)
      except Exception as exc:
        page.screenshot(path=str(OUT/f'headless_{mode}_failure.png'))
        failure(OUT/'browser_failure',start,exc)
        raise
      finally:
        context.close()
        browser.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--final',action='store_true')
    p.add_argument('--api-only',action='store_true')
    a = p.parse_args()
    api(a.final)
    if not a.api_only:
        browser(a.final)
