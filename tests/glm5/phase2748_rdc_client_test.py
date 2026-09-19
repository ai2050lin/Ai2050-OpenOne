"""Exact local API regression and isolated authored-app browser QA; no CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from server import rdc_question_service as service

OUT=service.BASE/'client'
API='http://127.0.0.1:5004/api/rdc-construction/questions'


def get(endpoint,**params):
    with urllib.request.urlopen(API+endpoint+'?'+urllib.parse.urlencode(params),timeout=60) as response:
        return json.load(response)


def equal(response,expected):
    actual=np.asarray(response['values'])
    assert actual.shape==expected.shape and np.array_equal(actual,expected),(actual.shape,expected.shape)
    assert response['native_width']==expected.shape[-1]
    assert response['start']==0 and response['end']==expected.shape[-1]


def main(browser_enabled):
    start=time.monotonic();checks=[];selected={}
    folder=OUT/('regression_'+str(time.time_ns()));folder.mkdir(parents=True)
    summary=get('/overview')
    assert summary['contexts']==400 and summary['questions']==1600
    assert not summary['confirmation_open'] # This qualification runs before first confirmation opening.
    for model in service.MODELS:
        train=get('/samples',model=model,split='train')
        diagnostic=get('/samples',model=model)
        assert len(train)==768 and len(diagnostic)==384
        row=next((r for r in diagnostic if r['captured']),None) or next((r for r in train if r['captured']),None)
        if row is None:
            checks.append(model+': metadata available; no formal committed native group claimed')
            continue
        qid=row['question_id'];sample=get('/sample',model=model,question=qid);q=sample['record']
        assert len(sample['siblings'])==4 and all(r['material']['group_id']==row['group_id'] for r in sample['siblings'])
        assert sample['tokens']['question_id']==qid
        expected=np.concatenate([service.array(q['field'],'hidden_BF16'),service.array(q['field'],'postnorm_BF16')[None]])
        equal(get('/field',model=model,question=qid),expected)
        for block in ([6,12,24,35] if model=='qwen4' else [12]):
            expected=np.stack([service.array(q['field'],f'block{block}_{n}_BF16') for n in ('gate','up','product')])
            equal(get('/field',model=model,question=qid,mode='MLP',block=block),expected)
        equal(get('/field',model=model,question=qid,mode='attention'),service.array(q['field'],'block12_attention_BF16'))
        full=service.array(sample['context_field'],'source_H12_BF16')
        for begin in [0,len(full)-1]:
            equal(get('/field',model=model,question=qid,mode='source_H12',start=begin),full[begin:begin+64])
        if q.get('history'):
            for mode,name in [('history_postnorm','postnorm_BF16'),('history_H12','H12_last_BF16'),('history_read','native_source_read_BF16')]:
                equal(get('/field',model=model,question=qid,mode=mode),service.array(q['history']['field'],name))
            equal(get('/field',model=model,question=qid,mode='teacher_postnorm'),service.array(q['teacher']['field'],'postnorm_BF16'))
        selected[model]=qid
        checks.append(model+': exact first-H, all selectedMLP, attention, first/lastsourcepages and retained histories match original full-coordinate arrays')
    # Full histories are chosen from predeclared IDs, not from favorable outcomes.
    full_example=None
    for row in get('/samples'):
        group,q=service.committed('qwen4',service.source(row['question_id']))
        if q.get('history',{}).get('full_H_all_layers_every_generated_step'):
            full_example=row['question_id'];h=q['history'];break
    assert full_example
    for step in (0,len(h['generated_ids'])-1):
        equal(get('/field',question=full_example,mode='history_all_hidden',step=step),service.array(h['field'],'all_hidden_BF16')[step])
    for cohort in ['drop','quoref']:
        for channel in ['hidden','MLP']:
            for statistic in ['within_RMS','train_standardized_RMS','mean']:
                a=get('/atlas',cohort=cohort,channel=channel,statistic=statistic)
                equal(a,np.asarray(service.atlas(model='qwen4',cohort=cohort,channel=channel,statistic=statistic,block=12)['values']))
    for figure in summary['models']['qwen4']['figures']:
        with urllib.request.urlopen(API+'/figure?'+urllib.parse.urlencode({'name':figure['name']})) as response:
            assert hashlib.sha256(response.read()).hexdigest()==figure['sha256']
    checks.append('Both full-H endpoints,12full-coordinate aggregate views,2actually reviewed PNG byte hashes verified')
    rejects=[('/samples',{'split':'confirmation'},409),('/samples',{'model':'../qwen4'},422),
        ('/sample',{'question':'../../bad'},404),('/field',{'question':selected['qwen4'],'mode':'missing'},422),
        ('/field',{'question':selected['qwen4'],'mode':'MLP','block':17},409),
        ('/field',{'question':selected['qwen4'],'mode':'source_H12','start':100000},422),
        ('/field',{'question':selected['qwen4'],'mode':'history_all_hidden'},409),
        ('/field',{'question':full_example,'mode':'history_all_hidden','step':127},422),
        ('/figure',{'name':'../../unknown'},404)]
    for route,params,status in rejects:
        try:get(route,**params);raise AssertionError(('Expected rejection',route,params))
        except urllib.error.HTTPError as e:assert e.code==status,(route,params,e.code,status)
    checks.append('Confirmation seal, unknown IDs/models, omitted axes, out-of-bounds steps/tokens and figure paths rejected')
    screenshots=[];errors=[]
    if browser_enabled:
        sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
        from playwright.sync_api import sync_playwright,expect
        with sync_playwright() as p:
            browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
            context=browser.new_context(viewport={'width':1440,'height':1080},device_scale_factor=1)
            page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
            try:
                page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
                page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
                section=page.locator('#construction-natural-questions');v=page.locator('#natural-question-fields');a=page.locator('#natural-question-aggregate')
                expect(v.get_by_label('自然问题',exact=True).locator('option')).to_have_count(384,timeout=60000)
                v.get_by_label('自然问题',exact=True).select_option(full_example)
                v.get_by_role('button',name='读取自然问题与完整坐标',exact=True).click()
                expect(v.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
                expect(v.locator('canvas')).to_have_attribute('height','38')
                v.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                path=folder/'native_first_all_H.png';page.screenshot(path=str(path));screenshots.append(path)
                v.get_by_label('自然场类型',exact=True).select_option('history_all_hidden')
                expect(v.locator('canvas')).to_have_count(0)
                v.get_by_label('自然生成步',exact=True).fill(str(len(h['generated_ids'])-1))
                v.get_by_role('button',name='读取自然问题与完整坐标',exact=True).click()
                expect(v.locator('canvas')).to_have_attribute('height','37',timeout=60000)
                a.get_by_role('button',name='读取自然语料全坐标图谱',exact=True).click()
                expect(a.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
                a.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                path=folder/'full_coordinate_standardized_atlas.png';page.screenshot(path=str(path));screenshots.append(path)
                a.get_by_label('图谱坐标空间',exact=True).select_option('MLP');expect(a.locator('canvas')).to_have_count(0)
                a.get_by_role('button',name='读取自然语料全坐标图谱',exact=True).click()
                expect(a.locator('canvas')).to_have_attribute('width','9728',timeout=60000)
                expect(a.locator('canvas')).to_have_attribute('height','3')
                section.get_by_label('自然研究模型',exact=True).select_option('glm4')
                expect(v.locator('canvas')).to_have_count(0);expect(a.locator('canvas')).to_have_count(0)
                expect(v.get_by_label('自然问题',exact=True).locator('option')).to_have_count(384,timeout=60000)
                expect(v.get_by_role('button',name='读取自然问题与完整坐标',exact=True)).to_be_disabled()
                section.get_by_label('自然研究模型',exact=True).select_option('qwen4')
                v.get_by_label('自然材料划分',exact=True).select_option('confirmation')
                expect(v.get_by_role('alert')).to_contain_text('Confirmation remains sealed',timeout=60000)
                expect(v.get_by_label('自然问题',exact=True).locator('option')).to_have_count(0)
                page.set_viewport_size({'width':390,'height':844})
                section.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                path=folder/'natural_mobile_progress.png';page.screenshot(path=str(path));screenshots.append(path)
                assert section.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
                assert not errors,errors
                checks.append('Isolated authored-app browser: full native/last-step/all9728MLP canvases; stale fields hidden; absent model not claimed; confirmation rejects;390pxmobile; no JS errors')
            finally:context.close();browser.close()
    result={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'all_passed':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks':checks,'selected':selected,'full_H_example':full_example,'browser_executed':browser_enabled,
        'screenshots':[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in screenshots],
        'visual_review_pending':bool(screenshots),'seconds':time.monotonic()-start}
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('NATURAL_CLIENT_TEST_PASS',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--browser',action='store_true');main(parser.parse_args().browser)
