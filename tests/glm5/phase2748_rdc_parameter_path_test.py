"""Independent native-checkpoint summand audit and isolated parameter-path UI."""
import json
import time
from pathlib import Path
import sys
import urllib.error
import numpy as np
from phase2748_rdc_client_test import ROOT,API,get,equal
from server import rdc_question_service as service
from rdc_relation_native_parameters import parameter,decode


def main():
    started=time.monotonic();checks=[];examples=[]
    folder=service.BASE/'client'/('parameter_path_'+str(time.time_ns()));folder.mkdir(parents=True)
    for model in ['qwen4','qwen14']:
        candidates=get('/samples',model=model,split='train')
        row=next(r for r in candidates if r['captured']);qid=row['question_id']
        material=get('/sample',model=model,question=qid)
        config=service.read(ROOT/'models/hf'/service.MODEL_FOLDERS[model]/'config.json')
        d,k=config['hidden_size'],config['intermediate_size']
        settings=[(block,unit,j)for block in [6,12,24,35]for unit,j in [(0,0),(k-1,d-1)]]if model=='qwen4'else[(12,0,0),(12,k-1,d-1)]
        for block,unit,j in settings:
            answer=get('/parameter-path',model=model,question=qid,block=block,unit=unit,output=j)
            pre=f'model.layers.{block}.mlp.'
            gate=decode(parameter(ROOT,pre+'gate_proj.weight',service.MODEL_FOLDERS[model])[unit]).astype(float)
            up=decode(parameter(ROOT,pre+'up_proj.weight',service.MODEL_FOLDERS[model])[unit]).astype(float)
            down=parameter(ROOT,pre+'down_proj.weight',service.MODEL_FOLDERS[model])
            column=decode(down[:,unit]).astype(float);outrow=decode(down[j]).astype(float)
            equal(answer['factors'],np.stack([gate,up,column]))
            iterms=[];wterms=[]
            for sibling,record in zip(material['siblings'],answer['records']):
                assert sibling['material']['question_id']==record['question_id']
                ref=sibling['record']['field'];path=ROOT/ref['path']
                with np.load(path,allow_pickle=False)as z:
                    x=decode(z[f'block{block}_MLP_input_BF16']).astype(float)
                    product=decode(z[f'block{block}_product_BF16']).astype(float)
                    actual=float(decode(z[f'block{block}_MLP_write_BF16'])[j])
                iterms.extend([gate*x,up*x]);wterms.append(outrow*product)
                assert record['all_unit_write_FP64_sum']==float(wterms[-1].sum())
                assert record['native_MLP_write_BF16_at_output']==actual
                assert record['selected_unit_write_term_FP64']==float(wterms[-1][unit])
                assert record['write_FP64_minus_native_BF16']==float(wterms[-1].sum()-actual)
            equal(answer['input_terms'],np.stack(iterms));equal(answer['write_terms'],np.stack(wterms))
            examples.append({'model':model,'question_id':qid,'block':block,'unit':unit,'output_coordinate':j,
                'width':d,'MLP_units':k,'all_four_question_records':answer['records']})
        checks.append(model+': all fixed edge-index settings exactly match independent original memmap parameter vectors and all4questions full summands')
        for params,status in [({'block':17,'unit':0,'output':0},409),({'block':12,'unit':k,'output':0},422),({'block':12,'unit':0,'output':d},422)]:
            try:get('/parameter-path',model=model,question=qid,**params);raise AssertionError('Expected rejection')
            except urllib.error.HTTPError as e:assert e.code==status
    sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
    from playwright.sync_api import sync_playwright,expect
    errors=[];screenshots=[]
    with sync_playwright()as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1080});page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            study=page.locator('#construction-natural-questions');history=study.get_by_test_id('natural-history-prediction')
            expect(history).to_contain_text('19.372',timeout=60000)
            history.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            path=folder/'native_history_failure.png';page.screenshot(path=str(path));screenshots.append(path)
            view=page.locator('#natural-parameter-path');view.get_by_role('button',name='读取自然问题的完整参数路径',exact=True).click()
            expect(view.locator('canvas')).to_have_count(2,timeout=60000)
            expect(view.locator('canvas').nth(0)).to_have_attribute('width','2560')
            expect(view.locator('canvas').nth(0)).to_have_attribute('height','8')
            expect(view.locator('canvas').nth(1)).to_have_attribute('width','9728')
            expect(view.locator('canvas').nth(1)).to_have_attribute('height','4')
            view.locator('h4').first.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            path=folder/'complete_native_parameter_path.png';page.screenshot(path=str(path));screenshots.append(path)
            view.get_by_label('路径 MLP 单元 k',exact=True).fill('9727');expect(view.locator('canvas')).to_have_count(0)
            view.get_by_role('button',name='读取自然问题的完整参数路径',exact=True).click()
            expect(view.locator('canvas')).to_have_count(2,timeout=60000)
            page.set_viewport_size({'width':390,'height':844})
            view.locator('h4').first.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            path=folder/'native_parameter_mobile.png';page.screenshot(path=str(path));screenshots.append(path)
            assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
            assert not errors,errors
        finally:context.close();browser.close()
    checks.append('New native-history failure table actually displayed; full input8x2560/write4x9728parameter terms; unit-change stale fields hidden;390pxmobile; no JS errors')
    result={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'all_passed':True,'checks':checks,'examples':examples,
        'screenshots':[p.relative_to(ROOT).as_posix()for p in screenshots],'visual_QA_pending':True,
        'scope':'Independent parameter/source summands and software QA, not BF16GPU replay or semantic identification.',
        'seconds':time.monotonic()-started}
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('NATURAL_PARAMETER_PATH_TEST_PASS',len(examples),str(folder),result['seconds'],flush=True)


if __name__=='__main__':main()
