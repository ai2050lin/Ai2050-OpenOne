"""Independent raw-field geometry audit, all API matrix entries and isolated UI."""
import sys
from itertools import permutations
import urllib.error
from urllib.request import urlopen
from urllib.parse import urlencode
from phase2748_rdc_client_test import ROOT,API,get
from rdc_question_common import OUT,read,sha,unbits,stamp,immutable
import numpy as np
import time


def main():
    start=time.monotonic();folder=OUT/'client'/('relation_geometry_'+str(time.time_ns()))
    result=read(OUT/'relation_geometry/qwen4/result.json');ref=result['field'];assert sha(ROOT/ref['path'])==ref['sha256']
    with np.load(ROOT/ref['path'])as z:arrays={k:z[k].copy()for k in z.files}
    checks=[]
    for split in ['train','validation','diagnostic']:
        for cohort in ['drop','quoref']:
            for metric in ['similarity','permutation_mean','excess']:
                response=get('/relation-geometry',model='qwen4',split=split,cohort=cohort,metric=metric)
                assert response['labels']==result['labels']
                counts=arrays[f'{split}__{cohort}__valid_contexts'];values=arrays[f'{split}__{cohort}__{metric}_mean']
                assert response['valid_context_counts']==counts.tolist()
                for i,row in enumerate(response['values']):
                    for j,v in enumerate(row):assert (v is None)if counts[i,j]==0 else(v==values[i,j])
    checks.append('18full52x52API matrices exactly agree with retained source values and validcounts; zeroenergy is JSONnull')
    for split in ['train','validation','diagnostic']:
        for cohort in ['drop','quoref']:
            n=next(i for i,r in enumerate(result['identities'])if r['split']==split and r['cohort']==cohort)
            row=result['identities'][n];group=read(OUT/'native/qwen4/nonconfirmation/groups'/(row['group_id']+'.json'))
            lookup={q['question_id']:q for q in group['questions']};samples=[]
            for qid in row['question_ids']:
                field=lookup[qid]['field'];assert sha(ROOT/field['path'])==field['sha256']
                with np.load(ROOT/field['path'])as z:samples.append({k:unbits(z[k]).astype(float)for k in ['hidden_BF16','postnorm_BF16']})
            for left in ['H12','H24','H32','H36']:
                a=np.stack([s['hidden_BF16'][int(left[1:])]for s in samples]);a-=a.mean(0)
                b=np.stack([s['postnorm_BF16']for s in samples]);b-=b.mean(0)
                ka=a@a.T/a.shape[1];kb=b@b.T/b.shape[1]
                ii=result['labels'].index(left);jj=result['labels'].index('postnorm')
                assert np.array_equal(ka,arrays['full_coordinate_Gram_by_context'][n,ii])
                assert np.array_equal(kb,arrays['full_coordinate_Gram_by_context'][n,jj])
                actual=np.sum(ka*kb)/(np.linalg.norm(ka)*np.linalg.norm(kb))
                baseline=np.mean([np.sum(ka*kb[np.ix_(p,p)])/(np.linalg.norm(ka)*np.linalg.norm(kb))for p in permutations(range(4))])
                assert abs(actual-arrays['similarity'][n,ii,jj])<1e-13
                assert abs(baseline-arrays['permutation_mean'][n,ii,jj])<1e-13
    checks.append('24fixed layer pairs across6cohort/split contexts rederived from raw all-coordinate fields; explicit24permutationmeans match formula')
    display=read(OUT/'relation_geometry/qwen4/display.json')
    for f in display['figures']:
        with urlopen(API+'/relation-figure?'+urlencode({'model':'qwen4','name':f['name']}),timeout=60)as r:payload=r.read()
        import hashlib
        assert hashlib.sha256(payload).hexdigest()==f['sha256']
    for params,status in [({'model':'qwen14'},409),({'split':'confirmation'},422),({'metric':'maximum'},422)]:
        try:get('/relation-geometry',**params);raise AssertionError('Expected refusal')
        except urllib.error.HTTPError as e:assert e.code==status
    checks.append('Both QA-reviewed PNGs HTTPbytes exactly match SHA; incomplete model/undeclaredmetric/confirmation refused')
    sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
    from playwright.sync_api import sync_playwright,expect
    folder.mkdir(parents=True);images=[];errors=[]
    with sync_playwright()as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1080});page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            study=page.locator('#construction-natural-questions');panel=page.locator('#natural-relation-geometry')
            expect(panel).to_contain_text('0.143390',timeout=60000)
            expect(panel.get_by_test_id('relation-value')).to_contain_text('0.174207',timeout=60000)
            panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            image=folder/'relation_table.png';page.screenshot(path=str(image));images.append(image)
            panel.get_by_label('关系行空间',exact=True).select_option('H0')
            expect(panel.get_by_test_id('relation-value')).to_contain_text('未定义（零响应能量）')
            panel.get_by_label('关系行空间',exact=True).select_option('H12');panel.get_by_label('关系语料',exact=True).select_option('quoref')
            expect(panel.get_by_test_id('relation-value')).to_contain_text('0.112572',timeout=60000)
            panel.get_by_text('excess · 全部观察空间矩阵（已实际核图）',exact=True).click()
            picture=panel.get_by_alt_text('qwen4 全坐标关系矩阵 excess',exact=True)
            expect(picture).to_be_visible();picture.evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.wait_for_function('()=>document.querySelector("#natural-relation-geometry img[alt*=excess]").naturalWidth===4200')
            image=folder/'relation_matrix.png';page.screenshot(path=str(image));images.append(image)
            study.get_by_label('自然研究模型',exact=True).select_option('qwen14');expect(panel).to_have_count(0)
            study.get_by_label('自然研究模型',exact=True).select_option('qwen4');expect(panel).to_be_visible()
            page.set_viewport_size({'width':390,'height':844});panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            image=folder/'relation_mobile.png';page.screenshot(path=str(image));images.append(image)
            assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')and not errors,errors
        finally:context.close();browser.close()
    checks.append('Actual isolated UI shows correctedvalues; H0undefined; switching cohort/model hides stalevalues; allmatriximage loaded;390pxlayout no overflow orJSerrors')
    immutable(folder/'result.json',{'timestamp':stamp(),'all_passed':True,'checks':checks,'screenshots':[p.relative_to(ROOT).as_posix()for p in images],
        'seconds':time.monotonic()-start,'visual_QA_pending':True})
    print('NATURAL_RELATION_GEOMETRY_TEST_PASS',str(folder),round(time.monotonic()-start,2),flush=True)


if __name__=='__main__':main()
