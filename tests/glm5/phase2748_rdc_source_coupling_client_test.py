"""Real source-coupling API/UI and independent native-field contractions."""
import argparse
import sys
import time
from urllib.error import HTTPError
import numpy as np
from phase2748_rdc_client_test import get, equal, ROOT
from rdc_question_common import OUT, read, sha, stamp, immutable, material, unbits

HEADS = ['passage_mass_fraction','question_mass_fraction','other_mass_fraction',
    'passage_conditional_effective_source_fraction','passage_coupling_L2','passage_permutation_delta_L2',
    'Cauchy_Schwarz_permutation_bound','FP64_read_minus_native_BF16_preO_L2']
COORDINATES = ['permuted_minus_native_read_mean_square','native_read_mean','permuted_read_mean']


def raw_check(model, result, saved):
    _, _, rows, _ = material(model); rr = {r['question_id']: r for r in rows}; checks = []
    for split in ['train','validation','diagnostic']:
        for cohort in ['drop','quoref']:
            selected = next(x for x in result['identities'] if x['split'] == split and x['cohort'] == cohort)
            group = read(OUT/'native'/model/'nonconfirmation/groups'/(selected['group_id']+'.json'))
            ref = group['context_field']; assert sha(ROOT/ref['path']) == ref['sha256']
            with np.load(ROOT/ref['path']) as z: prefix = unbits(z['block12_prefix_values_BF16']).astype(float)
            for q in group['questions']:
                qid = q['question_id']; i = next(i for i,r in enumerate(result['identities']) if r['question_id'] == qid)
                ref = q['field']; assert sha(ROOT/ref['path']) == ref['sha256']
                with np.load(ROOT/ref['path']) as z:
                    fields = {k: unbits(z[k]).astype(float) if z[k].dtype == np.uint16 else z[k].copy() for k in [
                        'block12_attention_BF16','block12_appended_values_BF16','source_value_permutation',
                        'block12_pre_O_BF16','native_source_read_BF16','source_value_pair_shuffle_BF16']}
                a = fields['block12_attention_BF16']; values = np.concatenate([prefix, fields['block12_appended_values_BF16']], axis=1)
                c = rr[qid]['tokens']['context_token_positions']; question = rr[qid]['tokens']['question_token_positions']
                order = fields['source_value_permutation']; hcount = len(a); group_size = hcount//len(values)
                original_pre = fields['block12_pre_O_BF16'].reshape(hcount,-1)
                expected = []
                for h in range(hcount):
                    v = values[h//group_size]; weights = a[h,c]; vv = v[c]; m = weights.sum()
                    center_weights = weights-m/len(c); centered = vv-vv.mean(0)
                    coupled = np.sum(center_weights[:,None]*centered,axis=0)
                    delta = np.sum(weights[:,None]*(v[order[c]]-vv),axis=0)
                    weighted = np.sum(weights[:,None]*vv,axis=0); uniform = m*vv.mean(0)
                    full = np.sum(a[h,:,None]*v,axis=0); other = sorted(set(range(len(v)))-set(c)-set(question))
                    total = a[h].sum(); effective = m*m/(np.sum(weights*weights)*len(c)) if np.any(weights) else 0.
                    expected.append([total,m/total,a[h,question].sum()/total,a[h,other].sum()/total,
                        np.linalg.norm(center_weights),effective,np.linalg.norm(weighted),np.linalg.norm(uniform),np.linalg.norm(coupled),
                        np.linalg.norm(delta),np.linalg.norm(center_weights)*np.linalg.norm(v[order[c]]-vv),np.linalg.norm(full),
                        np.linalg.norm(full-original_pre[h]),np.linalg.norm(vv.mean(0)),np.linalg.norm(centered)])
                np.testing.assert_allclose(expected,saved['per_question_all_heads'][i],atol=1e-12,rtol=1e-12)
                r=fields['native_source_read_BF16']; s=fields['source_value_pair_shuffle_BF16']; d=s-r
                expected_post=[np.mean(r*r),np.mean(s*s),np.mean(d*d),np.sum(d*d)/np.sum(r*r),np.max(np.abs(d))]
                np.testing.assert_array_equal(expected_post,saved['per_question_postO_statistics'][i])
                checks.append({'model':model,'split':split,'cohort':cohort,'question_id':qid,'all_native_heads':hcount})
    return checks


def cells(page, test_id, rows):
    expected=page.evaluate("rows=>rows.map(r=>r.map(v=>typeof v==='string'?v:v==null?'未定义':Number(v).toPrecision(6)))",rows)
    page.wait_for_function("""({testid,expected})=>{
      const t=document.querySelector('[data-testid="'+testid+'"]');
      if(!t)return false;
      return JSON.stringify(Array.from(t.querySelectorAll('tbody tr'),r=>Array.from(r.querySelectorAll('td'),c=>c.textContent.trim())))===JSON.stringify(expected);
    }""",arg={'testid':test_id,'expected':expected},timeout=60000)
    return sum(len(r) for r in rows)


def main(models):
    start=time.monotonic(); responses={}; api_checks=[]; raw_checks=[]
    for model in models:
        source=read(OUT/'source_coupling'/model/'result.json'); assert source['all_passed']
        relative=read(OUT/'source_coupling'/model/'relative_strength.json'); assert relative['all_passed']
        ref=source['field']; assert sha(ROOT/ref['path'])==ref['sha256']
        with np.load(ROOT/ref['path']) as z: saved={k:z[k].copy() for k in z.files}
        raw_checks.extend(raw_check(model,source,saved))
        for split in ['train','validation','diagnostic']:
            for cohort in ['drop','quoref']:
                prefix=split+'__'+cohort
                for coordinate in COORDINATES:
                    v=get('/source-coupling',model=model,split=split,cohort=cohort,coordinate=coordinate)
                    assert v['complete'] and v['summary']==next(r for r in source['summaries'] if r['split']==split and r['cohort']==cohort)
                    assert v['receipt']['sha256']==sha(OUT/'source_coupling'/model/'result.json')
                    assert v['relative_strength']['complete'] and v['relative_strength']['receipt']['sha256']==sha(OUT/'source_coupling'/model/'relative_strength.json')
                    assert v['relative_strength']['summaries']==[r for r in relative['summaries'] if r['split']==split and r['cohort']==cohort]
                    equal(v['field'],saved[prefix+'__'+coordinate][None])
                    count=saved[prefix+'__effective_source_valid_questions']; assert v['effective_source_valid_questions']==count.tolist()
                    j=source['head_columns'].index('passage_conditional_effective_source_fraction')
                    for h,row in enumerate(v['head_values']):
                        for k,x in enumerate(row): assert (x is None) if k==j and count[h]==0 else x==saved[prefix+'__head_means'][h,k]
                    responses[(model,split,cohort,coordinate)]=v; api_checks.append({'model':model,'split':split,'cohort':cohort,'coordinate':coordinate})
        print('NATURAL_SOURCE_COUPLING_CLIENT_API',model,flush=True)
    for query in [{'model':'bad'},{'split':'confirmation'},{'cohort':'all'},{'coordinate':'invented'}]:
        try:get('/source-coupling',**query)
        except HTTPError as exc:assert exc.code==422
        else:raise AssertionError('Invalid source query accepted')
    folder=OUT/'client'/('source_coupling_'+str(time.time_ns())); folder.mkdir(parents=True)
    sys.path.insert(0,str(ROOT/'tests/glm5_temp/rdc_update_browser_packages'))
    from playwright.sync_api import sync_playwright,expect
    images=[]; errors=[]; compared=0
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1080}); page=context.new_page(); page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            study=page.locator('#construction-natural-questions'); panel=page.locator('#natural-source-coupling')
            for model in models:
                study.get_by_label('自然研究模型',exact=True).select_option(model)
                for split in ['train','validation','diagnostic']:
                    for cohort in ['drop','quoref']:
                        panel.get_by_label('来源配对划分',exact=True).select_option(split); panel.get_by_label('来源配对语料',exact=True).select_option(cohort)
                        v=responses[(model,split,cohort,COORDINATES[0])]; expect(panel.locator('pre')).to_contain_text(v['receipt']['sha256'],timeout=60000)
                        ss=v['summary']; row=[ss['head_equal_means'][k] for k in HEADS[:3]]+[ss['postO_question_means'][k] for k in ['native_read_mean_square','native_permuted_read_MSE','native_permuted_read_relative_squared_L2']]
                        compared+=cells(page,'source-coupling-summary',[row])
                        rows=[[str(i)]+[r[v['head_columns'].index(k)] for k in HEADS] for i,r in enumerate(v['head_values'])]
                        compared+=cells(page,'source-coupling-heads',rows)
                        relrows=[[r['space']]+[r['values'][k] for k in ['permutation_MSE','common_context_change_MSE','within_native_mean_square','within_permutation_MSE','relative_question_permutation_energy','pooled_question_response_cosine']] for r in v['relative_strength']['summaries']]
                        compared+=cells(page,'source-coupling-relative',relrows)
                panel.get_by_label('来源配对语料',exact=True).select_option('drop')
                v=responses[(model,'diagnostic','drop',COORDINATES[0])]
                cells(page,'source-coupling-heads',[[str(i)]+[r[v['head_columns'].index(k)] for k in HEADS] for i,r in enumerate(v['head_values'])])
                panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                path=folder/(model+'_all_heads.png');page.screenshot(path=str(path));images.append(path)
                for coordinate in COORDINATES:
                    panel.get_by_label('来源配对坐标量',exact=True).select_option(coordinate)
                    expect(panel.locator('pre')).to_contain_text('"coordinate": "'+coordinate+'"',timeout=60000)
                    expect(panel.locator('canvas')).to_have_attribute('width',str(len(responses[(model,'diagnostic','drop',coordinate)]['field']['values'][0])))
                panel.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                path=folder/(model+'_all_coordinates.png');page.screenshot(path=str(path));images.append(path)
                panel.get_by_label('来源配对坐标量',exact=True).select_option(COORDINATES[0])
                expect(panel.locator('pre')).to_contain_text('"coordinate": "'+COORDINATES[0]+'"',timeout=60000)
                cells(page,'source-coupling-heads',[[str(i)]+[r[v['head_columns'].index(k)] for k in HEADS] for i,r in enumerate(v['head_values'])])
                expect(panel.locator('canvas')).to_have_attribute('width',str(len(v['field']['values'][0])))
            page.set_viewport_size({'width':390,'height':844});panel.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            path=folder/'source_coupling_mobile.png';page.screenshot(path=str(path));images.append(path)
            assert study.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1') and not errors,errors
        finally:context.close();browser.close()
    value={'timestamp':stamp(),'all_passed':True,'models':models,'API_checks':api_checks,'independent_raw_field_checks':raw_checks,
        'displayed_cells_compared_to_actual_sources':compared,'screenshots':[p.relative_to(ROOT).as_posix() for p in images],
        'visual_QA_pending':True,'seconds':time.monotonic()-start,'source_sha256':sha(__file__),
        'scope':'Every selected model has real completed analysis.18APIviews and6completeheadtables/model;24nativequestions/model independently recomputed, all heads/coordinates. No fabricated scientific data.'}
    immutable(folder/'result.json',value);print('NATURAL_SOURCE_COUPLING_CLIENT_PASS',str(folder),compared,round(value['seconds'],2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',choices=['qwen4','qwen14','glm4'],default=['qwen4','qwen14','glm4']);main(p.parse_args().models)
