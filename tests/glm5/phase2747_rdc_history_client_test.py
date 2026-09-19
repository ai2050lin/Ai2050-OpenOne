"""Completed native/trained/program histories: exact API arrays and authored UI."""
import argparse
from phase2747_rdc_client_test import *
from phase2747_rdc_evidence_client_test import get
from rdc_formation_readout import checked_arrays
from phase2747_rdc_program_own_history import BRANCHES


def main(program_only=False):
    start=time.monotonic();out=OUT/'client'/('history_regression_'+str(time.time_ns()));out.mkdir(parents=True)
    checks=[];program=read(OUT/'program_own_history/analysis/result.json');assert program['all_passed']
    own={'reports':[]} if program_only else read(OUT/'own_history/analysis/result.json')
    if not program_only:assert own['all_passed'] and own['complete_runs']==9
    progress=get('/formation-progress');figures=progress['figures'];assert len(figures) in ([15,17] if program_only else [17])
    for item in figures:
        with urllib.request.urlopen(API+'/figure?'+urllib.parse.urlencode({'stage':'2747','name':item['name']})) as response:
            assert hashlib.sha256(response.read()).hexdigest()==item['sha256']
    checks.append(f'All{len(figures)}actually reviewed scientific figures served with exact registered SHA256')
    selected={};program_selected={}
    for run in own['reports']:
        model,variant=run['model'],run['variant'];rows=get('/formation-own-samples',model=model,variant=variant)
        shown=next(r for r in progress['own_runs'] if r['model']==model and r['variant']==variant)
        assert shown['scoring_audit']==run['scoring_audit']
        assert len(rows)==512 and len({r['sample_id'] for r in rows})==512
        sid=next(r['sample_id'] for r in rows if r['kind']=='natural' and r['full_hidden_collected'])
        record=read(OUT/'own_history'/model/variant/'records'/(sid+'.json'));a=checked_arrays(record['field'])
        for field,step in [('postnorm',0),('all_hidden',0),('all_hidden',len(record['generated_ids'])-1)]:
            answer=get('/formation-own-field',model=model,variant=variant,sample=sid,field=field,step=step)
            expected=unbits(a['postnorm_BF16'] if field=='postnorm' else a['all_hidden_BF16'][step])
            assert np.array_equal(np.array(answer['values']),expected),(model,variant,field,step)
            assert answer['record']['generated_ids']==record['generated_ids']
        selected[model+'/'+variant]={'sample_id':sid,'steps':len(record['generated_ids']),
            'hidden_boundaries':a['all_hidden_BF16'].shape[1],'width':a['postnorm_BF16'].shape[1]}
        control=next(r['sample_id'] for r in rows if r['kind']=='controlled')
        control_record=read(OUT/'own_history'/model/variant/'records'/(control+'.json'))
        displayed=get('/formation-own-field',model=model,variant=variant,sample=control,field='postnorm')
        assert displayed['record']['answer_scoring']==control_record['answer_scoring']
        selected[model+'/'+variant]['controlled_sample_id']=control
    if not program_only:checks.append('All9complete native/trained runs list512correct IDs; allpostnorm and first/final full-H selected arrays match original BF16 bytes')
    language_review=read(OUT/'own_history/terminal_review/result.json') if not program_only else None
    if language_review:
        for item in language_review['reviews']:
            displayed=get('/formation-own-field',model=item['model'],variant=item['variant'],sample=item['sample_id'],field='postnorm')
            assert displayed['supplemental_terminal_review']==item
            assert displayed['record']['answer_scoring']==item['original_frozen_scoring']
        checks.append('All five GLM post-outcome terminal reviews are separate, with the frozen scores unchanged')
    for branch in BRANCHES:
        rows=get('/formation-program-samples',branch=branch);assert len(rows)==32
        sid=rows[0]['sample_id'];record=read(OUT/'program_own_history/records'/branch/(sid+'.json'));a=checked_arrays(record['field'])
        for field,key,index in [('postnorm','native_postnorm_BF16',None),('first_readout','first_actual_readout_BF16',None),
            ('first_hidden','first_final_all_hidden_BF16',0),('final_hidden','first_final_all_hidden_BF16',1)]:
            answer=get('/formation-program-field',branch=branch,sample=sid,field=field)
            expected=unbits(a[key] if index is None else a[key][index])
            if expected.ndim==1:expected=expected[None]
            assert np.array_equal(np.array(answer['values']),expected),(branch,field)
            assert answer['record']['generated_ids']==record['generated_ids']
            assert answer['record']['answer_scoring']==record['answer_scoring']
        program_selected[branch]={'sample_id':sid,'steps':len(record['generated_ids'])}
    checks.append('All6program paths list32heldoutgroups; every ownstep, first actual readout, first/final all37H and scoring match committed files')
    review=read(OUT/'program_own_history/terminal_review/result.json');assert review['all_passed']
    item=review['reviews'][0]
    reviewed=get('/formation-program-field',branch=item['branch'],sample=item['sample_id'],field='first_readout')
    assert reviewed['supplemental_terminal_review']==item
    assert not reviewed['record']['answer_scoring']['parsed_and_stopped_correct'] and reviewed['record']['answer_scoring']['EOS']
    checks.append('Supplemental unblinded terminal review appears separately; original unparsed-EOS score remains unchanged')
    for endpoint,params,status in [('/formation-program-field',{'branch':'missing'},422),
        ('/formation-program-field',{'branch':'native','sample':'missing'},404),
        ('/formation-program-field',{'branch':'native','sample':program_selected['native']['sample_id'],'field':'all_steps_hidden'},422),
        ('/formation-own-field',{'model':'qwen14','variant':'true_token_2747'},422)]:
        try:get(endpoint,**params);raise AssertionError('Expected rejection')
        except urllib.error.HTTPError as exc:assert exc.code==status,(endpoint,params,exc.code)
    checks.append('Missing samples, nonexistent paths, unsupported trained models and uncollected program axes reject without substitution')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1100},device_scale_factor=1);page=context.new_page()
        page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            view=page.locator('#formation-own-view');pv=page.locator('#formation-program-view')
            for model in ([] if program_only else ['qwen14','glm4']):
                row=selected[model+'/native']
                view.get_by_label('自身历史模型',exact=True).select_option(model)
                expect(view.get_by_label('自身历史表达',exact=True).locator('option')).to_have_count(512,timeout=60000)
                view.get_by_label('自身历史表达',exact=True).select_option(row['sample_id'])
                view.get_by_label('自身历史场',exact=True).select_option('all_hidden')
                view.get_by_label('自身生成步',exact=True).fill(str(row['steps']-1))
                view.get_by_role('button',name='读取真实自身历史',exact=True).click()
                expect(view.locator('canvas')).to_have_attribute('width',str(row['width']),timeout=60000)
                expect(view.locator('canvas')).to_have_attribute('height',str(row['hidden_boundaries']))
                view.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                page.screenshot(path=str(out/(model+'_last_all_hidden.png')))
            if not program_only:
                control=selected['qwen4/native']['controlled_sample_id']
                view.get_by_label('自身历史模型',exact=True).select_option('qwen4')
                expect(view.get_by_label('自身历史表达',exact=True).locator('option')).to_have_count(512,timeout=60000)
                view.get_by_label('自身历史表达',exact=True).select_option(control)
                view.get_by_label('自身历史场',exact=True).select_option('postnorm')
                view.get_by_role('button',name='读取真实自身历史',exact=True).click()
                expect(view.get_by_test_id('own-language-scoring')).to_be_visible(timeout=60000)
                expect(view.get_by_test_id('own-language-scoring')).to_contain_text('内容、格式和停止分别记录')
                view.get_by_test_id('own-language-scoring').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                page.screenshot(path=str(out/'native_controlled_scoring.png'))
                reviewed_language=next(r for r in language_review['reviews'] if r['supplemental_terminal_correct_and_stopped'])
                view.get_by_label('自身历史模型',exact=True).select_option('glm4')
                expect(view.get_by_label('自身历史表达',exact=True).locator('option')).to_have_count(512,timeout=60000)
                view.get_by_label('自身历史表达',exact=True).select_option(reviewed_language['sample_id'])
                view.get_by_role('button',name='读取真实自身历史',exact=True).click()
                expect(view.get_by_test_id('own-terminal-review')).to_be_visible(timeout=60000)
                expect(view.get_by_test_id('own-terminal-review')).to_contain_text(reviewed_language['exact_terminal_quote'])
                view.get_by_test_id('own-terminal-review').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                page.screenshot(path=str(out/'native_language_terminal_review.png'))
            for branch in ['native','mapped_code','mapped_digit_bias']:
                row=program_selected[branch]
                pv.get_by_label('程序读出路径',exact=True).select_option(branch)
                expect(pv.get_by_label('程序表达',exact=True).locator('option')).to_have_count(32,timeout=60000)
                pv.get_by_label('程序表达',exact=True).select_option(row['sample_id'])
                field='postnorm' if branch=='native' else 'first_readout'
                pv.get_by_label('程序原生场',exact=True).select_option(field)
                expect(pv.locator('canvas')).to_have_count(0)
                pv.get_by_role('button',name='读取程序自身历史',exact=True).click()
                expect(pv.locator('canvas')).to_have_attribute('width','2560',timeout=60000)
                expect(pv.locator('canvas')).to_have_attribute('height',str(row['steps'] if field=='postnorm' else 1))
                pv.locator('canvas').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
                page.screenshot(path=str(out/('program_'+branch+'.png')))
            pv.get_by_label('程序读出路径',exact=True).select_option(item['branch'])
            expect(pv.get_by_label('程序表达',exact=True).locator('option')).to_have_count(32,timeout=60000)
            pv.get_by_label('程序表达',exact=True).select_option(item['sample_id'])
            pv.get_by_role('button',name='读取程序自身历史',exact=True).click()
            expect(pv.get_by_test_id('program-terminal-review')).to_be_visible(timeout=60000)
            expect(pv.get_by_test_id('program-terminal-review')).to_contain_text(item['exact_terminal_quote'])
            pv.get_by_test_id('program-terminal-review').evaluate('(e)=>e.scrollIntoView({block:"center",behavior:"instant"})')
            page.screenshot(path=str(out/'program_terminal_review.png'))
            page.set_viewport_size({'width':390,'height':844})
            pv.locator('h3').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'program_mobile.png'))
            assert pv.evaluate('(e)=>e.scrollWidth<=e.clientWidth+1')
            assert not errors,errors
            checks.append(('Program-only: ' if program_only else 'Two native model last-step full-H and ')+
                'three program path views plus mobile rendering; identity changes never reuse stale fields; native controlled scoring separately shown when all runs complete')
        finally:context.close();browser.close()
    value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,
        'selected_native':selected,'selected_program':program_selected,'seconds':time.monotonic()-start,
        'program_only':program_only,'phase_client_complete':not program_only,
        'images':[p.name for p in sorted(out.glob('*.png'))],'visual_review':'Pending main-agent inspection of actual new images; not a science confirmation.'}
    save(out/'result.json',value)
    save(OUT/'client'/('current_program_regression.json' if program_only else 'current_history_regression.json'),{'timestamp':stamp(),'result':str((out/'result.json').relative_to(ROOT)),
        'result_sha256':sha(out/'result.json'),'image_directory':str(out.relative_to(ROOT))})
    print('FORMATION_HISTORY_CLIENT_PASS',len(checks),out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--program-only',action='store_true');main(parser.parse_args().program_only)
