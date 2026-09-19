"""Read-only new natural/cohort and original-attention-factor client checks."""
from phase2745_rdc_construction_client_test import get
from rdc_construction_common import *
from rdc_native_attention_parameters import entries
from rdc_construction_parameters import catalog


def main():
    from playwright.sync_api import sync_playwright,expect
    out=BASE/'phase2746/client';start=time.monotonic();checks=[]
    overview=get('/overview')
    for name in ['natural_interactions','natural_scrutiny','parameter_structure']:
        assert overview['phase2746'][name]==read(BASE/'phase2746'/name/'result.json')
    checks.append('Every new summary equals the original committed result without transformed counts')
    # Check real persisted C/F layouts, including the precisely registered
    # result-tree junction. Never create a fake uncommitted evidence archive.
    rotations=sorted((BASE/'phase2746/parameter_structure').rglob('*.npz'))
    found=False
    for file in rotations:
        with np.load(file) as z:
            for name in z.files:
                a=z[name]
                if a.ndim==2 and a.flags.f_contiguous and not a.flags.c_contiguous:
                    page=get('/array',path=file.relative_to(BASE).as_posix(),name=name,row_start=3,row_count=5,start=7,count=11)
                    assert np.array_equal(page['values'],a[3:8,7:18])
                    found=True;break
        if found:break
    assert found,'No actual native F-layout rotation fixture found'
    checks.append('Original F-layout native rotation page equals correctly indexed NumPy coordinates')
    for category in ['pilot','main']:
        commit_file=next((BASE/'phase2746/runtime'/category/'commits').glob('*.json'))
        rec=read(commit_file);headers=get('/arrays',path=rec['field_path'])
        assert any(r['name']=='units' for r in headers)
        with np.load(BASE/rec['field_path']) as z:
            a=unbits(z['units']).reshape(-1,z['units'].shape[-1])
            page=get('/array',path=rec['field_path'],name='units',row_start=4,row_count=3,start=93,count=17)
            assert np.array_equal(page['values'],a[4:7,93:110])
            get('/array',expected=422,path=rec['field_path'],name='units',row_start=len(a))
        checks.append(category+': committed overflow all-unit tensor exact page and native-axis boundary rejection')
    get('/arrays',expected=404,path='phase2746/field_store/unknown/missing.npz')
    get('/arrays',expected=404,path='phase2746/field_store/../../storage.json')
    checks.append('Unknown/missing overflow and parent traversal are rejected')
    for model in ['qwen4','qwen14','glm4']:
        c=catalog(model)['config'];head=c['num_attention_heads']-1;d=c['hidden_size'];block=c['num_hidden_layers']-1
        actual=get('/attention-parameter',model=model,block=block,head=head,input=d-1,input_r=0,output=d-1)
        reference=entries(model,block,head,d-1,0,d-1,113,17)
        assert actual['metadata']==reference['metadata']
        assert np.array_equal(actual['component_terms']['values'],np.stack([reference['qk_all_head_component_terms'],reference['ov_all_head_component_terms']]))
        assert np.array_equal(actual['factors']['values'],reference['read_write_factors'])
        assert actual['qk_numerator_coefficient']==reference['qk_numerator_coefficient']
        assert actual['ov_coefficient']==reference['ov_coefficient']
        get('/attention-parameter',expected=422,model=model,head=c['num_attention_heads'])
        checks.append(model+': complete last-head factor terms, native GQA map, bias/gains and boundary validation')
    errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-construction',wait_until='networkidle',timeout=60000)
            page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
            natural=page.locator('#construction-natural')
            expect(natural.locator('tbody').first.locator('tr')).to_have_count(4)
            expect(natural.locator('tbody').nth(1).locator('tr')).to_have_count(8)
            natural.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'natural.png'))
            attention=page.locator('#construction-attention-parameters')
            attention.get_by_label('attention 参数模型',exact=True).select_option('glm4')
            attention.get_by_label('查询头',exact=True).fill('31')
            attention.get_by_role('button',name='读取 QK / OV 因子',exact=True).click()
            expect(attention.locator('canvas')).to_have_count(2,timeout=60000)
            expect(attention.locator('canvas').first).to_have_attribute('width','128')
            expect(attention.locator('canvas').first).to_have_attribute('height','2')
            expect(attention.locator('canvas').nth(1)).to_have_attribute('height','9')
            attention.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'attention_parameters.png'))
            attention.get_by_label('查询位置',exact=True).fill('114')
            expect(attention.locator('canvas')).to_have_count(0)
            # The previously defective native GLM fused gate/up route.
            mlp=page.locator('#construction-parameters')
            mlp.get_by_label('参数模型',exact=True).select_option('glm4')
            mlp.get_by_label('MLP 单元',exact=True).fill('13695')
            mlp.get_by_role('button',name='查询原生参数因子',exact=True).click()
            expect(mlp.locator('canvas')).to_have_attribute('width','4096',timeout=60000)
            mlp.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
            page.screenshot(path=str(out/'glm_fused_mlp.png'))
            assert not errors,errors
            checks += ['Natural counts/weighting and exactly matched576depth table rendered',
                'GLM last GQA head shows both complete128component maps; changed condition clears stale maps',
                'GLM fused native lastMLPunit returns all4096read/write coordinates in the client']
        finally:
            context.close();browser.close()
    save(out/'result.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,
        'page_errors':errors,'seconds':time.monotonic()-start,'scope':'Ephemeral authored-app regression, not user browser control; manual image review separately required.'})
    print('PHASE2746_CLIENT_PASS',len(checks),flush=True)


if __name__=='__main__':main()
