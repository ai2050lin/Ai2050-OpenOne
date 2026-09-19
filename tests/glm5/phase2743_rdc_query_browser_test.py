"""Isolated headless Edge regression of the authored local application."""
import argparse,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5_temp/rdc_update_browser_packages'))
from rdc_query_common import *

def main(final=False):
    from playwright.sync_api import sync_playwright,expect
    import importlib.metadata
    out=BASE/'client';start=time.monotonic();errors=[];checks=[];mode='final' if final else 'preliminary'
    with sync_playwright() as p:
      browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
      context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1);page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
      try:
        page.goto('http://127.0.0.1:5173/rdc-query',wait_until='networkidle',timeout=60000)
        page.add_style_tag(content='html,body,main{scroll-behavior:auto !important}')
        expect(page.get_by_role('heading',name='条件查询与有序来源图谱',exact=True)).to_be_visible();page.screenshot(path=str(out/f'headless_{mode}_desktop.png'));checks.append('New route and actual committed counts visible')
        source=page.locator('#query-sources');source.get_by_role('button',name='读取完整查询场',exact=True).click()
        expect(source.locator('canvas')).to_have_count(1,timeout=60000);expect(source.locator('canvas')).to_have_attribute('width','2560');expect(source.locator('canvas')).to_have_attribute('height','100')
        source.get_by_label('场数值视图',exact=True).select_option('RMS');expect(source.locator('canvas')).to_have_count(0)
        source.get_by_role('button',name='读取完整查询场',exact=True).click();expect(source.locator('canvas')).to_have_count(1)
        source.locator('.prefix-field').first.scroll_into_view_if_needed();page.screenshot(path=str(out/f'headless_{mode}_queries.png'));checks.append('Full100x2560canvas and strict query/view identity clearing')
        source.get_by_label('显示对象',exact=True).select_option('layers');source.get_by_role('button',name='读取完整查询场',exact=True).click();expect(source.locator('canvas')).to_have_attribute('height','37')
        checks.append('All37rawlayer anchors shown, not postnorm mislabeled rawH36')
        archive=page.locator('#query-archive');archive.get_by_role('button',name='读取原序数值页',exact=True).click();expect(archive.locator('canvas')).to_have_count(1,timeout=30000)
        checks.append('All saved numerical archives reachable through original-axis paging')
        if final:
            findings=page.locator('#query-findings');expect(findings.get_by_text('ordered_softmax',exact=True).first).to_be_visible()
            expect(findings.get_by_text('3.008427 [2.954133, 3.063258]',exact=True)).to_be_visible()
            expect(findings.get_by_role('heading',name='真实中层训练：实际 BF16 部署后的自然内容 NLL',exact=True)).to_be_visible()
            findings.locator('h2').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})');page.screenshot(path=str(out/'headless_final_findings.png'));checks.append('Actual all-coordinate and full-vocabulary findings, training and own-history summaries remain separate')
            confirmation=findings.get_by_role('heading',name='独立确认 · 96 预留文档 · 20 未见查询',exact=True)
            expect(confirmation).to_be_visible();confirmation.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})');page.screenshot(path=str(out/'headless_final_confirmation.png'))
            if (BASE/'identifiability/analysis/result.json').exists():
                relation=findings.get_by_role('heading',name='严格相同 token 多重集 · 原生关系分离',exact=True)
                expect(findings.get_by_text('1.389e-8 [1.101e-8, 1.754e-8]',exact=True)).to_be_visible()
                expect(relation).to_be_visible();relation.evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})');page.screenshot(path=str(out/'headless_final_relation_results.png'))
                checks.append('Independent confirmation, strict matched-pair sensitivity and five native parameter behavior summaries')
            with urllib.request.urlopen('http://127.0.0.1:5003/api/rdc-query/samples?cohort=gum&split=test&limit=100') as r:rs=json.load(r)['rows']
            sid=next(r['sample_id'] for r in rs if r['captured'] and r['detail']);source.get_by_label('查询来源',exact=True).select_option(sid)
            source.get_by_role('button',name='读取全坐标预测对照',exact=True).click();expect(source.locator('canvas')).to_have_count(1);expect(source.locator('canvas')).to_have_attribute('height','3');checks.append('Frozen full-coordinate predictions visible')
            ordered=page.locator('#query-ordered');ordered.get_by_label('有序 MLP 单元',exact=True).fill('9727');ordered.get_by_label('有序输入坐标',exact=True).fill('2559');ordered.get_by_label('有序输出坐标',exact=True).fill('2559')
            ordered.get_by_role('button',name='读取有序参数路径',exact=True).click();expect(ordered.locator('canvas')).to_have_count(4,timeout=60000)
            ordered.locator('.prefix-field').first.scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_final_ordered_sources.png'));checks.append('Last-native-index scalar parameters and ordered fullsource matrix')
            events=page.locator('#query-events');events.get_by_role('button',name='读取时间锚点图谱',exact=True).click();expect(events.locator('canvas')).to_have_count(2,timeout=60000)
            options=events.get_by_label('实际时间锚点',exact=True).locator('option');last=options.last.get_attribute('value');events.get_by_label('实际时间锚点',exact=True).select_option(last)
            expect(events.locator('canvas')).to_have_count(0);events.get_by_role('button',name='读取时间锚点图谱',exact=True).click();expect(events.locator('canvas')).to_have_count(2)
            events.locator('.prefix-field').first.scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_final_event_time.png'));checks.append('Actual generation-time selection and100query temporal field')
            source.get_by_label('材料范围',exact=True).select_option('scale');source.get_by_label('显示对象',exact=True).select_option('queries')
            for model,width in [('qwen14','5120'),('glm4','4096')]:
                source.get_by_label('查询模型',exact=True).select_option(model);expect(source.get_by_role('button',name='读取完整查询场',exact=True)).to_be_enabled(timeout=30000)
                source.get_by_role('button',name='读取完整查询场',exact=True).click();expect(source.locator('canvas')).to_have_count(1);expect(source.locator('canvas')).to_have_attribute('width',width)
            checks.append('ActualQ14/GLM query coordinates with noQ4substitution')
            history=page.locator('#query-history');history.get_by_label('真实分支',exact=True).select_option('entropy_digit');history.get_by_role('button',name='读取自身历史生成',exact=True).click();expect(history.locator('blockquote')).to_have_count(1)
            history.get_by_role('slider').focus();history.get_by_role('slider').press('ArrowRight');history.scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_final_late_generation.png'))
            history.get_by_label('实验集合',exact=True).select_option('injection');history.get_by_label('真实分支',exact=True).select_option('mapped_code');expect(history.locator('blockquote')).to_have_count(0)
            history.get_by_role('button',name='读取自身历史生成',exact=True).click();expect(history.locator('blockquote')).to_have_count(1);checks.append('Paired late and mapped-code own-history branch identities, scores and step slider')
            if (BASE/'identifiability/analysis/result.json').exists():
                identity=page.locator('#query-identity');identity.get_by_label('实际训练变体',exact=True).select_option('natural_target_2743')
                identity.get_by_role('button',name='读取严格身份对照',exact=True).click();expect(identity.locator('canvas')).to_have_attribute('width','2560',timeout=30000)
                expect(identity.locator('canvas')).to_have_attribute('height','12')
                identity.get_by_label('身份对照字段',exact=True).select_option('units');expect(identity.locator('canvas')).to_have_count(0)
                identity.get_by_role('button',name='读取严格身份对照',exact=True).click();expect(identity.locator('canvas')).to_have_attribute('width','9728')
                identity.locator('.prefix-field').scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_final_identity_units.png'))
                checks.append('Strict token-matched paired worlds, real trained parameters and all9728units with stale-view clearing')
        page.set_viewport_size({'width':390,'height':844});page.evaluate('document.documentElement.style.scrollBehavior="auto";window.scrollTo(0,0)')
        page.locator('main').evaluate('(element)=>{element.style.scrollBehavior="auto";element.scrollTop=0}')
        page.wait_for_function('window.scrollY===0 && document.querySelector("main").scrollTop===0')
        page.screenshot(path=str(out/f'headless_{mode}_mobile.png'));layout=page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})');assert layout['scroll']<=layout['width']+1,layout
        checks.append('Mobile390px has no document-level horizontal overflow');assert not errors,errors
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'final':final,'checks':checks,'page_errors':errors,'mobile_layout':layout,
          'browser':browser.version,'playwright':importlib.metadata.version('playwright'),'seconds':time.monotonic()-start,
          'mode':'Independent ephemeral headless Edge, not attached to user profile or current browser; live connection failure remains separately recorded.'}
        save(out/f'browser_{mode}.json',result);print('QUERY_HEADLESS_PASS',mode,len(checks),flush=True)
      except Exception as exc:
        page.screenshot(path=str(out/f'headless_{mode}_failure.png'));failure(out/'browser_failure',start,exc);raise
      finally:context.close();browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
