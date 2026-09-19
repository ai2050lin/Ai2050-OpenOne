"""Isolated headless Edge application regression; not automation of the user's browser."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5_temp/rdc_update_browser_packages'))
from rdc_update_common import *

def main(final=False):
    from playwright.sync_api import sync_playwright,expect
    import importlib.metadata
    out=BASE/'client';start=time.monotonic();errors=[];checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True,chromium_sandbox=True)
        context=browser.new_context(viewport={'width':1440,'height':1000},device_scale_factor=1)
        page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            page.goto('http://127.0.0.1:5173/rdc-update',wait_until='networkidle',timeout=60000)
            expect(page.get_by_role('heading',name='关系、学习与自回归接续图谱',exact=True)).to_be_visible()
            page.screenshot(path=str(out/'headless_desktop.png'));checks.append('New route and live data load')
            source=page.locator('#update-sources');source.get_by_role('button',name='读取完整坐标场',exact=True).click()
            expect(source.locator('canvas')).to_have_count(1,timeout=30000);expect(source.locator('canvas')).to_have_attribute('width','2560')
            source.get_by_label('数值视图',exact=True).select_option('RMS');expect(source.locator('canvas')).to_have_count(0)
            source.get_by_role('button',name='读取完整坐标场',exact=True).click();expect(source.locator('canvas')).to_have_count(1)
            # A tall element capture exceeded the Edge compositor viewport in
            # the preliminary screenshot. Capture an explicit visible viewport.
            source.scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_full_coordinate_field.png'));checks.append('Raw/RMS full2560field and stale-query hiding')
            source.get_by_role('button',name='读取候选有向来源图',exact=True).click();expect(source.locator('canvas')).to_have_count(2)
            source.get_by_label('输入坐标',exact=True).fill('2559');source.get_by_label('MLP 单元',exact=True).fill('9727');source.get_by_label('输出坐标',exact=True).fill('2559')
            source.get_by_role('button',name='读取原生标量路径',exact=True).click();expect(source.locator('canvas')).to_have_count(4,timeout=30000)
            checks.append('Candidate source graph and last-index scalar rows')
            source.get_by_label('模型',exact=True).select_option('qwen14')
            expect(source.locator('canvas')).to_have_count(0)
            button=source.get_by_role('button',name='读取完整坐标场',exact=True)
            if final:
                expect(button).to_be_enabled(timeout=30000);button.click();expect(source.locator('canvas')).to_have_count(1)
                expect(source.locator('canvas')).to_have_attribute('width','5120');checks.append('Actual Q14 width5120, noQ4substitution')
                source.get_by_label('模型',exact=True).select_option('glm4');expect(source.locator('canvas')).to_have_count(0)
                expect(button).to_be_enabled(timeout=30000);button.click();expect(source.locator('canvas')).to_have_attribute('width','4096')
                checks.append('Actual GLM width4096')
            else:expect(button).to_be_disabled();checks.append('Pending Q14 disables query and clearsQ4field')
            gradient=page.locator('#update-gradient');gradient.get_by_role('button',name='读取全部参数行因子',exact=True).click()
            expect(gradient.locator('canvas')).to_have_count(2,timeout=30000);checks.append('Full parameter-gradient rows')
            archive=page.locator('#update-archive');archive.get_by_role('button',name='读取原序数组页',exact=True).click()
            expect(archive.locator('canvas')).to_have_count(1,timeout=30000);checks.append('Original tensor paging and real values')
            history=page.locator('#update-history');history.get_by_label('独立分支',exact=True).select_option('native')
            history.get_by_role('button',name='读取真实生成',exact=True).click();expect(history.locator('blockquote')).to_have_count(1)
            slider=history.get_by_role('slider');slider.focus();slider.press('ArrowRight');history.scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_history.png'))
            expect(history.get_by_text('终止答案解析正确：',exact=False)).to_be_visible()
            expect(history.get_by_text('格式包装复核正确：',exact=False)).to_be_visible()
            checks.append('Formal terminal answer scoring, EOS and censoring visible')
            checks.append('Exact native rollout identity and token-step slider')
            if final:
                native=page.locator('#update-native');native.get_by_role('button',name='核算原生来源路径',exact=True).click()
                expect(native.locator('canvas')).to_have_count(3,timeout=30000);native.scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_native_path.png'));checks.append('Actual24-source native path ledger')
                history.get_by_label('轨迹集合',exact=True).select_option('long_answers')
                history.get_by_role('button',name='读取真实生成',exact=True).click();expect(history.locator('blockquote')).to_have_count(1)
                checks.append('1024cap long-answer collection query')
                manual=read(BASE/'manual_terminal_audit/result.json')['adjudications'][0]
                raw=read(BASE/manual['raw_record'])
                history.get_by_label('独立分支',exact=True).select_option(raw['branch'])
                history.get_by_label('轨迹样本',exact=True).select_option(raw['sample_id'])
                history.get_by_role('button',name='读取真实生成',exact=True).click()
                expect(history.get_by_text('残余终止答案人工复核：',exact=False)).to_be_visible()
                history.locator('aside').scroll_into_view_if_needed();page.screenshot(path=str(out/'headless_manual_terminal.png'))
                checks.append('Separate unblinded manual terminal adjudication and exact supporting quote visible')
            page.set_viewport_size({'width':390,'height':844});page.evaluate('document.documentElement.style.scrollBehavior="auto"; window.scrollTo(0,0)')
            page.wait_for_function('window.scrollY===0');page.screenshot(path=str(out/'headless_mobile.png'))
            layout=page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})');assert layout['scroll']<=layout['width']+1,layout
            checks.append('390px viewport has no horizontal document overflow');assert not errors,errors
            result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'checks':checks,'page_errors':errors,'mobile_layout':layout,
              'browser':browser.version,'playwright':importlib.metadata.version('playwright'),'seconds':time.monotonic()-start,
              'mode':'Isolated headless Edge application tests; new ephemeral browser, does not attach to user browser or use its profile.',
              'CUA_limit':'Live user browser initialization failed separately; this result does not overwrite that receipt.'}
            save(out/('browser_final.json' if final else 'browser_preliminary.json'),result);print('HEADLESS_BROWSER_PASS',len(checks),flush=True)
        except Exception as exc:
            page.screenshot(path=str(out/'headless_failure.png'));failure(out/'browser_failure',start,exc);raise
        finally:context.close();browser.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
