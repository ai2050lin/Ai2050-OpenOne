# -*- coding: utf-8 -*-
"""官网优化：顶栏客户端入口、页脚死链、AGI 进展实测指标、计划页 webp。
   纪律：只做定点替换，每步断言命中数；原 PNG 与原有文案保留（可回退）。"""
import io, os, re, sys

os.chdir(r'D:\AI2050\Ai2050-OpenOne\frontend\website')
PAGES = ['index.html', 'ai2050.html', 'agi_project.html', 'about.html', 'industry_progress.html', 'join.html']
WB = 'https://server.ai2050.com/workbench'
report = []

def rw(name, fn):
    p = os.path.join(name)
    s = io.open(p, encoding='utf-8', newline='').read()
    out, hits = fn(s)
    if out != s:
        io.open(p, 'w', encoding='utf-8', newline='').write(out)
    report.append((name, hits))
    return out

# ---------- 1. 顶栏：加「研究工作台」次级入口 ----------
OLD_HDR = '      <a class="header-action dark" href="join.html" target="_blank" rel="noopener">参与讨论</a>'
OLD_HDR_JOIN = '      <a class="header-action dark" href="index.html">返回首页</a>'
WB_LINK = ('        <a class="header-wb" href="%s" title="研究报告工作台：进展、空间结构与研发工位">研究工作台</a>\n' % WB)

def hdr(s):
    if 'class="header-actions"' in s:                       # 幂等：已改过
        return s, '顶栏已就绪'
    if OLD_HDR in s:
        new = ('      <div class="header-actions">\n' + WB_LINK +
               '        <a class="header-action dark" href="join.html" target="_blank" rel="noopener">参与讨论</a>\n'
               '      </div>')
        return s.replace(OLD_HDR, new), '顶栏+1'
    if OLD_HDR_JOIN in s:
        new = ('      <div class="header-actions">\n' + WB_LINK +
               '        <a class="header-action dark" href="index.html">返回首页</a>\n'
               '      </div>')
        return s.replace(OLD_HDR_JOIN, new), '顶栏+1(join)'
    raise AssertionError('顶栏未命中')

for f in PAGES:
    rw(f, hdr)

# ---------- 2. 页脚：加研究工作台；未建页面不再伪装成链接 ----------
OLD_PLAN = ('<div class="footer-col"><h4>计划</h4><a href="ai2050.html">AI2050计划</a>'
            '<a href="agi_project.html">AGI研究</a><a href="industry_progress.html">行业进展</a>'
            '<a href="agi_project.html#research-route">研究路线</a></div>')
NEW_PLAN = ('<div class="footer-col"><h4>计划</h4><a href="ai2050.html">AI2050计划</a>'
            '<a href="agi_project.html">AGI研究</a><a href="industry_progress.html">行业进展</a>'
            '<a href="agi_project.html#research-route">研究路线</a>'
            '<a href="%s">研究工作台</a></div>') % WB

DEAD = ['论坛', '活动', '公告', '资源', '免责声明', '隐私政策']

def foot(s):
    hits = []
    if OLD_PLAN in s:
        s = s.replace(OLD_PLAN, NEW_PLAN); hits.append('页脚+工作台')
    for t in DEAD:
        a = '<a href="#">%s</a>' % t
        if a in s:
            s = s.replace(a, '<span class="footer-soon" title="筹备中，暂未开放">%s</span>' % t)
            hits.append(t)
    return s, ','.join(hits) if hits else '页脚无匹配'

for f in PAGES:
    rw(f, foot)

# ---------- 3. 首页首屏主按钮改指向客户端 ----------
def hero(s):
    old = '<a class="primary-button" href="about.html">参与建设</a>'
    new = '<a class="primary-button" href="%s">进入研究工作台</a>' % WB
    if old in s:
        return s.replace(old, new), '首屏 CTA'
    return s, '首屏 CTA 未命中'

rw('index.html', hero)

# ---------- 4. AGI 研究页：项目进展换成有口径的实测指标 ----------
OLD_PROG = '''      <section class="research-section">
        <h2>项目进展</h2>
        <div class="progress-panel">
          <div class="progress-list">
            <article><h3>语言理论</h3><p>如果通过有限的参数，实现知识网络、分析推理、语法规则三种能力，然后融合具备完整的语言能力。</p><strong class="progress-percent">57%</strong></article>
            <span class="progress-arrow" aria-hidden="true">→</span>
            <article><h3>编码机制</h3><p>在深度神经网络中，参数是怎么通过相对编码、差分复用、抽象详细三种能力，在编码维度实现语言能力的。</p><strong class="progress-percent">26%</strong></article>
            <span class="progress-arrow" aria-hidden="true">→</span>
            <article><h3>智能理论</h3><p>分析语言能力和编码机制，得到更系统的特征提取和网络结构，完成智能理论。</p><strong class="progress-percent">15%</strong></article>
          </div>
          <p class="research-callout">AGI研究的核心目标：破解语言编码机制，完成智能理论。</p>
        </div>
        
      </section>'''

NEW_PROG = '''      <section class="research-section" id="research-progress">
        <h2>项目进展 <span class="section-aside">实测 · 截至 2026-10-28</span></h2>
        <div class="progress-panel">
          <div class="progress-list">
            <article>
              <h3>语言理论</h3>
              <p>如果通过有限的参数，实现知识网络、分析推理、语法规则三种能力，然后融合具备完整的语言能力。</p>
              <strong class="progress-metric">350,208</strong>
              <span class="progress-unit">个 MLP 单元完成普查</span>
              <small class="progress-scope">qwen3-4b · 36 层 × 9,728 单元 · 1,904 万 token</small>
            </article>
            <span class="progress-arrow" aria-hidden="true">→</span>
            <article>
              <h3>编码机制</h3>
              <p>在深度神经网络中，参数是怎么通过相对编码、差分复用、抽象详细三种能力，在编码维度实现语言能力的。</p>
              <strong class="progress-metric">26</strong>
              <span class="progress-unit">条已登记的语言特征</span>
              <small class="progress-scope">E2 及以上 23 条 · 3 个主线模型 · 21 条失败同样入账</small>
            </article>
            <span class="progress-arrow" aria-hidden="true">→</span>
            <article>
              <h3>智能理论</h3>
              <p>分析语言能力和编码机制，得到更系统的特征提取和网络结构，完成智能理论。</p>
              <strong class="progress-metric">3</strong>
              <span class="progress-unit">条干预级（E3）证据</span>
              <small class="progress-scope">「读得出」到「控得住」仍有距离：这是当前最大瓶颈</small>
            </article>
          </div>
          <p class="research-callout">AGI研究的核心目标：破解语言编码机制，完成智能理论。</p>
          <p class="progress-source">口径与来源：单元普查取 qwen3-4b 的 19,042,014 个 token 前向（36 层 × 9,728）；特征数与失败账本取注册表 v1.39 与配套缺口账本。数字随研究账本更新，最新值以研究工作台「进展」页为准。</p>
          <a class="progress-link" href="''' + WB + '''?lens=route">在研究报告工作台中查看实时进展 →</a>
        </div>
      </section>'''

def prog(s):
    if OLD_PROG in s:
        return s.replace(OLD_PROG, NEW_PROG), '进展区块'
    return s, '进展区块未命中'

rw('agi_project.html', prog)

# ---------- 5. 重复 id：项目地址区块让出 research-route ----------
def dupid(s):
    old = '<section class="research-section" id="research-route">\n        <h2>项目地址</h2>'
    new = '<section class="research-section" id="project-repo">\n        <h2>项目地址</h2>'
    if old in s:
        return s.replace(old, new), 'dup-id 修复'
    return s, 'dup-id 未命中'

rw('agi_project.html', dupid)

# ---------- 6. 计划页图片：webp + 尺寸属性 ----------
def pics(s):
    n = 0
    def rep(m):
        nonlocal n
        src, alt = m.group(1), m.group(2)
        webp = src[:-4] + '.webp'
        if not os.path.exists(webp):
            return m.group(0)
        n += 1
        return ('<div class="plan-detail-visual"><picture>'
                '<source srcset="%s" type="image/webp" />'
                '<img src="%s" alt="%s" width="1536" height="1024" loading="lazy" decoding="async" />'
                '</picture></div>') % (webp, src, alt)
    s = re.sub(r'<div class="plan-detail-visual"><img src="(image/ai2050_\d+\.png)" alt="([^"]*)" loading="lazy" /></div>', rep, s)
    return s, 'picture×%d' % n

rw('ai2050.html', pics)

print('%-24s %s' % ('文件', '命中'))
for f, h in report:
    print('%-24s %s' % (f, h))
