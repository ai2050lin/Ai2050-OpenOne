# -*- coding: utf-8 -*-
"""Phase 3014 memory compression round 2."""
import io

P = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
     r'\.workbuddy\memory\MEMORY.md')
t = io.open(P, encoding='utf-8').read()
miss = []


def rep(old, new):
    global t
    if old in t:
        t = t.replace(old, new, 1)
    else:
        miss.append(old[:30])


rep('2938-2970：子空间/词盲/线性壳/阈值/头集中/承重/路由'
    '跳变/秩1/签名/载体h15/消融/塌缩/峰锁；',
    '2938-2970：子空间/词盲/线性壳/阈值/头集中/承重/路由'
    '跳变/秩1/签名/载体h15/消融/塌缩/峰锁；')
rep('2972-2991：主调制×词类特化/fr塌缩=重写/双轴注入/剂量窗/'
    '转正/h12载体/三要素否定/正交重定向/局部化/perp=重写/'
    '坍缩≠漂移/头谱=路由属性。',
    '2972-2991：主调制×词类特化/fr塌缩=重写/双轴注入/剂量窗/'
    '转正/h12载体/三要素否定/正交重定向/局部化/perp=重写。')
rep('2992 字典：符号率 0.405=快照。2993 逻辑签名长度稳健；'
    '2994 轴防御。2995-2999 Ω-F：重分级；xdir ~98% 洗消。',
    '2992 字典：符号率 0.405=快照。2993 逻辑签名稳健；2994 轴'
    '防御。2995-2999 Ω-F：重分级；xdir 98% 洗消。')
rep('3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；中带 xdir '
    '特异150×；3003：中带=xdir 特异剂量线性不对称；方案 v5。',
    '3001/3002 Ω-G：词携带 GLM4 86% vs qwen 63%；中带 xdir '
    '特异150×；3003 剂量线性不对称；方案 v5。')
rep('3004 Ω-P4：算子共享低秩（.715/.826）共享向≠xdir=重定向+'
    '放大；**T3 轴序 bug** bit 级修正。',
    '3004 Ω-P4：算子共享低秩（.715/.826）共享向≠xdir=重定向+'
    '放大；T3 轴序 bug 修正。')
rep('3007 Ω-P2a：锁定比 0.457（11.74<25.70）；扰动 2/4 '
    'divergent、3/4 零分歧。',
    '3007 Ω-P2a：锁定比 0.457（11.74<25.70）；扰动 3/4 零'
    '分歧。')
rep('3008 Ω-P2b：**held-out 轴生成态分离 logic>content'
    '（p 0.0002×2）**；KV 零化饱和。',
    '3008 Ω-P2b：**held-out 轴生成态分离 logic>content'
    '（p 0.0002×2）**；KV 零化饱和。')
rep(' 3009 Ω-P2c：KV 饱和与强度无关→token 分歧读出退役；'
    '坐标解离 logic<content<sham。',
    '3009 Ω-P2c：KV 饱和与强度无关→token 分歧读出退役；坐标'
    '解离 logic<sham。')
rep('3010 Ω-P2d：换读出保干预→逻辑位因果特异性确立'
    '（D=0.0297 p=0.0135）；读出解离=分布边界。',
    '3010 Ω-P2d：换读出→逻辑位因果特异（D=0.0297 p=0.0135）；'
    '读出解离=分布边界。')
rep('3011 Ω-P2e：**门控定位于 L3 KV**（D_l*=0.0137 '
    'p_maxT=1e-4；2×次峰 L31；K/V 双臂）；早层写+中带调',
    '3011 Ω-P2e：**门控定位于 L3 KV**（D=0.0137 p_maxT=1e-4；'
    '次峰 L31）；早层写+中带调')
rep('3014 Ω-P2h：**破坏剂量非单调且门控脆弱**——'
    'retain_joint(0.5)=1.011≥0.75（半擦除≥全擦除，半幅 KV毒'
    '信号）；**K/V 分岔**：KONLY 非单调（1.018/0.971/0.241，'
    'softmax 竞争）vs VONLY 单调（0.370/0.177/0.054，线性）'
    '——破坏=K 路由通道；sham 校准干净；npz8 跨 run4/5 位级。'
    '**手术不对称定标：破坏粗粒度易行，伪造须情景 K,V。**',
    '3014 Ω-P2h：**破坏剂量非单调门控脆弱**——retain(0.5)='
    '1.011（半擦除≥全擦除）；**K/V 分岔**：KONLY 非单调'
    '（softmax 竞争）vs VONLY 单调（线性）——破坏=K 路由；'
    'sham 干净；npz8 跨 run 位级。**破坏粗粒度易行，伪造须'
    '情景 K,V。**')

io.open(P, 'w', encoding='utf-8').write(t)
t2 = io.open(P, encoding='utf-8').read()
chk = {'has3014': '3014 Ω-P2h' in t2,
       'max14': 'max=3014' in t2}
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_mem14b.txt', 'w',
        encoding='utf-8').write(
    'len=%d miss=%s chk=%s'
    % (len(t2), miss,
       {k: v for k, v in chk.items() if not v}))
print('ok')
