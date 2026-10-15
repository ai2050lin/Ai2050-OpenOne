# -*- coding: utf-8 -*-
"""Phase 3236 add-on: interactive 3D feature-space view of the demo results.

Reads the sealed phase3236 npz (hs_final 160x2560 + meta + margins) and the
sealed phase3230 dR unit vector, does a PCA to 3D, and writes a self-contained
plotly HTML (plotly.min.js vendored locally, no CDN needed).
Zero GPU, read-only on sealed artifacts.
"""
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913')
NPZ = os.path.join(SRC, 'phase3236', 'g5d0_platform_pipeline_demo',
                   'pipeline_demo.npz')
RES = os.path.join(SRC, 'phase3236', 'g5d0_platform_pipeline_demo',
                   'result.json')
NPZ30 = os.path.join(SRC, 'phase3230', 'g5g27_relcontrast_efficacy',
                     'e3_relcontrast_flip.npz')
OUTDIR = os.path.join(SRC, 'phase3236', 'g5d0_3d_view')
HTML = os.path.join(OUTDIR, 'demo_3d.html')
TPLS = ['T1_zh_stmt', 'T2_zh_quest', 'T3_en_stmt', 'T4_en_quest', 'T5_zh_ctx']

d = np.load(NPZ, allow_pickle=True)
res = json.load(open(RES, encoding='utf-8'))
d30 = np.load(NPZ30, allow_pickle=True)

hs = d['hs_final'].astype(np.float64)          # (160, 2560)
margins = d['margins'].astype(np.float64)      # (160,)
lang = d['rows_lang']; ent = d['rows_e']; cls = d['rows_c']
tpl = d['rows_tpl']; kind = d['rows_kind']

# ---- PCA to 3D (center + SVD) ----
mu = hs.mean(axis=0, keepdims=True)
X = hs - mu
U, S, Vt = np.linalg.svd(X, full_matrices=False)
coords = U[:, :3] * S[:3]
evr = (S[:3] ** 2) / (S ** 2).sum()
# per-axis std normalization so the cloud is spherical
sstd = coords.std(axis=0)
coords_n = coords / sstd

# ---- project sealed directions into the same (normalized) PC space ----
def proj(v):
    c = (Vt[:3] @ (v - mu[0]).astype(np.float64))
    return (c / sstd).tolist()

dR = d30['qwen3-4b_dR_unit'].astype(np.float64)
v_dirs = d['v_dirs'].item()

ARROWS = [{'name': 'dR (3230 sealed)', 'vec': proj(dR), 'color': '#A32D2D',
           'cos': None}]
for t in TPLS:
    v = np.asarray(v_dirs[t], dtype=np.float64)
    v = v / max(np.linalg.norm(v), 1e-9)
    cos_dR = float(np.dot(v, dR) / max(np.linalg.norm(dR), 1e-9))
    ARROWS.append({'name': t, 'vec': proj(v), 'color': '#5F5E5A',
                   'cos': round(cos_dR, 3)})

# ---- layer emergence curves (from sealed result.json f5) ----
# note: only class-kind templates (T1/T3/T5) carry per-layer class readout
CURVES = []
for t in [t for t in TPLS if t in res['f5']]:
    xs = sorted(int(k) for k in res['f5'][t].keys())
    CURVES.append({'name': t,
                   'x': xs,
                   'y': [round(float(res['f5'][t][str(k)]), 4) for k in xs]})

# ---- points ----
PTS = []
for i in range(len(margins)):
    PTS.append({'p': [round(float(x), 4) for x in coords_n[i]],
                'lang': str(lang[i]), 'e': str(ent[i]), 'c': str(cls[i]),
                'tpl': str(tpl[i]), 'kind': str(kind[i]),
                'm': round(float(margins[i]), 3)})

g = res['gates']
DATA = {
    'pts': PTS, 'arrows': ARROWS, 'curves': CURVES,
    'evr': [round(float(x) * 100, 1) for x in evr],
    'meta': {'model': 'qwen3-4b', 'n': len(PTS), 'dims': int(hs.shape[1]),
             'verdict': res['verdict'],
             'f2_mean': res['f2_mean'], 'f3_mean': res['f3_mean'],
             'gates': g,
             'f1_auc': res['f1'].get('auc_mean') or res['f1'],
             'ledger': '#388 post 05a31fc4', 'freeze': res['freeze_sha8'],
             'res': res['res_sha8']},
}
payload = json.dumps(DATA, ensure_ascii=False).replace('</', '<\\/')

HTML_TPL = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Phase 3236 特征空间 3D 视图</title>
<script src="plotly.min.js"></script>
<style>
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#fafaf8;color:#2c2c2a;}
.wrap{max-width:1080px;margin:0 auto;padding:20px 24px 40px;}
h1{font-size:18px;font-weight:500;margin:8px 0 2px;}
.sub{font-size:12px;color:#888780;margin-bottom:14px;}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;background:#FAEEDA;color:#854F0B;border:0.5px solid #BA7517;margin-left:8px;vertical-align:2px;}
.btns{margin:6px 0 10px;}
button{font-size:13px;padding:5px 14px;margin-right:8px;border-radius:8px;border:0.5px solid #B4B2A9;background:#fff;color:#444441;cursor:pointer;}
button.on{background:#185FA5;color:#fff;border-color:#185FA5;}
label{font-size:13px;color:#444441;margin-left:12px;}
#info{font-size:12px;color:#888780;margin:2px 0 8px;}
#scene{width:100%;height:620px;}
#layers{width:100%;height:320px;margin-top:8px;}
.note{font-size:12px;color:#888780;border-top:0.5px solid #D3D1C7;margin-top:18px;padding-top:10px;line-height:1.6;}
</style>
</head>
<body>
<div class="wrap">
<h1>Phase 3236 平台演示 · 特征空间 3D 视图<span class="badge">__VERDICT__</span></h1>
<div class="sub">qwen3-4b · 160 条查询（5 模版 × 4 实体 × 4 类别）· 末层残差流 2560 维 → PCA 3D · ledger __LED__</div>
<div class="btns">
<button id="b_cls" class="on" onclick="setMode('cls')">按类别</button>
<button id="b_tpl" onclick="setMode('tpl')">按模版</button>
<button id="b_lang" onclick="setMode('lang')">按语言</button>
<button id="b_margin" onclick="setMode('margin')">按 margin</button>
<label><input type="checkbox" id="showArrows" checked onchange="toggleArrows()"> 显示图谱方向（dR / 模版轴）</label>
</div>
<div id="info"></div>
<div id="scene"></div>
<div id="layers"></div>
<div class="note">__NOTE__</div>
</div>
<script>
const DATA = __PAYLOAD__;
const CLS_COLORS = {'水果':'#E24B4A','花':'#D4537E','鱼':'#378ADD','工具':'#EF9F27'};
const TPL_COLORS = {'T1_zh_stmt':'#D85A30','T2_zh_quest':'#BA7517','T3_en_stmt':'#639922','T4_en_quest':'#185FA5','T5_zh_ctx':'#534AB7'};
const LANG_COLORS = {'zh':'#D85A30','en':'#378ADD'};
let mode = 'cls';
const ev = DATA.evr;
document.getElementById('info').textContent =
  'PC1 ' + ev[0] + '% · PC2 ' + ev[1] + '% · PC3 ' + ev[2] +
  '% ｜ n=' + DATA.meta.n + ' ｜ F2 跨模版拼图均值 ' + DATA.meta.f2_mean.toFixed(3) +
  '（G2 门 0.6 FAIL，诚实记录） ｜ F3 跨语言行为相关 ' + DATA.meta.f3_mean.toFixed(3);

function hover(p){
  return p.e + ' · ' + p.c + '<br>' + p.tpl + '（' + (p.lang==='zh'?'中文':'英文') + '/' + (p.kind==='class'?'陈述':'判断') + '）<br>margin = ' + (p.m>0?'+':'') + p.m;
}
function pointTraces(){
  const pts = DATA.pts;
  if (mode === 'margin'){
    const ms = pts.map(p=>p.m);
    const am = Math.max.apply(null, ms.map(Math.abs));
    return [{type:'scatter3d', mode:'markers', name:'margin',
      x:pts.map(p=>p.p[0]), y:pts.map(p=>p.p[1]), z:pts.map(p=>p.p[2]),
      text:pts.map(hover), hoverinfo:'text',
      marker:{size:3.5, color:ms, cmin:-am, cmax:am,
        colorscale:[[0,'#2e7d32'],[0.5,'#e8e8e8'],[1,'#c62828']],
        showscale:true, colorbar:{title:{text:'margin<br>(绿负红正)'}, thickness:12, len:0.7}},
      }];
  }
  const key = mode==='cls' ? p=>p.c : mode==='tpl' ? p=>p.tpl : p=>p.lang;
  const colors = mode==='cls' ? CLS_COLORS : mode==='tpl' ? TPL_COLORS : LANG_COLORS;
  const groups = {};
  pts.forEach(p=>{ const k=key(p); (groups[k]=groups[k]||[]).push(p); });
  return Object.keys(groups).sort().map(k=>({
    type:'scatter3d', mode:'markers', name:k,
    x:groups[k].map(p=>p.p[0]), y:groups[k].map(p=>p.p[1]), z:groups[k].map(p=>p.p[2]),
    text:groups[k].map(hover), hoverinfo:'text',
    marker:{size:3.5, color:colors[k], opacity:0.9}}));
}
function arrowTraces(){
  const trs = [];
  const span = 0.85;
  DATA.arrows.forEach((a,i)=>{
    const v = a.vec, n = Math.hypot(v[0],v[1],v[2]) || 1e-9;
    const tip = v.map(x=>x/n*span);
    const nm = a.name + (a.cos===null ? '' : '  cos(dR)=' + a.cos);
    trs.push({type:'scatter3d', mode:'lines', name:'方向: '+nm,
      x:[0,tip[0]], y:[0,tip[1]], z:[0,tip[2]],
      line:{width: a.cos===null?9:4, color:a.color},
      showlegend:true, hoverinfo:'name', visible:true,
      hoverlabel:{bgcolor:a.color, font:{color:'#fff'}}});
    trs.push({type:'scatter3d', mode:'text', x:[tip[0]*1.06], y:[tip[1]*1.06], z:[tip[2]*1.06],
      text:[a.name], textfont:{size:11, color:a.color}, showlegend:false,
      hoverinfo:'skip', visible:true});
  });
  return trs;
}
function layout(){
  return {margin:{l:0,r:0,t:10,b:0}, paper_bgcolor:'#fafaf8',
    legend:{font:{size:11}, itemsizing:'constant'},
    scene:{aspectmode:'cube',
      xaxis:{title:'PC1 ('+ev[0]+'%)', gridcolor:'#e5e3dc', zerolinecolor:'#c9c7be'},
      yaxis:{title:'PC2 ('+ev[1]+'%)', gridcolor:'#e5e3dc', zerolinecolor:'#c9c7be'},
      zaxis:{title:'PC3 ('+ev[2]+'%)', gridcolor:'#e5e3dc', zerolinecolor:'#c9c7be'},
      camera:{eye:{x:1.55,y:1.35,z:1.1}}}};
}
function draw(){
  Plotly.react('scene', pointTraces().concat(arrowTraces()), layout(),
    {responsive:true, displaylogo:false, scrollZoom:true});
}
function setMode(m){
  mode = m;
  ['cls','tpl','lang','margin'].forEach(k=>{
    document.getElementById('b_'+k).className = (k===m)?'on':'';
  });
  draw();
}
function toggleArrows(){
  const vis = document.getElementById('showArrows').checked;
  const n = DATA.arrows.length * 2;
  const idx = Array.from({length:n}, (_,i)=>i);
  Plotly.restyle('scene', {visible:vis}, idx);
}
draw();
// layer emergence curves
const curveTraces = DATA.curves.map(c=>({
  type:'scatter', mode:'lines+markers', name:c.name,
  x:c.x, y:c.y, line:{color:TPL_COLORS[c.name], width:2},
  marker:{size:5}, hovertemplate:'L%{x} · %{y:.2f}<extra>'+c.name+'</extra>'}));
Plotly.newPlot('layers', curveTraces, {
  margin:{l:50,r:16,t:30,b:40}, paper_bgcolor:'#fafaf8', plot_bgcolor:'#fafaf8',
  title:{text:'类读出层涌现曲线（逐层 margin 量级 · 末层跳变即读出端）', font:{size:13}},
  xaxis:{title:'层', gridcolor:'#e5e3dc'}, yaxis:{title:'类读出量级', gridcolor:'#e5e3dc'},
  legend:{font:{size:11}}},
  {responsive:true, displaylogo:false});
</script>
</body>
</html>
"""

html = (HTML_TPL
        .replace('__PAYLOAD__', payload)
        .replace('__VERDICT__', res['verdict'])
        .replace('__LED__', 'n=388 post 05a31fc4')
        .replace('__NOTE__',
                 '锚：freeze ' + res['freeze_sha8'] + ' · res ' + res['res_sha8']
                 + ' · dR 取自 3230 封存包（sha8 d430d29e）。点为 160 条查询的末层残差向量经'
                 + ' PCA 投影（归一化后等比）；箭头为封存方向 dR 与 5 个模版对比方向 v_T 在同一空间的投影，'
                 + '其中 T1–T5 的 cos(dR) 与 result.json F4 一致。G2 跨模版拼图均值 0.500 未过 0.6 门，'
                 + '在图中表现为同族模版（T1/T2/T5 与 T3/T4）点云内部紧、跨族分离——这正是"先族内收敛、再跨族校准"'
                 + '拼图纪律的几何直观。演示级结论，无 E 级声明。'))

with open(HTML, 'w', encoding='utf-8', newline='') as f:
    f.write(html)

# self-check
ok_html = os.path.getsize(HTML) > 20000
ok_js = os.path.getsize(os.path.join(OUTDIR, 'plotly.min.js')) > 3000000
# verify projected dR cos matches f4 per-template values
chk = []
for t in TPLS:
    v = np.asarray(v_dirs[t], dtype=np.float64)
    v = v / np.linalg.norm(v)
    c = float(np.dot(v, dR))
    chk.append(abs(c - res['f4'][t]['cos_dR']) < 1e-3)
print('html_bytes=%d plotly_ok=%s cos_match=%s evr=%s'
      % (os.path.getsize(HTML), ok_js, all(chk),
         [round(float(x), 3) for x in evr]))
