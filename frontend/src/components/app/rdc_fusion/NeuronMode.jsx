/* 神经元级模式：3D 空间精确到单个神经元 —— 单层内部全量展开 + 参数变化双轴
   三功能舱（全部为可独立 hover/选中的点，共 16,384 个 = qwen3-4b 单层全部神经元级单元）：
     舱A 残差流 2,560 dims（蓝系板阵）· 舱B MLP 9,728 SwiGLU units（emerald 簇云，主体）· 舱C attention 4,096 Q-dims（32 heads×128，紫系立方阵）

   「参数变化」双轴设计（固定权重推理中 W 不变，变化的是参数的调用）：
     ① t 轴（token 步）——底部时间轴播放器 + 三种编码模式：
        激活 act(i,t)：该参数单元在当前 token 步被调用的强度
        变化 Δact(t) ：相邻 token 步的差分（正在变化中的神经元）
        写入 w(i,t)  ：写入贡献（MLP: down 列范数 × act → 残差流；attn 写经 o_proj 混合不单列，舱置灰）
     ② ℓ 轴（层）——选中单元后显示跨层基座曲线 L0–L35 + 层扫描播放 ▶L

   语言模板驱动（design/client_template_tech_plan_v1.md §2，M5-P0）：
   token 序列 / 叙事峰位 / 注意力回看 / 焦点特征 / 激活示例全部来自 LANG_TEMPLATES
   （distributedData.js）——本组件不含任何模板叙事字面量，切换模板整体跟随。
   布局与激活为 demo 种子；接入点：torch hook 实测逐 unit 激活替换 actM[i*T+t]（collect.npz /
   预注册设计）；真实聚类布局替换 pts[]；门控曲线替换 gate()；head 注意力分布替换 attn 分布实测。 */
import { useEffect, useRef, useState } from 'react';
import { LANG_TEMPLATES } from './distributedData.js';

function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

const N_LAYERS=36, D_MODEL=2560, D_MLP=9728, HEADS=32, D_HEAD=128, N_Q=HEADS*D_HEAD;
const N_ALL=D_MODEL+D_MLP+N_Q;                     /* 16,384 */
const KIND_RES=0, KIND_MLP=1, KIND_Q=2;

/* 模块级缺省 = 默认语言模板（is_a）的 demo 包；组件内以 tpl 覆盖 */
const DEF_TPL=LANG_TEMPLATES[0];
const TOKENS=DEF_TPL.demo.tokens;
const T=TOKENS.length;
const DEF_CTX={T, tid:DEF_TPL.id, peaks:DEF_TPL.demo.peaks, focusUnit:DEF_TPL.demo.focus.unit, lookback:DEF_TPL.demo.attn_lookback, tokens:TOKENS};
const ENC_MODES=[
  {k:'act', t:'激活',  d:'act(i,t)：参数单元在当前 token 步的调用强度'},
  {k:'diff',t:'变化',  d:'|Δact(t)|：相邻 token 步差分，正在变化中的神经元'},
  {k:'write',t:'写入', d:'写入贡献 w(i,t)：MLP down 列范数 × act → 残差流；attn 写经 o_proj 混合，此舱置灰'},
];

/* 颜色分级（激活语言：低值淡、高值 emerald；蓝=残差流、紫=attn） */
const PALETTE=[
  /* res 蓝 0-4 */ '#cbd5e1','#7dd3fc','#38bdf8','#0284c7','#075985',
  /* mlp  5-9 */ '#cbd5e1','#a7f3d0','#34d399','#10b981','#059669',
  /* q 紫 10-14 */ '#ede9fe','#c4b5fd','#8b5cf6','#7c3aed','#5b21b6',
];
function actLevel(a){ return a>=0.75?4 : a>=0.5?3 : a>=0.25?2 : a>=0.08?1 : 0; }

/* demo 激活基座：L6 的 unit#<focusUnit> = 焦点特征载体；接入点：collect.npz 实测替换 */
function unitAct(l,i,fu=DEF_CTX.focusUnit){
  if(l===6){
    if(i===fu) return 0.97;
    const d=Math.abs(i-fu);
    if(d<=8) return 0.62+0.30*(1-d/9);
    if(i===fu+68||i===fu-74||i===fu+164) return 0.55;
  }
  const r=mulberry(l*7919+i);
  return 0.04+0.30*Math.pow(r(),2.2);
}
function resAct(l,d){ const r=mulberry(500+l*131+d); return 0.03+0.34*Math.pow(r(),2.6); }
function qAct(l,j){ const r=mulberry(900+l*211+j); const h=j>>7; return (h<4?0.22:0.05)+0.5*Math.pow(r(),2.4); }

/* t 轴环境函数：三舱在不同 token 步被调用的叙事曲线（峰位 ← 模板 peaks）
   MLP 峰 = 谓词完成步 · residual 写入峰在其后 · attn 峰 = 回看绑定步 */
function envOf(kind,t,peaks=DEF_CTX.peaks){
  const g=(c,w)=>Math.exp(-((t-c)*(t-c))/w);
  if(kind===KIND_MLP) return 0.35+0.65*g(peaks.mlp,1.2);
  if(kind===KIND_RES) return 0.42+0.58*g(peaks.res,1.6);
  return 0.35+0.65*g(peaks.attn,0.9);
}
function jit(i,t){ const r=mulberry(i*31+t*7+11); return 0.85+0.3*r(); }

/* 点云构建（每层×每模板缓存一次；actM = N×T 激活矩阵，wM = 写入基座） */
const CACHE={};
function buildLayer(l,ctx=DEF_CTX){
  const key=(ctx.tid||'def')+'|'+l;
  if(CACHE[key]) return CACHE[key];
  const tk=ctx.tokens||TOKENS, nT=ctx.T||tk.length;
  const pts=new Float32Array(N_ALL*3);
  const actM=new Float32Array(N_ALL*nT);
  const wM=new Float32Array(N_ALL);
  const kind=new Uint8Array(N_ALL);
  const uid=new Uint16Array(N_ALL);
  const rnd=mulberry(4200+l*17);
  const gauss=()=>{ let u=0,v=0; while(u===0)u=rnd(); while(v===0)v=rnd(); return Math.sqrt(-2*Math.log(u))*Math.cos(6.2832*v); };
  let p=0;
  /* 舱A 残差流：40 列 × 64 行板阵（x∈[-3.45,-2.61]，y∈[-1.28,1.28]） */
  for(let d=0;d<D_MODEL;d++){
    const c=d%40, r=(d/40)|0;
    pts[p*3]=-3.42+c*0.0208; pts[p*3+1]=1.28-r*0.0404; pts[p*3+2]=0;
    kind[p]=KIND_RES; uid[p]=d;
    for(let t=0;t<nT;t++) actM[p*nT+t]=Math.min(1,resAct(l,d)*envOf(KIND_RES,t,ctx.peaks)*jit(d,t));
    wM[p]=0.9; p++;   /* res 接收写入：write 模式显示 act 的接收侧 */
  }
  /* 舱B MLP：8 簇环状点云（demo 激活桶；接入点：真实聚类布局） */
  const bucket=i=>{ const r=mulberry(i*2654435761>>>0); return (r()*8)|0; };
  for(let i=0;i<D_MLP;i++){
    const b=bucket(i), ang=b/8*6.2832;
    const cx=Math.cos(ang)*1.18, cz=Math.sin(ang)*1.18;
    pts[p*3]=cx+gauss()*0.24; pts[p*3+1]=gauss()*0.30; pts[p*3+2]=cz+gauss()*0.24;
    kind[p]=KIND_MLP; uid[p]=i;
    for(let t=0;t<nT;t++) actM[p*nT+t]=Math.min(1,unitAct(l,i,ctx.focusUnit)*envOf(KIND_MLP,t,ctx.peaks)*jit(i,t));
    const rw=mulberry(l*7+i); wM[p]=0.25+0.75*rw(); p++;   /* down 列范数基座（demo） */
  }
  /* 舱C attention：32 head 立方（8×4 阵，每 head 128 dims=8×8×2），x∈[2.35,3.55] */
  for(let j=0;j<N_Q;j++){
    const h=j>>7, d=j&127;
    const hc=h%8, hr=(h/8)|0;
    pts[p*3]=2.40+hc*0.328+(d%4-1.5)*0.031;
    pts[p*3+1]=1.02-hr*0.66+(((d>>3)|0)%8-3.5)*0.030;
    pts[p*3+2]=(hr-1.5)*0.16+((d>>6)?0.05:-0.05);
    kind[p]=KIND_Q; uid[p]=j;
    for(let t=0;t<nT;t++) actM[p*nT+t]=Math.min(1,qAct(l,j)*envOf(KIND_Q,t,ctx.peaks)*jit(j,t));
    wM[p]=0; p++;   /* Q-dim 不直接写入残差流（写经 o_proj 混合），write 模式置灰 */
  }
  const data={pts,actM,wM,kind,uid,nT};
  CACHE[key]=data; return data;
}

/* demo 参数统计（确定性；接入点：真实权重切片统计） */
function paramStats(l,kindU,uid){
  const r=mulberry(l*100003+kindU*997+uid);
  const L2=(2.2+r()*6).toFixed(2), mean=((r()-0.5)*0.02).toFixed(4),
        mx=(0.4+r()*1.6).toFixed(3), mn=(-0.4-r()*1.6).toFixed(3);
  const vals=[...Array(8)].map(()=>+((r()-0.5)*1.4).toFixed(4));
  return {L2,mean,mx,mn,vals};
}

/* 时变曲线（面板用；与 3D actM 同源公式；ctx 携带 T / peaks / focusUnit） */
function unitCurve(l,i,ctx=DEF_CTX){ return [...Array(ctx.T)].map((_,t)=>Math.min(1,unitAct(l,i,ctx.focusUnit)*envOf(KIND_MLP,t,ctx.peaks)*jit(i,t))); }
function gateCurve(l,i,ctx=DEF_CTX){ const c=unitCurve(l,i,ctx); return c.map((v,t)=>v*(0.45+0.55*Math.abs(Math.sin(i*0.7+t*0.9)))); }
function writeCurve(l,i,ctx=DEF_CTX){ const rw=mulberry(l*7+i), w=0.25+0.75*rw(); return unitCurve(l,i,ctx).map(v=>v*w); }
function layerCurve(i,fu=DEF_CTX.focusUnit){ return [...Array(N_LAYERS)].map((_,l)=>unitAct(l,i,fu)); }
function resCurve(l,d,ctx=DEF_CTX){ return [...Array(ctx.T)].map((_,t)=>Math.min(1,resAct(l,d)*envOf(KIND_RES,t,ctx.peaks)*jit(d,t))); }
function qCurve(l,j,ctx=DEF_CTX){ return [...Array(ctx.T)].map((_,t)=>Math.min(1,qAct(l,j)*envOf(KIND_Q,t,ctx.peaks)*jit(j,t))); }

/* head 注意力分布（demo：t≥from 的步 head 回看 tokens[to]（← 模板 attn_lookback）；接入点：attn 实测） */
function attnDist(head,t,l,ctx=DEF_CTX){
  const nT=ctx.T, lb=ctx.lookback||{from:2,to:0};
  const sc=[...Array(nT)].map((_,j)=>{
    const r=mulberry(head*97+j*13+t+l);
    if(j===lb.to&&t>=lb.from) return 2.4+r()*0.4;
    if(j===t) return 1.2+r()*0.3;
    return 0.25+r()*0.4;
  });
  const mx=Math.max(...sc), e=sc.map(v=>Math.exp(v-mx)), s=e.reduce((a,b)=>a+b,0);
  return e.map(v=>v/s);
}

export default function NeuronMode({layer,onLayer,onOpenParam,onBack,tpl}){
  const tplDef=tpl||DEF_TPL;
  const demo=tplDef.demo;
  const tokens=demo.tokens;
  const ctx={T:tokens.length, tid:tplDef.id, peaks:demo.peaks, focusUnit:demo.focus.unit, lookback:demo.attn_lookback, tokens};
  const focus=demo.focus;
  const cvRef=useRef(null), tipRef=useRef(null);
  const camRef=useRef({th:0.62,ph:0.34,zoom:880,auto:true,panX:0,panY:0});
  const dataRef=useRef(null), scrRef=useRef(null), hoverRef=useRef(-1);
  const dragRef=useRef(null);
  const [sel,setSel]=useState(null);            /* {kind,uid} */
  const selRef=useRef(null);
  const [readout,setReadout]=useState({th:36,ph:19,panX:0,panY:0});
  const [t,setT]=useState(Math.min(3,tokens.length-1));  /* 当前 token 步 */
  const [play,setPlay]=useState(false);         /* t 轴播放 */
  const [enc,setEnc]=useState('act');           /* 编码模式 act/diff/write */
  const [scan,setScan]=useState(false);         /* ℓ 轴层扫描 */
  const tRef=useRef(t), encRef=useRef(enc);
  const ctxRef=useRef(ctx);

  useEffect(()=>{ selRef.current=sel; },[sel]);
  useEffect(()=>{ setSel(null); },[layer,tplDef.id]);   /* 换层/换模板清选中 */
  useEffect(()=>{ tRef.current=t; },[t]);
  useEffect(()=>{ encRef.current=enc; },[enc]);
  useEffect(()=>{ ctxRef.current=ctx; });               /* 绘制循环读最新模板上下文 */
  useEffect(()=>{ setT(v=>v<ctx.T?v:Math.max(0,ctx.T-1)); },[ctx.T]);   /* 换模板钳制 t 步 */
  /* t 轴播放：700ms/步 */
  useEffect(()=>{
    if(!play) return;
    const id=setInterval(()=>setT(v=>(v+1)%ctx.T),700);
    return ()=>clearInterval(id);
  },[play,ctx.T]);
  /* ℓ 轴扫描：800ms/层循环 */
  useEffect(()=>{
    if(!scan) return;
    const id=setInterval(()=>onLayer((layer+1)%N_LAYERS),800);
    return ()=>clearInterval(id);
  },[scan,layer,onLayer]);

  useEffect(()=>{
    const cv=cvRef.current; if(!cv) return;
    const ctx=cv.getContext('2d');
    const cam=camRef.current;
    dataRef.current=buildLayer(layer,ctxRef.current);
    scrRef.current={x:new Float32Array(N_ALL),y:new Float32Array(N_ALL),z:new Float32Array(N_ALL)};
    const oidx=new Uint16Array(N_ALL); for(let i=0;i<N_ALL;i++) oidx[i]=i;
    const vv=new Float32Array(N_ALL);       /* 当前帧编码值 */
    const lv=new Uint8Array(N_ALL);         /* 当前帧 PALETTE 桶 */

    function resize(){ const dpr=window.devicePixelRatio||1; cv.width=cv.clientWidth*dpr; cv.height=cv.clientHeight*dpr; }
    resize();

    let raf=0, pulse=0;
    function draw(){
      const dpr=window.devicePixelRatio||1;
      ctx.clearRect(0,0,cv.width,cv.height);
      if(cam.auto&&!dragRef.current) cam.th+=0.0011;
      pulse+=0.05;
      const D=dataRef.current, c=ctxRef.current;
      const nT=c.T;
      const tNow=tRef.current%nT, mode=encRef.current;
      const ct=Math.cos(cam.th),st=Math.sin(cam.th),cp=Math.cos(cam.ph),sp=Math.sin(cam.ph);
      const f=cam.zoom*(cv.height/900), cx=cv.width/2+(camRef.current.panX||0)*dpr, cy=cv.height*0.5+(camRef.current.panY||0)*dpr;
      const sx=scrRef.current.x, sy=scrRef.current.y, dep=scrRef.current.z;
      /* 投影 */
      for(let i=0;i<N_ALL;i++){
        const x=D.pts[i*3], y=D.pts[i*3+1], z=D.pts[i*3+2];
        const x1=x*ct+z*st, z1=-x*st+z*ct;
        const y2=y*cp-z1*sp, z2=y*sp+z1*cp;
        const d=6+z2*0.5, ff=f/d;
        sx[i]=cx+x1*ff*dpr; sy[i]=cy-y2*ff*dpr; dep[i]=z2;
      }
      /* 编码值 → 颜色桶（参数变化本体：每帧随 t/编码模式重算） */
      for(let i=0;i<N_ALL;i++){
        const a=D.actM[i*nT+tNow];
        let v;
        if(mode==='act') v=a;
        else if(mode==='diff') v=tNow>0?Math.abs(a-D.actM[i*nT+tNow-1])*2.6:0;
        else v=D.kind[i]===KIND_MLP?D.wM[i]*a:(D.kind[i]===KIND_RES?a*0.9:0);
        vv[i]=v; lv[i]=D.kind[i]*5+actLevel(v);
      }
      /* 深度排序（远→近） */
      oidx.sort((a,b)=>dep[a]-dep[b]);
      /* 按 PALETTE 级别分桶绘制（同色一次 fill，16384 点 / 15 色） */
      const buckets=PALETTE.map(()=>[]);
      for(let k=0;k<oidx.length;k++){ const i=oidx[k]; buckets[lv[i]].push(i); }
      const pt=2.1*(cam.zoom/880)*dpr;
      buckets.forEach((list,bIdx)=>{
        if(!list.length) return;
        ctx.fillStyle=PALETTE[bIdx];
        ctx.beginPath();
        const r=(bIdx%5)>=3?pt*1.15:pt*0.85;
        for(let k=0;k<list.length;k++){ const i=list[k]; ctx.rect(sx[i]-r,sy[i]-r,r*2,r*2); }
        ctx.fill();
      });
      /* 高编码值 top 点白描边（≥0.75） */
      ctx.strokeStyle='rgba(255,255,255,.75)'; ctx.lineWidth=0.8*dpr;
      for(let k=0;k<oidx.length;k++){ const i=oidx[k]; if(vv[i]>=0.75){ ctx.strokeRect(sx[i]-pt,sy[i]-pt,pt*2,pt*2); } }
      /* hover 光环 */
      if(hoverRef.current>=0){
        const i=hoverRef.current;
        ctx.strokeStyle='rgba(2,132,199,.9)'; ctx.lineWidth=1.4*dpr;
        ctx.beginPath(); ctx.arc(sx[i],sy[i],pt*2.6,0,6.283); ctx.stroke();
      }
      /* 选中：脉冲环 */
      if(selRef.current){
        const {kind,uid}=selRef.current;
        let target=-1;
        if(kind===KIND_RES) target=uid;
        else if(kind===KIND_MLP) target=D_MODEL+uid;
        else target=D_MODEL+D_MLP+uid;
        const r=(pt*3+Math.sin(pulse)*1.4*dpr);
        ctx.strokeStyle='#0ea5e9'; ctx.lineWidth=1.8*dpr;
        ctx.beginPath(); ctx.arc(sx[target],sy[target],r,0,6.283); ctx.stroke();
        ctx.strokeStyle='rgba(14,165,233,.35)'; ctx.lineWidth=1*dpr;
        ctx.beginPath(); ctx.arc(sx[target],sy[target],r*1.9,0,6.283); ctx.stroke();
      }
      /* 舱标注 */
      ctx.font=(11*dpr)+'px ui-monospace,Consolas,monospace';
      const lab=[
        [-3.0,-1.62,0,'残差流 2,560 dims'],
        [0,-1.95,0,'MLP 9,728 units · SwiGLU'],
        [2.97,-1.62,0,'attention 4,096 Q-dims · 32h×128'],
      ];
      if(mode==='write'){
        lab[2]=[-3.0,-1.92,0,'write 模式：attn 写经 o_proj 混合，此舱置灰'];
      }
      lab.forEach(([x,y,z,txt])=>{
        const x1=x*ct+z*st, z1=-x*st+z*ct;
        const y2=y*cp-z1*sp, z2=y*sp+z1*cp, ff=f/(6+z2*0.5);
        ctx.fillStyle='rgba(71,85,105,.9)';
        ctx.fillText(txt,cx+x1*ff*dpr,cy-y2*ff*dpr);
      });
      raf=requestAnimationFrame(draw);
    }
    raf=requestAnimationFrame(draw);
    const tId=setInterval(()=>setReadout({th:Math.round(cam.th*180/Math.PI)%360,ph:Math.round(cam.ph*180/Math.PI),panX:Math.round(cam.panX||0),panY:Math.round(cam.panY||0)}),500);

    /* 最近点查找 */
    function nearest(mx,my){
      const sx=scrRef.current.x, sy=scrRef.current.y;
      let best=-1, bd=64*(window.devicePixelRatio||1); bd*=bd;
      for(let i=0;i<N_ALL;i++){
        const dx=sx[i]-mx, dy=sy[i]-my, d=dx*dx+dy*dy;
        if(d<bd){bd=d;best=i;}
      }
      return best;
    }
    function toKind(i){ const D=dataRef.current; return {kind:D.kind[i],uid:D.uid[i],v:vv[i]}; }
    const onMove=e=>{
      const r=cv.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
      const mx=(e.clientX-r.left)*dpr, my=(e.clientY-r.top)*dpr;
      if(dragRef.current){
        if(dragRef.current.btn===2){
          /* 右键：平移视角位置（上下左右皆可），不改变任何角度 */
          cam.panX=Math.max(-800,Math.min(800,(cam.panX||0)+(e.clientX-dragRef.current.x)));
          cam.panY=Math.max(-800,Math.min(800,(cam.panY||0)+(e.clientY-dragRef.current.y)));
        }else{
          /* 左键：旋转视角，上下左右都改变角度 */
          cam.th+=(e.clientX-dragRef.current.x)*0.006;
          cam.ph=Math.max(-0.1,Math.min(1.15,cam.ph+(e.clientY-dragRef.current.y)*0.006));
        }
        dragRef.current={x:e.clientX,y:e.clientY,btn:dragRef.current.btn};
        hoverRef.current=-1; if(tipRef.current) tipRef.current.style.display='none';
        return;
      }
      const i=nearest(mx,my);
      hoverRef.current=i;
      if(i>=0&&tipRef.current){
        const k=toKind(i);
        const nm=k.kind===KIND_RES?('residual dim '+k.uid):k.kind===KIND_MLP?('MLP unit #'+k.uid):('Q-dim h'+((k.uid>>7)|0)+'·d'+(k.uid&127));
        const md=ENC_MODES.find(m=>m.k===encRef.current);
        const c=ctxRef.current;
        tipRef.current.textContent=nm+' · '+md.t+' '+k.v.toFixed(2)+' · t='+(tRef.current%c.T)+'「'+(c.tokens||TOKENS)[tRef.current%c.T]+'」';
        tipRef.current.style.display='block';
        tipRef.current.style.left=(e.clientX-r.left+12)+'px';
        tipRef.current.style.top=(e.clientY-r.top-8)+'px';
        cv.style.cursor='pointer';
      }else{ if(tipRef.current) tipRef.current.style.display='none'; cv.style.cursor='grab'; }
    };
    let autoT=null;
    const onDown=e=>{
      if(e.button!==0&&e.button!==2) return;
      dragRef.current={x:e.clientX,y:e.clientY,btn:e.button};
      cam.auto=false;
      if(autoT){clearTimeout(autoT);autoT=null;}   /* 取消旧的恢复定时器，防止拖拽中 auto-rotate 被唤醒 */
    };
    const onUp=e=>{
      if(dragRef.current){
        if(dragRef.current.btn===0){   /* 仅左键做点选 */
          const moved=Math.abs(e.clientX-dragRef.current.x)+Math.abs(e.clientY-dragRef.current.y);
          if(moved<5){
            const r=cv.getBoundingClientRect(), dpr=window.devicePixelRatio||1;
            const i=nearest((e.clientX-r.left)*dpr,(e.clientY-r.top)*dpr);
            if(i>=0){ const k=toKind(i); setSel({kind:k.kind,uid:k.uid}); }
            else setSel(null);
          }
        }
      }
      dragRef.current=null; autoT=setTimeout(()=>{cam.auto=true;},2500);
    };
    const onWheel=e=>{ e.preventDefault(); cam.zoom=Math.max(220,Math.min(3200,cam.zoom-e.deltaY*1.2)); };
    const onKey=e=>{ if(e.key==='Escape') setSel(null); };
    const onCtx=e=>e.preventDefault();
    window.addEventListener('keydown',onKey);
    cv.addEventListener('mousedown',onDown);
    cv.addEventListener('contextmenu',onCtx);
    window.addEventListener('mousemove',onMove);
    window.addEventListener('mouseup',onUp);
    cv.addEventListener('wheel',onWheel,{passive:false});
    window.addEventListener('resize',resize);
    return ()=>{
      cancelAnimationFrame(raf); clearInterval(tId);
      window.removeEventListener('keydown',onKey);
      cv.removeEventListener('mousedown',onDown);
      cv.removeEventListener('contextmenu',onCtx);
      window.removeEventListener('mousemove',onMove);
      window.removeEventListener('mouseup',onUp);
      cv.removeEventListener('wheel',onWheel);
      window.removeEventListener('resize',resize);
    };
  },[layer,tplDef.id]);

  const fu=ctx.focusUnit;
  return (
    <div className="fw-sp-row">
      <div className="fw-sp-canvasbox">
        <canvas ref={cvRef} className="fw-sp-canvas"/>
        <div ref={tipRef} className="fw-neu-tip" style={{display:'none'}}/>
        <div className="fw-neu-nav">
          <button className="fw-tbtn" onClick={onBack}>← 层平铺</button>
          <div className="fw-neu-lsel">
            <button className="fw-tbtn" onClick={()=>onLayer((layer+35)%36)} disabled={layer<=0}>{'<'}</button>
            <span className="fw-neu-cur">L{layer}</span>
            <button className="fw-tbtn" onClick={()=>onLayer((layer+1)%36)} disabled={layer>=35}>{'>'}</button>
            <button className={'fw-tbtn'+(scan?' on':'')} onClick={()=>setScan(v=>!v)} title="自动层扫描 L0→L35（ℓ 轴）">{scan?'⏸':'▶L'}</button>
          </div>
        </div>
        <div className="fw-sp-legend" style={{top:14,bottom:'auto',right:14,left:'auto'}}>
          <div className="li"><span className="fw-swatch" style={{background:'#059669',borderRadius:'50%'}}/>编码值高（emerald 梯度）</div>
          <div className="li"><span className="fw-swatch" style={{background:'#cbd5e1',borderRadius:'50%'}}/>低值基线</div>
        </div>
        <div className="fw-sp-readout">
          cam <b>θ {readout.th}° · φ {readout.ph}°</b> · 平移 <b>{readout.panX>0?'+':''}{readout.panX},{readout.panY>0?'+':''}{readout.panY}px</b> · <b>n={N_ALL.toLocaleString('en-US')} 点</b>
          {layer===6&&<span style={{color:'#059669'}}><br/>L6 · MLP#{fu} = {focus.id} 载体（emerald 簇）</span>}
        </div>
        {/* t 轴时间线（token 步播放器 + 编码模式） */}
        <div className="fw-neu-timeline">
          <button className={'fw-tbtn'+(play?' on':'')} onClick={()=>setPlay(v=>!v)} title="播放 token 步">{play?'⏸':'⏵'}</button>
          <div className="tl-toks">
            {tokens.map((tk,j)=>(
              <button key={j} className={'chip'+(j===t?' on':'')} onClick={()=>setT(j)} title={'t='+j}>{tk}</button>
            ))}
          </div>
          <span className="tl-pos">t={t+1}/{ctx.T}</span>
          <div className="fw-neu-enc">
            {ENC_MODES.map(m=>(
              <button key={m.k} className={enc===m.k?'on':''} onClick={()=>setEnc(m.k)} title={m.d}>{m.t}</button>
            ))}
          </div>
        </div>
      </div>

      {/* 右侧：神经元详情 / 舱总览 */}
      <aside className="fw-lp">
        {!sel?(
          <>
            <div className="fw-lp-hd">层内神经元总览 <small>L{layer} · {N_ALL.toLocaleString('en-US')} 单元</small></div>
            <div className="fw-lp-cfg">
              <span className="k">MLP units</span><span className="v">9,728 · SwiGLU</span>
              <span className="k">attention</span><span className="v">4,096 Q-dims = 32 heads × 128</span>
              <span className="k">残差流</span><span className="v">2,560 dims</span>
              <span className="k">层参数合计</span><span className="v">100.94M</span>
            </div>
            <div className="sec">参数映射 · 单元 → 权重<span>ALL</span></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">MLP unit #i</span><span className="pv">7,680 参</span></div>
              <div className="shp">up_proj[i,:] + gate_proj[i,:] + down_proj[:,i]，各 [2560]</div></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">Q-dim (h,d)</span><span className="pv">2,560 参</span></div>
              <div className="shp">q_proj[h·128+d,:] [2560] + q_proj.bias[h·128+d]</div></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">residual dim d</span><span className="pv">接收写</span></div>
              <div className="shp">← o_proj[:,d]（attn 写）+ down_proj[:,d]（MLP 写）+ 残差携带</div></div>
            <div className="sec">参数变化 · 双轴<span>t / ℓ</span></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">① t 轴 · token 步</span><span className="pv">{ctx.T} 步</span></div>
              <div className="shp">底部时间轴播放：激活 / 变化 / 写入 三种编码随 token 步刷新 3D 点色</div></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">② ℓ 轴 · 层扫描</span><span className="pv">L0–L35</span></div>
              <div className="shp">▶L 自动扫层；选中单元后显示跨层基座曲线（跨层=不同参数组的对比）</div></div>
            <div className="fw-lp-note">固定权重推理中参数 W 不变；「变化」= 参数被调用的时变（激活/门控/写入）与跨层参数组对比。<br/>点击任意点 = 单个神经元。接入点：torch hook 实测 actM[i,t]。</div>
            <div className="sec">研究标注</div>
            <div className="bdg-row">
              {layer===6&&<span className="fw-bdg" style={{background:'#0596691a',color:'#059669',border:'1px solid #05966955'}}>L6 · MLP#{fu} ↔ {focus.id}（候选）</span>}
              <span className="fw-bdg" style={{background:'#0d94881a',color:'#0d9488',border:'1px solid #0d948855'}}>写端峰值带 L6–L11 · P8–P11</span>
            </div>
          </>
        ):sel.kind===KIND_MLP?(
          <>
            <div className="fw-lp-hd"><span>MLP unit #{sel.uid}</span><button className="fw-tbtn" onClick={()=>setSel(null)}>×</button></div>
            <div className="fw-lp-cfg">
              <span className="k">位置</span><span className="v">L{layer} · mlp 中间层（SwiGLU）</span>
              <span className="k">t={t}「{tokens[t]}」激活</span><span className="v" style={{color:sel.uid===fu&&layer===6?'#059669':'inherit'}}>{unitCurve(layer,sel.uid,ctx)[t].toFixed(2)}</span>
              <span className="k">参数合计</span><span className="v">3 × [2560] = 7,680</span>
            </div>
            <div className="sec">时变曲线 · t 轴（SwiGLU 三段联动）<span>demo</span></div>
            <Spark curves={[
              {c:'#059669',v:unitCurve(layer,sel.uid,ctx)},
              {c:'#0d9488',v:gateCurve(layer,sel.uid,ctx)},
              {c:'#0284c7',v:writeCurve(layer,sel.uid,ctx)},
            ]}/>
            <div className="fw-spark-lg">
              <span><i style={{background:'#059669'}}/>act (up×gate)</span>
              <span><i style={{background:'#0d9488'}}/>gate 门控</span>
              <span><i style={{background:'#0284c7'}}/>write → 残差流</span>
            </div>
            <div className="sec">跨层基座 · ℓ 轴（L0–L35）<span>跨层参数组</span></div>
            <Spark curves={[{c:'#7c3aed',v:layerCurve(sel.uid,ctx.focusUnit)}]}/>
            <div className="fw-spark-lg"><span><i style={{background:'#7c3aed'}}/>unitAct(ℓ, #{sel.uid}) · 写端峰值带 L6–L11</span></div>
            <div className="sec">up_proj.weight[{sel.uid},:]<span>[2560]</span></div>
            <ValGrid s={paramStats(layer,1,sel.uid*3)}/>
            <div className="sec">gate_proj.weight[{sel.uid},:]<span>[2560]</span></div>
            <ValGrid s={paramStats(layer,1,sel.uid*3+1)}/>
            <div className="sec">down_proj.weight[:,{sel.uid}]<span>[2560]</span></div>
            <ValGrid s={paramStats(layer,1,sel.uid*3+2)}/>
            <div className="sec">激活示例 · demo token</div>
            <TokRows unit={sel.uid} layer={layer} actLines={demo.act_lines} focusUnit={fu}/>
            <div className="sec">研究关联</div>
            <div className="bdg-row">
              {layer===6&&sel.uid===fu
                ?<span className="fw-bdg" style={{background:'#0596691a',color:'#059669',border:'1px solid #05966955'}}>候选特征 {focus.id} · {focus.label}{tplDef.status==='measured'?' · mech_evidence':''}</span>
                :<span className="dim">暂无登记（接入 SAE/ledger 后自动点亮）</span>}
            </div>
            <div className="act">
              <span className="total">L{layer} · MLP#{sel.uid}</span>
              <button className="fw-tbtn primary" onClick={()=>onOpenParam&&onOpenParam(layer)}>在参数热图查看 →</button>
            </div>
          </>
        ):sel.kind===KIND_Q?(
          <>
            <div className="fw-lp-hd"><span>Q-dim h{(sel.uid>>7)|0}·d{sel.uid&127}</span><button className="fw-tbtn" onClick={()=>setSel(null)}>×</button></div>
            <div className="fw-lp-cfg">
              <span className="k">位置</span><span className="v">L{layer} · head {(sel.uid>>7)|0} / 32 · dim {sel.uid&127}/128</span>
              <span className="k">GQA 组</span><span className="v">kv head {((sel.uid>>7)|0)>>2} / 8</span>
              <span className="k">t={t}「{tokens[t]}」激活</span><span className="v">{qCurve(layer,sel.uid,ctx)[t].toFixed(2)}</span>
            </div>
            <div className="sec">时变曲线 · t 轴<span>demo</span></div>
            <Spark curves={[{c:'#7c3aed',v:qCurve(layer,sel.uid,ctx)}]}/>
            <div className="fw-spark-lg"><span><i style={{background:'#7c3aed'}}/>q-dim 激活(ℓ={layer}) · 随 token 步</span></div>
            <div className="sec">head {(sel.uid>>7)|0} 注意力分布 · t={t}「{tokens[t]}」<span>demo</span></div>
            <AttnBars head={(sel.uid>>7)|0} t={t} layer={layer} ctx={ctx}/>
            <div className="fw-lp-note">注意力权重随 token 步变化：本 demo 中 t≥{ctx.lookback.from} 的步 head 回看「{tokens[ctx.lookback.to]}」{ctx.lookback.note}。<br/>接入点：真实 attn 分布（softmax(q·k/√128)）。</div>
            <div className="sec">q_proj.weight[{sel.uid},:]<span>[2560]</span></div>
            <ValGrid s={paramStats(layer,2,sel.uid)}/>
            <div className="sec">q_proj.bias[{sel.uid}]<span>[1]</span></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">bias 值（demo）</span><span className="pv">{(paramStats(layer,2,sel.uid).vals[0]/40).toFixed(4)}</span></div></div>
          </>
        ):(
          <>
            <div className="fw-lp-hd"><span>Residual dim {sel.uid}</span><button className="fw-tbtn" onClick={()=>setSel(null)}>×</button></div>
            <div className="fw-lp-cfg">
              <span className="k">位置</span><span className="v">L{layer} · 残差流 {sel.uid}/2560</span>
              <span className="k">t={t}「{tokens[t]}」值</span><span className="v">{resCurve(layer,sel.uid,ctx)[t].toFixed(2)}</span>
              <span className="k">自有参数</span><span className="v">0（状态量）</span>
            </div>
            <div className="sec">时变曲线 · t 轴<span>demo</span></div>
            <Spark curves={[{c:'#0284c7',v:resCurve(layer,sel.uid,ctx)}]}/>
            <div className="fw-spark-lg"><span><i style={{background:'#0284c7'}}/>残差流值(d, ℓ={layer}) · 写入峰在「{tokens[ctx.peaks.res]}」后一步</span></div>
            <div className="sec">写入通道 · 谁写这个坐标</div>
            <div className="fw-lp-row"><div className="top"><span className="nm">← o_proj.weight[:,{sel.uid}]</span><span className="pv">[4096]</span></div>
              <div className="shp">本层 32 heads 输出对该维的写入混合（attn 写）</div></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">← down_proj.weight[:,{sel.uid}]</span><span className="pv">[9728]</span></div>
              <div className="shp">本层 9,728 个 MLP unit 对该维的写入混合（MLP 写）</div></div>
            <div className="fw-lp-row"><div className="top"><span className="nm">← 残差携带</span><span className="pv">恒等</span></div>
              <div className="shp">L{layer-1>=0?layer-1:0} 及更早层的写入沿残差流传入</div></div>
            <ValGrid s={paramStats(layer,0,sel.uid)}/>
            <div className="fw-lp-note">残差流坐标不是参数而是状态量：其值 = 上游全部写入的代数和，随 token 步演化。<br/>接入点：P8–P11 来源贡献分解（写入端分布式，MLP 0.472）。</div>
          </>
        )}
      </aside>
    </div>
  );
}

function ValGrid({s}){
  return (
    <div className="fw-neu-stats">
      <span>L2 <b>{s.L2}</b></span><span>mean <b>{s.mean}</b></span><span>max <b>{s.mx}</b></span><span>min <b>{s.mn}</b></span>
      <div className="vals">{s.vals.map((v,i)=><i key={i} style={{background:v>0?'rgba(2,132,199,'+(0.15+v*0.5)+')':'rgba(109,40,217,'+(0.15-v*0.5)+')'}} title={'w['+i+'] = '+v}/>)}</div>
    </div>
  );
}
function Spark({curves,height=46}){
  const W=300,H=height;
  const all=curves.flatMap(c=>c.v);
  const mx=Math.max(0.001,...all);
  return (
    <svg className="fw-neu-spark" viewBox={'0 0 '+W+' '+H} preserveAspectRatio="none">
      <line x1="0" y1={H-1} x2={W} y2={H-1} stroke="#e2e8f0" strokeWidth="1"/>
      {curves.map((c,ci)=>{
        const pts=c.v.map((v,i)=>((i/(c.v.length-1))*W).toFixed(1)+','+(H-3-(v/mx)*(H-8)).toFixed(1)).join(' ');
        return <polyline key={ci} points={pts} fill="none" stroke={c.c} strokeWidth="1.6" strokeLinejoin="round"/>;
      })}
    </svg>
  );
}
function AttnBars({head,t,layer,ctx}){
  const dist=attnDist(head,t,layer,ctx);
  return (
    <div className="fw-neu-attn">
      {ctx.tokens.map((tk,j)=>(
        <div className="bar-row" key={j}>
          <span className={'tk'+(j===t?' cur':'')}>{tk}</span>
          <div className="bar"><i style={{width:(dist[j]*100).toFixed(1)+'%'}}/></div>
          <span className="pv">{dist[j].toFixed(2)}</span>
        </div>
      ))}
    </div>
  );
}
function TokRows({unit,layer,actLines,focusUnit}){
  const fixed=(layer===6&&actLines&&(unit===focusUnit||Math.abs(unit-focusUnit)<=8));
  const rndToks=['的','在','模型','层','激活','特征','向量','token','残差','单元','权重','映射'];
  const rows = fixed ? actLines : (()=>{ const r=mulberry(unit*131+layer);
    return [...Array(2)].map(()=>[...Array(8)].map((_,j)=>[rndToks[(r()*rndToks.length)|0],j===((r()*8)|0)?0.4+r()*0.55:r()*0.1])); })();
  return (
    <div className="fw-neu-toks">
      {rows.map((row,i)=>(
        <div className="row" key={i}>
          {row.map(([tok,v],j)=>(
            <span key={j} style={{background:'rgba(52,211,153,'+(v>0.02?Math.max(0.06,v).toFixed(2):'0')+')',color:v>0.6?'#022c22':v>0.02?'#064e3b':'#94a3b8'}}>{tok}</span>
          ))}
        </div>
      ))}
    </div>
  );
}
