/* 层平铺全览模式（默认）：层阵排布参考旧版 DNNAnalysis3D（层沿信息流方向横排），
   层结构参考旧版 MultiLayer3DVisualization（半透明玻璃盒 + 盒内激活节点 + 层内连线 + 层间流动）
   —— qwen3-4b 全部 36 层平放为玻璃盒阵列（L0 → L35），残差流水平贯穿，流脉冲沿主干移动。
   右侧常驻面板 = 旁边模型：默认显示模型总览（config + 全参数树）；
   点击层盒 → 显示该层内部结构与全部参数（10 权重 + 4 偏置：形状/计数/占比）。
   研究标注（真实结论映射）：REACH 层集（P16 示意）、写端峰值带 L6–L11（P8–P11 · MLP 0.472）、
   L6 = F#3734 载体层、顶层读位槽 G−1=5（P4–P7）。层集成员为示意，接入点：P12–P16 结果表。 */
import { useEffect, useRef, useState } from 'react';

function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

const N=36;
const reachDemo=l=>l>=8;                 /* 接入点：P16 REACH={ℓ:ρ≥0.10} 结果表 */
const writePeak=l=>l>=6&&l<=11;          /* 接入点：P8–P11 写入端分布式结果 */
const HEADS=32, KVH=8, D_HEAD=128, D_MODEL=2560, D_MLP=9728;
const S=0.40, BX=0.17, BZ=0.30, BH=0.50; /* 层间距 / 盒 x 半宽 / 盒 z 半深 / 盒高 */

/* Qwen3-4B 单层全部参数（公开架构推导：q/k/v 带 bias，o_proj 与 MLP 无 bias，tied embeddings） */
const LAYER_GROUPS=[
  {k:'NORM · RMSNorm ×2',rows:[
    {n:'input_layernorm.weight',shape:'[2560]',p:2560},
    {n:'post_attention_layernorm.weight',shape:'[2560]',p:2560},
  ]},
  {k:'ATTENTION · GQA 32h / 8kv',rows:[
    {n:'self_attn.q_proj.weight',shape:'[4096, 2560]',p:10485760},
    {n:'self_attn.q_proj.bias',shape:'[4096]',p:4096,sub:true},
    {n:'self_attn.k_proj.weight',shape:'[1024, 2560]',p:2621440},
    {n:'self_attn.k_proj.bias',shape:'[1024]',p:1024,sub:true},
    {n:'self_attn.v_proj.weight',shape:'[1024, 2560]',p:2621440},
    {n:'self_attn.v_proj.bias',shape:'[1024]',p:1024,sub:true},
    {n:'self_attn.o_proj.weight',shape:'[2560, 4096]',p:10485760},
  ]},
  {k:'MLP · SwiGLU gate / up / down',rows:[
    {n:'mlp.gate_proj.weight',shape:'[9728, 2560]',p:24903680},
    {n:'mlp.up_proj.weight',shape:'[9728, 2560]',p:24903680},
    {n:'mlp.down_proj.weight',shape:'[2560, 9728]',p:24903680},
  ]},
];
const LAYER_TOTAL=LAYER_GROUPS.reduce((s,g)=>s+g.rows.reduce((a,r)=>a+r.p,0),0);   /* 100,936,704 */
const MODEL_TREE=[
  {n:'embed_tokens（= lm_head · tied）',shape:'[151936, 2560]',p:388956160},
  {n:'decoder.layers × 36',shape:'36 × 100.94M',p:3633721344,layerHint:true},
  {n:'final norm.weight',shape:'[2560]',p:2560},
];
const MODEL_TOTAL=MODEL_TREE.reduce((s,r)=>s+r.p,0);                               /* 4,022,680,064 */

const CONFIG=[
  ['n_layers','36'],['d_model','2560'],['attention','32 Q / 8 KV（GQA）'],['head_dim','128'],
  ['d_ff（MLP）','9,728 · SwiGLU'],['norm','RMSNorm（pre+post）'],['vocab','151,936'],
  ['context','32,768'],['tied embeddings','✓ embed = lm_head'],['q/k/v bias','✓（o_proj 无）'],
];

function fmt(p){
  if(p>=1e9) return (p/1e9).toFixed(2)+'B';
  if(p>=1e6) return (p/1e6).toFixed(2)+'M';
  if(p>=1e3) return (p/1e3).toFixed(1)+'K';
  return ''+p;
}
function layerBadges(l){
  const b=[];
  if(l===6) b.push({t:'F#3734 载体层',c:'#059669'});
  if(writePeak(l)) b.push({t:'写端峰值带 L6–L11 · P8–P11',c:'#0d9488'});
  if(reachDemo(l)) b.push({t:'REACH（ρ≥0.10）',c:'#0284c7'});
  if(l===N-1) b.push({t:'读出层 · 读位槽 G−1=5 · P4–P7',c:'#b45309'});
  return b;
}

/* 每层盒内节点（参考旧版玻璃盒：节点 0=LN，1–8=attn heads，9–16=MLP units；demo 种子，
   接入点：torch hook 实测逐 head / 分桶活性替换坐标与幅值） */
const LNODES=[...Array(N)].map((_,l)=>{
  const rnd=mulberry(9100+l*137);
  const nodes=[], edges=[];
  nodes.push({p:[0,0.10,0],c:'n',s:1.0});
  for(let i=0;i<8;i++) nodes.push({p:[(i-3.5)*0.038, 0.20+rnd()*0.05, -0.14], c:'a', s:0.8+rnd()*0.7, seed:rnd()*6.28});
  for(let j=0;j<8;j++){
    const col=j%4, row=(j/4)|0;
    nodes.push({p:[(col-1.5)*0.055, 0.31+row*0.11, 0.12], c:'m', s:0.9+rnd()*0.7, seed:rnd()*6.28});
  }
  for(let i=0;i<8;i++) edges.push([0,1+i]);                       /* LN → heads */
  for(let i=0;i<8;i++){
    edges.push([1+i, 9+(i%4)+(i>3?4:0)]);                         /* head → MLP */
    if(i%2) edges.push([1+i, 9+((i+3)%4)+(i>3?4:0)]);
  }
  return {nodes,edges};
});

export default function StackMode({onOpenParam,onOpenNeuron}){
  const cvRef=useRef(null);
  const camRef=useRef({th:0.55,ph:0.42,zoom:480,auto:true,pulse:0,panX:0,panY:0});
  const dragRef=useRef(null);
  const polysRef=useRef([]);           /* 每帧盒面多边形（点击命中用） */
  const selRef=useRef(-1);
  const [sel,setSel]=useState(-1);
  const [readout,setReadout]=useState({th:31,ph:24,panX:0,panY:0});

  useEffect(()=>{
    const cv=cvRef.current; if(!cv) return;
    const ctx=cv.getContext('2d');
    const cam=camRef.current;

    function resize(){
      const dpr=window.devicePixelRatio||1;
      cv.width=cv.clientWidth*dpr; cv.height=cv.clientHeight*dpr;
    }
    resize();

    function project(x,y,z){
      const dpr=window.devicePixelRatio||1;
      const ct=Math.cos(cam.th),st=Math.sin(cam.th),cp=Math.cos(cam.ph),sp=Math.sin(cam.ph);
      const x1=x*ct+z*st, z1=-x*st+z*ct;
      const y2=y*cp-z1*sp, z2=y*sp+z1*cp;
      const d=6+z2*0.5, f=cam.zoom*(cv.height/900)/d;
      return {sx:cv.width/2+(camRef.current.panX||0)*dpr+x1*f*dpr, sy:cv.height*0.52+(camRef.current.panY||0)*dpr-y2*f*dpr, depth:z2};
    }
    const xs=l=>(l-(N-1)/2)*S;

    function quad(pts,fill){
      ctx.beginPath();
      pts.forEach((q,i)=>i?ctx.lineTo(q.sx,q.sy):ctx.moveTo(q.sx,q.sy));
      ctx.closePath();
      if(fill){ctx.fillStyle=fill;ctx.fill();}
    }
    function line(a,b){
      ctx.beginPath(); ctx.moveTo(a.sx,a.sy); ctx.lineTo(b.sx,b.sy); ctx.stroke();
    }

    let raf=0;
    function draw(){
      const dpr=window.devicePixelRatio||1;
      ctx.clearRect(0,0,cv.width,cv.height);
      if(cam.auto&&!dragRef.current) cam.th+=0.0015;
      cam.pulse+=0.045;
      polysRef.current=[];
      const fC=cam.zoom*(cv.height/900)/6;

      /* 残差流主干：水平贯穿 L0 → L35（盒子半高处）+ 流动脉冲（参考旧版 flow paths） */
      const yS=BH*0.55;
      const b0=project(-8.3,yS,0), b1=project(8.3,yS,0);
      const spine=ctx.createLinearGradient(b0.sx,b0.sy,b1.sx,b1.sy);
      spine.addColorStop(0,'rgba(148,163,184,.18)'); spine.addColorStop(0.5,'rgba(100,116,139,.5)'); spine.addColorStop(1,'rgba(148,163,184,.18)');
      ctx.strokeStyle=spine; ctx.lineWidth=2.6*dpr;
      ctx.beginPath(); ctx.moveTo(b0.sx,b0.sy); ctx.lineTo(b1.sx,b1.sy); ctx.stroke();
      for(let k=0;k<3;k++){
        const tF=((cam.pulse*0.012)+(k/3))%1;
        const pp=project(-8.3+16.6*tF,yS,0);
        ctx.fillStyle='rgba(14,165,233,.25)';
        ctx.beginPath(); ctx.arc(pp.sx,pp.sy,5.5*dpr,0,6.283); ctx.fill();
        ctx.fillStyle='rgba(56,189,248,.95)';
        ctx.beginPath(); ctx.arc(pp.sx,pp.sy,2.4*dpr,0,6.283); ctx.fill();
      }
      const dx=b1.sx-b0.sx, dy=b1.sy-b0.sy, dl=Math.hypot(dx,dy)||1;
      const ax=b1.sx+8*dpr, ay=b1.sy+8*dpr, ah=9*dpr;
      ctx.fillStyle='rgba(100,116,139,.75)';
      ctx.beginPath();
      ctx.moveTo(ax+dx/dl*ah, ay+dy/dl*ah);
      ctx.lineTo(ax-dy/dl*ah*0.5, ay+dx/dl*ah*0.5);
      ctx.lineTo(ax+dy/dl*ah*0.5, ay-dx/dl*ah*0.5);
      ctx.closePath(); ctx.fill();
      ctx.fillStyle='#64748b'; ctx.font=(10*dpr)+'px ui-monospace,Consolas,monospace';
      ctx.fillText('残差流 L0 → L35', b0.sx+6*dpr, b0.sy-12*dpr);

      /* 玻璃盒阵列（按深度从远到近） */
      const order=[...Array(N).keys()].map(l=>({l,q:project(xs(l),BH*0.5,0)}));
      order.sort((a,b)=>a.q.depth-b.q.depth);
      order.forEach(({l})=>{
        const lift=(l===selRef.current)?0.1:0;
        const x=xs(l), y0=lift, y1=lift+BH;
        const pb=[[x+BX,y0,BZ],[x+BX,y0,-BZ],[x-BX,y0,-BZ],[x-BX,y0,BZ]].map(p=>project(p[0],p[1],p[2]));
        const pt=pb.map((p,i)=>project([x+BX,x+BX,x-BX,x-BX][i],y1,[BZ,-BZ,-BZ,BZ][i]));
        const q=project(x,lift+BH*0.5,0);
        const w=Math.abs(pt[0].sx-pt[2].sx);
        polysRef.current.push({l,pts:pt});
        polysRef.current.push({l,pts:[pt[1],pt[2],pb[2],pb[1]]});

        /* 玻璃面：顶 / 前 / 右（着色=研究标注带） */
        const tint = l===selRef.current ? '224,242,254'
          : writePeak(l) ? '209,250,229'
          : reachDemo(l) ? '224,242,254' : '255,255,255';
        quad(pt,'rgba('+tint+',.42)');
        quad([pt[1],pt[2],pb[2],pb[1]],'rgba('+tint+',.26)');
        quad([pt[0],pt[1],pb[1],pb[0]],'rgba('+tint+',.18)');

        /* 盒线框（可见三面 + 隐边弱化） */
        ctx.strokeStyle = l===6 ? 'rgba(5,150,105,.8)' : l===selRef.current ? '#0284c7' : 'rgba(100,116,139,.55)';
        ctx.lineWidth = ((l===6||l===selRef.current)?1.6:1)*dpr;
        quad(pt); ctx.stroke();
        quad([pt[1],pt[2],pb[2],pb[1]]); ctx.stroke();
        quad([pt[0],pt[1],pb[1],pb[0]]); ctx.stroke();
        ctx.globalAlpha=0.3; line(pb[3],pt[3]); ctx.globalAlpha=1;

        /* 盒内节点连线（结构参考旧版层内边） */
        const nd=LNODES[l];
        ctx.lineWidth=1*dpr;
        ctx.strokeStyle = l===selRef.current ? 'rgba(2,132,199,.42)' : 'rgba(148,163,184,.30)';
        nd.edges.forEach(([a,b])=>{
          const na=nd.nodes[a], nb=nd.nodes[b];
          line(
            project(x+na.p[0],lift+na.p[1],na.p[2]),
            project(x+nb.p[0],lift+nb.p[1],nb.p[2])
          );
        });
        /* 盒内节点（发光球：蓝=head，紫=MLP，灰=LN；呼吸=demo 激活） */
        nd.nodes.forEach(n=>{
          const p=project(x+n.p[0],lift+n.p[1],n.p[2]);
          const col=n.c==='a'?'2,132,199':n.c==='m'?'109,40,217':'100,116,139';
          const pul=n.seed!==undefined?0.62+0.38*Math.sin(cam.pulse*0.7+n.seed):0.9;
          const r=Math.max(1.2,n.s*2.3*(fC/70))*dpr;
          ctx.fillStyle='rgba('+col+','+(0.20*pul).toFixed(3)+')';
          ctx.beginPath(); ctx.arc(p.sx,p.sy,r*2.1,0,6.283); ctx.fill();
          ctx.fillStyle='rgba('+col+','+(0.55+0.4*pul).toFixed(3)+')';
          ctx.beginPath(); ctx.arc(p.sx,p.sy,r,0,6.283); ctx.fill();
        });

        /* 层号（盒前下方） */
        if(w>24){
          ctx.globalAlpha=0.92; ctx.fillStyle=l===selRef.current?'#0284c7':'#64748b';
          ctx.font=((w>40?9:7.5)*dpr)+'px ui-monospace,Consolas,monospace';
          const lm={sx:(pb[1].sx+pb[2].sx)/2, sy:(pb[1].sy+pb[2].sy)/2};
          ctx.fillText('L'+l, lm.sx, lm.sy+11*dpr);
          ctx.globalAlpha=1;
        }
        /* L6 脉冲环 + F#3734 标注 */
        if(l===6){
          const r=(Math.max(w,10)*0.55+Math.sin(cam.pulse)*2.2)*dpr;
          ctx.strokeStyle='rgba(5,150,105,.55)'; ctx.lineWidth=1.6*dpr;
          ctx.beginPath(); ctx.arc(q.sx,q.sy,r,0,6.283); ctx.stroke();
          ctx.fillStyle='#047857'; ctx.font='bold '+(10*dpr)+'px ui-monospace,Consolas,monospace';
          ctx.fillText('L6 · F#3734', q.sx-30*dpr, q.sy-BH*0.5*fC*dpr-8*dpr);
        }
        /* L35 读位槽 G−1=5（盒顶金点） */
        if(l===N-1&&w>40){
          for(let i=0;i<5;i++){
            const p=project(xs(l)-0.1+i*0.05,y1+0.03,0);
            ctx.fillStyle='rgba(180,83,9,.92)';
            ctx.beginPath(); ctx.arc(p.sx,p.sy,2.6*dpr,0,6.283); ctx.fill();
          }
        }
      });

      raf=requestAnimationFrame(draw);
    }
    raf=requestAnimationFrame(draw);

    const tId=setInterval(()=>{
      setReadout({th:Math.round(cam.th*180/Math.PI)%360, ph:Math.round(cam.ph*180/Math.PI), panX:Math.round(cam.panX||0), panY:Math.round(cam.panY||0)});
    },500);

    /* 命中检测 */
    function pick(mx,my){
      const polys=polysRef.current;
      for(let i=polys.length-1;i>=0;i--){
        const pts=polys[i].pts; let inside=false;
        for(let a=0,b=pts.length-1;a<pts.length;b=a++){
          if(((pts[a].sy>my)!==(pts[b].sy>my))&&(mx<(pts[b].sx-pts[a].sx)*(my-pts[a].sy)/(pts[b].sy-pts[a].sy)+pts[a].sx)) inside=!inside;
        }
        if(inside) return polys[i].l;
      }
      return -1;
    }
    const onMove=e=>{
      const r=cv.getBoundingClientRect();
      const mx=(e.clientX-r.left)*(cv.width/r.width);
      const my=(e.clientY-r.top)*(cv.height/r.height);
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
        return;
      }
      cv.style.cursor=pick(mx,my)>=0?'pointer':'grab';
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
            const r=cv.getBoundingClientRect();
            const dpr=window.devicePixelRatio||1;
            const mx=(e.clientX-r.left)*dpr, my=(e.clientY-r.top)*dpr;
            const l=pick(mx,my);
            if(l>=0){ selRef.current=(selRef.current===l?-1:l); setSel(selRef.current); }
            else if(selRef.current>=0){ selRef.current=-1; setSel(-1); }
          }
        }
      }
      dragRef.current=null; autoT=setTimeout(()=>{cam.auto=true;},2500);
    };
    const onWheel=e=>{ e.preventDefault(); cam.zoom=Math.max(220,Math.min(1900,cam.zoom-e.deltaY*1.2)); };
    const onResize=()=>resize();

    const onKey=e=>{ if(e.key==='Escape'&&selRef.current>=0){ selRef.current=-1; setSel(-1); } };
    const onCtx=e=>e.preventDefault();
    window.addEventListener('keydown',onKey);

    cv.addEventListener('mousedown',onDown);
    cv.addEventListener('contextmenu',onCtx);
    window.addEventListener('mousemove',onMove);
    window.addEventListener('mouseup',onUp);
    cv.addEventListener('wheel',onWheel,{passive:false});
    window.addEventListener('resize',onResize);
    return ()=>{
      cancelAnimationFrame(raf); clearInterval(tId);
      window.removeEventListener('keydown',onKey);
      cv.removeEventListener('mousedown',onDown);
      cv.removeEventListener('contextmenu',onCtx);
      window.removeEventListener('mousemove',onMove);
      window.removeEventListener('mouseup',onUp);
      cv.removeEventListener('wheel',onWheel);
      window.removeEventListener('resize',onResize);
    };
  },[]);

  /* 选中层 demo 内部数据（确定性） */
  const rnd=mulberry(600+sel*77);
  const heads=sel>=0?[...Array(HEADS).keys()].map(()=>rnd()):[];
  const bars=sel>=0?[...Array(16).keys()].map(()=>0.15+rnd()*0.85):[];
  const badges=sel>=0?layerBadges(sel):[];

  return (
    <div className="fw-sp-row">
      <div className="fw-sp-canvasbox">
        <canvas ref={cvRef} className="fw-sp-canvas"/>
        <div className="fw-sp-legend" style={{top:14,bottom:'auto'}}>
          <div className="li"><span className="fw-swatch" style={{background:'#0284c7',borderRadius:'50%'}}/>attention head 节点（示意 8 / 实际 32）</div>
          <div className="li"><span className="fw-swatch" style={{background:'#6d28d9',borderRadius:'50%'}}/>MLP unit 节点（示意 8 / 实际 9,728）</div>
          <div className="li"><span className="fw-swatch" style={{background:'#94a3b8',borderRadius:'50%'}}/>RMSNorm 节点 · 盒内连线=层内结构</div>
        </div>
        <div className="fw-sp-readout">
          cam <b>θ {readout.th}° · φ {readout.ph}°</b> · 平移 <b>{readout.panX>0?'+':''}{readout.panX},{readout.panY>0?'+':''}{readout.panY}px</b> · 平铺 <b>n=36 layers</b>
        </div>
      </div>

      {/* 旁边模型：内部结构与全部参数 */}
      <aside className="fw-lp">
        {sel<0?(
          <>
            <div className="fw-lp-hd">模型总览 <small>qwen3-4b · 4.02B</small></div>
            <div className="fw-lp-cfg">
              {CONFIG.flatMap(([k,v])=>[
                <span className="k" key={k}>{k}</span>,
                <span className="v" key={'v'+k}>{v}</span>,
              ])}
            </div>
            <div className="sec">参数树 · ALL PARAMETERS<span>{fmt(MODEL_TOTAL)}</span></div>
            {MODEL_TREE.map(r=>(
              <div className="fw-lp-row" key={r.n}>
                <div className="top">
                  <span className="nm" title={r.n}>{r.n}</span>
                  <span className="pv">{fmt(r.p)}</span>
                </div>
                <div className="shp">{r.shape} · {(r.p/MODEL_TOTAL*100).toFixed(1)}%</div>
                <div className="fw-lp-bar"><i style={{width:(r.p/MODEL_TOTAL*100).toFixed(2)+'%'}}/></div>
              </div>
            ))}
            <div className="fw-lp-note">点击左侧任意层盒 → 该层内部结构与全部参数。<br/>形状/计数按 Qwen3-4B 公开架构推导（config.json 口径）；节点激活为 demo 种子，接入点：torch hook 实测。</div>
            <div className="sec">研究标注 · 全局</div>
            <div className="bdg-row">
              <span className="fw-bdg" style={{background:'#0596691a',color:'#059669',border:'1px solid #05966955'}}>F#3734 载体层 L6</span>
              <span className="fw-bdg" style={{background:'#0d94881a',color:'#0d9488',border:'1px solid #0d948855'}}>写端峰值带 L6–L11 · P8–P11</span>
              <span className="fw-bdg" style={{background:'#0284c71a',color:'#0284c7',border:'1px solid #0284c755'}}>REACH ρ≥0.10（示意）</span>
              <span className="fw-bdg" style={{background:'#b453091a',color:'#b45309',border:'1px solid #b4530955'}}>读位槽 G−1=5 · 顶层 P4–P7</span>
            </div>
          </>
        ):(
          <>
            <div className="fw-lp-hd">
              <span>L{sel} · TransformerBlock</span>
              <button className="fw-tbtn" onClick={()=>{setSel(-1);selRef.current=-1;}} title="返回模型总览">×</button>
            </div>
            <div className="fw-lp-cfg">
              <span className="k">层参数合计</span><span className="v">{fmt(LAYER_TOTAL)}（{LAYER_TOTAL.toLocaleString('en-US')}）</span>
              <span className="k">占全模型</span><span className="v">{(LAYER_TOTAL/MODEL_TOTAL*100).toFixed(2)}% · ×36 层 {(LAYER_TOTAL*36/MODEL_TOTAL*100).toFixed(1)}%</span>
            </div>
            {LAYER_GROUPS.map(g=>{
              const sum=g.rows.reduce((a,r)=>a+r.p,0);
              return (
                <div key={g.k} style={{marginTop:8}}>
                  <div className="sec">{g.k}<span>{fmt(sum)} · {(sum/LAYER_TOTAL*100).toFixed(1)}%</span></div>
                  {g.rows.map(r=>(
                    <div className="fw-lp-row" key={r.n}>
                      <div className="top">
                        <span className={'nm'+(r.sub?' sub':'')} title={r.n}>{r.n}</span>
                        <span className="pv">{fmt(r.p)}</span>
                      </div>
                      <div className="shp">{r.shape} · {r.p.toLocaleString('en-US')}</div>
                      {!r.sub&&<div className="fw-lp-bar"><i style={{width:(r.p/LAYER_TOTAL*100).toFixed(2)+'%'}}/></div>}
                    </div>
                  ))}
                </div>
              );
            })}
            <div className="sec">ATTENTION · {HEADS} heads（GQA {KVH} kv · head_dim {D_HEAD}）<span>demo 激活</span></div>
            <div className="fw-heads-grid">
              {heads.map((v,i)=>(
                <span key={i} className="cell" style={{background:'rgba(2,132,199,'+(0.12+v*0.75).toFixed(2)+')'}} title={'head '+i+' · demo 激活 '+(v*4.2).toFixed(2)}/>
              ))}
            </div>
            <div className="sec">MLP · {D_MLP} units<span>demo 活性</span></div>
            <div className="fw-mlp-bars">
              {bars.map((v,i)=><span key={i} style={{height:(12+v*30)+'px',background:v>0.75?'#6d28d9':'rgba(109,40,217,.45)'}} title={'unit group '+i}/>)}
            </div>
            <div className="sec">研究标注</div>
            <div className="bdg-row">
              {badges.length?badges.map((b,i)=><span key={i} className="fw-bdg" style={{background:b.c+'1a',color:b.c,border:'1px solid '+b.c+'55'}}>{b.t}</span>)
                :<span className="dim">本层暂无登记标注（接入 atlas_ledger 后自动点亮）</span>}
            </div>
            <div className="act">
              <span className="total">≈{fmt(LAYER_TOTAL)} / 层 · 36 层 ≈ {fmt(LAYER_TOTAL*36)}</span>
              <span style={{display:'flex',gap:6}}>
                <button className="fw-tbtn" onClick={()=>onOpenNeuron&&onOpenNeuron(sel)} title="该层内部 3D 全量展开，精确到单个神经元">进入神经元空间 →</button>
                <button className="fw-tbtn primary" onClick={()=>onOpenParam&&onOpenParam(sel)}>进入参数热图 →</button>
              </span>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
