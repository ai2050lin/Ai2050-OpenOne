/* 特征点云模式：canvas 2D 伪 3D 特征族点云（原空间透镜主体平移，接 collect.npz 前的 demo 数据）
   语言模板驱动（design/client_template_tech_plan_v1.md §2，M5-P0）：
   点云族 / 焦点特征 / 邻居 / 图例全部来自 LANG_TEMPLATES（distributedData.js）——
   本组件不含任何模板叙事字面量，切换模板整体跟随。 */
import { useEffect, useRef, useState } from 'react';
import { LANG_TEMPLATES } from './distributedData.js';

function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

/* 邻居相对位置（族结构骨架，跨模板共用；特征 id ← 模板 demo.neighbors） */
const NB_POS=[[0.6,-0.9,0.5],[0.3,-1.5,0.9],[0.8,-1.0,0.3],[0.2,-0.8,0.6]];
const BG_CLUSTER={c:[0,0,0],col:'#cbd5e1',n:180,s:1.9};

export default function CloudMode({tpl}){
  const tplDef=tpl||LANG_TEMPLATES[0];
  const demo=tplDef.demo;
  const focus=demo.focus;
  const cvRef=useRef(null);
  const camRef=useRef({th:0.6,ph:0.35,zoom:190,auto:true,pulse:0,panX:0,panY:0});
  const dragRef=useRef(null);
  const [readout,setReadout]=useState({th:38,ph:22,panX:0,panY:0});

  useEffect(()=>{
    const cv=cvRef.current; if(!cv) return;
    const ctx2=cv.getContext('2d');
    const cam=camRef.current;

    /* 生成点云（每次模板切换重建；族结构 ← 模板 demo.cloud） */
    const rnd=mulberry(2050), PTS=[];
    demo.cloud.forEach(cl=>{
      for(let i=0;i<cl.n;i++){
        PTS.push({x:cl.c[0]+(rnd()-0.5)*cl.s*2,y:cl.c[1]+(rnd()-0.5)*cl.s*2,z:cl.c[2]+(rnd()-0.5)*cl.s*2,
          col:cl.col,r:2.2,a:0.8});
      }
    });
    for(let i=0;i<BG_CLUSTER.n;i++){
      PTS.push({x:BG_CLUSTER.c[0]+(rnd()-0.5)*BG_CLUSTER.s*2,y:BG_CLUSTER.c[1]+(rnd()-0.5)*BG_CLUSTER.s*2,z:BG_CLUSTER.c[2]+(rnd()-0.5)*BG_CLUSTER.s*2,
        col:BG_CLUSTER.col,r:1.3,a:0.35});
    }
    const FOCUS={x:focus.pos[0],y:focus.pos[1],z:focus.pos[2]};
    const NEIGHBORS=NB_POS.map((p,i)=>({x:p[0],y:p[1],z:p[2],id:demo.neighbors[i]||('F#'+((i*917+109)%4000))}));

    function resize(){
      const dpr=window.devicePixelRatio||1;
      cv.width=cv.clientWidth*dpr; cv.height=cv.clientHeight*dpr;
    }
    resize();

    function project(p){
      const dpr=window.devicePixelRatio||1;
      const ct=Math.cos(cam.th),st=Math.sin(cam.th),cp=Math.cos(cam.ph),sp=Math.sin(cam.ph);
      const x1=p.x*ct+p.z*st, z1=-p.x*st+p.z*ct;
      const y2=p.y*cp-z1*sp, z2=p.y*sp+z1*cp;
      const d=4+z2*0.35, f=cam.zoom/d;
      return {sx:cv.width/2+(camRef.current.panX||0)*dpr+x1*f*dpr, sy:cv.height*0.5+(camRef.current.panY||0)*dpr+y2*f*dpr, depth:z2};
    }

    let raf=0;
    function draw(){
      const dpr=window.devicePixelRatio||1;
      ctx2.clearRect(0,0,cv.width,cv.height);
      if(cam.auto&&!dragRef.current) cam.th+=0.0022;
      cam.pulse+=0.05;
      const f=project(FOCUS);
      const nbp=NEIGHBORS.map(n=>({q:project(n),n}));
      /* 邻居连线 */
      ctx2.lineWidth=1.2*dpr;
      nbp.forEach(({q})=>{
        const g=ctx2.createLinearGradient(f.sx,f.sy,q.sx,q.sy);
        g.addColorStop(0,'rgba(5,150,105,.9)'); g.addColorStop(1,'rgba(5,150,105,.15)');
        ctx2.strokeStyle=g; ctx2.beginPath(); ctx2.moveTo(f.sx,f.sy); ctx2.lineTo(q.sx,q.sy); ctx2.stroke();
      });
      /* 点（按深度排序） */
      const items=PTS.map(p=>({q:project(p),p})).concat(nbp.map(({q})=>({q,p:{col:'#059669',r:2.6,a:0.95}})));
      items.sort((a,b)=>a.q.depth-b.q.depth).forEach(({q,p})=>{
        ctx2.globalAlpha=Math.max(0.08,p.a*(1-0.35*((q.depth+2)/4)));
        ctx2.fillStyle=p.col;
        ctx2.beginPath(); ctx2.arc(q.sx,q.sy,p.r*0.016*cam.zoom*0.9*dpr,0,6.283); ctx2.fill();
      });
      /* 高亮焦点特征 */
      ctx2.globalAlpha=1;
      const r=(7+Math.sin(cam.pulse)*1.6)*dpr;
      ctx2.strokeStyle='rgba(5,150,105,.5)'; ctx2.lineWidth=1.6*dpr;
      ctx2.beginPath(); ctx2.arc(f.sx,f.sy,r+5*dpr,0,6.283); ctx2.stroke();
      ctx2.fillStyle='#059669';
      ctx2.beginPath(); ctx2.arc(f.sx,f.sy,4.5*dpr,0,6.283); ctx2.fill();
      ctx2.fillStyle='#022c22';
      ctx2.font='bold '+(10*dpr)+'px ui-monospace,Consolas,monospace';
      ctx2.fillText(focus.id+' · '+focus.short,f.sx+9*dpr,f.sy-8*dpr);
      raf=requestAnimationFrame(draw);
    }
    raf=requestAnimationFrame(draw);

    /* 读数节流更新 */
    const tId=setInterval(()=>{
      setReadout({th:Math.round(cam.th*180/Math.PI)%360, ph:Math.round(cam.ph*180/Math.PI), panX:Math.round(cam.panX||0), panY:Math.round(cam.panY||0)});
    },500);

    const onMove=e=>{
      if(!dragRef.current) return;
      if(dragRef.current.btn===2){
        /* 右键：平移视角位置（上下左右皆可），不改变任何角度 */
        cam.panX=Math.max(-800,Math.min(800,(cam.panX||0)+(e.clientX-dragRef.current.x)));
        cam.panY=Math.max(-800,Math.min(800,(cam.panY||0)+(e.clientY-dragRef.current.y)));
      }else{
        /* 左键：旋转视角，上下左右都改变角度 */
        cam.th+=(e.clientX-dragRef.current.x)*0.006;
        cam.ph=Math.max(-1.2,Math.min(1.2,cam.ph+(e.clientY-dragRef.current.y)*0.006));
      }
      dragRef.current={x:e.clientX,y:e.clientY,btn:dragRef.current.btn};
    };
    let autoT=null;
    const onUp=()=>{ dragRef.current=null; autoT=setTimeout(()=>{cam.auto=true;},2500); };
    const onWheel=e=>{ e.preventDefault(); cam.zoom=Math.max(110,Math.min(340,cam.zoom-e.deltaY*0.4)); };
    const onDown=e=>{
      if(e.button!==0&&e.button!==2) return;
      dragRef.current={x:e.clientX,y:e.clientY,btn:e.button};
      cam.auto=false;
      if(autoT){clearTimeout(autoT);autoT=null;}   /* 取消旧的恢复定时器，防止拖拽中 auto-rotate 被唤醒 */
    };
    const onCtx=e=>e.preventDefault();
    const onResize=()=>resize();

    cv.addEventListener('mousedown',onDown);
    cv.addEventListener('contextmenu',onCtx);
    window.addEventListener('mousemove',onMove);
    window.addEventListener('mouseup',onUp);
    cv.addEventListener('wheel',onWheel,{passive:false});
    window.addEventListener('resize',onResize);
    return ()=>{
      cancelAnimationFrame(raf); clearInterval(tId);
      cv.removeEventListener('mousedown',onDown);
      cv.removeEventListener('contextmenu',onCtx);
      window.removeEventListener('mousemove',onMove);
      window.removeEventListener('mouseup',onUp);
      cv.removeEventListener('wheel',onWheel);
      window.removeEventListener('resize',onResize);
    };
  },[tplDef.id]);

  return (
    <div style={{position:'absolute',inset:0}}>
      <canvas ref={cvRef} className="fw-sp-canvas"/>
      <div className="fw-sp-focus">
        <button className="fw-tbtn" onClick={()=>window.alert('以 '+focus.id+' 为中心重置相机（接入真实坐标后生效）')}>◎ 以 {focus.id} 为中心</button>
        <button className="fw-tbtn" onClick={()=>window.alert('E_ar(k) 方向叠加：接入 Q05 四臂结果后，在点云上渲染当前语言模板的关系方向箭头')}>E_ar(k) 方向叠加</button>
      </div>
      <div className="fw-sp-legend">
        {demo.cloud.map((cl,i)=>(
          <div className="li" key={i}><span className="fw-swatch" style={{background:cl.col}}/>{cl.label}</div>
        ))}
        <div className="li"><span className="fw-swatch" style={{background:'#cbd5e1'}}/>其他</div>
      </div>
      <div className="fw-sp-readout">
        cam <b>θ {readout.th}° · φ {readout.ph}°</b> · 平移 <b>{readout.panX>0?'+':''}{readout.panX},{readout.panY>0?'+':''}{readout.panY}px</b><br/>
        {focus.id} <b>({focus.pos.map(v=>v.toFixed(2)).join(', ').replace(/-/g,'−')})</b>
      </div>
    </div>
  );
}
