/* 特征点云模式：canvas 2D 伪 3D 特征族点云（原空间透镜主体平移，接 collect.npz 前的 demo 数据） */
import { useEffect, useRef, useState } from 'react';

/* 确定性伪随机 */
function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

const CLUSTERS=[
  {c:[0.9,-0.6,0.4],col:'#10b981',n:120,s:0.85,fam:'is-a'},
  {c:[-1.0,0.5,-0.5],col:'#0ea5e9',n:90,s:0.95,fam:'attr'},
  {c:[-0.2,-0.9,1.0],col:'#f59e0b',n:70,s:0.75,fam:'syntax'},
  {c:[0,0,0],col:'#cbd5e1',n:180,s:1.9,fam:'bg'},
];
const FOCUS={x:0.42,y:-1.18,z:0.73};
const NEIGHBORS=[
  {x:0.6,y:-0.9,z:0.5,id:'F#1092'},{x:0.3,y:-1.5,z:0.9,id:'F#2210'},
  {x:0.8,y:-1.0,z:0.3,id:'F#3655'},{x:0.2,y:-0.8,z:0.6,id:'F#0821'},
];

export default function CloudMode(){
  const cvRef=useRef(null);
  const camRef=useRef({th:0.6,ph:0.35,zoom:190,auto:true,pulse:0,panX:0,panY:0});
  const dragRef=useRef(null);
  const [readout,setReadout]=useState({th:38,ph:22,panX:0,panY:0});

  useEffect(()=>{
    const cv=cvRef.current; if(!cv) return;
    const ctx=cv.getContext('2d');
    const cam=camRef.current;

    /* 生成点云（一次） */
    const rnd=mulberry(2050), PTS=[];
    CLUSTERS.forEach(cl=>{
      for(let i=0;i<cl.n;i++){
        PTS.push({x:cl.c[0]+(rnd()-0.5)*cl.s*2,y:cl.c[1]+(rnd()-0.5)*cl.s*2,z:cl.c[2]+(rnd()-0.5)*cl.s*2,
          col:cl.col,r:cl.fam==='bg'?1.3:2.2,a:cl.fam==='bg'?0.35:0.8});
      }
    });

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
      ctx.clearRect(0,0,cv.width,cv.height);
      if(cam.auto&&!dragRef.current) cam.th+=0.0022;
      cam.pulse+=0.05;
      const f=project(FOCUS);
      const nbp=NEIGHBORS.map(n=>({q:project(n),n}));
      /* 邻居连线 */
      ctx.lineWidth=1.2*dpr;
      nbp.forEach(({q})=>{
        const g=ctx.createLinearGradient(f.sx,f.sy,q.sx,q.sy);
        g.addColorStop(0,'rgba(5,150,105,.9)'); g.addColorStop(1,'rgba(5,150,105,.15)');
        ctx.strokeStyle=g; ctx.beginPath(); ctx.moveTo(f.sx,f.sy); ctx.lineTo(q.sx,q.sy); ctx.stroke();
      });
      /* 点（按深度排序） */
      const items=PTS.map(p=>({q:project(p),p})).concat(nbp.map(({q})=>({q,p:{col:'#059669',r:2.6,a:0.95}})));
      items.sort((a,b)=>a.q.depth-b.q.depth).forEach(({q,p})=>{
        ctx.globalAlpha=Math.max(0.08,p.a*(1-0.35*((q.depth+2)/4)));
        ctx.fillStyle=p.col;
        ctx.beginPath(); ctx.arc(q.sx,q.sy,p.r*0.016*cam.zoom*0.9*dpr,0,6.283); ctx.fill();
      });
      /* 高亮 F#3734 */
      ctx.globalAlpha=1;
      const r=(7+Math.sin(cam.pulse)*1.6)*dpr;
      ctx.strokeStyle='rgba(5,150,105,.5)'; ctx.lineWidth=1.6*dpr;
      ctx.beginPath(); ctx.arc(f.sx,f.sy,r+5*dpr,0,6.283); ctx.stroke();
      ctx.fillStyle='#059669';
      ctx.beginPath(); ctx.arc(f.sx,f.sy,4.5*dpr,0,6.283); ctx.fill();
      ctx.fillStyle='#022c22';
      ctx.font='bold '+(10*dpr)+'px ui-monospace,Consolas,monospace';
      ctx.fillText('F#3734 · is-a 水果族',f.sx+9*dpr,f.sy-8*dpr);
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
  },[]);

  return (
    <div style={{position:'absolute',inset:0}}>
      <canvas ref={cvRef} className="fw-sp-canvas"/>
      <div className="fw-sp-focus">
        <button className="fw-tbtn" onClick={()=>window.alert('以 F#3734 为中心重置相机（接入真实坐标后生效）')}>◎ 以 F#3734 为中心</button>
        <button className="fw-tbtn" onClick={()=>window.alert('E_ar(k) 方向叠加：接入 Q05 四臂结果后，在点云上渲染 is-a / attr / syntax 方向箭头')}>E_ar(k) 方向叠加</button>
      </div>
      <div className="fw-sp-legend">
        <div className="li"><span className="fw-swatch" style={{background:'#10b981'}}/>水果族 · is-a</div>
        <div className="li"><span className="fw-swatch" style={{background:'#0ea5e9'}}/>属性轴 · size/speed</div>
        <div className="li"><span className="fw-swatch" style={{background:'#f59e0b'}}/>语法功能</div>
        <div className="li"><span className="fw-swatch" style={{background:'#cbd5e1'}}/>其他</div>
      </div>
      <div className="fw-sp-readout">
        cam <b>θ {readout.th}° · φ {readout.ph}°</b> · 平移 <b>{readout.panX>0?'+':''}{readout.panX},{readout.panY>0?'+':''}{readout.panY}px</b><br/>
        F#3734 <b>(0.42, −1.18, 0.73)</b>
      </div>
    </div>
  );
}
