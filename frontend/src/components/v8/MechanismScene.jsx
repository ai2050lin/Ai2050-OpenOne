import {useEffect,useRef,useState} from 'react';
import { panFixedCanvas, pitchedScreenY, useScenePointer, SCENE_MOUSE_HINT, useSceneWheel } from '../scenePointer.js';

const colour=(v,max)=>v>=0?`rgba(42,120,214,${.08+.85*Math.abs(v)/(max||1)})`:`rgba(235,104,52,${.08+.85*Math.abs(v)/(max||1)})`;

export function NativeArray({values:inputValues,selected=0,onPick,label='原生阵列'}){
  const values=inputValues??[];
  const ref=useRef(null),cols=128,cell=5;
  useEffect(()=>{const c=ref.current;if(!c)return;const ctx=c.getContext('2d');if(!ctx)return;ctx.clearRect(0,0,c.width,c.height);
    const max=Math.max(1e-12,...values.map(Math.abs));values.forEach((v,i)=>{ctx.fillStyle=colour(v,max);ctx.fillRect(i%cols*cell,Math.floor(i/cols)*cell,4,4)});
    if(selected>=0&&selected<values.length){ctx.strokeStyle='#111827';ctx.lineWidth=1;ctx.strokeRect(selected%cols*cell,Math.floor(selected/cols)*cell,4,4)}
  },[values,selected]);
  return <div style={{overflowX:'auto'}}><canvas ref={ref} width={640} height={Math.max(10,Math.ceil(values.length/cols)*cell)} aria-label={`${label}，全部${values.length}维，未进行TopK截断`}
    style={{maxWidth:'100%',cursor:'crosshair'}} onClick={e=>{const r=e.currentTarget.getBoundingClientRect();const x=(e.clientX-r.left)*640/r.width,y=(e.clientY-r.top)*e.currentTarget.height/r.height;
      const i=Math.floor(y/cell)*cols+Math.floor(x/cell);if(i<values.length)onPick?.(i)}}/><p className="v8-note">原生索引顺序；蓝为正、橙为负，按本阵列最大绝对值 {values.length?Math.max(...values.map(Math.abs)).toPrecision(5):'—'} 缩放颜色。色深不跨阵列直接比较，原始数值保留。</p></div>;
}

export default function MechanismScene({manifest:d,block,head,unit,embedding=[],activation=[],headZ=[],headOverlay,caption,connections,onBlock,onHead,onUnit,showUnitMarker=true}){
  const ref=useRef(null),hits=useRef([]);const [view, setView] = useState({ yaw: -.45, pitch: 0, zoom: 1 });
  const { yaw, pitch, panX = 0, panY = 0, zoom } = view;
  useEffect(()=>{
    const cv=ref.current;if(!cv||!d)return;const ctx=cv.getContext('2d');if(!ctx)return;const W=760,H=600;ctx.clearRect(0,0,W,H);hits.current=[];
    const scale=135*zoom;const P=(x,y,z)=>[W/2+panX+scale*(x*Math.cos(yaw)-z*Math.sin(yaw)), H/2+panY+scale*pitchedScreenY(y,x*Math.sin(yaw)+z*Math.cos(yaw),pitch,.88,.35)];
    const Y=l=>-1.65+3.3*l/(d.n_layers+1);
    const polygon=(pts,fill,stroke)=>{ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.stroke()};
    for(let l=0;l<=d.n_layers;l++){
      const y=Y(l),selected=l===block+1,pts=[P(-1,y,-.72),P(1,y,-.72),P(1,y,.72),P(-1,y,.72)];
      polygon(pts,l===0?'#eef1f5':selected?'#eaf3ff':'rgba(245,247,250,.7)',selected?'#397bbd':'#cbd5e1');
      ctx.fillStyle='#64748b';ctx.font='10px sans-serif';ctx.fillText(l===0?'Embedding':`B${l-1}`,...P(-1.45,y,.75));
      const c=P(1.2,y,.1);hits.current.push({x:c[0],y:c[1],r:9,kind:'block',i:Math.max(0,l-1)});
      if(l>0){polygon([P(-.92,y,-.62),P(.92,y,-.62),P(.92,y,-.13),P(-.92,y,-.13)],selected?'#c6dcf3':'#dce7f4','#9bb7d5');
        polygon([P(-.92,y,.13),P(.92,y,.13),P(.92,y,.62),P(-.92,y,.62)],selected?'#f6d9c9':'#f0e0d7','#d8b69d')}
    }
    // Every native coordinate/unit is drawn on its declared index grid.
    const em=Math.max(1e-12,...embedding.map(Math.abs));embedding.forEach((v,i)=>{const p=P(-.92+1.84*(i%128)/127,Y(0),-.58+1.16*Math.floor(i/128)/Math.max(1,Math.ceil(embedding.length/128)-1));ctx.fillStyle=colour(v,em);ctx.fillRect(p[0],p[1],1,1)});
    const am=Math.max(1e-12,...activation.map(Math.abs)),uy=Y(block+1);
    const UP=i=>P(-.92+1.84*(i%128)/127,uy,.16+.44*Math.floor(i/128)/Math.max(1,Math.ceil(d.d_ff/128)-1));
    activation.forEach((v,i)=>{const p=UP(i);ctx.fillStyle=colour(v,am);ctx.fillRect(p[0],p[1],1.1,1.1);hits.current.push({x:p[0],y:p[1],r:2,kind:'unit',i})});
    const HP=i=>P(-.8+1.6*(i%8)/7,uy,-.56+.35*Math.floor(i/8)/Math.max(1,Math.ceil(d.n_heads/8)-1));
    const hm=Math.max(1e-12,...(headOverlay||[]).map(Math.abs));
    for(let h=0;h<d.n_heads;h++){
      const p=HP(h);ctx.beginPath();ctx.arc(...p,h===head?5:2.7,0,Math.PI*2);ctx.fillStyle=headOverlay?colour(headOverlay[h],hm):h===head?'#17457f':'#5294d6';ctx.fill();
      hits.current.push({x:p[0],y:p[1],r:7,kind:'head',i:h});
    }
    const u=UP(unit),h=HP(head);
    if(showUnitMarker){ctx.beginPath();ctx.arc(...u,5,0,Math.PI*2);ctx.strokeStyle='#b74413';ctx.lineWidth=2;ctx.stroke();ctx.lineWidth=1;}
    if(connections){ctx.beginPath();ctx.moveTo(...h);ctx.lineTo(...u);ctx.strokeStyle='#4b6a90';ctx.setLineDash([5,3]);ctx.stroke();ctx.setLineDash([]);
      ctx.fillStyle='#334155';ctx.font='12px sans-serif';ctx.fillText(`H${head} → U${unit}：固定RMS的局部贡献`,35,35)}
    ctx.fillStyle='#334155';ctx.font='12px sans-serif';ctx.fillText(`B${block} · ${d.n_heads}Q / ${d.n_kv_heads}KV · 完整MLP索引 ${d.d_ff}`,35,560);
    if(caption){ctx.font='11px sans-serif';ctx.fillText(caption,35,540)}
    ctx.fillStyle='#64748b';ctx.font='11px sans-serif';ctx.fillText('层高=原生层序；板内=索引布局；距离不是语义几何。点蓝色头联动。',35,584);
  },[d,block,head,unit,embedding,activation,headZ,headOverlay,caption,connections,yaw,pitch,panX,panY,zoom,showUnitMarker]);
  const pointer = useScenePointer({
    onPan: (dx, dy, canvas) => setView(v => panFixedCanvas(v, dx, dy, canvas)),
    onRotate: (dx, dy) => setView(v => ({ ...v, yaw: v.yaw + dx / 180, pitch: Math.max(-1, Math.min(1, v.pitch + dy / 180)) })),
    onPick: e => {
      const r=e.currentTarget.getBoundingClientRect(),x=(e.clientX-r.left)*760/r.width,y=(e.clientY-r.top)*600/r.height;
            const candidates=hits.current.filter(h=>Math.hypot(h.x-x,h.y-y)<h.r),hit=candidates.find(h=>h.kind==='head')||candidates.find(h=>h.kind==='block')||candidates.sort((a,b)=>Math.hypot(a.x-x,a.y-y)-Math.hypot(b.x-x,b.y-y))[0];if(hit?.kind==='head')onHead(hit.i);if(hit?.kind==='block')onBlock(hit.i);if(hit?.kind==='unit')onUnit(hit.i)
    },
  });
  const zoomBy = delta => setView(v => ({ ...v, zoom: Math.max(0.7, Math.min(1.7, v.zoom + delta)) }));
  useSceneWheel(ref, delta => zoomBy(delta < 0 ? 0.1 : -0.1));
  return <div><canvas ref={ref} width="760" height="600" style={{width:'100%',maxWidth:'74vh',display:'block',margin:'0 auto',touchAction:'none',cursor:'grab'}} aria-label="真实轨迹三模块3D空间，灰Embedding、蓝Attention、橙MLP"
    title={SCENE_MOUSE_HINT} {...pointer} />
    <div className="v6-toolbar"><button className="fw-tbtn" onClick={()=>zoomBy(0.1)}>放大轨迹</button><button className="fw-tbtn" onClick={()=>zoomBy(-0.1)}>缩小轨迹</button><button className="fw-tbtn" onClick={()=>setView({ yaw: -.45, pitch: 0, zoom: 1 })}>重置轨迹视角</button></div><p className="v8-note">颜色按当前Embedding、MLP、Head各自的最大绝对值缩放，未采集的模块只显示结构位置；板内不按激活排序。</p>
  </div>;
}
