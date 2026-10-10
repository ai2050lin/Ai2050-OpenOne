/* 参数热图模式：单层权重分块统计热图（双轨）
   ── live（实时分析已连接本地模型）：POST /api/live/tensor 懒加载真实权重——
      32×64 行块切片、超限均匀抽样；格值=行带×列带的 |W| 均值，色标按实际值域归一。
   ── demo（未连接）：结构化种子噪声示意（原 demo 保留，显式标注）。
   F#3734 行簇高亮仅 demo（真实位置来自写入端来源贡献分析 P8–P11）。 */
import { useEffect, useRef, useState } from 'react';

function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const COLS=56, ROWS=30;   /* demo 网格 */

export default function ParamMode({layer, live, scan}){
  const L=layer>=0?layer:6;
  const cvRef=useRef(null);
  const [hover,setHover]=useState(null);
  const [wt,setWt]=useState(null);          // 真实 tensor 详情 {name,shape,dtype,stats,heatmap,rownorm}
  const [wtErr,setWtErr]=useState('');

  const recName = scan && scan.layers ? (scan.layers.find(x=>x.layer===L)||{}).name : null;

  useEffect(()=>{
    if(!live||!recName){ setWt(null); setWtErr(''); return; }
    let stop=false; setWt(null); setWtErr('');
    fetch(`${API_BASE}/api/live/tensor`,{
      method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:recName}),
    }).then(async r=>{
      const d=await r.json().catch(()=>({}));
      if(!r.ok) throw new Error(d.detail||('HTTP '+r.status));
      if(!stop) setWt(d);
    }).catch(e=>{ if(!stop) setWtErr(String(e.message||e)); });
    return ()=>{ stop=true; };
  },[live, recName]);

  useEffect(()=>{
    const cv=cvRef.current; if(!cv) return;
    const ctx=cv.getContext('2d');
    const grid=wt?wt.heatmap:null;
    const NC=grid?grid.cols:COLS, NR=grid?grid.rows:ROWS;
    let vmax=1;
    if(grid) vmax=Math.max(...grid.data, 1e-12);

    function resize(){
      const dpr=window.devicePixelRatio||1;
      cv.width=cv.clientWidth*dpr; cv.height=cv.clientHeight*dpr;
      draw();
    }
    /* demo 值（原种子）；live 值=|W| 行带×列带均值 → /vmax 归一进色带 */
    function val(r,c){
      if(grid) return (grid.data[r*NC+c]||0)/vmax;
      const rnd=mulberry(9000+L*131+r*17+c);
      const base=0.30+0.42*rnd();
      const band=0.12*Math.sin(c*0.35+L)+0.08*Math.sin(r*0.5);
      const l6=(L===6&&r>=12&&r<=14)?0.45:0;         /* F#3734 写入行簇（demo 位置） */
      return Math.max(0.03,Math.min(1,base+band*0.4+l6));
    }
    function rawVal(r,c){
      if(grid) return grid.data[r*NC+c]||0;
      return val(r,c);
    }
    function ramp(v){
      /* 白→sky 蓝基础段；v>0.7 转 emerald 高亮（本项目高亮惯例） */
      if(v>0.7){
        const t=Math.min(1,(v-0.7)/0.3);
        const r=Math.round(2+(52-2)*t), g=Math.round(132+(211-132)*t), b=Math.round(199+(153-199)*t);
        return 'rgb('+r+','+g+','+b+')';
      }
      const t=v/0.7;
      const r=Math.round(241-(241-2)*t), g=Math.round(245-(245-132)*t), b=Math.round(248-(248-199)*t);
      return 'rgb('+r+','+g+','+b+')';
    }
    function draw(){
      const dpr=window.devicePixelRatio||1;
      const W=cv.width, H=cv.height;
      ctx.clearRect(0,0,W,H);
      const cw=W/NC, ch=H/NR;
      for(let r=0;r<NR;r++)for(let c=0;c<NC;c++){
        ctx.fillStyle=ramp(val(r,c));
        ctx.fillRect(c*cw, r*ch, Math.ceil(cw), Math.ceil(ch));
      }
      /* F#3734 行簇标注框（仅 demo：真实位置待来源贡献接入） */
      if(!grid&&L===6){
        ctx.strokeStyle='#059669'; ctx.lineWidth=2*dpr;
        ctx.strokeRect(0,12*ch,W,3*ch);
        ctx.fillStyle='#047857'; ctx.font='bold '+(10*dpr)+'px ui-monospace,Consolas,monospace';
        ctx.fillText('F#3734 写入行簇（demo 位置 · 接入来源贡献后精确）', 8*dpr, 12*ch-6*dpr);
      }
      /* 色标条 */
      const bw=180*dpr, bh=8*dpr, bx=W-bw-20*dpr, by=H-24*dpr;
      for(let i=0;i<60;i++){
        ctx.fillStyle=ramp(0.03+i/59*0.97);
        ctx.fillRect(bx+i/60*bw, by, Math.ceil(bw/60), bh);
      }
      ctx.fillStyle='#475569'; ctx.font=(9*dpr)+'px ui-monospace,Consolas,monospace';
      if(grid){
        ctx.fillText('|W| 块均值 0', bx-70*dpr, by+bh);
        ctx.fillText(vmax.toExponential(2), bx+bw+6*dpr, by+bh);
      }else{
        ctx.fillText('块均值 0.03', bx-52*dpr, by+bh);
        ctx.fillText('1.00', bx+bw+6*dpr, by+bh);
      }
    }
    resize();
    const onMove=e=>{
      const r=cv.getBoundingClientRect();
      const c=Math.floor((e.clientX-r.left)/r.width*NC);
      const rr=Math.floor((e.clientY-r.top)/r.height*NR);
      if(c>=0&&c<NC&&rr>=0&&rr<NR) setHover({r:rr,c,v:rawVal(rr,c)});
      else setHover(null);
    };
    const onLeave=()=>setHover(null);
    const ro=new ResizeObserver(()=>resize());
    ro.observe(cv);
    cv.addEventListener('mousemove',onMove);
    cv.addEventListener('mouseleave',onLeave);
    return ()=>{
      ro.disconnect();
      cv.removeEventListener('mousemove',onMove);
      cv.removeEventListener('mouseleave',onLeave);
    };
  },[L, wt]);

  const st=wt&&wt.stats;
  return (
    <div style={{position:'absolute',inset:0}}>
      <canvas ref={cvRef} className="fw-sp-canvas" style={{cursor:'crosshair'}}/>
      <div className="fw-sp-readout">
        {wt&&recName?(<>
          <b className="fw-mono">{recName}</b>（真实权重）<br/>
          shape [{(wt.shape||[]).join('×')}] · {wt.dtype} · {st.sampled?'32×64 行块抽样':'全量遍历'}<br/>
          RMS <b>{st.rms}</b> · mean|W| <b>{st.mean_abs}</b> · max|W| <b>{st.max_abs}</b><br/>
          {hover?(<span>带 [r{hover.r}, c{hover.c}] · |W| 均值 <b>{hover.v.toExponential(3)}</b></span>)
                :<span style={{color:'#94a3b8'}}>悬停读取块数值</span>}
        </>):(
          <>
            L{L} · <b>mlp.down_proj</b> 块统计（demo）<br/>
            9728 × 2560 ≈ <b>24.9M 参数</b><br/>
            {wtErr?<span style={{color:'#b91c1c'}}>数值加载失败：{wtErr}</span>
                  :(hover?(<span>块 [r{hover.r}, c{hover.c}] · 值 <b>{hover.v.toFixed(3)}</b></span>)
                          :<span style={{color:'#94a3b8'}}>悬停读取块数值</span>)}
          </>
        )}
      </div>
      <div className="fw-param-note">
        <div className="hd"><b>参数热图 · L{L}</b>
          <span className="dim-tag" style={wt?{color:'#047857',background:'#ecfdf5'}:null}>{wt?'真实权重':(live&&recName?'数值加载中…':'demo 数据')}</span>
        </div>
        {wt&&<div className="hint" style={{marginTop:6}}>
          格值=行带×列带的 |W| 均值（safetensors 懒加载，不整体加载权重）；
          行维 {wt.heatmap.rows} 带 × 列维 {wt.heatmap.cols} 带{wt.rownorm&&(' · 行范数 '+wt.rownorm.n+' 点降采样')}。
        </div>}
        {live&&!recName&&!wtErr&&<div className="hint" style={{marginTop:6}}>该层未找到 mlp.down_proj 权重——当前显示 demo。</div>}
        {!live&&<div className="hint" style={{marginTop:6}}>
          每格 = 权重矩阵一个分块的均值统计（示意纹理）。<br/>
          连接本地模型后自动切换为真实权重数值（懒加载 32×64 行块）；F#3734 行簇位置来自写入端来源贡献（P8–P11），当前为演示排布。
        </div>}
      </div>
    </div>
  );
}
