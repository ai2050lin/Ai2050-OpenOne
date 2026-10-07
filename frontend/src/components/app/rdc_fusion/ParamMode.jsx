/* 参数热图模式：单层权重分块统计热图（demo = 结构化种子噪声）
   接入点：torch hook 逐块统计真实 W（如 L{layer} · mlp.down_proj 9728×2560 ≈ 24.9M），
   F#3734 相关行簇以 emerald 高亮（来源：写入端来源贡献分析 P8–P11）。 */
import { useEffect, useRef, useState } from 'react';

function mulberry(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

const COLS=56, ROWS=30;

export default function ParamMode({layer}){
  const L=layer>=0?layer:6;
  const cvRef=useRef(null);
  const [hover,setHover]=useState(null);

  useEffect(()=>{
    const cv=cvRef.current; if(!cv) return;
    const ctx=cv.getContext('2d');
    function resize(){
      const dpr=window.devicePixelRatio||1;
      cv.width=cv.clientWidth*dpr; cv.height=cv.clientHeight*dpr;
      draw();
    }
    function val(r,c){
      const rnd=mulberry(9000+L*131+r*17+c);
      const base=0.30+0.42*rnd();
      const band=0.12*Math.sin(c*0.35+L)+0.08*Math.sin(r*0.5);
      const l6=(L===6&&r>=12&&r<=14)?0.45:0;         /* F#3734 写入行簇（demo 位置） */
      return Math.max(0.03,Math.min(1,base+band*0.4+l6));
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
      const cw=W/COLS, ch=H/ROWS;
      for(let r=0;r<ROWS;r++)for(let c=0;c<COLS;c++){
        ctx.fillStyle=ramp(val(r,c));
        ctx.fillRect(c*cw, r*ch, Math.ceil(cw), Math.ceil(ch));
      }
      /* F#3734 行簇标注框 */
      if(L===6){
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
      ctx.fillText('块均值 0.03', bx-52*dpr, by+bh);
      ctx.fillText('1.00', bx+bw+6*dpr, by+bh);
    }
    resize();
    const onMove=e=>{
      const r=cv.getBoundingClientRect();
      const c=Math.floor((e.clientX-r.left)/r.width*COLS);
      const rr=Math.floor((e.clientY-r.top)/r.height*ROWS);
      if(c>=0&&c<COLS&&rr>=0&&rr<ROWS) setHover({r:rr,c,v:val(rr,c)});
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
  },[L]);

  return (
    <div style={{position:'absolute',inset:0}}>
      <canvas ref={cvRef} className="fw-sp-canvas" style={{cursor:'crosshair'}}/>
      <div className="fw-sp-readout">
        L{L} · <b>mlp.down_proj</b> 块统计（demo）<br/>
        9728 × 2560 ≈ <b>24.9M 参数</b><br/>
        {hover?(<span>块 [r{hover.r}, c{hover.c}] · 值 <b>{hover.v.toFixed(3)}</b></span>):<span style={{color:'#94a3b8'}}>悬停读取块数值</span>}
      </div>
      <div className="fw-param-note">
        <div className="hd"><b>参数热图 · L{L}</b><span className="dim-tag">demo 数据</span></div>
        <div className="hint" style={{marginTop:6}}>
          每格 = 权重矩阵一个分块的均值统计（示意纹理）。<br/>
          接入点：torch hook 逐块统计真实 <b>down_proj</b>；F#3734 行簇位置来自写入端来源贡献（P8–P11），当前为演示排布。
        </div>
      </div>
    </div>
  );
}
