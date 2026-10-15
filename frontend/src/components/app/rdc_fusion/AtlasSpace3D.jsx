/* 特征谱图空间（atlas.v1 → graphs[].coords[层]）。每个点 = 一个特征：某模式族某因子某水平在该层的对比方向；
   坐标 = 该层全部特征单位方向的经典 MDS（距离 √(2−2cos)），方向相近的特征靠在一起。只在同一模型内有意义。
   颜色 = 模式族（类别色板固定顺序）；实心 = 稳定特征（未见词表复现），空心环 = 不稳定，菱形 = 只出现在留出句式的水平；
   点大小 = 效应/噪声；连线 = 跨族边（同一共享编码轴的成员之间），轴在成员中心处直接标注概念名。
   拖动旋转、滚轮缩放、悬停查看、点击选中。 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { familyPalette } from './atlasUtil.js';

const OTHER = '#94a3b8';

function drawNode(ctx, shape, x, y, r, color) {
  ctx.beginPath();
  if (shape === 'diamond') {
    ctx.moveTo(x, y - r * 1.3); ctx.lineTo(x + r * 1.3, y); ctx.lineTo(x, y + r * 1.3); ctx.lineTo(x - r * 1.3, y); ctx.closePath();
    ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#ffffff'; ctx.stroke();
  } else if (shape === 'ring') {
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff'; ctx.fill(); ctx.lineWidth = 2.2; ctx.strokeStyle = color; ctx.stroke();
  } else {
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = '#ffffff'; ctx.stroke();
  }
}

export default function AtlasSpace3D({ graph, layer, axisId, nodeId, onSelectNode, showUnstable = true, height = 540 }) {
  const wrapRef = useRef(null);
  const cvRef = useRef(null);
  const cam = useRef({ yaw: 0.6, pitch: 0.3, zoom: 1, drag: null, moved: false });
  const projected = useRef([]);
  const [hover, setHover] = useState(null);
  const [tick, setTick] = useState(0);

  const scene = useMemo(() => {
    const coords = graph.coords?.[layer];
    if (!coords) return null;
    const pal = familyPalette(graph.families);
    const axes = (graph.axes || []).filter(a => a.layer === layer);
    const axisOf = {};
    axes.forEach(a => a.members.forEach(m => { axisOf[m] = a; }));
    const pts = Object.values(coords);
    const maxR = Math.max(1e-6, ...pts.map(p => Math.hypot(p[0], p[1], p[2])));
    const k = 1.8 / maxR;
    const nodes = graph.nodes.filter(n => n.layer === layer && coords[n.id] && (showUnstable || n.stable)).map(n => {
      const c = coords[n.id];
      const snr = n.snr || 0;
      const r = 4 + Math.max(0, Math.min(1, Math.log10(1 + snr) / 2)) * 4;
      return { n, x: c[0] * k, y: c[1] * k, z: c[2] * k, color: pal[n.family] || OTHER, r,
        shape: n.heldout_only ? 'diamond' : n.stable ? 'dot' : 'ring', axis: axisOf[n.id] };
    });
    const byId = Object.fromEntries(nodes.map(p => [p.n.id, p]));
    const edges = (graph.edges || []).filter(e => e.layer === layer && byId[e.source] && byId[e.target]);
    return { nodes, byId, edges, axes, pal };
  }, [graph, layer, showUnstable]);

  useEffect(() => {
    const cv = cvRef.current;
    const wrap = wrapRef.current;
    if (!cv || !wrap || !scene) return;
    const dpr = window.devicePixelRatio || 1;
    const W = wrap.clientWidth, H = height;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const { yaw, pitch, zoom } = cam.current;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const scale = Math.min(W, H) * 0.24 * zoom;
    const proj = (x, y, z) => {
      const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
      const y1 = y * cp - z1 * sp, z2 = y * sp + z1 * cp;
      const f = 6 / (6 + z2);
      return [W / 2 + x1 * scale * f, H / 2 - y1 * scale * f, z2, f];
    };
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, W, H);
    // 参考轴（MDS 1–3）
    [[2, 0, 0], [0, 2, 0], [0, 0, 2]].forEach((v, i) => {
      const a = proj(-v[0], -v[1], -v[2]), b = proj(v[0], v[1], v[2]);
      ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      ctx.fillStyle = '#94a3b8'; ctx.font = '10px system-ui, sans-serif';
      ctx.fillText(`MDS${i + 1}`, b[0] + 4, b[1] - 4);
    });
    const P = Object.fromEntries(scene.nodes.map(p => [p.n.id, proj(p.x, p.y, p.z)]));
    const focusAxis = axisId || null;
    const focusNode = nodeId || null;
    // 边：同轴成员之间；选中轴高亮，其余变淡
    scene.edges.forEach(e => {
      const a = P[e.source], b = P[e.target];
      const on = !focusAxis || e.axis === focusAxis;
      ctx.strokeStyle = on ? 'rgba(15,23,42,0.62)' : 'rgba(148,163,184,0.22)';
      ctx.lineWidth = on ? 1 + 2.5 * Math.max(0, e.cos || 0) : 1;
      ctx.setLineDash(e.declared ? [] : [5, 4]);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    });
    ctx.setLineDash([]);
    // 点（远处淡化；不在选中轴的点变淡）
    const drawn = scene.nodes.map(p => ({ p, q: P[p.n.id] })).sort((a, b) => b.q[2] - a.q[2]);
    drawn.forEach(({ p, q }) => {
      const dim = focusAxis && p.axis?.id !== focusAxis;
      ctx.globalAlpha = dim ? 0.18 : Math.max(0.45, Math.min(1, 0.6 + q[3] * 0.4));
      drawNode(ctx, p.shape, q[0], q[1], p.r, p.color);
      if (focusNode === p.n.id) {
        ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = '#0f172a';
        ctx.beginPath(); ctx.arc(q[0], q[1], p.r + 4, 0, Math.PI * 2); ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
    projected.current = drawn.map(({ p, q }) => ({ p, sx: q[0], sy: q[1] }));
    // 共享轴直接标注：成员中心处标概念名（未预注册 = 新发现）
    scene.axes.forEach(ax => {
      const ms = ax.members.map(m => P[m]).filter(Boolean);
      if (!ms.length) return;
      const x = ms.reduce((s, q) => s + q[0], 0) / ms.length, y = ms.reduce((s, q) => s + q[1], 0) / ms.length;
      const text = `${ax.concepts.length ? ax.concepts.join('/') : '新发现'}${ax.supported ? '' : '（未达判据）'}`;
      ctx.font = '600 11px system-ui, sans-serif';
      const w = ctx.measureText(text).width + 12;
      const on = !focusAxis || ax.id === focusAxis;
      ctx.globalAlpha = on ? 1 : 0.35;
      ctx.fillStyle = '#ffffff'; ctx.strokeStyle = ax.supported ? '#0f172a' : '#94a3b8'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(x - w / 2, y - 26, w, 18, 9); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#0f172a'; ctx.fillText(text, x - w / 2 + 6, y - 13);
      ctx.globalAlpha = 1;
    });
  }, [scene, height, tick, axisId, nodeId]);

  useEffect(() => {
    const onResize = () => setTick(t => t + 1);
    window.addEventListener('resize', onResize);
    const cv = cvRef.current;
    const onWheel = e => {
      e.preventDefault();
      cam.current.zoom = Math.max(0.3, Math.min(5, cam.current.zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
      setTick(t => t + 1);
    };
    if (cv) cv.addEventListener('wheel', onWheel, { passive: false });
    return () => { window.removeEventListener('resize', onResize); if (cv) cv.removeEventListener('wheel', onWheel); };
  }, [scene]);

  if (!scene) return <div className="v6-empty">该层没有谱图坐标。</div>;

  const nearest = (e) => {
    const box = cvRef.current.getBoundingClientRect();
    const mx = e.clientX - box.left, my = e.clientY - box.top;
    let best = null, bd = 144;
    for (const it of projected.current) {
      const d = (it.sx - mx) ** 2 + (it.sy - my) ** 2;
      if (d < bd) { bd = d; best = it; }
    }
    return { best, mx, my };
  };
  const onDown = e => { cam.current.drag = { x: e.clientX, y: e.clientY, yaw: cam.current.yaw, pitch: cam.current.pitch }; cam.current.moved = false; };
  const onUp = e => {
    const d = cam.current.drag;
    cam.current.drag = null;
    if (d && !cam.current.moved && e) {
      const { best } = nearest(e);
      onSelectNode?.(best ? best.p.n.id : null);
    }
  };
  const onMove = e => {
    const d = cam.current.drag;
    if (d) {
      if (Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 3) cam.current.moved = true;
      cam.current.yaw = d.yaw + (e.clientX - d.x) * 0.01;
      cam.current.pitch = Math.max(-1.4, Math.min(1.4, d.pitch + (e.clientY - d.y) * 0.01));
      setTick(t => t + 1);
      return;
    }
    const { best, mx, my } = nearest(e);
    setHover(best ? { x: mx, y: my, p: best.p, scene } : null);
  };
  const fams = Object.entries(scene.pal);
  return (
    <div className="v6-space3d v6-atlas3d" ref={wrapRef}>
      <canvas ref={cvRef} onMouseDown={onDown} onMouseUp={onUp} onMouseLeave={() => { cam.current.drag = null; setHover(null); }}
        onMouseMove={onMove} role="img" aria-label={`特征谱图空间：${layer} 层，${scene.nodes.length} 个特征，${scene.edges.length} 条跨族边`} />
      {hover && hover.scene === scene && (
        <div className="v6-tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>
          {hover.p.n.family} · {hover.p.n.factor}={hover.p.n.level}{hover.p.n.concept ? `（概念：${hover.p.n.concept}）` : ''}{'\n'}
          {hover.p.n.heldout_only ? '只出现在留出句式（未见句式预测对象）' : `未见词表复现 cos ${hover.p.n.rep ?? '—'} · 效应/噪声 ${hover.p.n.snr ?? '—'}`}{'\n'}
          {hover.p.axis ? `共享轴：${hover.p.axis.label}` : '未并入共享轴'}
        </div>
      )}
      <div className="v6-space3d-legend">
        <div>{fams.map(([f, c]) => <span key={f}><i style={{ background: c }} />{f}</span>)}</div>
        <div>
          <span><i className="shape dot" />稳定特征</span>
          <span><i className="shape ring" />未复现</span>
          <span><i className="shape diamond" />留出句式水平</span>
          <span><i className="v6-line-key" />预注册边</span>
          <span><i className="v6-line-key dashed" />未预注册边</span>
        </div>
      </div>
    </div>
  );
}
