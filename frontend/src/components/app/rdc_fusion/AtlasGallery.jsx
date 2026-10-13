/* AtlasGallery 图谱立体馆（gallery3d.v1 驱动；design/atlas_gallery3d_scheme_v1.md）：
   数据：GET /api/ai-rnd/atlas/tree → gallery3d 段（派生器确定性布局，人手禁写）。
   结构建筑化融合（图形 × 3D）：三座域塔=KN/AR/GR（120° 环布）· 塔内楼层盘=语言族 ·
   证据节点=FTR（E 级着色，√anchors 大小）· 机制竖井=MEC 贯穿柱+横杆连节点 ·
   交叉天桥=X1–X3 虚线弧 · 馆外三圈轨道环=模型 scope 光束 · 时间滑块=建馆回放（首锚 phase 点亮）。
   相机：左键旋转 / 右键平移 / 滚轮缩放 / 闲置自转 / 相机预设（全景·俯瞰·中庭，缓动飞位）。
   交互：hover tooltip；点击选中 → 与全景四模共用同一 sel 与 FeatureDetail 抽屉。
   诚实口径：立体馆=结构示意（派生布局），楼层高度=族序（装饰性）、塔高不编码信息；
   npz 真实坐标接入后仅替换节点 pos，馆体骨架不变。LIVE/DEMO 不伪造。
   相机/交互态走 ref（防 effect 重建渲染循环）；面板 UI 态走 state 并镜像到 ref。 */
import { useEffect, useRef, useState } from 'react';

const API_BASE = (import.meta.env.VITE_API_BASE || 'http://localhost:5001').replace(/\/$/, '');

const E_COLOR = { E3_causal_scoped: '#b45309', E2_predictive: '#15803d', E2_held_out: '#15803d', E1_repeatable: '#24558a' };
const DOMAIN_COLOR = { KN: '#24558a', AR: '#7c3aed', GR: '#0f766e' };
const DOMAIN_NAME = { KN: '知识网络', AR: '分析推理', GR: '语法系统' };
const MODEL_RINGS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b'];       // 轨道环半径自内向外
const RING_R = [20, 24.5, 29];

/* 相机预设（gallery3d §4）：全景 / 俯瞰 / 中庭 */
const CAM_PRESETS = {
  overview: { yaw: 0.7, pitch: 0.42, dist: 46, tgt: [0, 9, 0] },
  topdown:  { yaw: 0.9, pitch: 1.15, dist: 58, tgt: [0, 4, 0] },
  atrium:   { yaw: 0.3, pitch: 0.22, dist: 20, tgt: [0, 5, 0] },
};

export default function AtlasGallery({ sel, onPick, onGo }) {
  const wrapRef = useRef(null);
  const cvRef = useRef(null);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [layers, setLayers] = useState({ tower: true, floor: true, shaft: true, bridge: true, orbit: false });
  const [colorMode, setColorMode] = useState('elevel');
  const [preset, setPreset] = useState('overview');
  const [phaseIdx, setPhaseIdx] = useState(null);
  const [tip, setTip] = useState(null);
  const uiRef = useRef({});
  uiRef.current = { sel, layers, colorMode, phaseIdx, data };
  const cam = useRef({ ...CAM_PRESETS.overview, lastAct: 0 });
  const camGoal = useRef(null);
  const drag = useRef(null);

  useEffect(() => {
    let stop = false;
    fetch(`${API_BASE}/api/ai-rnd/atlas/tree`).then(r => r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)))
      .then(d => { if (!stop) setData(d.gallery3d || null); })
      .catch(e => { if (!stop) setErr(String(e.message || e)); });
    return () => { stop = true; };
  }, []);

  const phases = data ? [...new Set(data.nodes.flatMap(n => n.anchor_phases || []))].sort((a, b) => a - b) : [];
  useEffect(() => { setPhaseIdx(phases.length ? phases.length - 1 : null); }, [data && phases.length]); // eslint-disable-line

  const flyTo = (k) => { setPreset(k); camGoal.current = { ...CAM_PRESETS[k] }; cam.current.lastAct = 0; };

  useEffect(() => {
    const cv = cvRef.current, wrap = wrapRef.current;
    if (!cv || !wrap || !data) return undefined;
    const ctx = cv.getContext('2d');
    const domOf = {};
    data.towers.forEach(t => { domOf[t.domain] = t; });

    let raf = 0, W = 0, H = 0;
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      W = wrap.clientWidth; H = wrap.clientHeight;
      cv.width = W * dpr; cv.height = H * dpr;
      cv.style.width = W + 'px'; cv.style.height = H + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    const project = (x, y, z) => {
      const c = cam.current;
      const dx = x - c.tgt[0], dy = y - c.tgt[1], dz = z - c.tgt[2];
      const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
      const x1 = dx * cy - dz * sy, z1 = dx * sy + dz * cy;
      const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
      const y2 = dy * cp - z1 * sp, z2 = dy * sp + z1 * cp;
      const per = 26 / (26 + z2 + c.dist);
      return [W / 2 + x1 * per * 9, H / 2 - y2 * per * 9, per];
    };
    const ell = (x, y, z, r, fill, stroke, lw) => {
      const p = project(x, y, z);
      ctx.beginPath(); ctx.ellipse(p[0], p[1], r * p[2] * 9, r * p[2] * 9 * 0.32, 0, 0, Math.PI * 2);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1; ctx.stroke(); }
      return p;
    };
    const line = (a, b, style, width, dash) => {
      const p1 = project(...a), p2 = project(...b);
      ctx.strokeStyle = style; ctx.lineWidth = width;
      ctx.setLineDash(dash || []); ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke(); ctx.setLineDash([]);
    };

    const draw = () => {
      const ui = uiRef.current;
      const c = cam.current;
      if (camGoal.current) {                                   // 预设缓动飞位
        const g = camGoal.current, L = (a, b) => a + (b - a) * 0.09;
        c.yaw = L(c.yaw, g.yaw); c.pitch = L(c.pitch, g.pitch); c.dist = L(c.dist, g.dist);
        c.tgt = c.tgt.map((v, i) => L(v, g.tgt[i]));
        if (Math.abs(c.yaw - g.yaw) < 0.002 && Math.abs(c.dist - g.dist) < 0.05) camGoal.current = null;
      } else if (!drag.current && performance.now() - c.lastAct > 2500) {
        c.yaw += 0.002;                                        // 闲置自转
      }
      ctx.clearRect(0, 0, W, H);

      /* 参考轴 */
      line([-30, 0, 0], [30, 0, 0], '#dde3ea', 1); line([0, -2, 0], [0, 26, 0], '#e3e8ee', 1); line([0, 0, -30], [0, 0, 30], '#dde3ea', 1);

      /* 馆外模型轨道环（地面三圈同心，虚线） */
      if (ui.layers.orbit) {
        RING_R.forEach((r, i) => {
          const on = true;
          ell(0, 0, 0, r, null, on ? (i % 2 ? '#9db6cf' : '#b7c9da') : '#e3e8ee', 1);
          ctx.setLineDash([]);
          const lp = project(r * Math.cos(-0.5), 0, r * Math.sin(-0.5));
          ctx.fillStyle = '#52616f'; ctx.font = '10px ui-monospace,Consolas,monospace';
          ctx.fillText(MODEL_RINGS[i], lp[0] + 4, lp[1]);
        });
      }

      /* 三座域塔（半透明线框壳：底/顶椭圆 + 切线） */
      if (ui.layers.tower) {
        data.towers.forEach(t => {
          const col = DOMAIN_COLOR[t.domain] || '#8a93a3';
          const yTop = 3.0 + (t.n_floors - 1) * 6.0 + 5.0;
          const pb = ell(t.pos[0], 0, t.pos[1], 6, col + '10', col + '66', 1);
          const pt = ell(t.pos[0], yTop, t.pos[1], 6, col + '08', col + '55', 1);
          line([t.pos[0] - 6, 0, t.pos[1]], [t.pos[0] - 6, yTop, t.pos[1]], col + '44', 1);
          line([t.pos[0] + 6, 0, t.pos[1]], [t.pos[0] + 6, yTop, t.pos[1]], col + '44', 1);
          ctx.fillStyle = col; ctx.font = '11px system-ui';
          ctx.fillText(DOMAIN_NAME[t.domain] + ' · ' + t.n_features + ' 特征', pt[0] - 40, pt[1] - 8);
        });
      }

      /* 证据回放截止相位（建馆：节点按首锚 phase 逐层点亮） */
      const cutoff = ui.phaseIdx == null ? Infinity : phases[ui.phaseIdx];
      const lit = n => (n.anchor_phases || []).length && Math.min(...n.anchor_phases) <= cutoff;

      /* 楼层盘 + 族名匾额 */
      if (ui.layers.floor) {
        data.floors.forEach(f => {
          const col = DOMAIN_COLOR[f.tower] || '#8a93a3';
          const p = ell(f.center[0], f.center[1], f.center[2], f.radius, col + '22', col + '77', 1);
          ctx.fillStyle = col; ctx.font = '10.5px system-ui';
          ctx.fillText(f.family + ' (' + f.n_features + ')', p[0] - f.radius * p[2] * 9, p[1] - f.radius * p[2] * 9 * 0.32 - 4);
        });
      }

      /* 交叉天桥（两塔顶间虚线弧）+ 桥面节点平台 */
      if (ui.layers.bridge) {
        data.bridges.forEach(b => {
          const ta = domOf[b.from], tb = domOf[b.to];
          const yA = 3.0 + (ta.n_floors - 1) * 6.0 + 5.0, yB = 3.0 + (tb.n_floors - 1) * 6.0 + 5.0;
          const p1 = project(ta.pos[0], yA, ta.pos[1]), p2 = project(tb.pos[0], yB, tb.pos[1]);
          const pm = project(b.mid[0], b.mid[1], b.mid[2]);
          ctx.strokeStyle = 'rgba(122,132,148,0.65)'; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
          ctx.beginPath(); ctx.moveTo(p1[0], p1[1]);
          ctx.quadraticCurveTo((p1[0] + p2[0]) / 2, Math.min(p1[1], p2[1]) - 30, p2[0], p2[1]);
          ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = '#52616f'; ctx.font = '10px ui-monospace,Consolas,monospace';
          ctx.fillText(b.edge + '·' + b.type, pm[0] + 8, pm[1] - 8);
          ell(b.mid[0], b.mid[1] - 1.2, b.mid[2], 1.8, '#f4f6f9cc', '#9aa4b2', 0.8);
        });
      }

      /* 机制竖井（柱 + 横杆连各特征节点） */
      if (ui.layers.shaft) {
        data.shafts.forEach(sh => {
          line(sh.base, [sh.base[0], sh.top, sh.base[2]], 'rgba(124,58,237,0.55)', 2);
          ell(sh.base[0], 0, sh.base[2], 0.9, 'rgba(124,58,237,0.15)', 'rgba(124,58,237,0.6)', 1);
        });
      }

      /* 证据节点（深度排序：远→近）+ 竖井横杆 */
      const picks = [];
      const alive = data.nodes.filter(lit);
      const projd = alive.map(n => ({ n, p: project(...n.pos.xyz) }));
      projd.sort((a, b) => a.p[2] - b.p[2]);
      if (ui.layers.shaft) {
        projd.forEach(q => {
          const n = q.n;
          (n.shafts || []).forEach(mid => {
            const sh = data.shafts.find(s => s.id === mid);
            if (!sh) return;
            line([sh.base[0], n.pos.xyz[1], sh.base[2]], n.pos.xyz, 'rgba(124,58,237,0.28)', 1);
          });
          if (ui.layers.orbit) {
            const th = Math.atan2(n.pos.xyz[2], n.pos.xyz[0]);
            (n.models || []).forEach(m => {
              const ri = MODEL_RINGS.indexOf(m);
              if (ri < 0) return;
              line(n.pos.xyz, [RING_R[ri] * Math.cos(th), 0.1, RING_R[ri] * Math.sin(th)], 'rgba(36,85,138,0.16)', 1);
            });
          }
        });
      }
      projd.forEach(({ n, p }) => {
        const r = (3.2 + 1.15 * Math.sqrt(n.n_anchors || 0)) * Math.max(0.55, p[2]);
        const col = ui.colorMode === 'domain'
          ? (DOMAIN_COLOR[(data.floors.find(f => f.family === n.family) || {}).tower] || '#8a93a3')
          : (E_COLOR[n.e_level] || '#8a93a3');
        ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        ctx.fillStyle = col; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1; ctx.stroke();
        if (ui.sel === n.ftr_id) {
          ctx.beginPath(); ctx.arc(p[0], p[1], r + 4.5, 0, Math.PI * 2);
          ctx.strokeStyle = '#b45309'; ctx.lineWidth = 2; ctx.stroke();
        }
        picks.push({ x: p[0], y: p[1], n, r });
      });

      /* 相位标签 */
      if (ui.phaseIdx != null) {
        ctx.fillStyle = '#8a93a3'; ctx.font = '10px ui-monospace,Consolas,monospace';
        ctx.fillText('建馆回放 · 截至 P' + cutoff + ' · 点亮 ' + picks.length + '/' + data.nodes.length, 12, H - 34);
      }

      hoverRef.current.picks = picks;
      raf = requestAnimationFrame(draw);
    };

    /* 指针交互（与 AtlasSpace 同范式） */
    const hoverRef = { current: { picks: [], x: -1, y: -1 } };
    const onDown = e => { drag.current = { btn: e.button, x: e.clientX, y: e.clientY, moved: 0 }; cam.current.lastAct = performance.now(); camGoal.current = null; };
    const onMove = e => {
      const rect = cv.getBoundingClientRect();
      hoverRef.current.x = e.clientX - rect.left; hoverRef.current.y = e.clientY - rect.top;
      const d = drag.current;
      if (!d) { hoverPick(); return; }
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      d.moved += Math.abs(dx) + Math.abs(dy);
      cam.current.lastAct = performance.now();
      if (d.btn === 2 || e.shiftKey) {
        const c = cam.current, s = c.dist * 0.0016;
        const cy = Math.cos(c.yaw), sy = Math.sin(c.yaw);
        c.tgt[0] -= dx * cy * s; c.tgt[2] -= dx * sy * s; c.tgt[1] += dy * s;
      } else {
        cam.current.yaw += dx * 0.0055;
        cam.current.pitch = Math.max(-1.35, Math.min(1.35, cam.current.pitch + dy * 0.0045));
      }
      d.x = e.clientX; d.y = e.clientY;
    };
    const hoverPick = () => {
      const h = hoverRef.current;
      let best = null, bd = 12;
      (h.picks || []).forEach(q => {
        const dd = Math.hypot(q.x - h.x, q.y - h.y) - q.r;
        if (dd < bd) { bd = dd; best = q; }
      });
      setTip(cur => {
        const nx = best ? { x: h.x, y: h.y, fid: best.n.ftr_id, stmt: best.n.statement } : null;
        return (cur && nx && cur.fid === nx.fid) || (!cur && !nx) ? cur : nx;
      });
    };
    const onUp = e => {
      const d = drag.current;
      drag.current = null;
      if (d && d.moved < 5) {
        const h = hoverRef.current;
        let best = null, bd = 14;
        (h.picks || []).forEach(q => {
          const dd = Math.hypot(q.x - h.x, q.y - h.y) - q.r;
          if (dd < bd) { bd = dd; best = q; }
        });
        if (best && onPick) onPick(best.n.ftr_id);
      }
    };
    const onWheel = e => {
      e.preventDefault();
      cam.current.dist = Math.max(10, Math.min(95, cam.current.dist * Math.exp(e.deltaY * 0.001)));
      cam.current.lastAct = performance.now(); camGoal.current = null;
    };
    const onCtx = e => e.preventDefault();

    cv.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    cv.addEventListener('wheel', onWheel, { passive: false });
    cv.addEventListener('contextmenu', onCtx);
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf); ro.disconnect();
      cv.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      cv.removeEventListener('wheel', onWheel);
      cv.removeEventListener('contextmenu', onCtx);
    };
  }, [data, phases, onPick]);

  const LK = [['tower', '塔体'], ['floor', '楼层匾额'], ['shaft', '机制竖井'], ['bridge', '交叉天桥'], ['orbit', '模型轨道环']];
  const selPt = data ? data.nodes.find(n => n.ftr_id === sel) : null;

  return (
    <div className="at-space" ref={wrapRef}>
      {!data && !err && <div className="at-space-empty">图谱立体馆数据载入中…</div>}
      {!data && err && (
        <div className="at-space-empty">图谱服务不可达（{err}）——启动 :5001 后切换视图重试。<br/>
          <small>立体馆由派生档 atlas_tree_v1.json（gallery3d 段）驱动，不伪造数据。</small></div>
      )}
      <canvas ref={cvRef} className="at-space-cv" style={data ? undefined : { display: 'none' }}/>
      {data && (
        <>
          <div className="at-space-cap">gallery3d.v1 · <b>结构示意（派生布局）</b> — 楼层高度=族序（装饰性）· 塔高不编码信息 · npz 接入后仅替换节点坐标</div>
          <div className="at-space-panel">
            <b>图层板</b>
            {LK.map(([k, lb]) => (
              <label key={k}><input type="checkbox" checked={layers[k]}
                onChange={e => setLayers(o => ({ ...o, [k]: e.target.checked }))}/> {lb}</label>
            ))}
            <div className="at-space-seg">
              <button type="button" className={colorMode === 'elevel' ? 'on' : ''}
                onClick={() => setColorMode('elevel')}>E级</button>
              <button type="button" className={colorMode === 'domain' ? 'on' : ''}
                onClick={() => setColorMode('domain')}>域</button>
            </div>
            <div className="at-gal-presets" title="相机预设：全景俯瞰 / 高角俯瞰 / 中庭特写（缓动飞位）">
              {[['overview', '全景'], ['topdown', '俯瞰'], ['atrium', '中庭']].map(([k, lb]) => (
                <button key={k} type="button" className={preset === k ? 'on' : ''} onClick={() => flyTo(k)}>{lb}</button>
              ))}
            </div>
            <div className="at-space-legend">
              {colorMode === 'elevel'
                ? <><i style={{ background: E_COLOR.E3_causal_scoped }}/>E3<i style={{ background: E_COLOR.E2_predictive }}/>E2<i style={{ background: E_COLOR.E1_repeatable }}/>E1</>
                : <><i style={{ background: DOMAIN_COLOR.KN }}/>KN<i style={{ background: DOMAIN_COLOR.AR }}/>AR<i style={{ background: DOMAIN_COLOR.GR }}/>GR</>}
            </div>
          </div>
          {sel && selPt && <div className="at-space-selhint" title="详情见下方特征抽屉（与全景四模共用）">
            <span className="mono">{sel}</span> 已选中 · 位置={selPt.home} · 跨模型连线={(selPt.models || []).length}
          </div>}
          <div className="at-space-time">
            <b>建馆回放</b>
            <input type="range" min={0} max={Math.max(0, phases.length - 1)} value={phaseIdx ?? 0}
              disabled={!phases.length}
              onChange={e => setPhaseIdx(Number(e.target.value))}/>
            <span className="mono">{phases.length ? 'P' + phases[phaseIdx ?? phases.length - 1] : '—'}</span>
            <span className="dim">拖回过去 = 截至 该 phase 的馆</span>
          </div>
          {tip && (
            <div className="at-space-tip" style={{ left: tip.x + 12, top: tip.y + 10 }}>
              <b>{tip.fid}</b><span>{tip.statement}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
