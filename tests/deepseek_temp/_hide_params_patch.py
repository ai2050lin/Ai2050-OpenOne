# -*- coding: utf-8 -*-
"""M9g: 隐藏「全部参数」浏览器（LiveAnalysis），注释保留可恢复。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\frontend\src\components\app\rdc_fusion\LensSpatial.jsx'
s = io.open(P, encoding='utf-8', newline='').read()


def sub_once(s, old, new, tag):
    for joiner in ('\n', '\r\n'):
        oo = old.replace('\n', joiner)
        nn = new.replace('\n', joiner)
        if oo in s:
            assert s.count(oo) == 1, (tag, 'COUNT', s.count(oo))
            return s.replace(oo, nn)
    raise AssertionError(tag + ' MISS')


# ── op1 fmtW 注释（模块级，仅参数表数值列引用） ──
s = sub_once(s,
    "/* 权重标量数值显示：rms 一般 1e-2 量级 */\n"
    "const fmtW = (v) => v == null ? '—' : (v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(2) : String(+v.toFixed(4)));",
    "/* 权重标量数值显示（「全部参数」隐藏后暂无引用；恢复参数表数值列时取消注释）\n"
    "const fmtW = (v) => v == null ? '—' : (v !== 0 && Math.abs(v) < 0.001 ? v.toExponential(2) : String(+v.toFixed(4)));\n"
    "*/",
    'fmtW')

# ── op2 状态注释 ──
s = sub_once(s,
    "  const [filter, setFilter] = useState('');\n"
    "  const [open, setOpen] = useState({});        // prefix → bool；未登记时默认展开第一组\n"
    "  const [wt, setWt] = useState({});            // name → 真实权重标量统计（展开组时懒加载 summary）",
    "  /* 「全部参数」已隐藏（2026-10-09）：过滤/展开/权重懒加载状态随模块停用（恢复时取消注释）\n"
    "  const [filter, setFilter] = useState('');\n"
    "  const [open, setOpen] = useState({});        // prefix → bool；未登记时默认展开第一组\n"
    "  const [wt, setWt] = useState({});            // name → 真实权重标量统计（展开组时懒加载 summary）\n"
    "  */",
    'state')

# ── op3 头注释同步 ──
s = sub_once(s,
    "/* 实时分析 · 连接本机模型并浏览全部参数",
    "/* 实时分析 · 连接本机模型（3D 舞台 + 真实权重数值；「全部参数」浏览器已隐藏）",
    'head1')
s = sub_once(s,
    "   middle 插槽 = 共享 3D 舞台，插在「模型总览统计」与「全部参数浏览器」之间。 */",
    "   middle 插槽 = 共享 3D 舞台（置顶）。 */",
    'head2')

# ── op4 连接条注释文案 ──
s = sub_once(s,
    "      {/* 未连接时的精简连接条（连接成功后整条隐藏，界面回到 3D + 统计 + 参数表） */}",
    "      {/* 未连接时的精简连接条（连接成功后整条隐藏，界面回到 3D 舞台 + 右侧模型总览） */}",
    'connbar')

# ── op5 顶层模式描述 ──
s = sub_once(s,
    "  { k: 'live', t: '实时分析', d: '连接本地模型 · 全部参数' },",
    "  { k: 'live', t: '实时分析', d: '连接本地模型 · 真实权重数值' },",
    'topmode')

# ── op6 派生逻辑 + 懒加载 useEffect 注释（内嵌块注释 */ 先转 //） ──
s = sub_once(s,
    "  const data = conn || LIVE_MODEL_DEMO;\n"
    "  const f = filter.trim().toLowerCase();\n"
    "  const groups = f\n"
    "    ? (data.groups || [])\n"
    "        .map(g => ({ ...g, tensors: g.tensors.filter(t => t.name.toLowerCase().includes(f)) }))\n"
    "        .filter(g => g.tensors.length || g.prefix.toLowerCase().includes(f))\n"
    "        .map(g => (g.tensors.length ? g : { ...g, tensors: g.tensors }))\n"
    "    : (data.groups || []);\n"
    "  const isOpen = (g, i) => (f ? true : open[g.prefix] !== undefined ? open[g.prefix] : i === 0);\n"
    "  const toggle = (p) => setOpen(o => ({ ...o, [p]: !(o[p] !== undefined ? o[p] : false) }));\n"
    "\n"
    "  /* 展开组时懒加载真实权重标量（POST /live/weights/summary：32×64 行块切片，超限均匀抽样） */\n"
    "  useEffect(() => {\n"
    "    if (!conn) return;\n"
    "    const names = [];\n"
    "    (conn.groups || []).forEach((g, i) => {\n"
    "      if (isOpen(g, i)) (g.tensors || []).forEach(t => {\n"
    "        if (t.params >= 4096 && !t.name.endsWith('bias') && !wt[t.name] && names.length < 16) names.push(t.name);\n"
    "      });\n"
    "    });\n"
    "    if (!names.length) return;\n"
    "    let stop = false;\n"
    "    fetch(`${API_BASE}/api/live/weights/summary`, {\n"
    "      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names }),\n"
    "    }).then(r => r.ok ? r.json() : null).then(d => {\n"
    "      if (d && d.results && !stop) setWt(w => ({ ...w, ...d.results }));\n"
    "    }).catch(() => {});\n"
    "    return () => { stop = true; };\n"
    "    // eslint-disable-next-line react-hooks/exhaustive-deps\n"
    "  }, [conn, open, filter]);",
    "  /* 「全部参数」逻辑随模块隐藏（2026-10-09）：派生分组/展开状态/懒加载 fetch 一并停用（恢复时取消注释）\n"
    "  const data = conn || LIVE_MODEL_DEMO;\n"
    "  const f = filter.trim().toLowerCase();\n"
    "  const groups = f\n"
    "    ? (data.groups || [])\n"
    "        .map(g => ({ ...g, tensors: g.tensors.filter(t => t.name.toLowerCase().includes(f)) }))\n"
    "        .filter(g => g.tensors.length || g.prefix.toLowerCase().includes(f))\n"
    "        .map(g => (g.tensors.length ? g : { ...g, tensors: g.tensors }))\n"
    "    : (data.groups || []);\n"
    "  const isOpen = (g, i) => (f ? true : open[g.prefix] !== undefined ? open[g.prefix] : i === 0);\n"
    "  const toggle = (p) => setOpen(o => ({ ...o, [p]: !(o[p] !== undefined ? o[p] : false) }));\n"
    "\n"
    "  // 展开组时懒加载真实权重标量（POST /live/weights/summary：32×64 行块切片，超限均匀抽样）\n"
    "  useEffect(() => {\n"
    "    if (!conn) return;\n"
    "    const names = [];\n"
    "    (conn.groups || []).forEach((g, i) => {\n"
    "      if (isOpen(g, i)) (g.tensors || []).forEach(t => {\n"
    "        if (t.params >= 4096 && !t.name.endsWith('bias') && !wt[t.name] && names.length < 16) names.push(t.name);\n"
    "      });\n"
    "    });\n"
    "    if (!names.length) return;\n"
    "    let stop = false;\n"
    "    fetch(`${API_BASE}/api/live/weights/summary`, {\n"
    "      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names }),\n"
    "    }).then(r => r.ok ? r.json() : null).then(d => {\n"
    "      if (d && d.results && !stop) setWt(w => ({ ...w, ...d.results }));\n"
    "    }).catch(() => {});\n"
    "    return () => { stop = true; };\n"
    "    // eslint-disable-next-line react-hooks/exhaustive-deps\n"
    "  }, [conn, open, filter]);\n"
    "  */",
    'logic')

# ── op7a 注释 stash（const tbl）插入 return 之前 ──
BLOCK = (
    '    <div className="fw-pl-card">\n'
    '      <h6>\n'
    '        全部参数\n'
    '        <span>{(data.groups || []).reduce((a, g) => a + g.tensor_count, 0)} tensors · 按层分组 · 悬停看完整形状</span>\n'
    '      </h6>\n'
    '      <input className="fw-live-filter" placeholder="过滤 tensor 名（如 q_proj / layers.5 / norm）"\n'
    '             value={filter} onChange={e => setFilter(e.target.value)}/>\n'
    '      <div className="fw-live-groups">\n'
    '        {groups.map((g, i) => (\n'
    '          <div key={g.prefix} className="fw-live-group">\n'
    '            <button type="button" className="fw-live-ghd" onClick={() => toggle(g.prefix)}>\n'
    '              <span className="fw-tr">{isOpen(g, i) ? \'▾\' : \'▸\'}</span>\n'
    '              <b className="fw-mono">{g.prefix}</b>\n'
    '              <span className="dim2">{g.tensor_count} tensors</span>\n'
    '              <span className="fw-mono dim2">{fmtP(g.params)}</span>\n'
    '            </button>\n'
    '            {isOpen(g, i) && (\n'
    '              <div className="fw-live-table">\n'
    '                {g.tensors.map(t => {\n'
    '                  const w = wt[t.name];\n'
    '                  return (\n'
    '                    <div key={t.name} className="fw-live-trow" title={t.name + \' · shape [\' + t.shape.join(\', \') + \'] · \' + t.dtype + \' · \' + t.params + \' params\'}>\n'
    '                      <span className="fw-mono fw-live-tname">{t.name}</span>\n'
    '                      <span className="fw-mono dim2">[{t.shape.join(\'×\')}]</span>\n'
    '                      <span className="fw-live-tdtype">{t.dtype}</span>\n'
    '                      <span className="fw-mono r">{fmtP(t.params)}</span>\n'
    '                      {conn && (\n'
    '                        <span className={\'fw-mono fw-live-tval\' + (w ? \'\' : \' pend\')}\n'
    '                              title={w ? (\'rms \' + w.rms + \' · mean|W| \' + w.mean_abs + \' · max|W| \' + w.max_abs + \' · \' + (w.sampled ? \'32×64 行块抽样\' : \'全量遍历\'))\n'
    '                                      : (t.params >= 4096 && !t.name.endsWith(\'bias\') ? \'真实权重统计加载中…\' : \'\')}>\n'
    '                          {w ? fmtW(w.rms) : (t.params >= 4096 && !t.name.endsWith(\'bias\') ? \'…\' : \'\')}\n'
    '                        </span>\n'
    '                      )}\n'
    '                    </div>\n'
    '                  );\n'
    '                })}\n'
    '              </div>\n'
    '            )}\n'
    '          </div>\n'
    '        ))}\n'
    '        {!groups.length && <div className="dim2" style={{ padding: \'6px 2px\', fontSize: 10 }}>无匹配 tensor</div>}\n'
    '      </div>\n'
    '      <p className="fw-tut-note">实时范围：参数结构实时读取（config + 权重头扫描）+ 数值列/热图/3D 着色的<b>真实权重懒加载</b>\n'
    '        （safetensors 按行块切片读，rms/mean|W| 为 32×64 行块统计，超限均匀抽样；模型文件不变即零重复成本）；\n'
    '        激活级实时流（真实 forward、逐层 norm 曲线）需 runner 侧加载权重，属下一期（P2）。</p>\n'
    '    </div>'
)
STASH = (
    '  /* ===== 「全部参数」浏览器（已隐藏 2026-10-09）：分组（层）→ 逐 tensor 行 + 真实权重数值列\n'
    '     恢复方式：取消本段注释得 const tbl，并在下方 return 中 {middle} 之后引用 {tbl}\n'
    '  const tbl = (\n'
    + BLOCK + '\n'
    '  );\n'
    '  ===== */\n'
    '\n'
    '  return (\n'
    '    <div className="fw-live">'
)
s = sub_once(s, '  return (\n    <div className="fw-live">', STASH, 'stash')

# ── op7b JSX 原位替换为隐藏注记 ──
OLD_JSX = (
    '      {/* 全部参数浏览器：分组（层）→ 逐 tensor 行 */}\n'
    + BLOCK.replace('\n    <div', '\n      <div').replace('\n      <h6>', '\n        <h6>')
)
# 上面的 replace 拼接易错，直接显式构造（与磁盘一致的原始 6 空格缩进版本）
OLD_JSX = (
    '      {/* 全部参数浏览器：分组（层）→ 逐 tensor 行 */}\n'
    '      <div className="fw-pl-card">\n'
    '        <h6>\n'
    '          全部参数\n'
    '          <span>{(data.groups || []).reduce((a, g) => a + g.tensor_count, 0)} tensors · 按层分组 · 悬停看完整形状</span>\n'
    '        </h6>\n'
    '        <input className="fw-live-filter" placeholder="过滤 tensor 名（如 q_proj / layers.5 / norm）"\n'
    '               value={filter} onChange={e => setFilter(e.target.value)}/>\n'
    '        <div className="fw-live-groups">\n'
    '          {groups.map((g, i) => (\n'
    '            <div key={g.prefix} className="fw-live-group">\n'
    '              <button type="button" className="fw-live-ghd" onClick={() => toggle(g.prefix)}>\n'
    '                <span className="fw-tr">{isOpen(g, i) ? \'▾\' : \'▸\'}</span>\n'
    '                <b className="fw-mono">{g.prefix}</b>\n'
    '                <span className="dim2">{g.tensor_count} tensors</span>\n'
    '                <span className="fw-mono dim2">{fmtP(g.params)}</span>\n'
    '              </button>\n'
    '              {isOpen(g, i) && (\n'
    '                <div className="fw-live-table">\n'
    '                  {g.tensors.map(t => {\n'
    '                    const w = wt[t.name];\n'
    '                    return (\n'
    '                      <div key={t.name} className="fw-live-trow" title={t.name + \' · shape [\' + t.shape.join(\', \') + \'] · \' + t.dtype + \' · \' + t.params + \' params\'}>\n'
    '                        <span className="fw-mono fw-live-tname">{t.name}</span>\n'
    '                        <span className="fw-mono dim2">[{t.shape.join(\'×\')}]</span>\n'
    '                        <span className="fw-live-tdtype">{t.dtype}</span>\n'
    '                        <span className="fw-mono r">{fmtP(t.params)}</span>\n'
    '                        {conn && (\n'
    '                          <span className={\'fw-mono fw-live-tval\' + (w ? \'\' : \' pend\')}\n'
    '                                title={w ? (\'rms \' + w.rms + \' · mean|W| \' + w.mean_abs + \' · max|W| \' + w.max_abs + \' · \' + (w.sampled ? \'32×64 行块抽样\' : \'全量遍历\'))\n'
    '                                        : (t.params >= 4096 && !t.name.endsWith(\'bias\') ? \'真实权重统计加载中…\' : \'\')}>\n'
    '                            {w ? fmtW(w.rms) : (t.params >= 4096 && !t.name.endsWith(\'bias\') ? \'…\' : \'\')}\n'
    '                          </span>\n'
    '                        )}\n'
    '                      </div>\n'
    '                    );\n'
    '                  })}\n'
    '                </div>\n'
    '              )}\n'
    '            </div>\n'
    '          ))}\n'
    '          {!groups.length && <div className="dim2" style={{ padding: \'6px 2px\', fontSize: 10 }}>无匹配 tensor</div>}\n'
    '        </div>\n'
    '        <p className="fw-tut-note">实时范围：参数结构实时读取（config + 权重头扫描）+ 数值列/热图/3D 着色的<b>真实权重懒加载</b>\n'
    '          （safetensors 按行块切片读，rms/mean|W| 为 32×64 行块统计，超限均匀抽样；模型文件不变即零重复成本）；\n'
    '          激活级实时流（真实 forward、逐层 norm 曲线）需 runner 侧加载权重，属下一期（P2）。</p>\n'
    '      </div>'
)
s = sub_once(s, OLD_JSX,
    '      {/* 「全部参数」浏览器已隐藏（2026-10-09）：恢复=取消上方 tbl 注释并在此处引用 {tbl} */}',
    'jsx-block')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('OK 8 ops applied')
