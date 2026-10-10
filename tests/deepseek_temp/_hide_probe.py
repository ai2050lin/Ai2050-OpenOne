# -*- coding: utf-8 -*-
"""探针：逐行定位 OLD_JSX 与磁盘的差异行。"""
import io

P = r'D:\AI2050\Ai2050-OpenOne\frontend\src\components\app\rdc_fusion\LensSpatial.jsx'
disk = io.open(P, encoding='utf-8', newline='').read()
dlines = disk.replace('\r\n', '\n').split('\n')

# 与补丁脚本相同的 OLD_JSX（第二版构造）
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
    '                                        : (t.params >= 4096 && !t.name.endsWith(\'bias\') ? \'真实权重统计加载中…\' : \'\')}\n'
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
olines = OLD_JSX.replace('\r\n', '\n').split('\n')

out = []
start = None
for idx, l in enumerate(dlines):
    if l == olines[0]:
        start = idx
        break
out.append('first line found at disk line %s' % ((start + 1) if start is not None else 'NONE'))
if start is not None:
    for k, ol in enumerate(olines):
        dl = dlines[start + k] if start + k < len(dlines) else '<EOF>'
        if dl != ol:
            out.append('MISMATCH at block line %d (disk line %d):' % (k + 1, start + k + 1))
            out.append('  disk: %r' % dl)
            out.append('  want: %r' % ol)
            break
    else:
        out.append('ALL %d lines match' % len(olines))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_probe_out.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('probe done')
