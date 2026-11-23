export const graphModes = [
  ['live', '实时模式'],
  ['archive', '测试记录'],
  ['features', '特征分析'],
];

const siteLabels = {
  prompt_end: '输入末尾',
  before_dth_printed_step: '最后一步前',
  before_value: 'Value 前',
  before_code: 'Code 前',
};
const artifact = window.__graphObservation;

function decodeWords(encoded) {
  const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
  const reader = new DataView(bytes.buffer);
  const floats = new Float32Array(bytes.length / 2);
  const words = new Uint32Array(floats.buffer);
  for (let index = 0; index < floats.length; index++) words[index] = reader.getUint16(index * 2, true) << 16;
  return floats;
}

let decodedFields;
export function observationField(kind, layer, position) {
  const [layers, positions, hiddenSize] = artifact.hiddenShape;
  const [, mlpSize] = artifact.mlpShape;
  if (position < 0 || position >= positions || layer < 0 || layer >= layers) return null;
  if (kind !== 'hs' && (kind !== 'mlp' || layer !== artifact.block + 1)) return null;
  decodedFields ||= { hs: decodeWords(artifact.hidden), mlp: decodeWords(artifact.mlp) };
  const length = kind === 'hs' ? hiddenSize : mlpSize;
  const offset = (kind === 'hs' ? layer * positions + position : position) * length;
  return decodedFields[kind].subarray(offset, offset + length);
}

export const graphObservation = {
  snapshot: artifact.snapshot,
  positions: artifact.positions,
  block: artifact.block,
  units: artifact.units,
};

function Source() {
  return <details className="ow-provenance">
    <summary>运行与数据来源</summary>
    <dl>
      <div><dt>归档运行</dt><dd>{artifact.run}</dd></div>
      <div><dt>样本</dt><dd>{artifact.caseId}</dd></div>
      <div><dt>模型快照</dt><dd>{artifact.snapshot}</dd></div>
      <div><dt>原始文件 SHA-256</dt><dd>{artifact.source.sha256}</dd></div>
    </dl>
    <a href="../../../../tests/glm5/result/b36_transition14_20261114_v1/report.html">查看 Phase 2933 实验报告 →</a>
  </details>;
}

export function GraphObservation({matches, position, setPosition, sampleSelected, onLocate, onModel, draft, onDraft}) {
  const point = artifact.positions[position];
  return <section className="ow-mode-panel" aria-label="实时模式工作区">
    <div className="ow-section-head">
      <div><h3>实时模式</h3><p>静态稿回放 Phase 2933 实测。运行新输入需要正式客户端的模型服务。</p></div>
      {matches ? <button className="fw-tbtn primary" onClick={onLocate}>观察 B36/U{artifact.units[0]} →</button>
        : <button className="fw-tbtn primary" onClick={onModel}>切换到样例模型 Qwen3-14B</button>}
    </div>
    {matches && <>
      <div className="ow-step-list" role="group" aria-label="采集位置">
        {artifact.positions.map((site, index) => <button className="fw-tbtn" key={site.site} aria-pressed={position === index}
          onClick={() => setPosition(index)}>{siteLabels[site.site]}</button>)}
      </div>
      <div className="ow-run-sample">
        <dl><div><dt>当前位置</dt><dd>{point.index} · Token ID {point.tokenId}</dd></div>
          <div><dt>已采集</dt><dd>逐层 HiddenState · B36 MLP</dd></div></dl>
        {sampleSelected && <p className="ow-inline-value">U{sampleSelected.index} 激活 <output>{sampleSelected.value.toFixed(4)}</output></p>}
      </div>
      <details className="ow-provenance"><summary>查看已生成的前缀</summary><pre>{artifact.prefix}</pre></details>
      <details className="ow-provenance"><summary>查看已记录的最终输出分数</summary>
        <table className="ow-value-table"><thead><tr><th>候选</th><th>最终 logit</th></tr></thead>
          <tbody>{artifact.tokens.map((token, index) => <tr key={token}><td><code>{token}</code></td><td>{artifact.scores[index]}</td></tr>)}</tbody></table>
      </details>
      <Source/>
    </>}
    <details className="ow-provenance"><summary>编辑本机输入</summary>
      <label className="ow-draft-input">运行文本<textarea rows={3} value={draft} onChange={event=>onDraft(event.target.value)}
        placeholder="输入要在桌面节点执行的文本"/></label>
      <button className="fw-tbtn" disabled>开始运行</button>
      <p>此静态稿未连接模型服务。编辑文本只影响输入草稿。</p>
    </details>
  </section>;
}

export function ObservationAnalysis({plugin, matches, layer, object, position, onLocate}) {
  const id = plugin === 'heads' ? 'Head' : plugin === 'mlp' ? 'MLP' : plugin === 'hs' ? 'HiddenState' : plugin === 'lens' ? 'Logit Lens' : '';
  if (!id) return null;
  const kind = plugin === 'hs' || plugin === 'mlp' ? plugin : null;
  const data = matches && kind ? observationField(kind, layer, position) : null;
  const selectedIndex = data && (object.module === kind || kind === 'hs' && object.module === 'embedding') && object.index != null
    ? object.module === 'embedding' ? object.coordinate : object.index
    : kind === 'mlp' ? artifact.units[0] : 0;
  return <section className="ow-mode-panel" aria-label={id + ' 实测读数'}>
    <div className="ow-section-head"><div><h3>{id} · Phase 2933 观测样例</h3>
      <p>{siteLabels[artifact.positions[position].site]} · {layer === 0 ? '词嵌入' : '第 ' + layer + ' 层'}</p></div></div>
    {!matches ? <p>当前模型没有随稿附带的观测轨迹。</p> : data ? <>
      <p className="ow-inline-value">{kind === 'hs' ? 'HiddenState 坐标 ' : '原生神经元 U'}{selectedIndex} <output>{data[selectedIndex]?.toFixed(4) ?? '未记录'}</output></p>
      <p>点击 3D 网格中的单元，或使用精确定位选择其他索引。</p>
    </> : <>
      <p>{plugin === 'mlp' ? '这份样例只在 B36 采集了 MLP 激活。' :
        plugin === 'heads' ? '这份观测样例未保存注意力权重。' : '这份观测样例未保存逐层 Logit Lens。'}</p>
      {plugin === 'mlp' && <button className="fw-tbtn" onClick={onLocate}>定位 B36 MLP</button>}
    </>}
  </section>;
}

export function GraphFeatures({matches, featureKind, setFeatureKind, position, unit, onLocate, onGroup, onModel}) {
  const names = [['unit', '原生神经元'], ['group', '原生单元组'], ['sae', '稀疏字典特征']];
  const values = matches && featureKind === 'unit' ? artifact.positions.map((site, index) => ({
    name: siteLabels[site.site], value: observationField('mlp', artifact.block + 1, index)[unit],
  })) : [];
  return <section className="ow-mode-panel" aria-label="特征分析工作区">
    <div className="ow-section-head"><div><h3>特征分析</h3><p>按模型快照查看原生索引、实测例子和证据范围。</p></div></div>
    <div className="ow-step-list" role="group" aria-label="特征类型">
      {names.map(([id, name]) => <button key={id} className="fw-tbtn" aria-pressed={id === featureKind}
        onClick={() => setFeatureKind(id)}>{name}</button>)}
    </div>
    {!matches && <p>当前模型没有随稿附带的特征档案。<button className="fw-tbtn" onClick={onModel}>切换到 Qwen3-14B</button></p>}
    {matches && featureKind === 'unit' && <>
      <div className="ow-feature-entry"><div><h4>B36/U{unit}</h4><p>原生 MLP 神经元 · Phase 2933 单题观测</p></div>
        <button className="fw-tbtn primary" onClick={() => onLocate(unit)}>在 3D 中定位</button></div>
      <table className="ow-value-table"><thead><tr><th>采集位置</th><th>激活值</th></tr></thead>
        <tbody>{values.map((row, index) => <tr key={row.name} data-active={index === position}><td>{row.name}</td><td>{row.value.toFixed(4)}</td></tr>)}</tbody>
      </table>
      <p>这一单元的语义功能和单独因果作用仍待验证。</p>
      <Source/>
    </>}
    {matches && featureKind === 'group' && <>
      <div className="ow-feature-entry"><div><h4>B36 固定 32 个 MLP 单元</h4><p>冻结原生索引 · Phase 2933 组干预</p></div>
        <button className="fw-tbtn primary" onClick={onGroup}>在 3D 中标出整组</button></div>
      <dl className="ow-feature-results">
        {artifact.comparisons.map(item => <div key={item.arm}><dt>{item.arm === 'mask0' ? '原模型' : '三段位置联合屏蔽'}</dt>
          <dd>{item.joint_family_correct} / {item.families} 家族正确</dd></div>)}
      </dl>
      <p>在本次 Code 读出条件下，正确率未提供该组必要性的证据。</p>
      <details className="ow-provenance"><summary>32 个原生索引</summary>
        <div className="ow-step-list">{artifact.units.map(index => <button className="fw-tbtn" key={index}
          onClick={() => onLocate(index)}>U{index}</button>)}</div><p className="ow-feature-id">{artifact.featureId}</p>
      </details>
      <Source/>
    </>}
    {matches && featureKind === 'sae' && <p>此快照的静态稿尚未载入 SAE 字典。相关特征卡需要绑定字典版本、读出层和样本。</p>}
  </section>;
}
