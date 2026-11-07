/* 空间三张图与总览共用的说明部件：编码图谱的证据阶梯、关联机制的三种运行、四组成绩。
   约束：总量来自模型结构接口，已检查数来自解释检验接口；没有统计的写“未统计”，没有测的写“未测”。 */
import { useApi } from '../app/rdc_fusion/api.js';
import { useCoverage } from './coverage.js';
import { SCORES } from './data/mainline.js';
import './client_guide.css';

const n = value => Number.isFinite(value) ? value.toLocaleString() : '—';
const Unset = ({ children = '未统计' }) => <span className="mp-tag">{children}</span>;

/* LLM 编码图谱：三类原生对象 + 并列的字典特征，按证据阶梯记录。 */
export function EncodingLadder() {
  const coverage = useCoverage();
  const arch = useApi(coverage.model ? `/api/model/arch?model_key=${encodeURIComponent(coverage.model)}` : null).data;
  const a = arch?.available ? arch : null;
  const heads = a ? a.n_layers * a.n_heads : null, kv = a ? a.n_layers * a.n_kv_heads : null;
  return <section className="v8-card cg-section" aria-label="证据阶梯">
    <h3>三类对象的证据阶梯</h3>
    <p className="v8-note">每个对象按“已检查 → 读得出 → 改得动 → 能预测”记录，不用一个破解百分比。总量来自模型结构，没有按对象汇总的写“未统计”。</p>
    <div className="v8-weight-scroll"><table className="v6-table">
      <thead><tr><th>对象</th><th>总量</th><th>已检查</th><th>读得出</th><th>改得动</th><th>能预测</th></tr></thead>
      <tbody>
        <tr><td>词嵌入</td><td>{a ? `${n(a.vocab_size)} 行 × ${n(a.d_model)} 维` : '—'}</td><td><Unset/></td><td><Unset/></td><td><Unset/></td><td><Unset>未测</Unset></td></tr>
        <tr><td>注意力头</td><td>{heads ? `${n(heads)} 个查询头 · ${n(kv)} 组 KV` : '—'}</td><td><Unset/></td><td><Unset/></td><td>部分头有 QK 与门控链记录</td><td><Unset>未测</Unset></td></tr>
        <tr><td>MLP 单元</td><td>{n(coverage.neuronTotal)}</td><td>{coverage.checked == null ? <Unset/> : n(coverage.checked)}</td><td><Unset/></td><td>部分单元有干预记录</td><td><Unset>未测</Unset></td></tr>
        <tr><td>字典特征（并列）</td><td>第 30 块 16,384 个 transcoder 特征</td><td>1 层（Phase 262）</td><td><Unset/></td><td><Unset>未测</Unset></td><td><Unset>未测</Unset></td></tr>
      </tbody></table></div>
    <p className="v8-note">更接近“破解”的指标是：只保留已解释的对象，留出句子的下一词分布还能恢复多少，并与词身份、位置、n-gram 基线比较。这一项<b>未测</b>。</p>
  </section>;
}

/* 关联机制：三种运行方式的边界。 */
export function AssocIntro() {
  const modes = [
    ['观察', '一次前向，几秒', '只看读数和逐层读出，不产生研究结论'],
    ['对照', '只改一个因素，分钟级', '能比较因素的作用，仍是未登记观察'],
    ['正式', '冻结决策点加修补', '进入研究记录，按预注册判据判定'],
  ];
  return <section className="v8-card cg-section" aria-label="关联机制的运行方式">
    <h3>一段文字怎样决定下一个词</h3>
    <p className="v8-note">“竞争”需要对照：单次前向只能看到读数，比较因素要有对照文本，判断因果要有修补。下面三种运行方式分开记录。</p>
    <div className="cg-grid3">{modes.map(([name, cost, use]) => <div key={name}><strong>{name}</strong><small className="v8-note"> · {cost}</small><p className="v8-note">{use}</p></div>)}</div>
  </section>;
}

/* 四组成绩：v6 第 11 节的主指标。 */
export function ScoreBoard({ bare = false }) {
  return <section className={bare ? "cg-section" : "v8-card cg-section"} aria-label="关联机制的四组成绩">
    {!bare && <h3>关联机制的四组成绩</h3>}
    <p className="v8-note">这是衡量破解程度的主指标。覆盖数和连接数只表示执行量，不计入这里。</p>
    <div className="mp-scores">{SCORES.map(score => <div key={score.name}><b>{score.name}</b><span className="v8-note">{score.desc}</span>
      <div className="mp-v muted">{score.value ?? '未测'}</div></div>)}</div>
    <p className="v8-note">最接近的旧结果：跨事件迁移恢复 0.43–0.60（Phase 282、283）。材料已暴露，只作起点，不计入四组成绩。</p>
  </section>;
}
