/* 神经元图谱 · 归因优先级：按单元对答案对数几率差的直接作用排序（机制案例报告的 neurons.top），
   每行给四轴（可读 / 读出 / 控制 / 连接）的检验状态与证据等级。这里只有直接作用（读出轴的近似）是报告里的实测值；
   其余三轴在单元卡里看（模板选择性、干预、上下游电路），本表不推断、不晋级。 */
import { Empty } from '../rdc_fusion/ui.jsx';
import { CaseHeader, CaseMissing } from './MechCase.jsx';
import { useConflictCase } from './mechData.js';
import { fmt, pct } from './mechUtil.js';

export default function NeuronPriority({ st, set, onUnit }) {
  const { index, cases, caseId, api, report } = useConflictCase(st.case);
  if (!cases.length) return <div className="wb9-page"><CaseMissing index={index} /></div>;
  const top = report?.neurons?.top || [];
  return (
    <div className="wb9-page">
      <CaseHeader cases={cases} caseId={caseId} onCase={id => set({ case: id })} report={report} />
      {api.error && <Empty>案例报告不在本机数据包里：{api.error}</Empty>}
      {report && !top.length && <Empty>未接入：报告里没有神经元直接作用。</Empty>}
      {top.length > 0 && (
        <section className="wb9-sec">
          <header><h3>按归因优先级排序的 MLP 单元</h3>
            <small>直接作用 = 单元输出经最终归一化投到（上下文 − 记忆）答案方向，按发现划分 |值| 排序；留出划分同号 {pct(report.neurons.sign_replication_heldout)}</small></header>
          <table className="wb9-table">
            <thead><tr><th>#</th><th>块.单元</th><th>直接作用（发现）</th><th>直接作用（留出）</th><th>可读</th><th>读出</th><th>控制</th><th>连接</th><th>证据等级</th><th /></tr></thead>
            <tbody>{top.map((u, i) => {
              const rep = Math.sign(u.dla_discovery) === Math.sign(u.dla_heldout);
              return (
                <tr key={`${u.layer}.${u.unit}`}>
                  <td>{i + 1}</td><td className="mono">L{u.layer}.U{u.unit}</td>
                  <td>{fmt(u.dla_discovery, 4)}</td><td>{fmt(u.dla_heldout, 4)}</td>
                  <td className="muted">见单元卡</td>
                  <td className={rep ? 'good' : 'bad'}>{rep ? '✓ 直接作用留出同号' : '✗ 留出反号'}</td>
                  <td className="muted">未检验</td>
                  <td className="muted">见单元卡</td>
                  <td className="muted">未登记</td>
                  <td><button type="button" className="fw-tbtn ghost" onClick={() => onUnit(u.layer, u.unit)}>单元卡 →</button></td>
                </tr>
              );
            })}</tbody>
          </table>
          <p className="v6-note">「控制」需要预注册干预（单元或单元组消融 / steering，正负双向）；直接作用排名靠前不表示单元被模型使用。
            证据等级在 Research OS 登记，本表不自动晋级。</p>
        </section>
      )}
    </div>
  );
}
