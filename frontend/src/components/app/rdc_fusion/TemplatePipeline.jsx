/* 研发 · 模板流水线（原「模板工作室 / 试跑与冻结 / 分发与节点 / 审计队列」四工位合并）。
   这四步本来就是同一条闭环：建模板 → 冻结 → 分发 → 审计，且共享同一份 draft（草稿模板规格）、
   同一个 NodeCommand 组件、同一批 /api/grammar|templates 接口。分开成四个 tab 时每跳一次就丢一次状态，
   现在把 draft 下沉到本组件内部，四步用步骤条切换。
   纪律：审计只决定执行产物能否进入合并统计，不改变 Evidence 等级；第四步仅管理员可用。
   v9 工作台把全部管理功能收成一个分步骤向导：steps 在四步之后追加「普查」「自动测试」等步骤，
   其内容由 panels[step id] 提供（见 Workbench.jsx 与 workbenchUtil.ADMIN_STEPS）。 */
import { useCallback, useState } from 'react';

import AuditQueue from './AuditQueue.jsx';
import DispatchPanel from './DispatchPanel.jsx';
import FreezePanel from './FreezePanel.jsx';
import TemplateStudio from './TemplateStudio.jsx';

export const PIPE_STEPS = [
  { id: 'design', label: '建模板', desc: '句式框架 × 填充词表 × 因子水平' },
  { id: 'freeze', label: '冻结', desc: '先试跑、再正式；冻结后不可修改' },
  { id: 'dispatch', label: '分发', desc: '分片覆盖、节点与租约' },
  { id: 'audit', label: '审计', desc: '自动检查 + 人工裁决', admin: true },
];

export function isPipeStep(id) {
  return PIPE_STEPS.some(s => s.id === id);
}

export default function TemplatePipeline({ step = 'design', onStep, onGo, canAudit = false, steps = PIPE_STEPS, panels = {} }) {
  /* draft 原在 Workbench 顶层，只为跨 tab 传递；合并后不必再上抛 */
  const [draft, setDraft] = useState(null);
  const all = steps.map(s => ({ ...s, admin: s.admin || PIPE_STEPS.find(p => p.id === s.id)?.admin }));
  const cur = all.some(s => s.id === step) ? step : 'design';
  const usable = all.filter(s => !s.admin || canAudit);
  const blocked = all.find(s => s.id === cur)?.admin && !canAudit;

  /* 子面板内部的 onGo('studio'|'dispatch') 一律先当站内步骤处理，其余上抛给工作台。
     旧工位名（studio = 建模板）保留为别名，面板代码不必改写。 */
  const routeGo = useCallback((a, b, c, extra) => {
    const target = a === 'studio' ? 'design' : a;
    if (isPipeStep(target) || steps.some(s => s.id === target)) { onStep?.(target); return; }
    onGo?.(a, b, c, extra);
  }, [onStep, onGo, steps]);

  return (
    <div className="v6-pipe">
      <nav className="v6-pipe-rail" aria-label="管理向导">
        <ol>
          {usable.map(s => {
            const i = all.indexOf(s);
            return (
              <li key={s.id} className={'v6-pipe-step' + (cur === s.id ? ' on' : '')}>
                <button type="button" onClick={() => onStep?.(s.id)} aria-current={cur === s.id ? 'step' : undefined}>
                  <b>{i + 1}</b>
                  <span>{s.label}</span>
                  <small>{s.desc}</small>
                </button>
              </li>
            );
          })}
        </ol>
        <p className="v6-pipe-note">
          前四步是一条闭环：模板冻结后语料、runner 与预注册阈值都不可修改；审计、普查校验与复算通过只表示产物可进入合并统计，不改变证据等级。
        </p>
      </nav>

      <div className="v6-pipe-body">
        {cur === 'design' && <TemplateStudio draft={draft} onDraft={setDraft} onNext={() => onStep?.('freeze')} />}
        {cur === 'freeze' && <FreezePanel draft={draft} onDraft={setDraft} onGo={routeGo} />}
        {cur === 'dispatch' && <DispatchPanel onGo={routeGo} />}
        {cur === 'audit' && !blocked && <AuditQueue />}
        {!isPipeStep(cur) && !blocked && panels[cur]}
        {blocked && (
          <p className="v6-note warn">
            审计队列只对管理员开放（写操作需要管理令牌或管理员账号）。当前账号不是管理员，请用管理员登录后再看。
          </p>
        )}
      </div>
    </div>
  );
}
