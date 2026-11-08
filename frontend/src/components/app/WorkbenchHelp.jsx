/* 研究工作台的「?」窗口（Phase 338）：按当前页签显示这一页的原理与模块说明。
   内容在 helpPages.js，键 = 一级界面 / 二级页签（/ 页内页签，由 PageTabs 登记）；只描述已实现的页面与数据流。
   没有专门说明的页面，退回显示 PageHead 登记的一句话说明。 */
import { useEffect } from 'react';

import { HELP_PAGES, helpKey } from './helpPages.js';
import { usePageInfo, usePageTab } from '../v8/pageInfo.js';

export default function WorkbenchHelp({ base, onClose }) {
  const page = usePageInfo();
  const tab = usePageTab();
  const help = HELP_PAGES[helpKey(base, tab)];
  const title = help?.title || page?.title || '本页说明';
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="wb-help-mask" onClick={onClose} role="presentation">
      <div className="wb-help-modal" role="dialog" aria-modal="true" aria-label={title} onClick={e => e.stopPropagation()}>
        <header className="wb-help-head">
          <div>
            <h2>{title}</h2>
            <p>{help?.ask || page?.sub || ''}</p>
          </div>
          <button type="button" className="wb-help-close" onClick={onClose} aria-label="关闭说明">×</button>
        </header>
        <div className="wb-help-body">
          {!help && <p className="wb-help-intro">{page?.note || '这一页还没有专门的说明。'}</p>}
          {help?.principle?.length > 0 && <section>
            <h3>原理</h3>
            <ul>{help.principle.map((p, i) => <li key={i}>{p}</li>)}</ul>
          </section>}
          {help?.modules?.length > 0 && <section>
            <h3>页面模块</h3>
            <dl className="wb-help-mods">{help.modules.map(([name, desc]) => <div key={name}><dt>{name}</dt><dd>{desc}</dd></div>)}</dl>
          </section>}
          {help?.read?.length > 0 && <section>
            <h3>怎样读 · 证据边界</h3>
            <ul>{help.read.map((p, i) => <li key={i}>{p}</li>)}</ul>
          </section>}
        </div>
      </div>
    </div>
  );
}
