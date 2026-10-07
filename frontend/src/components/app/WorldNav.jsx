import { BookOpen, ImageOff, Map, FlaskConical } from 'lucide-react';

import './WorldNav.css';

/**
 * 三世界共享导航（M2）：
 *   学习 /learn   —— 封存案例 + 领域方法图谱（引导式，默认入口之一）
 *   证据 /        —— Canonical Snapshot 驱动的证据总览（默认落地页）
 *   研发 /build   —— 行业情报 + 研发闭环 + 本地复现指引
 *   画廊 /gallery —— C 档概念件隔离区（弱化展示）
 * 纪律：三个世界读同一 registry 投影；概念画廊不属于三世界，只作归档出口。
 */
const WORLDS = [
  { href: '/learn', key: 'learn', label: '学习世界', desc: '案例 · 方法', Icon: BookOpen },
  { href: '/', key: 'evidence', label: '证据世界', desc: '图谱 · 快照', Icon: Map },
  { href: '/build', key: 'build', label: '研发世界', desc: '闭环 · 复现', Icon: FlaskConical },
];

export function WorldNav({ active }) {
  return (
    <nav className="world-nav" aria-label="三世界导航">
      <div className="world-nav__worlds">
        {WORLDS.map(({ href, key, label, desc, Icon }) => (
          <a
            key={key}
            href={href}
            className={`world-nav__item${active === key ? ' world-nav__item--active' : ''}`}
            aria-current={active === key ? 'page' : undefined}
          >
            <Icon size={14} />
            <span>
              <strong>{label}</strong>
              <em>{desc}</em>
            </span>
          </a>
        ))}
      </div>
      <a className="world-nav__gallery" href="/gallery" title="概念示意与演示数据隔离区（不得作为研究事实引用）">
        <ImageOff size={12} />
        概念画廊
      </a>
    </nav>
  );
}

export default WorldNav;
