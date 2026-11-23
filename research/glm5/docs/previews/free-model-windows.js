(() => {
  'use strict';

  const icons = {
    pin: '<path d="m8 3 5 5-2 .6-2.7 3.3-1.9-1.9L3 13.5l3.5-3.6-1.8-1.8L8 5.4z"/>',
    collapse: '<path d="M3 8h10"/>',
    expand: '<path d="M3 8h10M8 3v10"/>',
    maximize: '<rect x="3" y="3" width="10" height="10" rx="1"/>',
    restore: '<path d="M5.5 3H13v7.5M3 5.5h7.5V13H3z"/>',
    close: '<path d="m4 4 8 8M12 4l-8 8"/>',
    resize: '<path d="m5 12 7-7m-3 7 3-3"/>',
  };
  const svg = name => '<svg aria-hidden="true" viewBox="0 0 16 16">' + icons[name] + '</svg>';
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function create(main) {
    const layer = main.querySelector('#free-windows');
    const panels = new Map();
    const events = new AbortController();
    const listen = (node, event, handler) => node.addEventListener(event, handler, {signal: events.signal});
    let dock = '', floating = false, sequence = 0;
    let drag = null;

    function dimensions() {
      return {w: layer.clientWidth, h: layer.clientHeight};
    }

    function defaultRect(p, {w, h}) {
      const gap = w < 520 ? 8 : 14;
      const width = Math.min(w - 2 * gap, p.id === 'locator' ? 350 : 426);
      const limit = p.id === 'tools' ? 398 : (p.id === 'locator' ? 405 : 470);
      const height = Math.min(h - 2 * gap, Math.max(240, Math.min(limit, Math.round(h * .71))));
      const left = p.id === 'locator' || p.id === 'evidence';
      const top = p.id === 'locator' || p.id === 'tools';
      if (p.id === 'readout') {
        return {x: Math.round((w - width) / 2), y: gap, w: width, h: height};
      }
      return {x: left ? gap : w - gap - width, y: top ? gap : h - gap - height, w: width, h: height};
    }

    function fit(rect, size, collapsedHeight = 0) {
      const w = clamp(rect.w, Math.min(246, size.w), size.w);
      const h = clamp(rect.h, Math.min(150, size.h), size.h);
      return {w, h, x: clamp(rect.x, 0, size.w - w), y: clamp(rect.y, 0, size.h - (collapsedHeight || h))};
    }

    function place(p) {
      if (!floating || !p.open) return;
      const size = dimensions();
      if (!size.w || !size.h) return;
      p.rect ||= defaultRect(p, size);
      const barHeight = Math.min(size.h, p.bar.offsetHeight + p.el.offsetHeight - p.el.clientHeight);
      p.rect = fit(p.rect, size, p.collapsed ? barHeight : 0);
      const rect = p.maximized ? {x: 0, y: 0, ...size} : p.rect;
      const style = p.el.style;
      style.left = rect.x + 'px';
      style.top = rect.y + 'px';
      style.width = rect.w + 'px';
      style.height = (p.collapsed ? barHeight : rect.h) + 'px';
    }

    function showState(p, action, label, icon, pressed = null) {
      const button = p.controls.querySelector('[data-fw-action="' + action + '"]');
      button.title = label;
      button.setAttribute('aria-label', label);
      if (button.dataset.icon !== icon) {
        button.innerHTML = svg(icon);
        button.dataset.icon = icon;
      }
      if (pressed !== null) button.setAttribute('aria-pressed', String(pressed));
    }

    function update() {
      const ordered = [...panels.values()].filter(p => p.open).sort((a, b) =>
        Number(a.pinned) - Number(b.pinned) || a.order - b.order);
      const front = ordered[ordered.length - 1];
      for (const p of panels.values()) {
        const visible = floating ? p.open : p.id === dock;
        p.el.hidden = !visible;
        p.el.classList.toggle('fw-front', floating && front === p);
        p.el.classList.toggle('fw-collapsed', floating && p.collapsed);
        p.el.classList.toggle('fw-maximized', floating && p.maximized);
        p.bar.tabIndex = floating && visible ? 0 : -1;
        p.bar.title = floating ? '拖动以移动窗口。聚焦标题后，方向键移动，Shift 加方向键调整大小。' : '';
        if (floating) p.el.style.zIndex = String(ordered.indexOf(p) + 1);
        else for (const key of ['left', 'top', 'width', 'height', 'z-index']) p.el.style.removeProperty(key);
        showState(p, 'pin', p.pinned ? '取消置顶' : '置顶窗口', 'pin', p.pinned);
        showState(p, 'collapse', p.collapsed ? '展开窗口' : '收起窗口', p.collapsed ? 'expand' : 'collapse');
        p.controls.querySelector('[data-fw-action="collapse"]').setAttribute('aria-expanded', String(!p.collapsed));
        showState(p, 'maximize', p.maximized ? '还原窗口' : '最大化窗口', p.maximized ? 'restore' : 'maximize');
        place(p);
      }
      for (const launcher of main.querySelectorAll('[data-open-window]')) {
        const p = panels.get(launcher.dataset.openWindow);
        if (!p) continue;
        const visible = !p.el.hidden;
        const attr = launcher.getAttribute('role') === 'tab' ? 'aria-selected' : 'aria-expanded';
        launcher.setAttribute(attr, String(visible));
        launcher.setAttribute('aria-controls', p.el.id);
        launcher.classList.toggle('fw-open', visible);
      }
    }

    function notify() {
      main.dispatchEvent(new CustomEvent('freewindowchange', {
        detail: {floating, active: dock, open: [...panels.values()].filter(p => !p.el.hidden).map(p => p.id)}
      }));
    }

    function activate(p) {
      if (dock === p.id && p.order === sequence) return;
      dock = p.id;
      p.order = ++sequence;
      update();
      notify();
    }

    function open(id) {
      const p = panels.get(id);
      if (!p) return false;
      dock = p.id;
      if (floating) { p.open = true; p.collapsed = false; }
      p.order = ++sequence;
      update();
      if (floating) p.bar.focus({preventScroll: true});
      notify();
      return true;
    }

    function close(id) {
      const p = panels.get(id);
      if (!p) return false;
      p.open = false;
      if (id === dock) {
        const remaining = [...panels.values()].filter(q => q !== p && (floating ? q.open : true));
        const next = remaining.sort((a, b) => b.order - a.order)[0];
        dock = next?.id || '';
      }
      update();
      const launcher = [...main.querySelectorAll('[data-open-window]')].find(el =>
        el.dataset.openWindow === id && el.getClientRects().length && !el.closest('[hidden]'));
      launcher?.focus({preventScroll: true});
      notify();
      return true;
    }

    function setFullscreen(value) {
      const next = Boolean(value);
      if (next === floating) return;
      drag = null;
      floating = next;
      if (!panels.has(dock)) dock = panels.keys().next().value;
      const p = panels.get(dock);
      if (floating && p) { p.open = true; p.order = ++sequence; }
      main.classList.toggle('fw-floating', floating);
      update();
      notify();
    }

    function reset() {
      drag = null;
      for (const p of panels.values()) {
        p.rect = null;
        p.pinned = p.maximized = p.collapsed = false;
      }
      update();
      notify();
    }

    function destroy() {
      drag = null;
      setFullscreen(false);
      observer.disconnect();
      events.abort();
      for (const p of panels.values()) {
        p.controls.remove();
        p.grip.remove();
        p.bar.removeAttribute('tabindex');
        p.bar.removeAttribute('title');
      }
    }

    function action(p, type) {
      if (type === 'close') return close(p.id);
      if (type === 'pin') p.pinned = !p.pinned;
      if (type === 'maximize') { p.maximized = !p.maximized; p.collapsed = false; }
      if (type === 'collapse') { p.collapsed = !p.collapsed; p.maximized = false; }
      p.order = ++sequence;
      dock = p.id;
      update();
      notify();
    }

    function pointerStart(e, p, resize) {
      if (!floating || e.button !== 0 || !e.isPrimary || p.maximized || drag) return;
      if (!resize && e.target.closest('button,input,select,textarea,a,[contenteditable="true"]')) return;
      const handle = e.currentTarget;
      handle.focus({preventScroll: true});
      drag = {p, resize, start: {...p.rect}, x: e.clientX, y: e.clientY, pointer: e.pointerId, handle};
      handle.setPointerCapture(e.pointerId);
      e.preventDefault();
    }

    function pointerMove(e) {
      if (!drag || e.pointerId !== drag.pointer) return;
      const {p, start} = drag;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (drag.resize) resizeBy(p, start, dx, dy);
      else p.rect = {...start, x: start.x + dx, y: start.y + dy};
      place(p);
    }

    function resizeBy(p, start, dx, dy) {
      const size = dimensions();
      p.rect = {...start, w: Math.min(size.w - start.x, start.w + dx), h: Math.min(size.h - start.y, start.h + dy)};
    }

    function keyboard(e, p, grip) {
      if (!floating || e.target !== e.currentTarget) return;
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key) || p.maximized) return;
      const dx = e.key === 'ArrowLeft' ? -12 : e.key === 'ArrowRight' ? 12 : 0;
      const dy = e.key === 'ArrowUp' ? -12 : e.key === 'ArrowDown' ? 12 : 0;
      if (grip || e.shiftKey) {
        if (p.collapsed) return;
        resizeBy(p, p.rect, dx, dy);
      } else { p.rect.x += dx; p.rect.y += dy; }
      e.preventDefault();
      place(p);
    }

    for (const el of layer.querySelectorAll('.free-window[data-window]')) {
      const id = el.dataset.window;
      const bar = el.querySelector('.window-bar');
      const h2 = bar.querySelector('h2');
      el.id ||= 'fw-window-' + id;
      if (h2) { h2.id ||= el.id + '-title'; el.setAttribute('aria-labelledby', h2.id); }
      const controls = document.createElement('div');
      controls.className = 'fw-controls';
      for (const type of ['pin', 'collapse', 'maximize', 'close']) {
        const button = document.createElement('button');
        button.type = 'button';
        button.dataset.fwAction = type;
        controls.append(button);
      }
      bar.append(controls);
      const grip = document.createElement('button');
      grip.type = 'button';
      grip.className = 'fw-resize';
      grip.setAttribute('aria-label', '调整窗口大小，支持方向键');
      grip.title = '拖动或使用方向键调整大小';
      grip.innerHTML = svg('resize');
      el.append(grip);
      const p = {id, el, bar, controls, grip, rect: null, open: false, collapsed: false, pinned: false, maximized: false, order: 0};
      panels.set(id, p);
      if (!dock && !el.hidden) dock = id;
      showState(p, 'close', '关闭窗口', 'close');
      listen(el, 'focusin', () => { if (floating && p.open) activate(p); });
      listen(el, 'pointerdown', () => { if (floating && p.open) activate(p); });
      listen(controls, 'click', e => {
        const button = e.target.closest('[data-fw-action]');
        if (button) action(p, button.dataset.fwAction);
      });
      listen(bar, 'pointerdown', e => pointerStart(e, p, false));
      listen(grip, 'pointerdown', e => pointerStart(e, p, true));
      listen(bar, 'keydown', e => keyboard(e, p, false));
      listen(grip, 'keydown', e => keyboard(e, p, true));
      for (const target of [bar, grip]) {
        listen(target, 'pointermove', pointerMove);
        for (const end of ['pointerup', 'pointercancel', 'lostpointercapture']) {
          listen(target, end, e => { if (drag?.pointer === e.pointerId) drag = null; });
        }
      }
    }
    dock ||= panels.keys().next().value;
    listen(main, 'click', e => {
      const button = e.target.closest('[data-open-window]');
      if (button && !button.disabled) open(button.dataset.openWindow);
    });
    const observer = new ResizeObserver(() => { if (floating) for (const p of panels.values()) place(p); });
    observer.observe(layer);
    update();
    return {open, close, setFullscreen, reset, destroy};
  }
  window.FreeWindowManager = {create};
})();
