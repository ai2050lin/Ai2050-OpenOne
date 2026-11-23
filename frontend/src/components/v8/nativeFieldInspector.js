// Shared by the desktop workbench and the standalone file:// design.
(() => {
  'use strict';

  const PAGE_SIZE = 12;
  const HISTOGRAM_BINS = 20;
  const countText = n => n.toLocaleString('en-US');
  const valueText = value => Number.isFinite(value) ? String(value) : '未采集';
  const brief = v => Number.isFinite(v) ? Number(v.toPrecision(4)).toString() : '—';
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function summarize(values, size) {
    let count = 0, min = Infinity, max = -Infinity, zero = 0;
    if (values) {
      for (let i = 0; i < size; i++) {
        const x = values[i];
        if (!Number.isFinite(x)) continue;
        count++;
        if (x === 0) zero++;
        min = Math.min(min, x);
        max = Math.max(max, x);
      }
    }
    return {count, min, max, zero, peak: count ? Math.max(Math.abs(min), Math.abs(max)) : 0};
  }

  function palette(end) {
    const start = [249, 250, 252];
    return Array.from({length: 256}, (_, i) =>
      'rgb(' + start.map((v, k) => Math.round(v + (end[k] - v) * i / 255)).join(',') + ')');
  }
  const positive = palette([32, 91, 173]);
  const negative = palette([191, 78, 39]);

  function create(host, {onPick} = {}) {
    const root = document.createElement('section');
    root.className = 'native-field';
    root.setAttribute('data-native-inspector', '');
    root.setAttribute('aria-label', '原生数值观察器');
    root.innerHTML = `
      <div class="nf-head"><div><h3 data-field-title></h3><p data-field-source></p></div>
        <span class="nf-size" data-field-size></span></div>
      <div class="nf-toolbar"><div class="nf-modes" role="group" aria-label="数值视图">
        <button type="button" data-field-view="heat">热力图</button>
        <button type="button" data-field-view="distribution">分布图</button>
        <button type="button" data-field-view="table">数值表</button><button type="button" data-field-view="positions" hidden>位置对照</button>
      </div><div class="nf-color-options"><label class="nf-scale">色标 <select data-field-scale aria-label="热力图色标">
        <option value="signed">原值 · 区分正负</option><option value="absolute">绝对值 · 强度</option>
      </select></label><label class="nf-contrast" title="反双曲正弦色标放大靠近零的小数值，定位时仍显示原值"><input type="checkbox" data-field-contrast>增强弱响应</label></div></div>
      <p class="nf-notice" data-field-missing role="status" hidden></p>
      <div data-field-heat><div class="nf-map"><canvas data-field-canvas role="group" tabindex="0"
        aria-label="全部原生编号热力图，方向键移动，Home 与 End 跳至两端"></canvas>
        <span class="nf-tooltip" data-field-tooltip hidden></span></div>
        <div class="nf-map-meta"><span data-field-grid-label></span>
          <span class="nf-legend"><span data-field-low></span><i data-field-ramp></i><span data-field-high></span>
          <span data-field-mapping></span><i class="nf-empty-swatch"></i>缺值</span></div></div>
      <div class="nf-hist" data-field-distribution hidden><p data-field-hist-caption></p>
        <div class="nf-hist-bars" data-field-bars></div><div class="nf-hist-axis"><span data-hist-min></span><span data-hist-max></span></div>
        <p class="nf-hist-focus" data-field-bin aria-live="polite"></p></div>
      <div data-field-positions hidden><p data-field-compare-caption></p><div data-field-compare-rows></div></div>
      <div class="nf-table-wrap" data-field-table hidden>
        <table><thead><tr><th scope="col">原生编号</th><th scope="col">原始数值</th><th scope="col">绝对值</th></tr></thead><tbody></tbody></table>
        <div class="nf-pager"><button type="button" data-field-page-prev>上一页</button>
          <span data-field-page></span><button type="button" data-field-page-next>下一页</button></div></div>
      <div class="nf-selection"><form class="nf-locate" data-field-form><label><span data-field-index-label>原生编号</span>
        <input type="number" data-field-index aria-label="原生编号" min="0" step="1" required></label>
        <button type="submit">定位</button><button type="button" data-field-prev aria-label="上一个原生编号">‹</button>
        <button type="button" data-field-next aria-label="下一个原生编号">›</button></form>
        <div class="nf-reading" aria-live="polite"><span data-field-selection></span>
          <span data-field-data-label>原值</span><output data-field-value></output></div></div>
      <p class="nf-status" data-field-stats></p>`;
    host.replaceChildren(root);
    const find = selector => root.querySelector(selector);
    const attr = name => find('[data-field-' + name + ']');
    const canvas = attr('canvas'), context = canvas.getContext('2d');
    const views = [...root.querySelectorAll('[data-field-view]')];
    let current = {}, values = null, summary = summarize(null, 0);
    let index = 0, page = 0, view = 'heat', columns = 1, rows = 1, frame = null, closed = false;
    const prefix = () => current.kind === 'ffn' ? 'U' : 'D';
    const noun = () => current.kind === 'ffn' ? 'FFN 神经元' : 'Head 通道';
    const indexLabel = i => current.headColumns
      ? 'H'+Math.floor(i/current.headColumns)+'/D'+(i%current.headColumns) : prefix()+i;

    function pick(next) {
      if (!Number.isSafeInteger(next) || next < 0 || next >= current.size) return;
      index = next;
      page = Math.floor(index / PAGE_SIZE);
      renderSelection();
      if (view === 'table') renderTable();
      if (view === 'heat') queueDraw();
      if (typeof onPick === 'function') onPick(next);
    }

    function renderSelection() {
      attr('index').max = String(Math.max(0, current.size - 1));
      attr('index').value = current.size ? String(index) : '';
      attr('index').disabled = !current.size;
      attr('form').querySelector('[type="submit"]').disabled = !current.size;
      attr('prev').disabled = !current.size || index === 0;
      attr('next').disabled = !current.size || index >= current.size - 1;
      attr('selection').textContent = current.size ? indexLabel(index) : '无编号';
      attr('index-label').textContent = current.headColumns ? '行优先编号' : '原生编号';
      attr('data-label').textContent = current.dataLabel || '原值';
      attr('value').textContent = valueText(values?.[index]);
      root.dataset.selected = current.size ? String(index) : '';
    }

    function renderTable() {
      const body = find('tbody');
      body.replaceChildren();
      const start = page * PAGE_SIZE, end = Math.min(start + PAGE_SIZE, current.size);
      for (let i = start; i < end; i++) {
        const row = body.insertRow();
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = indexLabel(i);
        button.dataset.fieldRow = String(i);
        button.setAttribute('aria-current', String(i === index));
        row.insertCell().appendChild(button);
        const x = values?.[i];
        row.insertCell().textContent = valueText(x);
        row.insertCell().textContent = Number.isFinite(x) ? valueText(Math.abs(x)) : '—';
        if (i === index) row.className = 'nf-selected-row';
      }
      attr('page').textContent = current.size ? (start + 1) + '–' + end + ' / ' + countText(current.size) : '无可列出的编号';
      attr('page-prev').disabled = page === 0;
      attr('page-next').disabled = end >= current.size;
    }

    function renderHistogram() {
      const bars = attr('bars');
      bars.replaceChildren();
      attr('bin').textContent = '';
      find('[data-hist-min]').textContent = '';
      find('[data-hist-max]').textContent = '';
      if (!summary.count) {
        attr('hist-caption').textContent = '此测量没有可统计的已采集数值。';
        return;
      }
      const {min, max, count} = summary;
      const bins = new Array(min === max ? 1 : HISTOGRAM_BINS).fill(0);
      const span = max - min;
      for (let i = 0; i < current.size; i++) {
        const v = values[i];
        if (!Number.isFinite(v)) continue;
        const b = span ? Math.min(bins.length - 1, Math.floor((v - min) / span * bins.length)) : 0;
        bins[b]++;
      }
      const highest = Math.max(...bins);
      attr('hist-caption').textContent = '横轴为原始数值，柱高为编号数。已计入 ' + countText(count) + ' 项。';
      find('[data-hist-min]').textContent = brief(min);
      find('[data-hist-max]').textContent = brief(max);
      bins.forEach((freq, i) => {
        const button = document.createElement('button');
        button.type = 'button';
        const lower = min + span * i / bins.length;
        const upper = i === bins.length - 1 ? max : min + span * (i + 1) / bins.length;
        const interval = bins.length === 1 ? '数值 ' + brief(min) :
          '[' + brief(lower) + ', ' + brief(upper) + (i === bins.length - 1 ? ']' : ')');
        button.title = interval + ' · ' + countText(freq) + ' 项';
        button.setAttribute('aria-label', button.title);
        const bar = document.createElement('i');
        bar.style.height = (100 * freq / highest) + '%';
        button.appendChild(bar);
        bars.appendChild(button);
      });
      attr('bin').textContent = '选择柱形可查看区间和数量。';
    }

    function renderPositions() {
      const container = attr('compare-rows');
      container.replaceChildren();
      attr('compare-caption').textContent = prefix()+index+' 在同一输入各位置的原值 · 中线为零';
      const peak = current.comparisons.reduce((p,r) => Number.isFinite(r.value) ? Math.max(p,Math.abs(r.value)) : p,0);
      for (const {label,value,active} of current.comparisons) {
        const row = document.createElement('div'), name = document.createElement('span');
        row.className = 'nf-site-row'; name.textContent = label;
        if (active) row.setAttribute('aria-current','true');
        const axis = document.createElement('span'), bar = document.createElement('i');
        axis.className = 'nf-site-axis'; axis.setAttribute('aria-hidden','true');
        if (Number.isFinite(value)) {
          bar.className = value < 0 ? 'nf-site-negative' : 'nf-site-positive';
          bar.style.width = (peak ? 50 * Math.abs(value) / peak : 0)+'%';
          axis.appendChild(bar);
        }
        const output = document.createElement('output');
        output.setAttribute('data-field-compare-value',''); output.textContent = valueText(value);
        row.append(name,axis,output); container.appendChild(row);
      }
    }

    function cellAt(event) {
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      const x = Math.floor((event.clientX - rect.left) * columns / rect.width);
      const y = Math.floor((event.clientY - rect.top) * rows / rect.height);
      const candidate = y * columns + x;
      return x >= 0 && x < columns && y >= 0 && y < rows && candidate < current.size ? candidate : null;
    }

    function draw() {
      frame = null;
      if (closed || view !== 'heat' || !context) return;
      const rect = canvas.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const w = rect.width, h = rect.height, size = current.size;
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rasterW = Math.round(w * dpr), rasterH = Math.round(h * dpr);
      if (canvas.width !== rasterW || canvas.height !== rasterH) {
        canvas.width = rasterW; canvas.height = rasterH;
      }
      context.setTransform(rasterW / w, 0, 0, rasterH / h, 0, 0);
      context.fillStyle = '#edf1f5';
      context.fillRect(0, 0, w, h);
      columns = current.headColumns || (size ? Math.min(size, Math.max(1, Math.round(Math.sqrt(size * w / h)))) : 1);
      rows = Math.ceil(size / columns) || 1;
      canvas.dataset.columns = String(columns); canvas.dataset.rows = String(rows); canvas.dataset.count = String(size);
      attr('grid-label').textContent = current.headColumns ? '从上到下 H0–H'+(rows-1)+' · 从左到右 D0–D'+(columns-1) :
        size ? '每行 ' + columns + ' 个 · 从左到右按原生编号排列' : '尚无可定位的编号';
      if (!size) return;
      const cellW = w / columns, cellH = h / rows, gap = Math.min(cellW, cellH) > 7 ? 1 : 0;
      const absolute = attr('scale').value === 'absolute';
      const enhance = attr('contrast').checked;
      for (let i = 0; i < size; i++) {
        const col = i % columns, row = Math.floor(i / columns), x = col * cellW, y = row * cellH;
        const v = values?.[i];
        if (Number.isFinite(v)) {
          const intensity = summary.peak ? Math.min(1, Math.abs(v) / summary.peak) : 0;
          const shade = Math.round(255 * (enhance ? Math.asinh(32 * intensity) / Math.asinh(32) : intensity));
          context.fillStyle = !absolute && v < 0 ? negative[shade] : positive[shade];
        } else context.fillStyle = (col + row) % 2 ? '#abb8c5' : '#bfccd6';
        context.fillRect(x, y, cellW - gap + (gap ? 0 : 0.15), cellH - gap + (gap ? 0 : 0.15));
      }
      const x = (index % columns) * cellW, y = Math.floor(index / columns) * cellH;
      context.strokeStyle = '#fff'; context.lineWidth = 3;
      context.strokeRect(x + .75, y + .75, Math.max(1, cellW - 1.5), Math.max(1, cellH - 1.5));
      context.strokeStyle = '#142e49'; context.lineWidth = 1;
      context.strokeRect(x + .75, y + .75, Math.max(1, cellW - 1.5), Math.max(1, cellH - 1.5));
    }

    function queueDraw() {
      if (!closed && frame === null) frame = requestAnimationFrame(draw);
    }

    function renderLegend() {
      const absolute = attr('scale').value === 'absolute';
      attr('ramp').className = absolute ? 'nf-ramp nf-ramp-abs' : 'nf-ramp';
      attr('ramp').title = absolute ? '0 到最大绝对值' : '负值为橙色，0 为白色，正值为蓝色';
      attr('ramp').hidden = !summary.count;
      attr('low').textContent = !summary.count ? '' : absolute ? '0' : brief(-summary.peak);
      attr('high').textContent = summary.count ? brief(summary.peak) : '';
      attr('mapping').textContent = attr('contrast').checked && summary.count ? 'asinh 增强' : '';
    }

    function chooseView(next) {
      const hasPositions = current.comparisons?.length > 1;
      find('[data-field-view="positions"]').hidden = !hasPositions;
      if (next === 'positions' && !hasPositions) next = 'heat';
      view = next;
      views.forEach(btn => btn.setAttribute('aria-pressed', String(btn.dataset.fieldView === next)));
      attr('heat').hidden = next !== 'heat';
      attr('distribution').hidden = next !== 'distribution';
      attr('table').hidden = next !== 'table';
      attr('positions').hidden = next !== 'positions';
      find('.nf-color-options').hidden = next !== 'heat';
      attr('tooltip').hidden = true;
      if (next === 'table') renderTable();
      if (next === 'distribution') renderHistogram();
      if (next === 'positions') renderPositions();
      if (next === 'heat') queueDraw();
    }

    function update(data) {
      if (closed) return;
      const size = Number.isSafeInteger(data?.size) && data.size > 0 ? data.size : 0;
      const incoming = data?.values;
      const supplied = incoming != null;
      const complete = supplied && (Array.isArray(incoming) || ArrayBuffer.isView(incoming)) && incoming.length === size;
      const nextValues = complete ? incoming : null;
      const nextIndex = Number.isSafeInteger(data?.selected) ? clamp(data.selected, 0, Math.max(0, size - 1)) : 0;
      if (current.identity !== data?.identity || index !== nextIndex || size !== current.size)
        page = Math.floor(nextIndex / PAGE_SIZE);
      if (nextValues !== values || size !== current.size) summary = summarize(nextValues, size);
      current = {...data, size};
      index = nextIndex; values = nextValues;
      attr('title').textContent = current.vectorLabel || noun() + ' · ' + (current.measure || '原生数值');
      attr('source').textContent = [current.vectorLabel ? current.measure : '', current.source].filter(Boolean).join(' · ');
      attr('source').hidden = !attr('source').textContent;
      attr('size').textContent = '共 ' + countText(size) + ' 个' + (current.kind === 'ffn' ? '单元' : '通道');
      root.dataset.kind = current.kind || ''; root.dataset.size = String(size);
      const warning = supplied && !complete ? '向量长度与原生维度不一致，无法对齐编号。' : current.missing ||
        (!summary.count ? '该位置尚无可显示的原生数值。' : '');
      attr('missing').textContent = warning;
      attr('missing').hidden = !warning;
      attr('stats').textContent = summary.count ?
        '已采集 '+countText(summary.count)+' / '+countText(size)+' · 零值 '+countText(summary.zero)+' · 最小值 '+brief(summary.min)+' · 最大值 '+brief(summary.max) :
        '已采集 0 / '+countText(size);
      renderLegend(); renderSelection(); chooseView(view);
    }

    views.forEach(btn => btn.onclick = () => chooseView(btn.dataset.fieldView));
    attr('scale').onchange = () => { renderLegend(); queueDraw(); };
    attr('contrast').onchange = () => { renderLegend(); queueDraw(); };
    attr('form').onsubmit = event => {
      event.preventDefault();
      if (attr('index').reportValidity()) pick(attr('index').valueAsNumber);
    };
    attr('prev').onclick = () => pick(index - 1);
    attr('next').onclick = () => pick(index + 1);
    attr('page-prev').onclick = () => { page--; renderTable(); };
    attr('page-next').onclick = () => { page++; renderTable(); };
    find('tbody').onclick = event => {
      const button = event.target.closest('[data-field-row]');
      if (button) { pick(Number(button.dataset.fieldRow)); find('[data-field-row="'+index+'"]')?.focus({preventScroll:true}); }
    };
    const showBin = event => {
      const button = event.target.closest('button');
      if (button) attr('bin').textContent = button.title;
    };
    attr('bars').onclick = showBin; attr('bars').onfocusin = showBin; attr('bars').onpointerover = showBin;
    canvas.onclick = event => { const hit = cellAt(event); if (hit !== null) pick(hit); };
    canvas.onpointermove = event => {
      const hit = cellAt(event), bubble = attr('tooltip');
      if (hit === null || event.pointerType === 'touch') { bubble.hidden = true; return; }
      const rect = canvas.getBoundingClientRect();
      bubble.textContent = indexLabel(hit)+' · '+valueText(values?.[hit]);
      bubble.hidden = false;
      bubble.style.left = clamp(event.clientX - rect.left + 10, 0, Math.max(0, rect.width - bubble.offsetWidth))+'px';
      bubble.style.top = clamp(event.clientY - rect.top - bubble.offsetHeight - 5, 0, rect.height-bubble.offsetHeight)+'px';
    };
    canvas.onpointerleave = () => { attr('tooltip').hidden = true; };
    canvas.onkeydown = event => {
      const keys = {ArrowRight:index+1, ArrowLeft:index-1, ArrowDown:index+columns, ArrowUp:index-columns, Home:0, End:current.size-1};
      if (Object.hasOwn(keys, event.key)) {
        event.preventDefault(); pick(clamp(keys[event.key],0,current.size-1));
      }
    };
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(queueDraw) : null;
    observer?.observe(canvas);
    window.addEventListener('resize', queueDraw);
    update({size:0});
    return {update, destroy() {
      closed = true;
      observer?.disconnect();
      window.removeEventListener('resize', queueDraw);
      if (frame !== null) cancelAnimationFrame(frame);
      root.remove();
    }};
  }
  globalThis.AI2050NativeField = Object.freeze({create});
})();
