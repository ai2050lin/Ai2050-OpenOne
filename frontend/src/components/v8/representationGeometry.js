/* Shared by the application and its file:// design review. Rows must already be
   paired observations, in exactly the same order in the two representations. */
(() => {
  'use strict';

  function dot(a, b) {
    let value = 0;
    for (let i = 0; i < a.length; i++) value += a[i] * b[i];
    return value;
  }

  function describe(rows) {
    const n = rows.length;
    const width = rows[0]?.length;
    if (!Number.isSafeInteger(width) || width < 1) return null;
    const rowMax = [];
    let maximum = 0;
    for (const row of rows) {
      if (!(Array.isArray(row) || ArrayBuffer.isView(row)) || row.length !== width) return null;
      let peak = 0;
      for (const v of row) {
        if (!Number.isFinite(v)) return null;
        peak = Math.max(peak, Math.abs(v));
      }
      rowMax.push(peak);
      maximum = Math.max(maximum, peak);
    }
    // Scaling before the dot products avoids overflow and preserves both metrics.
    const unit = rows.map((row, i) => {
      if (rowMax[i] === 0) return null;
      const v = Float64Array.from(row, x => x / rowMax[i]);
      const length = Math.sqrt(dot(v, v));
      for (let j = 0; j < width; j++) v[j] /= length;
      return v;
    });
    // Center each feature across observations. Translate by the first row
    // before taking the mean so identical observations center to exact zero.
    const scale = maximum || 1;
    const mean = new Float64Array(width);
    const centered = rows.map(row => Float64Array.from(row, (v, j) => v / scale - rows[0][j] / scale));
    for (const row of centered) for (let j = 0; j < width; j++) mean[j] += row[j] / n;
    let maxCentered = 0;
    for (const row of centered) {
      for (let j = 0; j < width; j++) {
        row[j] -= mean[j];
        maxCentered = Math.max(maxCentered, Math.abs(row[j]));
      }
    }
    if (maxCentered > 0) {
      for (const row of centered) for (let j = 0; j < width; j++) row[j] /= maxCentered;
    }
    const gram = new Float64Array(n * n);
    const distances = rows.map(() => Array(n).fill(null));
    const pairs = [];
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        gram[i*n+j] = gram[j*n+i] = dot(centered[i], centered[j]);
        if (unit[i] && unit[j]) {
          distances[i][j] = distances[j][i] = i === j ? 0 : Math.max(0, Math.min(2, 1 - dot(unit[i], unit[j])));
        }
        if (j > i) pairs.push(distances[i][j]);
      }
    }
    return {gram, distances, pairs};
  }

  function centeredRanks(values) {
    if (values.includes(null)) return null;
    const ordered = values.map((value, index) => ({value, index})).sort((a, b) => a.value - b.value);
    const ranks = new Float64Array(ordered.length);
    let first = 0;
    while (first < ordered.length) {
      let end = first + 1;
      // Distances lie in [0,2]. Treat floating-point noise at machine precision as a tie.
      while (end < ordered.length && ordered[end].value - ordered[first].value <= 16 * Number.EPSILON) end++;
      const rank = (first + end - ordered.length) / 2;
      for (let p = first; p < end; p++) ranks[ordered[p].index] = rank;
      first = end;
    }
    return ranks;
  }

  function normalizedDot(a, b) {
    if (!a || !b) return null;
    const divisor = Math.sqrt(dot(a, a)) * Math.sqrt(dot(b, b));
    return divisor > 0 ? Math.max(-1, Math.min(1, dot(a, b) / divisor)) : null;
  }

  function compare(rowsA, rowsB) {
    const count = Array.isArray(rowsA) ? rowsA.length : 0;
    const empty = {count, cka:null, rsa:null, rdmA:[], rdmB:[]};
    if (!Array.isArray(rowsA) || !Array.isArray(rowsB) || count !== rowsB.length || count < 3) {
      return {...empty, error:'两层至少需要三个一一对应的读出位置。'};
    }
    const a = describe(rowsA), b = describe(rowsB);
    if (!a || !b) return {...empty, error:'有位置缺少完整的有限向量，或同层各位置的坐标数不一致。'};
    const cka = normalizedDot(a.gram, b.gram);
    return {
      count, error:null, cka:cka == null ? null : Math.max(0, cka),
      rsa:normalizedDot(centeredRanks(a.pairs), centeredRanks(b.pairs)),
      rdmA:a.distances, rdmB:b.distances,
    };
  }

  globalThis.AI2050RepresentationGeometry = Object.freeze({compare});
})();
