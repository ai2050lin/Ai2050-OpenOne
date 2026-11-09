const { test } = require('node:test');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const path = require('node:path');
test('all native cells stay selectable after pan, rotation and zoom; padded cells are rejected', async () => {
  const { assocHit, assocProject } = await import(pathToFileURL(path.resolve(__dirname, '../../frontend/src/components/v8/assocLayerGeometry.js')));
  for (const n of [32, 2560, 9728]) {
    const w = n === 32 ? 8 : Math.ceil(Math.sqrt(n * 1.6)), gh = Math.ceil(n / w);
    const h = { o: { id: 'test', grid: [w, gh], count: n }, l: 35, x0: -8, x1: 491, z0: 31, z1: 203, y: 20 };
    for (const [yaw, pitch, zoom, panX, panY] of [[-.26, .71, 1, 0, 0], [1.7, .2, 4, 210, -61], [-3.9, 1.4, .5, -85, 102]]) {
      const project = (x, y, z) => assocProject({ W: 1348, H: 560, yaw, pitch, zoom, panX, panY }, x, y, z);
      for (let i = 0; i < w * gh; i++) {
        const xy = project(h.x0 + (h.x1 - h.x0) * (i % w + .5) / w, 20, h.z0 + (h.z1 - h.z0) * (Math.floor(i / w) + .5) / gh);
        const result = assocHit(h, project, ...xy);
        assert.equal(result?.idx ?? null, i < n ? i : null);
      }
    }
  }
});
