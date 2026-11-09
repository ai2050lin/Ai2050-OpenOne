export const ASSOC_VIEWS = { iso: [-0.26, 0.71, '斜视'], side: [0, 0.22, '侧视'], top: [-0.25, 1.2, '俯视'] };
// Both rendering and hit testing use this affine projection; grid order stays native.
export function assocProject(s, x, y, z) {
  const cy = Math.cos(s.yaw), sy = Math.sin(s.yaw), cp = Math.cos(s.pitch), sp = Math.sin(s.pitch);
  const x1 = x * cy - z * sy, z1 = x * sy + z * cy;
  const k = Math.min(s.W / 1060, s.H / 500) * s.zoom;
  return [s.W * 0.455 + (s.panX || 0) + x1 * k, s.H * 0.52 + (s.panY || 0) - (y * cp - z1 * sp) * k];
}

export function assocHit(h, project, mx, my) {
  const p = project(h.x0, h.y, h.z0), a = project(h.x1, h.y, h.z0), b = project(h.x0, h.y, h.z1);
  const ax = a[0] - p[0], ay = a[1] - p[1], bx = b[0] - p[0], by = b[1] - p[1], det = ax * by - ay * bx;
  if (Math.abs(det) < 1e-7) return null;
  const dx = mx - p[0], dy = my - p[1], u = (dx * by - dy * bx) / det, v = (ax * dy - ay * dx) / det;
  if (u < 0 || u >= 1 || v < 0 || v >= 1) return null;
  if (!h.o.grid) return { o: h.o.id, l: h.l, part: u < 0.5 ? '注意力' : 'MLP' };
  const [gw, gh] = h.o.grid, idx = Math.floor(v * gh) * gw + Math.floor(u * gw);
  return idx < h.o.count ? { o: h.o.id, l: h.l, idx } : null;
}
