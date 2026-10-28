/* 账号管理页的筛选与状态（纯函数，组件文件只导出组件）。 */

export const ROLE_LABEL = { admin: '管理员', member: '成员' };

/* 状态优先级：停用 > 锁定 > 正常；符号与文字同时给出 */
export function userState(u) {
  if (u.disabled) return { key: 'disabled', status: 'fail', label: '已停用' };
  if (u.locked) return { key: 'locked', status: 'warn', label: '锁定中' };
  return { key: 'active', status: 'pass', label: '正常' };
}

export function filterUsers(users, { q = '', role = '', state = '' } = {}) {
  const needle = q.trim().toLowerCase();
  return (users || []).filter(u => {
    if (role && u.role !== role) return false;
    if (state && userState(u).key !== state) return false;
    if (!needle) return true;
    return [u.username, u.display_name, u.email].some(v => String(v || '').toLowerCase().includes(needle));
  });
}

export function countBy(users) {
  const out = { total: 0, admin: 0, active: 0, disabled: 0, locked: 0 };
  for (const u of users || []) {
    out.total += 1;
    if (u.role === 'admin') out.admin += 1;
    out[userState(u).key] += 1;
  }
  return out;
}
