/* 研发 · 贡献算力：本机节点服务的桥接（非组件）。
   桌面客户端的 preload 暴露 window.ai2050Desktop.node，主进程只转发白名单内的本机接口，本机令牌不进页面；
   浏览器里的网页没有这个桥，也就不能操控节点（只显示下载入口）。 */
import { useCallback, useEffect, useRef, useState } from 'react';

export function nodeBridge() {
  if (typeof window === 'undefined') return null;
  return window.ai2050Desktop?.node || null;
}

export async function nodeCall(method, path, body) {
  const bridge = nodeBridge();
  if (!bridge) throw new Error('只有桌面客户端可以控制本机节点');
  const r = await bridge.request(method, path, body);
  if (!r || r.status < 200 || r.status >= 300) {
    const err = new Error(r?.data?.detail || `本机节点服务返回 ${r?.status ?? '无响应'}`);
    err.status = r?.status;
    throw err;
  }
  return r.data;
}

/* 监护状态（主进程推送）+ 节点状态与日志（每 2 秒读一次本机接口，不访问中心） */
export function useNode() {
  const bridge = nodeBridge();
  const [sup, setSup] = useState(null);
  const [status, setStatus] = useState(null);
  const [logs, setLogs] = useState([]);
  const [error, setError] = useState('');
  const seq = useRef(0);
  const lastPort = useRef(null);

  useEffect(() => {
    if (!bridge) return undefined;
    let dead = false;
    bridge.info().then(i => { if (!dead) setSup(i); });
    const off = bridge.onInfo(i => setSup(i));
    return () => { dead = true; off?.(); };
  }, [bridge]);

  const live = sup && (sup.state === 'running' || sup.state === 'attached');
  const poll = useCallback(async port => {
    try {
      const s = await nodeCall('GET', '/status');
      setStatus(s);
      setError('');
      let fresh = false;
      if (port !== lastPort.current || s.log_seq < seq.current) {   // 换了一个服务进程：日志从头读
        lastPort.current = port;
        seq.current = 0;
        fresh = true;
      }
      const items = s.log_seq > seq.current ? (await nodeCall('GET', `/logs?after=${seq.current}`)).items : [];
      if (items.length) seq.current = items[items.length - 1].seq;
      if (fresh || items.length) setLogs(prev => (fresh ? items : [...prev, ...items]).slice(-500));
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    if (!live) return undefined;
    const port = sup?.port;
    const first = setTimeout(() => poll(port), 0);
    const t = setInterval(() => poll(port), 2000);
    return () => { clearTimeout(first); clearInterval(t); };
  }, [live, poll, sup?.port]);

  return { bridge, sup, live, status, logs, error, setStatus, poll: () => poll(sup?.port) };
}

export const NODE_STATE = {
  stopped: { label: '未开始', tone: 'muted', icon: '○' },
  running: { label: '运行中', tone: 'good', icon: '●' },
  waiting: { label: '等待中', tone: 'warn', icon: '◐' },
  paused: { label: '已暂停', tone: 'muted', icon: '❚❚' },
  error: { label: '出错', tone: 'bad', icon: '✗' },
};

export const SUP_STATE = {
  idle: '尚未启动', starting: '启动中…', running: '运行中（由客户端启动）', attached: '运行中（已接管命令行启动的服务）',
  stopped: '已停止', 'no-python': '没有找到 Python', 'no-script': '安装包缺少节点程序', failed: '出错',
};

export const TASK_KIND_LABEL = { census_shard: '神经元普查分片', family_test: '模式族测试' };

export const RESULT_LABEL = {
  accepted: ['✓', '中心校验通过', 'good'], done: ['✓', '已上传', 'good'],
  rejected: ['✗', '中心拒收', 'bad'], failed: ['✗', '失败', 'bad'],
};

export function fmtTime(t) {
  if (!t) return '—';
  const d = new Date(t * 1000);
  const p = n => String(n).padStart(2, '0');
  return `${d.getMonth() + 1}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}
