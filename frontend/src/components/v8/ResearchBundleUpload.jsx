import { useState } from 'react';
import { API_BASE, authJson } from '../app/rdc_fusion/api.js';
import { useAuth } from '../app/rdc_fusion/authContext.js';
import ResearchDisclosure from './ResearchDisclosure.jsx';
import { prepareResearchBundle, uploadResearchBundle } from './researchBundleUpload.js';
import { workbenchHref } from './v8Util.js';

export default function ResearchBundleUpload() {
  const { user } = useAuth();
  const [bundle, setBundle] = useState(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [published, setPublished] = useState([]);
  if (user?.role !== 'admin') return null;

  async function select(event) {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    setBundle(null);
    setPublished([]);
    setError('');
    setBusy(true);
    setStatus('正在校验研究包…');
    try {
      const verified = await prepareResearchBundle(file);
      setBundle(verified);
      setStatus(`已校验 ${verified.objects.size} 个对象，可登记到中心。`);
    } catch (reason) {
      setError(reason.message);
      setStatus('');
    } finally {
      setBusy(false);
      input.value = '';
    }
  }

  async function publish() {
    setError('');
    setBusy(true);
    setStatus('正在连接研究中心…');
    try {
      const center = new URL(API_BASE || '/', window.location.href);
      const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(center.hostname);
      if (center.username || center.password || (center.protocol !== 'https:' && !(center.protocol === 'http:' && loopback)))
        throw new Error('上传研究包需要 HTTPS 中心或本机服务。');
      const result = await uploadResearchBundle(bundle, (path, options) => authJson(path, { ...options, redirect:'error' }),
        (done, total) => setStatus(`已登记或确认 ${done} / ${total} 个对象…`));
      setPublished(result.studies);
      setStatus(`已确认 ${result.total} 个对象，本次新增 ${result.uploaded} 个。公开索引已可读取。`);
      setBundle(null);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  }

  return <ResearchDisclosure title="上传研究包">
    <p className="v8-note">选择发布用的 .tar.gz。客户端校验全部对象后，通过当前中心登记。中断后可重试，已登记对象按哈希跳过。</p>
    <div className="v6-toolbar">
      <label className="fw-tbtn">选择研究包<input aria-label="选择研究包" type="file" accept=".tar.gz,.tgz" hidden disabled={busy} onChange={select}/></label>
      <button className="fw-tbtn" type="button" disabled={busy || !bundle} onClick={publish}>登记到中心</button>
    </div>
    {bundle && <p className="v8-note">准备登记 {bundle.studies.map(study => study.run_id).join('、')}</p>}
    {status && <p className="v8-note" role="status">{status}</p>}
    {error && <p className="v6-note warn" role="alert">{error}</p>}
    {published.map(study => <p key={study.run_id+'|'+study.model_snapshot}><a href={workbenchHref(new URLSearchParams({
      lens:'spatial',view:'graph',gview:'internal',run:study.run_id,snapshot:study.model_snapshot,
    }).toString())}>打开 3D 研究成果 · {study.run_id}</a></p>)}
  </ResearchDisclosure>;
}
