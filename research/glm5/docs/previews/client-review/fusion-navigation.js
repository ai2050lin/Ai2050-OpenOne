/* Shared by the file preview and the native workbench URL reader. */
window.openEvidenceWorkspace = function (destination, entry) {
  const url = new URL(window.reviewRoute.href);
  const q = url.searchParams;
  q.set('wb', '1');
  if (destination === 'object' || destination === 'head') {
    const model = window.__reviewEntries['/api/model/skeleton'].models.find(m => m.model_snapshot === entry.snapshot);
    if (!model) return '离线稿缺少这个快照的模型结构。';
    const {layer, module, index} = entry.location;
    const study = q.get('lens') === 'spatial' && window.__evidenceData.studyDownloads?.find(s =>
      s.run === q.get('run') && String(s.phase) === q.get('studyPack') && s.snapshot === entry.snapshot);
    for (const key of ['scoord', 'strace', 'straceb', 'run', 'trace', 'context_trace', 'rtab', 'reviewPage']) q.delete(key);
    Object.entries({lens:'spatial', view:'graph', gview:'internal', mdl:model.model, snapshot:entry.snapshot,
      sblock:layer, smodule:module, sindex:index, reviewFocus:destination}).forEach(([k,v])=>q.set(k,v));
    if (destination === 'head') q.set('observeMode', 'archive');
    if (study) q.set('run', study.run);
  } else {
    q.set('lens','route');
    q.set('rtab','rules');
    q.delete('reviewFocus');
    if (destination === 'ledger') q.set('reviewPage','ledger');
    else q.delete('reviewPage');
  }
  window.reviewHistory.pushState(null, '', url);
  window.dispatchEvent(new PopStateEvent('popstate'));
  requestAnimationFrame(()=>document.querySelector('.wb-main')?.scrollTo(0,0));
  return '';
};
