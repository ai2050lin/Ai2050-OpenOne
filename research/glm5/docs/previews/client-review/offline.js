/* Native workbench links use query strings. Under file://, store them in the fragment. */
(() => {
  const initial = 'wb=1&lens=spatial&view=graph&gview=internal&mdl=Qwen3-14B';
  function readRoute() {
    const page = new URL(window.location.href);
    page.search = page.hash.startsWith('#?') ? page.hash.slice(2) : initial;
    page.hash = '';
    return page;
  }
  function setRoute(action, state, to) {
    if (!to) return;
    const logical = new URL(to, readRoute());
    const current = new URL(window.location.href);
    current.hash = '?' + logical.searchParams.toString();
    if (current.href !== window.location.href) window.history[action](state, '', current);
  }
  window.reviewRoute = {
    get href() { return readRoute().href; },
    get search() { return readRoute().search; }
  };
  window.reviewHistory = {
    get state() { return window.history.state; },
    replaceState(state, title, to) { setRoute('replaceState', state, to); },
    pushState(state, title, to) { setRoute('pushState', state, to); }
  };
  // Native data probes receive an offline response without contacting any node.
  window.fetch = async () => new Response(JSON.stringify({detail:'静态评审稿只提供已嵌入的历史资料。'}), {
    status:503, headers:{'Content-Type':'application/json'}
  });
  // Open at the affected console. The original scene remains immediately above.
  const observer = new MutationObserver(() => {
    const host = document.querySelector('.client-evidence-host');
    if (!host) return;
    observer.disconnect();
    requestAnimationFrame(() => {
      const scroller = document.querySelector('.wb-main');
      const target = document.querySelector(window.innerWidth > 760 ? '.aw-body' : '.aw-right');
      if (scroller && target) scroller.scrollTop += target.getBoundingClientRect().top -
        scroller.getBoundingClientRect().top - (window.innerWidth > 760 ? 65 : 8);
    });
  });
  observer.observe(document.getElementById('root'), {childList:true,subtree:true});
})();
