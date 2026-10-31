import { useEffect, useState } from 'react';

const SNAPSHOT_URL = '/research_data/current/snapshot.json';
const EMPTY_SNAPSHOT = {
  framework: { title: 'Research Framework', stages: [] },
  current: null,
  roadmap: [],
  summaries: { evidence: { latest: [] } },
  counts: {},
};

export function useResearchSnapshot() {
  const [state, setState] = useState({ snapshot: EMPTY_SNAPSHOT, loading: true, error: '' });

  useEffect(() => {
    const controller = new AbortController();
    fetch(SNAPSHOT_URL, { cache: 'no-store', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Canonical Snapshot ${response.status}`);
        return response.json();
      })
      .then((snapshot) => {
        setState({
          snapshot: {
            framework: snapshot?.framework || EMPTY_SNAPSHOT.framework,
            current: snapshot?.current || null,
            roadmap: snapshot?.roadmap || EMPTY_SNAPSHOT.roadmap,
            summaries: snapshot?.summaries || EMPTY_SNAPSHOT.summaries,
            counts: snapshot?.counts || EMPTY_SNAPSHOT.counts,
            ...snapshot,
          },
          loading: false,
          error: '',
        });
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setState({ snapshot: EMPTY_SNAPSHOT, loading: false, error: error.message });
      });
    return () => controller.abort();
  }, []);

  return state;
}
