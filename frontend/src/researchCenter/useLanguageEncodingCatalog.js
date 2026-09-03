import { useEffect, useState } from 'react';

const CATALOG_URL = '/research_data/current/language_encoding_catalog.json';

export function useLanguageEncodingCatalog() {
  const [state, setState] = useState({ catalog: null, loading: true, error: '' });

  useEffect(() => {
    const controller = new AbortController();
    fetch(CATALOG_URL, { cache: 'no-store', signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Encoding catalog ${response.status}`);
        return response.json();
      })
      .then((catalog) => setState({ catalog, loading: false, error: '' }))
      .catch((error) => {
        if (error.name !== 'AbortError') setState({ catalog: null, loading: false, error: error.message });
      });
    return () => controller.abort();
  }, []);

  return state;
}
