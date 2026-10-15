import { useEffect, useState } from 'react';

const COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

function readCookie(name, fallback) {
  if (typeof document === 'undefined') return fallback;
  const prefix = `${encodeURIComponent(name)}=`;
  const entry = document.cookie.split(';').map(value => value.trim()).find(value => value.startsWith(prefix));
  if (!entry) return fallback;
  try {
    return JSON.parse(decodeURIComponent(entry.slice(prefix.length)));
  } catch {
    return fallback;
  }
}

/** Persist small, non-sensitive UI preferences across page reloads. */
export default function useCookiePreference(name, fallback) {
  const [value, setValue] = useState(() => readCookie(name, fallback));

  useEffect(() => {
    if (typeof document === 'undefined') return;
    try {
      document.cookie = `${encodeURIComponent(name)}=${encodeURIComponent(JSON.stringify(value))}; Max-Age=${COOKIE_MAX_AGE}; Path=/; SameSite=Lax`;
    } catch {
      // Cookie storage can be disabled; the in-memory preference still works.
    }
  }, [name, value]);

  return [value, setValue];
}
