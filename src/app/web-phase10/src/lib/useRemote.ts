import { useEffect, useRef, useState } from "react";
import type { Lru } from "./replay";

/** Fetch `key` once (cached); aborts the previous request when the key changes, so an older response can never
 * replace newer data. `last` keeps the previous result to display while the next one loads. */
export function useRemote<T>(key: string | null, cache: Lru<T>, load: (s: AbortSignal) => Promise<T>, delay = 0) {
  const [st, setSt] = useState<{ key: string | null; data?: T; error?: string }>({ key: null });
  const last = useRef<T | undefined>(undefined);
  const loadRef = useRef(load);
  loadRef.current = load;
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (key === null) return;
    const hit = cache.get(key);
    if (hit !== undefined) {
      setSt({ key, data: hit });
      return;
    }
    const ac = new AbortController();
    const id = window.setTimeout(() => {
      loadRef.current(ac.signal).then(
        (d) => {
          if (ac.signal.aborted) return;
          cache.set(key, d);
          setSt({ key, data: d });
        },
        (e: Error) => {
          if (e.name !== "AbortError") setSt({ key, error: e.message });
        },
      );
    }, delay);
    return () => {
      clearTimeout(id);
      ac.abort();
    };
  }, [key, cache, delay, retry]);
  const fresh = st.key === key && key !== null;
  if (fresh && st.data !== undefined) last.current = st.data;
  return {
    data: fresh ? st.data : undefined,
    last: last.current,
    error: fresh ? st.error : undefined,
    loading: key !== null && !fresh,
    retry: () => setRetry((r) => r + 1),
  };
}
