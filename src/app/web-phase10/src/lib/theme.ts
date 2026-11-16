import { useSyncExternalStore } from "react";

/* Light / dark theme. The choice lives on <html data-theme>; index.html sets it before the first paint (stored choice,
   else the system preference) so there is no flash. This module only reads, toggles and persists it. */
export type Theme = "light" | "dark";
const KEY = "aiori-theme";

const read = (): Theme => (document.documentElement.dataset.theme === "dark" ? "dark" : "light");

function subscribe(cb: () => void) {
  const mo = new MutationObserver(cb);
  mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  // Follow the system setting until the user picks a theme themselves.
  const mq = matchMedia("(prefers-color-scheme: dark)");
  const onSystem = () => {
    let stored: string | null = null;
    try { stored = localStorage.getItem(KEY); } catch { /* storage blocked: keep following the system */ }
    if (!stored) document.documentElement.dataset.theme = mq.matches ? "dark" : "light";
  };
  mq.addEventListener("change", onSystem);
  return () => { mo.disconnect(); mq.removeEventListener("change", onSystem); };
}

export function setTheme(t: Theme) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem(KEY, t); } catch { /* the choice just won't persist */ }
}

export const useTheme = () => useSyncExternalStore(subscribe, read, () => "light" as Theme);

/** Current value of a design token, for colours drawn outside CSS (the Leaflet canvas). */
export const token = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(`--color-${name}`).trim();
