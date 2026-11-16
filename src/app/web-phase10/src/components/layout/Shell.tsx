import { Menu, Moon, Sun, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router";
import { cn } from "../../lib/cn";
import { useStudy } from "../../lib/data";
import { setTheme, useTheme } from "../../lib/theme";

/* Navigation follows what people come to do: explore the network and its history, make a forecast, check the evidence,
   read how it was done. `title` is the page name (document title, heading focus); `nav` is the short sign label. */
export const ROUTES = [
  { to: "/", title: "Overview", nav: "Overview", group: "Explore" },
  // Replay and Disruptions exist only in `--mode replay` builds; a default production build leaves them out.
  ...(import.meta.env.MODE === "replay"
    ? ([
        { to: "/replay", title: "Historical Traffic Replay", nav: "Replay", group: "Explore" },
        { to: "/disruptions", title: "Historical Disruption Explorer", nav: "Disruptions", group: "Explore" },
      ] as const)
    : []),
  { to: "/forecast", title: "Forecasting", nav: "Forecasting", group: "Forecast" },
  { to: "/performance", title: "Model performance", nav: "Performance", group: "Evidence" },
  { to: "/federated", title: "Federated learning", nav: "Federated", group: "Evidence" },
  { to: "/methodology", title: "Methodology", nav: "Methodology", group: "Reference" },
  { to: "/about", title: "About", nav: "About", group: "Reference" },
] as const;
const GROUPS = ["Explore", "Forecast", "Evidence", "Reference"] as const;

function Wordmark() {
  return (
    <NavLink to="/" className="flex shrink-0 items-center gap-2 rounded-control px-1.5 py-1 focus-visible:outline-white" aria-label="AIORI Traffic Intelligence, overview">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
        <path d="M5 21 10 3M14 3l5 18" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" />
        <path d="M12 8v2.4M12 14v2.6" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <span className="pt-0.5 text-[17px] font-extrabold tracking-[0.01em] text-white">
        AIORI<span className="ml-1.5 hidden font-medium text-white/80 sm:inline">Traffic Intelligence</span>
      </span>
    </NavLink>
  );
}

function ServiceStatus({ className }: { className?: string }) {
  const { state } = useStudy();
  const [dot, text] =
    state.status === "ready"
      ? ["bg-white", `Model service ready · ${state.data.health.device.toUpperCase()}`]
      : state.status === "error"
        ? ["bg-[#ffb4a8]", "Model service unavailable"]
        : ["bg-white/60 animate-pulse", "Connecting to model service"];
  return (
    <p className={cn("flex items-center gap-2 whitespace-nowrap text-xs", className)} role="status">
      <span className={cn("size-1.5 rounded-full", dot)} aria-hidden />
      {text}
    </p>
  );
}

function ThemeToggle() {
  const theme = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="grid size-8 cursor-pointer place-items-center rounded-[3px] text-white/90 hover:bg-white/14 hover:text-white focus-visible:outline-white"
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
      onClick={() => setTheme(next)}
    >
      {theme === "dark" ? <Sun className="size-4" aria-hidden /> : <Moon className="size-4" aria-hidden />}
    </button>
  );
}

const REPO_URL = "https://github.com/aiori-hackathon/a3-ps003-tc188";

const GitHubMark = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 16 16" className={className} fill="currentColor" aria-hidden>
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
  </svg>
);

function Footer() {
  return (
    <footer className="border-t border-line-strong bg-raised">
      <div className="mx-auto grid w-full max-w-[1320px] gap-8 px-4 py-10 sm:px-6 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1.2fr)] lg:px-8">
        <div className="flex flex-col gap-1.5">
          <p className="text-sm font-bold">AIORI Traffic Intelligence</p>
          <p className="text-sm text-fg-2">Privacy-aware freeway traffic forecasting research.</p>
          <p className="text-xs text-fg-3">Everything shown is historical XTraffic data. Nothing here is live.</p>
        </div>
        <nav aria-label="Footer" className="flex flex-col gap-2 text-sm">
          <NavLink to="/performance" className="w-fit text-fg-2 underline decoration-line-strong hover:text-fg">Model performance</NavLink>
          <NavLink to="/methodology" className="w-fit text-fg-2 underline decoration-line-strong hover:text-fg">Methodology</NavLink>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="inline-flex w-fit items-center gap-1.5 text-fg-2 underline decoration-line-strong hover:text-fg">
            <GitHubMark className="size-3.5" />
            GitHub repository
          </a>
        </nav>
        <p className="text-xs leading-relaxed text-fg-3">
          Forecasts are research demonstrations, not traffic advisories. Reported error rates are averages over a held-out test period, not a guarantee for any
          single forecast.
        </p>
      </div>
    </footer>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLButtonElement>(null);
  const firstLoad = useRef(true);
  const route = ROUTES.find((r) => r.to === pathname);
  const title = route?.title ?? "Page not found";

  useEffect(() => {
    document.title = `${title} · AIORI Traffic Intelligence`;
    window.scrollTo(0, 0);
    setOpen(false);
    // After navigation, move focus to the new page's heading so screen-reader and keyboard users land on the content.
    // Pages load lazily, so keep looking for the heading for about a second; the first load keeps the browser's focus.
    if (firstLoad.current) {
      firstLoad.current = false;
      return;
    }
    let id = 0;
    let tries = 50;
    const seek = () => {
      const h1 = document.querySelector<HTMLElement>("main h1");
      if (h1) h1.focus({ preventScroll: true });
      else if (tries-- > 0) id = window.setTimeout(seek, 20);
    };
    id = window.setTimeout(seek, 0);
    return () => clearTimeout(id);
  }, [pathname, title]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        menuRef.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  const link = (r: (typeof ROUTES)[number], sheet = false) => (
    <NavLink
      key={r.to}
      to={r.to}
      end
      className={({ isActive }) =>
        sheet
          ? cn("flex h-10 items-center rounded-control px-3 text-md", isActive ? "bg-accent font-bold text-accent-ink" : "font-medium text-fg hover:bg-surface-2")
          : cn(
              "flex h-8 items-center rounded-[3px] px-2.5 pt-px text-sm font-semibold transition-colors focus-visible:outline-white",
              isActive ? "bg-white text-sign-deep" : "text-white/90 hover:bg-white/14 hover:text-white",
            )
      }
    >
      {sheet ? r.title : r.nav}
    </NavLink>
  );

  return (
    <div className="flex min-h-dvh flex-col">
      <a href="#main" className="sr-only z-50 rounded-control bg-fg px-3 py-2 text-sm text-surface focus:not-sr-only focus:fixed focus:left-3 focus:top-3">
        Skip to content
      </a>
      {/* The navigation is drawn as an overhead guide sign: green panel, white inset border, white lettering. */}
      <div className="sticky top-0 z-30 bg-bg px-2 pt-2 sm:px-3">
        <header className="relative mx-auto max-w-[1320px] rounded-[10px] bg-sign p-[3px] shadow-[0_1px_0_rgba(20,32,27,0.25)]">
          <div className="flex h-12 items-center gap-2 rounded-[7px] border-2 border-white/90 px-2 text-white sm:px-3">
            <Wordmark />
            <nav aria-label="Main navigation" className="ml-3 hidden min-w-0 items-center xl:flex">
              {GROUPS.map((g, i) => (
                <div key={g} role="group" aria-label={g} className={cn("flex items-center gap-0.5", i > 0 && "ml-2.5 border-l border-white/35 pl-2.5")}>
                  {ROUTES.filter((r) => r.group === g).map((r) => link(r))}
                </div>
              ))}
            </nav>
            <div className="ml-auto flex items-center gap-3">
              <ServiceStatus className="hidden text-white/85 min-[1440px]:flex" />
              <span className="hidden whitespace-nowrap text-xs text-white/85 md:inline xl:hidden min-[1560px]:inline" title="All examples, readings and metrics come from recorded XTraffic data">
                Historical data · not live
              </span>
              {pathname !== "/forecast" && pathname !== "/" && (
                <NavLink to="/forecast" className="hidden h-8 items-center rounded-[3px] bg-white px-3 pt-px text-sm font-bold text-sign-deep hover:bg-white/90 focus-visible:outline-white sm:inline-flex xl:hidden min-[1360px]:inline-flex">
                  Run a forecast
                </NavLink>
              )}
              <a
                href={REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub repository (opens in a new tab)"
                className="grid size-8 place-items-center rounded-[3px] text-white/90 hover:bg-white/14 hover:text-white focus-visible:outline-white"
              >
                <GitHubMark className="size-4" />
              </a>
              <ThemeToggle />
              <button
                ref={menuRef}
                type="button"
                className="grid size-8 cursor-pointer place-items-center rounded-[3px] text-white hover:bg-white/14 focus-visible:outline-white xl:hidden"
                aria-expanded={open}
                aria-controls="nav-sheet"
                aria-label={open ? "Close navigation" : "Open navigation"}
                onClick={() => setOpen((o) => !o)}
              >
                {open ? <X className="size-[18px]" aria-hidden /> : <Menu className="size-[18px]" aria-hidden />}
              </button>
            </div>
          </div>
          <div
            id="nav-sheet"
            inert={!open}
            className={cn(
              "absolute inset-x-0 top-full mt-2 max-h-[calc(100dvh-5.5rem)] origin-top overflow-auto rounded-card border border-line-strong bg-surface p-3 shadow-float transition-[opacity,transform] duration-200 ease-[var(--ease-out-expo)] xl:hidden",
              open ? "opacity-100" : "pointer-events-none -translate-y-1 opacity-0",
            )}
          >
            <nav aria-label="Main navigation" className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
              {GROUPS.map((g) => (
                <div key={g}>
                  <p className="mb-1 px-3 text-xs font-semibold text-fg-3">{g}</p>
                  <div className="flex flex-col gap-0.5">{ROUTES.filter((r) => r.group === g).map((r) => link(r, true))}</div>
                </div>
              ))}
            </nav>
            <p className="mt-3 border-t border-line px-3 pt-3 text-xs text-fg-3">Historical data · not live. Recorded XTraffic readings, 2022–2024.</p>
          </div>
        </header>
      </div>
      <div aria-hidden onClick={() => setOpen(false)} className={cn("fixed inset-0 z-20 bg-fg/30 transition-opacity xl:hidden", open ? "opacity-100" : "pointer-events-none opacity-0")} />
      <main id="main" className="mx-auto w-full max-w-[1320px] flex-1 px-4 pb-20 pt-8 sm:px-6 lg:px-8 lg:pt-10">
        {children}
      </main>
      <Footer />
    </div>
  );
}
