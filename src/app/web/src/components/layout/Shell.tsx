import { BookOpenText, ChartNoAxesColumn, Info, LayoutGrid, Menu, Network, TrendingUp, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router";
import { cn } from "../../lib/cn";
import { useStudy } from "../../lib/data";
import { ButtonLink } from "../ui";

export const ROUTES = [
  { to: "/", title: "Overview", icon: LayoutGrid, group: "Product" },
  { to: "/forecast", title: "Forecasting", icon: TrendingUp, group: "Product" },
  { to: "/performance", title: "Model performance", icon: ChartNoAxesColumn, group: "Product" },
  { to: "/federated", title: "Federated learning", icon: Network, group: "Research" },
  { to: "/methodology", title: "Methodology", icon: BookOpenText, group: "Research" },
  { to: "/about", title: "About", icon: Info, group: "Research" },
] as const;

function Wordmark() {
  return (
    <NavLink to="/" className="flex items-center gap-2.5 rounded-control px-2 py-1" aria-label="AIORI Traffic Intelligence, overview">
      <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
        <rect width="32" height="32" rx="8" fill="#1f1f1f" stroke="#353535" />
        <path d="M9 24 15 8M17 8l6 16" stroke="#3bb27e" strokeWidth="2.6" strokeLinecap="round" />
        <path d="M16 13v2M16 19v2" stroke="#ededed" strokeWidth="2" strokeLinecap="round" />
      </svg>
      <span className="text-[15px] font-semibold tracking-tight">
        aiori<span className="ml-1 font-normal text-fg-3">traffic</span>
      </span>
    </NavLink>
  );
}

function ServiceStatus() {
  const { state } = useStudy();
  const [dot, text] =
    state.status === "ready"
      ? ["bg-accent", `Model service ready · ${state.data.health.device.toUpperCase()}`]
      : state.status === "error"
        ? ["bg-danger", "Model service unavailable"]
        : ["bg-fg-3 animate-pulse", "Connecting to model service"];
  return (
    <p className="flex items-center gap-2 text-xs text-fg-2" role="status">
      <span className={cn("size-1.5 rounded-full", dot)} aria-hidden />
      {text}
    </p>
  );
}

function Sidebar({ open, onNavigate }: { open: boolean; onNavigate: () => void }) {
  const [desktop, setDesktop] = useState(() => window.matchMedia("(min-width: 1024px)").matches);
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const on = () => setDesktop(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return (
    <aside
      id="sidebar"
      inert={!desktop && !open}
      aria-label="Main navigation"
      className={cn(
        "fixed inset-y-0 left-0 z-40 flex w-[248px] flex-col border-r border-line bg-raised px-3 py-4 transition-transform duration-300 ease-[var(--ease-out-expo)]",
        "lg:sticky lg:top-0 lg:h-dvh lg:w-[232px] lg:translate-x-0",
        open ? "translate-x-0 shadow-[0_16px_48px_rgba(0,0,0,0.55)]" : "-translate-x-full",
      )}
    >
      <Wordmark />
      {(["Product", "Research"] as const).map((group) => (
        <nav key={group} aria-label={group} className="mt-6">
          <p className="mb-1.5 px-2.5 text-xs font-medium text-fg-3">{group}</p>
          <ul className="flex flex-col gap-0.5">
            {ROUTES.filter((r) => r.group === group).map(({ to, title, icon: Icon }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "flex h-8 items-center gap-2.5 rounded-control px-2.5 text-sm transition-colors",
                      isActive ? "bg-surface-2 font-medium text-fg [&>svg]:text-accent" : "text-fg-2 hover:bg-surface-2/60 hover:text-fg",
                    )
                  }
                >
                  <Icon className="size-4 text-fg-3" strokeWidth={1.6} aria-hidden />
                  {title}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      ))}
      <div className="mt-auto flex flex-col gap-2 border-t border-line px-2.5 pt-4">
        <ServiceStatus />
        <p className="text-xs text-fg-3">XTraffic 2023, January–September. Historical data; nothing is live.</p>
      </div>
    </aside>
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
    <footer className="border-t border-line bg-raised/60">
      <div className="mx-auto grid w-full max-w-[1240px] gap-8 px-4 py-10 sm:px-6 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_minmax(0,1fr)] lg:px-10">
        <div className="flex flex-col gap-2">
          <p className="text-sm font-semibold">AIORI Traffic Intelligence</p>
          <p className="text-sm text-fg-2">Privacy-aware traffic forecasting research</p>
          <p className="text-xs text-fg-3">Historical research data · 2023</p>
        </div>
        <nav aria-label="Footer" className="flex flex-col gap-2 text-sm">
          <NavLink to="/performance" className="w-fit text-fg-2 hover:text-fg">Model Performance</NavLink>
          <NavLink to="/methodology" className="w-fit text-fg-2 hover:text-fg">Methodology</NavLink>
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer" className="inline-flex w-fit items-center gap-1.5 text-fg-2 hover:text-fg">
            <GitHubMark className="size-3.5" />
            GitHub repository
          </a>
        </nav>
        <p className="text-xs leading-relaxed text-fg-3 md:text-right">Forecasts are research demonstrations, not live traffic advisories.</p>
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

  return (
    <div className="flex min-h-dvh">
      <a href="#main" className="sr-only z-50 rounded-control bg-accent px-3 py-2 text-sm text-accent-ink focus:not-sr-only focus:fixed focus:left-3 focus:top-3">
        Skip to content
      </a>
      <Sidebar open={open} onNavigate={() => setOpen(false)} />
      <div
        aria-hidden
        onClick={() => setOpen(false)}
        className={cn("fixed inset-0 z-30 bg-black/60 transition-opacity lg:hidden", open ? "opacity-100" : "pointer-events-none opacity-0")}
      />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-3 border-b border-line bg-raised/90 px-4 backdrop-blur-md sm:px-6">
          <button
            ref={menuRef}
            type="button"
            className="-ml-1.5 grid size-9 cursor-pointer place-items-center rounded-control text-fg-2 hover:bg-surface-2 hover:text-fg lg:hidden"
            aria-expanded={open}
            aria-controls="sidebar"
            aria-label={open ? "Close navigation" : "Open navigation"}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="size-[18px]" aria-hidden /> : <Menu className="size-[18px]" aria-hidden />}
          </button>
          <nav aria-label="Breadcrumb" className="min-w-0 flex-1 truncate text-sm">
            <span className="hidden text-fg-3 md:inline">Traffic Intelligence</span>
            <span className="mx-2 hidden text-fg-3 md:inline" aria-hidden>
              /
            </span>
            <span className="font-medium text-fg" aria-current="page">
              {title}
            </span>
          </nav>
          <a
            href={REPO_URL}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub repository (opens in a new tab)"
            className="inline-flex h-7 items-center gap-1.5 rounded-full border border-line-strong px-2.5 text-xs text-fg-2 transition-colors hover:border-[#454545] hover:text-fg"
          >
            <GitHubMark className="size-3.5" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
          <span
            className="inline-flex h-7 items-center gap-2 rounded-full border border-line-strong px-2.5 text-xs text-fg-2"
            title="Examples and metrics come from historical data (January–September 2023)"
          >
            <span className="size-1.5 rounded-full bg-observed" aria-hidden />
            Historical data<span className="hidden md:inline">&nbsp;· Jan–Sep 2023</span>
          </span>
          {pathname !== "/forecast" && pathname !== "/" && (
            <ButtonLink to="/forecast" variant="primary" className="hidden sm:inline-flex">
              Run a forecast
            </ButtonLink>
          )}
        </header>
        <main id="main" className="mx-auto w-full max-w-[1240px] flex-1 px-4 pb-20 pt-8 sm:px-6 lg:px-10 lg:pt-12">
          {children}
        </main>
        <Footer />
      </div>
    </div>
  );
}
