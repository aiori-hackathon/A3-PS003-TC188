/* Shared primitives in the shadcn/ui style (variants via props, cn() merging), styled from DESIGN.md tokens. */
import { AlertCircle, Loader2, RotateCw } from "lucide-react";
import { useId, type ButtonHTMLAttributes, type ComponentProps, type ReactNode } from "react";
import { Link } from "react-router";
import { cn } from "../../lib/cn";

/* ---------- buttons ---------- */
type Variant = "primary" | "secondary" | "ghost";
const BTN =
  "inline-flex h-9 shrink-0 cursor-pointer items-center justify-center gap-2 whitespace-nowrap rounded-control px-3.5 text-sm font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-55";
const VARIANT: Record<Variant, string> = {
  primary: "bg-accent text-accent-ink hover:bg-accent-hover",
  secondary: "border border-line-strong bg-surface-2 text-fg hover:border-[#454545] hover:bg-surface-3",
  ghost: "text-fg-2 hover:bg-surface-2 hover:text-fg",
};

export function Button({ variant = "secondary", loading, className, children, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  return (
    <button className={cn(BTN, VARIANT[variant], className)} disabled={loading || rest.disabled} aria-busy={loading || undefined} {...rest}>
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
}

export function ButtonLink({ variant = "secondary", className, ...rest }: ComponentProps<typeof Link> & { variant?: Variant }) {
  return <Link className={cn(BTN, VARIANT[variant], className)} {...rest} />;
}

/* ---------- surfaces ---------- */
export function Card({ className, ...rest }: ComponentProps<"section">) {
  return <section className={cn("min-w-0 rounded-card border border-line bg-surface", className)} {...rest} />;
}

export function CardHeader({ title, meta, action, id }: { title: ReactNode; meta?: ReactNode; action?: ReactNode; id?: string }) {
  return (
    <header className="flex min-h-13 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-line px-5 py-3">
      <div className="min-w-0">
        <h3 id={id} className="text-base font-semibold tracking-tight">
          {title}
        </h3>
        {meta && <p className="text-xs text-fg-3">{meta}</p>}
      </div>
      {action}
    </header>
  );
}

export const CardBody = ({ className, ...rest }: ComponentProps<"div">) => <div className={cn("p-5", className)} {...rest} />;

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="mb-8 max-w-[70ch]">
      <h1 tabIndex={-1} className="text-2xl font-semibold outline-none sm:text-[32px] sm:leading-tight">
        {title}
      </h1>
      {children && <p className="mt-3 text-md leading-relaxed text-fg-2">{children}</p>}
    </header>
  );
}

export function SectionHeader({ title, meta }: { title: string; meta?: string }) {
  return (
    <div className="mb-4 mt-12 flex flex-wrap items-baseline justify-between gap-2">
      <h2 className="text-lg font-semibold">{title}</h2>
      {meta && <p className="text-sm text-fg-3">{meta}</p>}
    </div>
  );
}

/* ---------- small labels ---------- */
type Tone = "neutral" | "accent" | "observed" | "danger";
const TONE: Record<Tone, string> = {
  neutral: "border-line-strong bg-surface-2 text-fg-2",
  accent: "border-accent/35 bg-accent/12 text-accent",
  observed: "border-observed/35 bg-observed/10 text-observed",
  danger: "border-danger/40 bg-danger/10 text-[#f08680]",
};
export function Tag({ tone = "neutral", className, ...rest }: ComponentProps<"span"> & { tone?: Tone }) {
  return <span className={cn("inline-flex h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full border px-2 text-xs font-medium", TONE[tone], className)} {...rest} />;
}

export const Swatch = ({ color, dash }: { color: string; dash?: boolean }) => (
  <span aria-hidden className={cn("inline-block w-4 shrink-0 border-t-2", dash && "border-dashed")} style={{ borderColor: color }} />
);

export function Legend({ items }: { items: { label: string; color: string; dash?: boolean }[] }) {
  return (
    <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-xs text-fg-2" aria-label="Legend">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-2">
          <Swatch color={i.color} dash={i.dash} />
          {i.label}
        </li>
      ))}
    </ul>
  );
}

/* ---------- stats ---------- */
export function Stat({ label, value, unit, note, accent }: { label: string; value: ReactNode; unit?: string; note?: ReactNode; accent?: boolean }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 bg-surface px-5 py-4">
      <span className="text-sm text-fg-2">{label}</span>
      <span className={cn("tnum text-2xl font-semibold tracking-tight", accent && "text-accent")}>
        {value}
        {unit && <span className="ml-1.5 text-sm font-normal text-fg-3">{unit}</span>}
      </span>
      {note && <span className="text-xs text-fg-3">{note}</span>}
    </div>
  );
}

/** Stats sharing one bordered strip with 1px dividers, instead of a row of identical cards. */
export const StatStrip = ({ children, className }: { children: ReactNode; className?: string }) => (
  <div className={cn("grid gap-px overflow-hidden rounded-card border border-line bg-line", className)}>{children}</div>
);

/* ---------- form controls ---------- */
const CONTROL =
  "h-9 w-full min-w-0 rounded-control border border-line-strong bg-surface-2 px-3 text-sm text-fg transition-[border-color,box-shadow] outline-none focus-visible:border-accent focus-visible:ring-3 focus-visible:ring-accent/20 focus-visible:outline-none aria-invalid:border-danger";

export function Field({ label, hint, error, children, className }: { label: string; hint?: ReactNode; error?: string; className?: string; children: (id: string, describedBy?: string) => ReactNode }) {
  const id = useId();
  const desc = hint || error ? `${id}-d` : undefined;
  return (
    <div className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <label htmlFor={id} className="text-sm font-medium text-fg-2">
        {label}
      </label>
      {children(id, desc)}
      {(hint || error) && (
        <p id={desc} className={cn("text-xs", error ? "text-[#f08680]" : "text-fg-3")}>
          {error ?? hint}
        </p>
      )}
    </div>
  );
}

export const Input = ({ className, ...rest }: ComponentProps<"input">) => <input className={cn(CONTROL, "tnum", className)} {...rest} />;

export const Select = ({ className, ...rest }: ComponentProps<"select">) => (
  <select
    className={cn(
      CONTROL,
      "cursor-pointer appearance-none bg-[url('data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20width=%2212%22%20height=%2212%22%20fill=%22none%22%20stroke=%22%238a8a8a%22%20stroke-width=%221.5%22%3E%3Cpath%20d=%22m3%204.5%203%203%203-3%22/%3E%3C/svg%3E')] bg-[position:right_10px_center] bg-no-repeat pr-8",
      className,
    )}
    {...rest}
  />
);

export function Switch({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 py-1">
      <span className="flex flex-col">
        <span className="text-sm text-fg">{label}</span>
        {hint && <span className="text-xs text-fg-3">{hint}</span>}
      </span>
      <span className="relative inline-flex">
        <input type="checkbox" role="switch" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="h-5 w-9 rounded-full border border-line-strong bg-surface-3 transition-colors peer-checked:border-accent peer-checked:bg-accent peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent" />
        <span className="pointer-events-none absolute left-[3px] top-[3px] size-3.5 rounded-full bg-fg-2 transition-transform duration-200 ease-out peer-checked:translate-x-4 peer-checked:bg-accent-ink" />
      </span>
    </label>
  );
}

/** Radio group drawn as a segmented control; arrow keys move between options. */
export function Segmented<T extends string>({ value, onChange, options, label, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode }[]; label: string; className?: string }) {
  const name = useId();
  return (
    <div role="radiogroup" aria-label={label} className={cn("grid auto-cols-fr grid-flow-col gap-1 rounded-[8px] border border-line bg-bg p-1", className)}>
      {options.map((o) => (
        <label
          key={o.value}
          className={cn(
            "flex h-8 cursor-pointer items-center justify-center gap-2 rounded-control px-3 text-sm font-medium transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent",
            value === o.value ? "bg-surface-3 text-fg" : "text-fg-3 hover:text-fg-2",
          )}
        >
          <input type="radio" className="sr-only" name={name} value={o.value} checked={value === o.value} onChange={() => onChange(o.value)} />
          {o.label}
        </label>
      ))}
    </div>
  );
}

/* ---------- tables ---------- */
export const TableWrap = ({ children, label }: { children: ReactNode; label?: string }) => (
  <div className="overflow-x-auto" role={label ? "region" : undefined} aria-label={label} tabIndex={label ? 0 : undefined}>
    <table className="w-full min-w-[560px] border-collapse text-sm [&_td]:border-t [&_td]:border-line [&_td]:px-5 [&_td]:py-2.5 [&_th]:bg-surface-2/60 [&_th]:px-5 [&_th]:py-2.5 [&_th]:text-left [&_th]:text-xs [&_th]:font-medium [&_th]:text-fg-3">
      {children}
    </table>
  </div>
);

export const Num = ({ className, ...rest }: ComponentProps<"td">) => <td className={cn("tnum text-right whitespace-nowrap", className)} {...rest} />;
export const NumTh = ({ className, ...rest }: ComponentProps<"th">) => <th className={cn("!text-right whitespace-nowrap", className)} {...rest} />;

/* ---------- states ---------- */
export const Skeleton = ({ className, label = "Loading" }: { className?: string; label?: string }) => (
  <div role="status" aria-label={label} className={cn("animate-pulse rounded-card bg-surface-2/70", className)} />
);

export function ErrorState({ title, message, onRetry, className }: { title: string; message: string; onRetry?: () => void; className?: string }) {
  return (
    <div role="alert" className={cn("flex items-start gap-3 rounded-card border border-danger/35 bg-danger/[0.06] p-4", className)}>
      <AlertCircle className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />
      <div className="min-w-0 flex-1">
        <p className="text-sm font-semibold text-fg">{title}</p>
        <p className="mt-0.5 text-sm text-fg-2 [overflow-wrap:anywhere]">{message}</p>
      </div>
      {onRetry && (
        <Button variant="ghost" className="h-8 px-2.5" onClick={onRetry}>
          <RotateCw className="size-3.5" aria-hidden />
          Retry
        </Button>
      )}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-14 text-center">
      <span className="mb-1 grid size-10 place-items-center rounded-card border border-line bg-surface-2 text-fg-3">{icon}</span>
      <p className="text-sm font-semibold">{title}</p>
      {children && <p className="max-w-[44ch] text-sm text-fg-3">{children}</p>}
    </div>
  );
}

export function Note({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn("text-xs leading-relaxed text-fg-3", className)}>{children}</p>;
}

/** A short ordered procedure: bold lead-in, then the sentence. */
export function Steps({ items }: { items: string[][] }) {
  return (
    <ol className="flex flex-col">
      {items.map(([b, t], i) => (
        <li key={b} className="flex gap-3.5 py-2">
          <span className="tnum grid size-6 shrink-0 place-items-center rounded-full border border-line-strong text-xs text-fg-2">{i + 1}</span>
          <p className="text-sm leading-relaxed text-fg-2">
            <b className="font-semibold text-fg">{b}</b> {t}
          </p>
        </li>
      ))}
    </ol>
  );
}
