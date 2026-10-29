const nf = new Map<number, Intl.NumberFormat>();

/** Fixed-decimal number with grouping; "—" for missing values. */
export function num(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  let f = nf.get(digits);
  if (!f) nf.set(digits, (f = new Intl.NumberFormat("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })));
  return f.format(v);
}

export const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? "—" : `${num(v, digits)}%`);

/** Signed difference with a true minus sign. */
export const signed = (v: number, digits = 3) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${num(Math.abs(v), digits)}`;

export const int = (v: number) => Math.round(v).toLocaleString("en-US");
