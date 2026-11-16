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

/** Public name for a frozen model id, e.g. "aiori2-mlp-fedavgm-A-geo-s20261027" -> "AIORI MLP + FedAvgM · geographic · seed 20261027". */
export function modelName(id: string): string {
  const m = /^aiori2-mlp-fedavgm-A-(geo|iid)-s(\d+)$/.exec(id);
  return m ? `AIORI MLP + FedAvgM · ${m[1] === "geo" ? "geographic" : "IID"} · seed ${m[2]}` : id;
}

/** Catalog provenance without internal development labels. */
export const publicSource = (s: string) => s.replace(/AIORI 2\.0 Phase \d+ preprocessing/g, "AIORI preprocessing");

export const int = (v: number) => Math.round(v).toLocaleString("en-US");
