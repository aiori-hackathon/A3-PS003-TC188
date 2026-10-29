import { addSteps, fromMinutes, HORIZON, STEP_MIN, toMinutes } from "./clock";

/** Same limit as the API (MAX_FLOW): vehicles per 5-minute interval at one sensor. */
export const MAX_FLOW = 5000;
const MAX_BYTES = 1_000_000;
const MISSING = new Set(["", "na", "n/a", "nan", "null", "none", "-"]);

export type Reading = number | null;

/** One typed reading: a number in range, null for a deliberately missing value, or an error message. */
export function parseReading(raw: string): { value: Reading } | { error: string } {
  const s = raw.trim();
  if (MISSING.has(s.toLowerCase())) return { value: null };
  if (!/^[+]?(\d+(\.\d*)?|\.\d+)$/.test(s)) return { error: `"${s}" is not a number from 0 to ${MAX_FLOW}` };
  const v = Number(s);
  if (v > MAX_FLOW) return { error: `${s} is above ${MAX_FLOW}` };
  return { value: v };
}

export interface ParsedReadings {
  ok: true;
  /** Twelve readings, oldest first; null = missing (filled the way the model was trained). */
  observations: Reading[];
  /** Clock time of each reading. */
  timestamps: string[];
  /** First forecast interval: five minutes after the last reading. */
  start: string;
  rowsRead: number;
  missing: number;
  notes: string[];
}

export type ParseResult = ParsedReadings | { ok: false; errors: string[] };

const TIME_COL = /^(timestamp|time|datetime|date_time|ts|interval_start)$/i;
const FLOW_COL = /^(flow|value|vehicles|count|reading|observation|volume)$/i;
const show = (min: number) => fromMinutes(min).replace("T", " ");

/**
 * Parses a CSV of 5-minute readings for one sensor: a timestamp column ("2024-03-05 07:55" or "2024-03-05T07:55")
 * and a flow column. A header row is optional. The last 12 rows are used; they must be consecutive 5-minute intervals.
 */
export function parseReadingsCsv(input: string): ParseResult {
  if (input.length > MAX_BYTES) return { ok: false, errors: ["The file is larger than 1 MB. Keep only the most recent readings."] };
  const lines = input
    .replace(/^﻿/, "")
    .split(/\r?\n/)
    .map((l, i) => ({ text: l.trim(), line: i + 1 }))
    .filter((l) => l.text && !l.text.startsWith("#"));
  if (!lines.length) return { ok: false, errors: ["The file is empty."] };
  const delim = [",", ";", "\t"].find((d) => lines[0].text.includes(d)) ?? ",";
  const split = (t: string) => t.split(delim).map((c) => c.trim().replace(/^"(.*)"$/, "$1"));

  let tCol = 0;
  let fCol = 1;
  let rows = lines;
  const head = split(lines[0].text);
  if (toMinutes(head[0] ?? "") === null) {
    tCol = head.findIndex((h) => TIME_COL.test(h));
    fCol = head.findIndex((h) => FLOW_COL.test(h));
    if (tCol < 0 || fCol < 0)
      return {
        ok: false,
        errors: [`Line ${lines[0].line}: expected a header with "timestamp" and "flow" columns, or rows that start with a timestamp.`],
      };
    rows = lines.slice(1);
  }
  if (rows.length < HORIZON)
    return { ok: false, errors: [`The file has ${rows.length} reading${rows.length === 1 ? "" : "s"}; the model needs the latest ${HORIZON} (one hour).`] };

  const notes: string[] = [];
  if (rows.length > HORIZON) notes.push(`Used the latest ${HORIZON} of ${rows.length} rows.`);
  const last = rows.slice(-HORIZON);
  const errors: string[] = [];
  const minutes: number[] = [];
  const observations: Reading[] = [];
  for (const r of last) {
    const cells = split(r.text);
    const t = toMinutes(cells[tCol] ?? "");
    if (t === null) errors.push(`Line ${r.line}: "${cells[tCol] ?? ""}" is not a time like 2024-03-05 07:55.`);
    else if (t % STEP_MIN) errors.push(`Line ${r.line}: ${show(t)} is not on a 5-minute boundary.`);
    minutes.push(t ?? NaN);
    const v = parseReading(cells[fCol] ?? "");
    if ("error" in v) errors.push(`Line ${r.line}: ${v.error}. Leave the value blank if it is missing.`);
    observations.push("value" in v ? v.value : null);
  }
  if (!errors.length) {
    for (let i = 1; i < minutes.length; i++) {
      const gap = minutes[i] - minutes[i - 1];
      if (gap !== STEP_MIN) {
        errors.push(
          `Line ${last[i].line}: expected ${show(minutes[i - 1] + STEP_MIN)} next. Readings must be consecutive 5-minute intervals, oldest first` +
            (gap <= 0 ? " (found a repeated or earlier time)." : "."),
        );
        break;
      }
    }
  }
  if (!errors.length) {
    if (fromMinutes(minutes[0]) < "2023-01-01T00:00") errors.push("Readings must start on or after 1 January 2023, the start of the study data.");
    if (fromMinutes(minutes[HORIZON - 1]) >= "2099-12-31T23:00") errors.push("Readings must end before the year 2100.");
  }
  if (!errors.length && observations.every((v) => v === null)) errors.push("All 12 readings are missing; at least one is needed.");
  if (errors.length) return { ok: false, errors: errors.slice(0, 5).concat(errors.length > 5 ? [`…and ${errors.length - 5} more.`] : []) };

  const timestamps = minutes.map(fromMinutes);
  const missing = observations.filter((v) => v === null).length;
  if (missing) notes.push(`${missing} missing reading${missing === 1 ? " is" : "s are"} filled the way the model was trained.`);
  return { ok: true, observations, timestamps, start: addSteps(timestamps[HORIZON - 1], 1), rowsRead: rows.length, missing, notes };
}

/** A blank CSV for the hour before `start`, ready to fill in. */
export function readingsTemplate(start: string): string {
  const rows = Array.from({ length: HORIZON }, (_, i) => `${addSteps(start, i - HORIZON).replace("T", " ")},`);
  return ["timestamp,flow", ...rows].join("\n") + "\n";
}
