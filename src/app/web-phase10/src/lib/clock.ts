/**
 * Local clock times as "YYYY-MM-DDTHH:MM" strings, matching the API. Arithmetic runs in UTC so the
 * viewer's own time zone and daylight-saving changes never shift a reading.
 */
export const STEP_MIN = 5;
export const HORIZON = 12;
export const STUDY_END = "2023-09-30T23:55";
const RE = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?$/;

/** Minutes since the Unix epoch, or null if the text is not a valid clock time. */
export function toMinutes(text: string): number | null {
  const m = RE.exec(text.trim());
  if (!m) return null;
  const [y, mo, d, h, mi, s] = m.slice(1).map((x) => (x === undefined ? 0 : Number(x)));
  const ms = Date.UTC(y, mo - 1, d, h, mi);
  const back = new Date(ms);
  if (back.getUTCFullYear() !== y || back.getUTCMonth() !== mo - 1 || back.getUTCDate() !== d || h > 23 || mi > 59 || s !== 0) return null;
  return ms / 60000;
}

export function fromMinutes(min: number): string {
  return new Date(min * 60000).toISOString().slice(0, 16);
}

export const addSteps = (t: string, steps: number) => fromMinutes((toMinutes(t) ?? 0) + steps * STEP_MIN);

export const hhmm = (t: string) => t.slice(11, 16);

const dayFmt = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
export const niceDate = (t: string) => dayFmt.format(new Date((toMinutes(t) ?? 0) * 60000));

const CA = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/Los_Angeles", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});

/** The next 5-minute boundary on California clock time (the sensors' local time, which the model's time features use). */
export function nextInterval(now: Date = new Date()): string {
  const v = Object.fromEntries(CA.formatToParts(now).map((x) => [x.type, x.value]));
  const min = toMinutes(`${v.year}-${v.month}-${v.day}T${v.hour}:${v.minute}`) ?? 0;
  return fromMinutes(Math.ceil(min / STEP_MIN) * STEP_MIN);
}

/** True when any forecast step falls after the last held-out interval of the study. */
export const beyondStudy = (start: string) => addSteps(start, HORIZON - 1) > STUDY_END;
