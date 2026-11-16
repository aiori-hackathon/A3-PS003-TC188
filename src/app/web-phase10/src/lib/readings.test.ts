import { describe, expect, it } from "vitest";
import { parseReading, parseReadingsCsv, readingsTemplate } from "./readings";
import { addSteps, beyondStudy, niceDate, toMinutes } from "./clock";

const rows = (start: string, values: (string | number)[]) =>
  values.map((v, i) => `${addSteps(start, i).replace("T", " ")},${v}`).join("\n");

describe("clock", () => {
  it("parses only real clock times", () => {
    expect(toMinutes("2024-02-29 23:55")).not.toBeNull();
    expect(toMinutes("2023-02-29 10:00")).toBeNull();
    expect(toMinutes("2024-03-05T24:00")).toBeNull();
    expect(toMinutes("yesterday")).toBeNull();
  });
  it("steps across days and years without time-zone drift", () => {
    expect(addSteps("2023-12-31T23:55", 1)).toBe("2024-01-01T00:00");
    expect(addSteps("2024-03-10T01:55", 1)).toBe("2024-03-10T02:00"); // US daylight-saving night
    expect(niceDate("2024-03-05T08:00")).toBe("Tue, 5 Mar 2024");
  });
  it("flags forecasts that leave the study period", () => {
    expect(beyondStudy("2023-09-30T23:00")).toBe(false); // last step 23:55, still inside
    expect(beyondStudy("2023-09-30T23:05")).toBe(true);
    expect(beyondStudy("2024-01-01T00:00")).toBe(true);
  });
});

describe("parseReading", () => {
  it("accepts numbers and blanks, rejects the rest", () => {
    expect(parseReading(" 412.5 ")).toEqual({ value: 412.5 });
    expect(parseReading("")).toEqual({ value: null });
    expect(parseReading("NA")).toEqual({ value: null });
    expect(parseReading("-3")).toHaveProperty("error");
    expect(parseReading("1e3")).toHaveProperty("error");
    expect(parseReading("5001")).toHaveProperty("error");
  });
});

describe("parseReadingsCsv", () => {
  const vals = [410, 422, 418, "", 430, 441, 437, 452, 460, 455, 470, 468];

  it("reads a CSV with a header and forecasts from the next interval", () => {
    const r = parseReadingsCsv(`timestamp,flow\n${rows("2024-03-05T07:00", vals)}\n`);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.observations).toEqual([410, 422, 418, null, 430, 441, 437, 452, 460, 455, 470, 468]);
    expect(r.start).toBe("2024-03-05T08:00");
    expect(r.missing).toBe(1);
  });

  it("accepts rows without a header, semicolons, ISO times, a BOM and CRLF", () => {
    const body = rows("2025-01-01T00:00", vals).replaceAll(",", ";").replaceAll(" ", "T").replaceAll("\n", "\r\n");
    const r = parseReadingsCsv(`﻿${body}`);
    expect(r.ok && r.start).toBe("2025-01-01T01:00");
  });

  it("finds named columns in any order and keeps the latest 12 rows", () => {
    const extra = Array.from({ length: 15 }, (_, i) => `${100 + i},${addSteps("2024-06-01T10:00", i).replace("T", " ")},x`).join("\n");
    const r = parseReadingsCsv(`vehicles,time,note\n${extra}`);
    expect(r.ok).toBe(true);
    if (!r.ok) return;
    expect(r.observations[0]).toBe(103);
    expect(r.start).toBe("2024-06-01T11:15");
    expect(r.notes[0]).toMatch(/latest 12 of 15/);
  });

  it("rejects too few rows, gaps, repeats, bad values and off-grid times", () => {
    expect(parseReadingsCsv(rows("2024-03-05T07:00", vals.slice(0, 11)))).toMatchObject({ ok: false });
    const gap = rows("2024-03-05T07:00", vals).replace("07:30", "07:35");
    expect(parseReadingsCsv(gap)).toMatchObject({ ok: false, errors: [expect.stringMatching(/consecutive/)] });
    const repeat = rows("2024-03-05T07:00", vals).replace("07:30", "07:25");
    expect(parseReadingsCsv(repeat)).toMatchObject({ ok: false, errors: [expect.stringMatching(/repeated/)] });
    const bad = rows("2024-03-05T07:00", [...vals.slice(0, 11), -4]);
    expect(parseReadingsCsv(bad)).toMatchObject({ ok: false, errors: [expect.stringMatching(/Line 12/)] });
    const off = rows("2024-03-05T07:00", vals).replace("07:00", "07:02");
    expect(parseReadingsCsv(off).ok).toBe(false);
    expect(parseReadingsCsv(rows("2024-03-05T07:00", Array(12).fill("")))).toMatchObject({ ok: false, errors: [expect.stringMatching(/missing/)] });
    expect(parseReadingsCsv(rows("2022-12-31T23:00", vals)).ok).toBe(false);
    expect(parseReadingsCsv("date,count\n").ok).toBe(false);
    expect(parseReadingsCsv("").ok).toBe(false);
  });

  it("round-trips its own template once filled", () => {
    const tpl = readingsTemplate("2024-03-05T08:00");
    expect(tpl.split("\n")[1]).toBe("2024-03-05 07:00,");
    const filled = tpl.trim().split("\n").map((l, i) => (i ? `${l}${100 + i}` : l)).join("\n");
    const r = parseReadingsCsv(filled);
    expect(r.ok && r.start).toBe("2024-03-05T08:00");
  });
});
