import { afterEach, describe, expect, it, vi } from "vitest";
import { compareRows, DEFAULT_FILTERS, disruptionsApi, duration, listParams, pct, replayHref, seriesRows, type EventCompare, type EventSeries } from "./disruptions";
import { parseReplayLink, ReplayApiError } from "./replay";

describe("disruption API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("sends only set filters, with bounded paging, and surfaces the server's message (e.g. a missing catalog)", async () => {
    const fetch = vi.fn(async () => ({ ok: false, status: 503, json: async () => ({ error: { message: "The disruption catalog has not been built yet." } }) }) as unknown as Response);
    vi.stubGlobal("fetch", fetch);
    await expect(disruptionsApi.list({ ...DEFAULT_FILTERS, sensor: "LA-05", from: "2024-03-01" })).rejects.toMatchObject({ status: 503, message: expect.stringContaining("not been built") });
    const url = String((fetch.mock.calls[0] as unknown[])[0]);
    expect(url).toContain("/api/replay/disruptions?");
    const q = new URLSearchParams(url.split("?")[1]);
    expect(Object.fromEntries(q)).toEqual({ sort: "-score", page: "1", page_size: "25", quality: "exclude", holiday: "include", sensor: "LA-05", from: "2024-03-01" });
  });

  it("propagates aborts and rejects unreadable bodies", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, json: async () => { throw new DOMException("aborted", "AbortError"); } }) as unknown as Response));
    await expect(disruptionsApi.meta()).rejects.toMatchObject({ name: "AbortError" });
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, status: 200, json: async () => { throw new SyntaxError("x"); } }) as unknown as Response));
    await expect(disruptionsApi.event("20240101T0000-LA-D-abcdef")).rejects.toBeInstanceOf(ReplayApiError);
  });
});

describe("listParams", () => {
  it("drops empty filters and keeps defaults explicit", () => {
    expect(listParams({ ...DEFAULT_FILTERS, min_sensors: " 3 ", region: "" })).toEqual({ sort: "-score", page: 1, page_size: 25, quality: "exclude", holiday: "include", min_sensors: "3" });
  });
});

describe("formatting", () => {
  it("formats durations and signed percentages", () => {
    expect(duration(15)).toBe("15 min");
    expect(duration(60)).toBe("1 h");
    expect(duration(95)).toBe("1 h 35 min");
    expect(duration(1440 * 2 + 180)).toBe("2 d 3 h");
    expect(pct(-0.624)).toBe("-62%");
    expect(pct(1.5)).toBe("+150%");
  });
});

describe("chart rows", () => {
  const s = {
    timestamps: ["2024-01-01T08:00", "2024-01-01T08:05", "2024-01-01T08:10"],
    observed: [100, null, 0],
    status: ["ok", "missing", "ok"],
    expected: [120, 120, null],
    threshold_low: [-10, 60, null],
    threshold_high: [200, 200, null],
    anomalous: [0, 0, -1],
  } as unknown as EventSeries;

  it("keeps missing readings null (never zero), marks them, keeps real zeros, and clamps the band at zero", () => {
    const r = seriesRows(s);
    expect(r[1]).toMatchObject({ observed: null, missing: 0, anomaly: null });
    expect(r[2]).toMatchObject({ observed: 0, missing: null, anomaly: 0, band: null, expected: null });
    expect(r[0].band).toEqual([0, 200]);
  });

  it("builds one column per chosen sensor", () => {
    const c = { timestamps: ["a", "b"], sensors: [{ sensor: "LA-01", ratio: [1, null], available: 1 }, { sensor: "LA-02", ratio: [0.5, 2], available: 1 }] } as unknown as EventCompare;
    expect(compareRows(c, ["LA-02"])).toEqual([{ t: "a", "LA-02": 0.5 }, { t: "b", "LA-02": 2 }]);
  });
});

describe("replay deep links", () => {
  const dates = ["2024-03-09", "2024-03-10"];
  const ids = ["LA-01", "BA-07"];

  it("opens the event's date, time and sensor", () => {
    const href = replayHref({ id: "20240310T0805-LA-D-abc123", replay: { date: "2024-03-10", time: "2024-03-10T08:10", sensor: "LA-01" } }, "BA-07");
    expect(href).toBe("/replay?date=2024-03-10&time=2024-03-10T08%3A10&sensor=BA-07&event=20240310T0805-LA-D-abc123");
    const link = parseReplayLink(new URLSearchParams(href.split("?")[1]), dates, ids);
    expect(link).toEqual({ date: "2024-03-10", slot: 98, sensor: "BA-07", event: "20240310T0805-LA-D-abc123", problems: [] });
  });

  it("ignores and reports invalid parameters instead of guessing", () => {
    expect(parseReplayLink(new URLSearchParams("time=2024-03-10T08:07&sensor=XX-01&event=../x"), dates, ids)).toEqual({ problems: ["time", "sensor", "event"] });
    expect(parseReplayLink(new URLSearchParams("time=2025-01-01T08:00"), dates, ids)).toEqual({ problems: ["date"] });
    expect(parseReplayLink(new URLSearchParams("date=2024-03-09&time=2024-03-10T08:00"), dates, ids).problems).toEqual(["time"]);
    expect(parseReplayLink(new URLSearchParams(""), dates, ids)).toEqual({ problems: [] });
  });
});
