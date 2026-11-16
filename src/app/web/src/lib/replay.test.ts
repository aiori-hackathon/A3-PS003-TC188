import { afterEach, describe, expect, it, vi } from "vitest";
import { adjacentDate, forecastRows, Lru, nextAvailable, NO_DATA, REL_HIGH, REL_LOW, REL_MID, relColor, replayApi, ReplayApiError, slotTime, speedDelay, timeSlot, type Forecast } from "./replay";

describe("replay API client", () => {
  afterEach(() => vi.unstubAllGlobals());
  const respond = (ok: boolean, status: number, json: () => Promise<unknown>) =>
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok, status, json }) as unknown as Response));

  it("propagates an abort that happens while the body is read (never an empty success)", async () => {
    respond(true, 200, async () => {
      throw new DOMException("aborted", "AbortError");
    });
    await expect(replayApi.frame("2024-01-01T00:00", "observed", "geo", 12)).rejects.toMatchObject({ name: "AbortError" });
  });

  it("rejects an unreadable 200 response and surfaces the server's error message", async () => {
    respond(true, 200, async () => {
      throw new SyntaxError("bad json");
    });
    await expect(replayApi.meta()).rejects.toBeInstanceOf(ReplayApiError);
    respond(false, 422, async () => ({ error: { message: "No valid reading" } }));
    await expect(replayApi.forecast("LA-01", "2024-01-01T00:00", "geo")).rejects.toMatchObject({ message: "No valid reading", status: 422 });
  });
});

describe("replay time helpers", () => {
  it("maps slots to wall-clock strings and back", () => {
    expect(slotTime("2024-03-10", 0)).toBe("2024-03-10T00:00");
    expect(slotTime("2024-03-10", 24)).toBe("2024-03-10T02:00");
    expect(slotTime("2024-03-10", 287)).toBe("2024-03-10T23:55");
    for (const s of [0, 1, 96, 287]) expect(timeSlot(slotTime("2024-08-14", s))).toBe(s);
  });

  it("skips slots without valid readings in both directions", () => {
    const c = [3, 0, 0, 5, 0, 1];
    expect(nextAvailable(c, 0, 1)).toBe(3);
    expect(nextAvailable(c, 3, 1)).toBe(5);
    expect(nextAvailable(c, 5, 1)).toBeNull();
    expect(nextAvailable(c, 3, -1)).toBe(0);
    expect(nextAvailable(c, 0, -1)).toBeNull();
  });

  it("moves between dataset dates and stops at the ends", () => {
    const d = ["2022-01-01", "2022-01-02", "2022-01-03"];
    expect(adjacentDate(d, "2022-01-02", 1)).toBe("2022-01-03");
    expect(adjacentDate(d, "2022-01-03", 1)).toBeNull();
    expect(adjacentDate(d, "2022-01-01", -1)).toBeNull();
    expect(adjacentDate(d, "2021-12-31", 1)).toBeNull();
  });

  it("converts speeds to frame delays", () => {
    expect(speedDelay(1)).toBe(1000);
    expect(speedDelay(8)).toBe(125);
  });
});

describe("relative-flow colour", () => {
  it("uses the scale ends, the typical midpoint, and grey for missing values", () => {
    expect(relColor(1)).toBe(REL_MID);
    expect(relColor(0.5)).toBe(REL_LOW);
    expect(relColor(0.1)).toBe(REL_LOW);
    expect(relColor(2)).toBe(REL_HIGH);
    expect(relColor(null)).toBe(NO_DATA);
    expect(relColor(0)).toBe(REL_LOW);                                  // a real zero reading is low, not missing
    expect(relColor(-0.2)).toBe(REL_LOW);
    expect(relColor(Number.NaN)).toBe(NO_DATA);
  });
});

describe("bounded cache", () => {
  it("evicts the least recently used entry", () => {
    const c = new Lru<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    c.get("a");
    c.set("c", 3);
    expect(c.get("b")).toBeUndefined();
    expect(c.get("a")).toBe(1);
    expect(c.size).toBe(2);
  });
});

describe("forecast chart rows", () => {
  const ts = (h: number, m: number) => `2024-12-31T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  const f = {
    issued_at: ts(23, 25),
    input: {
      timestamps: Array.from({ length: 12 }, (_, i) => ts(22 + Math.floor((30 + i * 5) / 60), (30 + i * 5) % 60)),
      model_input: Array(12).fill(50),
      observed: [...Array(11).fill(50), null],
      imputed: [...Array(11).fill(false), true],
    },
    forecast: { timestamps: [ts(23, 30), ts(23, 35), ts(23, 40), ts(23, 45), ts(23, 50), ts(23, 55), null, null, null, null, null, null], values: Array(12).fill(60) },
    actual: { label: "", values: [55, null, 57, 58, 59, 60, null, null, null, null, null, null], status: [] },
  } as unknown as Forecast;

  it("keeps inputs and targets on separate rows in time order, marks filled inputs and drops steps beyond the dataset", () => {
    const rows = forecastRows(f);
    expect(rows).toHaveLength(18);
    expect(rows[11]).toMatchObject({ observed: null, filled: 50 });
    expect(rows[10]).toMatchObject({ observed: 50, filled: null });
    expect(rows[12]).toMatchObject({ t: ts(23, 30), forecast: 60, actual: 55 });
    expect(rows[13].actual).toBeNull();
    expect(rows.every((r, i) => i === 0 || r.t > rows[i - 1].t)).toBe(true);
    expect(rows.slice(0, 12).every((r) => r.forecast === undefined && r.actual === undefined)).toBe(true);
  });
});
