import { describe, expect, it } from "vitest";
import { forecastRows } from "./series";
import type { Prediction } from "./types";

const ts = (h: number, m: number) => `2024-03-05T${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
const pred = (actual: boolean, last: number | null): Prediction => ({
  sensor: { id: "BA-03", client_name: "Bay Area", type_group: "mainline" },
  mode: actual ? "historical" : "custom",
  data_label: "",
  period: null,
  units: "vehicles per 5-minute interval",
  forecast_start: ts(8, 0),
  history: { timestamps: Array.from({ length: 12 }, (_, i) => ts(7, i * 5)), values: [...Array(11).fill(100), last] },
  forecast_timestamps: Array.from({ length: 12 }, (_, i) => ts(8, i * 5)),
  forecasts: { final: { label: "", values: Array(12).fill(120), negative_clipping: true }, centralised: { label: "", values: Array(12).fill(130), negative_clipping: false } },
  actual: actual ? { label: "", values: Array(12).fill(125) } : null,
  beyond_study_period: !actual,
  validated: actual,
  accuracy_note: "",
  inference_ms: 1,
  device: "cpu",
});

describe("forecastRows", () => {
  it("joins forecasts to the last observed reading and keeps truth out of the history", () => {
    const rows = forecastRows(pred(true, 110), { models: ["final", "centralised"], showActual: true });
    expect(rows).toHaveLength(24);
    expect(rows[11]).toMatchObject({ observed: 110, final: 110, centralised: 110, actual: null });
    expect(rows[12]).toMatchObject({ observed: null, final: 120, centralised: 130, actual: 125 });
  });
  it("omits hidden series and never bridges from a missing reading", () => {
    const rows = forecastRows(pred(false, null), { models: ["final"], showActual: true });
    expect(rows[11].final).toBeNull();
    expect(rows[12]).toMatchObject({ final: 120, centralised: null, actual: null });
  });
});
