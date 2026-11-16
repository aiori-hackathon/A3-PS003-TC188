import { describe, expect, it } from "vitest";
import { nextInterval } from "./clock";

describe("nextInterval", () => {
  it("uses California clock time, not the viewer's", () => {
    // 2024-01-15 20:02 UTC = 12:02 PST
    expect(nextInterval(new Date(Date.UTC(2024, 0, 15, 20, 2)))).toBe("2024-01-15T12:05");
  });
  it("follows daylight saving and keeps exact boundaries", () => {
    // 2024-07-01 00:10 UTC = 2024-06-30 17:10 PDT
    expect(nextInterval(new Date(Date.UTC(2024, 6, 1, 0, 10)))).toBe("2024-06-30T17:10");
  });
});
