import { describe, expect, it } from "vitest";
import { modelName, publicSource } from "./format";

describe("public labels", () => {
  it("names frozen models without internal ids", () => {
    expect(modelName("aiori2-mlp-fedavgm-A-geo-s20261027")).toBe("AIORI MLP + FedAvgM · geographic · seed 20261027");
    expect(modelName("aiori2-mlp-fedavgm-A-iid-s20261027")).toBe("AIORI MLP + FedAvgM · IID · seed 20261027");
    expect(modelName("something-else")).toBe("something-else");
  });
  it("drops development-phase labels from the catalog source", () => {
    expect(publicSource("XTraffic historical observations, AIORI 2.0 Phase 2 preprocessing (conservative mask)")).toBe(
      "XTraffic historical observations, AIORI preprocessing (conservative mask)",
    );
  });
});
