import type { ArmKey, BaselineKey, ClientKey } from "./types";

export const ARMS: ArmKey[] = ["local", "centralised", "fedavg_iid", "fedavg_geo"];
export const BASELINES: BaselineKey[] = ["persistence", "seasonal_naive_daily", "historical_average"];
export const CLIENTS: ClientKey[] = ["client_0_la_ventura", "client_1_bay_area", "client_2_coastal_south_central", "client_3_inland_central"];

export const CLIENT_NAME: Record<ClientKey, string> = {
  client_0_la_ventura: "LA / Ventura",
  client_1_bay_area: "Bay Area",
  client_2_coastal_south_central: "Coastal / South Central",
  client_3_inland_central: "Inland / Central",
};

export const BASELINE_LABEL: Record<BaselineKey, string> = {
  persistence: "Persistence",
  seasonal_naive_daily: "Same time yesterday",
  historical_average: "Historical average",
};

/** Chart series styling from DESIGN.md: the final model is the only coloured series. */
export const SERIES: Record<ArmKey | BaselineKey | "observed" | "actual", { color: string; dash?: string; width: number }> = {
  local: { color: "#3bb27e", width: 2.4 },
  centralised: { color: "#bdbdbd", width: 1.5 },
  fedavg_iid: { color: "#8a8a8a", dash: "6 4", width: 1.5 },
  fedavg_geo: { color: "#8a8a8a", dash: "2 3", width: 1.5 },
  persistence: { color: "#616161", dash: "1 4", width: 1.6 },
  seasonal_naive_daily: { color: "#4f4f4f", width: 1.4 },
  historical_average: { color: "#444444", width: 1.4 },
  observed: { color: "#ededed", width: 1.6 },
  actual: { color: "#d6a24a", dash: "4 4", width: 1.5 },
};

export const REPO = "https://github.com/aiori-hackathon/a3-ps003-tc188";
