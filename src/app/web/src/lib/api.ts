import type { Demo, DeployedModel, Health, Metrics, PredictRequest, Prediction, Sensor } from "./types";

/** Empty in development (Vite proxies /api) and when the Python server hosts this build; set VITE_API_BASE for a separate API host. */
const BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/api${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("The model service is not reachable. Start it with .venv\\Scripts\\python.exe src\\app\\backend\\server.py, then retry.", 0);
  }
  const data = (await res.json().catch(() => ({}))) as { error?: { message?: string } };
  if (!res.ok)
    throw new ApiError(data.error?.message ?? `The model service is not available at this address (HTTP ${res.status}).`, res.status);
  return data as T;
}

export const api = {
  health: (s?: AbortSignal) => request<Health>("/health", undefined, s),
  metrics: (s?: AbortSignal) => request<Metrics>("/metrics", undefined, s),
  sensors: (s?: AbortSignal) => request<Sensor[]>("/sensors", undefined, s),
  demo: (s?: AbortSignal) => request<Demo>("/demo", undefined, s),
  models: (s?: AbortSignal) => request<DeployedModel[]>("/models", undefined, s),
  predict: (body: PredictRequest, s?: AbortSignal) => request<Prediction>("/predict", body, s),
};
