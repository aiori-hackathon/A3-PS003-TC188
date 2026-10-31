import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { api } from "./api";
import type { Demo, DeployedModel, Health, Metrics, Sensor } from "./types";

export interface StudyData {
  health: Health;
  metrics: Metrics;
  sensors: Sensor[];
  demo: Demo;
  models: DeployedModel[];
}

type State = { status: "loading" } | { status: "error"; message: string } | { status: "ready"; data: StudyData };

const Ctx = createContext<{ state: State; retry: () => void } | null>(null);

/** Loads the five read-only endpoints once; every page reads verified numbers from here. */
export function StudyProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const ctrl = new AbortController();
    setState({ status: "loading" });
    Promise.all([api.health(ctrl.signal), api.metrics(ctrl.signal), api.sensors(ctrl.signal), api.demo(ctrl.signal), api.models(ctrl.signal)])
      .then(([health, metrics, sensors, demo, models]) => setState({ status: "ready", data: { health, metrics, sensors, demo, models } }))
      .catch((e: Error) => {
        if (e.name !== "AbortError") setState({ status: "error", message: e.message });
      });
    return () => ctrl.abort();
  }, [attempt]);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  return <Ctx.Provider value={{ state, retry }}>{children}</Ctx.Provider>;
}

export function useStudy() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useStudy needs <StudyProvider>");
  return v;
}

/** Runs one request at a time; a newer call cancels the older one so stale results never land. */
export function useRequest<A extends unknown[], T>(fn: (...args: [...A, AbortSignal]) => Promise<T>) {
  const [state, setState] = useState<{ loading: boolean; data: T | null; error: string | null }>({ loading: false, data: null, error: null });
  const ctrl = useRef<AbortController | null>(null);
  useEffect(() => () => ctrl.current?.abort(), []);
  const run = useCallback(
    async (...args: A) => {
      ctrl.current?.abort();
      const c = (ctrl.current = new AbortController());
      setState((s) => ({ ...s, loading: true, error: null }));
      try {
        const data = await fn(...args, c.signal);
        if (!c.signal.aborted) setState({ loading: false, data, error: null });
        return data;
      } catch (e) {
        if ((e as Error).name !== "AbortError" && !c.signal.aborted) setState({ loading: false, data: null, error: (e as Error).message });
        return null;
      }
    },
    [fn],
  );
  const reset = useCallback((error: string | null = null) => {
    ctrl.current?.abort(); // an in-flight request from before the reset must not land afterwards
    setState({ loading: false, data: null, error });
  }, []);
  return { ...state, run, reset };
}
