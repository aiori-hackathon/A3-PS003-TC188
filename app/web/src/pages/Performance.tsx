import { BarCompare, HorizonChart, type BarDatum } from "../components/charts";
import { ForecastingPerformance, ResultsTable, StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, Legend, Note, Num, NumTh, PageHeader, TableWrap } from "../components/ui";
import { num, pct } from "../lib/format";
import { ARMS, BASELINE_LABEL, BASELINES, CLIENT_NAME, CLIENTS, SERIES } from "../lib/study";
import type { Arm, ArmKey, BaselineKey, Metrics } from "../lib/types";

const kind = (k: ArmKey | BaselineKey): BarDatum["kind"] => (k === "local" ? "final" : k in BASELINE_LABEL ? "ref" : "arm");

function bars(m: Metrics, pick: (k: ArmKey | BaselineKey) => number): BarDatum[] {
  return [...ARMS, ...BASELINES].map((k) => ({ label: k in BASELINE_LABEL ? BASELINE_LABEL[k as BaselineKey] : m.arms[k as ArmKey].label, value: pick(k), kind: kind(k) }));
}

function WapeTarget({ m }: { m: Metrics }) {
  const pe = m.percentage_error!;
  const fin = pe.arms.local.clipped!;
  const scale = 20;
  return (
    <Card className="flex flex-col">
      <CardHeader title="Percentage error, original GRU" meta="Held-out test, Aug 8 – Sep 30, 2023" />
      <CardBody className="flex flex-1 flex-col gap-4">
        <p className="text-sm leading-relaxed text-fg-2">
          The original GRU (local-only, clipped at 0) has a held-out WAPE of <b className="tnum font-semibold text-fg">{pct(fin.wape_pct, 3)}</b>, from{" "}
          {pct(fin.by_horizon_wape_pct[0], 2)} at 5 minutes to {pct(fin.by_horizon_wape_pct[11], 2)} at 60 minutes. For reference, the project's 6% WAPE target
          has not been reached by any evaluated model; the closest independent result is the optimized MLP at{" "}
          {m.optimized ? pct(m.optimized.independent.optimized.wape_pct, 3) : "—"} (Oct–Dec 2023).
        </p>
        <div>
          <div className="relative h-2.5 rounded-full bg-surface-3" role="img" aria-label={`WAPE ${num(fin.wape_pct, 2)} percent against a 6 percent target`}>
            <div className="h-full rounded-full bg-g2" style={{ width: `${(100 * fin.wape_pct) / scale}%` }} />
            <div className="absolute -top-1.5 h-5.5 w-0.5 rounded bg-accent" style={{ left: `${(100 * 6) / scale}%` }} />
          </div>
          <div className="tnum mt-1.5 flex justify-between text-xs text-fg-3">
            <span>0%</span>
            <span>10%</span>
            <span>20%</span>
          </div>
          <Legend
            items={[
              { label: `Original GRU ${pct(fin.wape_pct, 2)}`, color: "#8a8a8a" },
              { label: "Reference 6%", color: "#3bb27e" },
            ]}
          />
        </div>
        <Note className="mt-auto border-t border-line pt-4">
          WAPE is total absolute error divided by total observed traffic over {pe.cells.observed.toLocaleString("en-US")} test cells; sMAPE excludes cells where
          forecast and observation are both 0. MAPE is not shown because {pct((100 * pe.cells.actual_zero) / pe.cells.observed, 2)} of observations are zero. An
          MAE of {num(m.arms.local.mae)} is in vehicles, not percent, and "{pct(m.arms.local.improvement_over_baselines_pct.persistence)} lower than persistence"
          is an improvement, not an error rate.
        </Note>
      </CardBody>
    </Card>
  );
}

export default function Performance() {
  return (
    <StudyGate>
      {({ metrics: m }) => {
        const pe = m.percentage_error;
        const fin = pe?.arms.local.clipped;
        const period: [string, (a: Arm) => number][] = [
          ["Aug 8–31", (a) => a.month["August (8-31)"]],
          ["September", (a) => a.month.September],
          ["Weekdays", (a) => a.weekday_weekend.weekday],
          ["Weekends", (a) => a.weekday_weekend.weekend],
        ];
        const shown: ArmKey[] = ["local", "centralised", "fedavg_geo"];
        return (
          <>
            <PageHeader title="Model performance">
              How the optimized forecasting model compares with the original GRU and naive forecasting on independent data, followed by the federated and
              independent training approaches on {m.test_windows.toLocaleString("en-US")} held-out windows (8 August – 30 September 2023). Lower is better
              everywhere on this page.
            </PageHeader>
            {m.optimized && (
              <div className="mb-12">
                <h2 className="mb-4 text-lg font-semibold">Forecasting Model Comparison</h2>
                <ForecastingPerformance o={m.optimized} />
              </div>
            )}
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
              <h2 className="text-lg font-semibold">Federated vs Independent Training</h2>
              <p className="text-sm text-fg-3">Original GRU approaches and historical baselines · held-out test, Aug 8 – Sep 30, 2023</p>
            </div>

            <div className="grid gap-6 lg:grid-cols-2 2xl:grid-cols-3">
              <Card>
                <CardHeader title="MAE" meta="vehicles per 5 minutes" />
                <CardBody>
                  <BarCompare label="Mean absolute error" data={bars(m, (k) => (k in m.arms ? m.arms[k as ArmKey].mae : m.baselines[k as BaselineKey].mae))} />
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="RMSE" meta="penalises large misses" />
                <CardBody>
                  <BarCompare label="Root mean squared error" data={bars(m, (k) => (k in m.arms ? m.arms[k as ArmKey].rmse : m.baselines[k as BaselineKey].rmse))} />
                </CardBody>
              </Card>
              {pe && (
                <Card className="lg:col-span-2 2xl:col-span-1">
                  <CardHeader title="WAPE" meta="error as a share of traffic" />
                  <CardBody>
                    <BarCompare
                      label="Weighted absolute percentage error"
                      digits={2}
                      suffix="%"
                      data={bars(m, (k) => (k in pe.arms ? pe.arms[k as ArmKey].raw.wape_pct : pe.baselines[k as BaselineKey].wape_pct))}
                    />
                  </CardBody>
                </Card>
              )}
            </div>

            {pe && (
              <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
                <Card>
                  <CardHeader title="All results" meta="three-seed means · held-out test" />
                  <ResultsTable m={m} />
                </Card>
                <WapeTarget m={m} />
              </div>
            )}

            <Card className="mt-6">
              <CardHeader title="Error by forecast horizon" meta="MAE at each 5-minute step ahead" />
              <CardBody>
                <HorizonChart
                  series={[
                    { key: "persistence", name: "Persistence", values: m.baselines.persistence.horizon, ...SERIES.persistence },
                    ...[...ARMS].reverse().map((k) => ({ key: k, name: m.arms[k].label, values: m.arms[k].horizon, ...SERIES[k] })),
                  ]}
                />
                <Legend
                  items={[
                    ...ARMS.map((k) => ({ label: m.arms[k].label, color: SERIES[k].color, dash: !!SERIES[k].dash })),
                    { label: "Persistence", color: SERIES.persistence.color, dash: true },
                  ]}
                />
                {fin && (
                  <Note className="mt-3">
                    Final-model WAPE rises from {pct(fin.by_horizon_wape_pct[0], 1)} at 5 minutes to {pct(fin.by_horizon_wape_pct[11], 1)} at 60 minutes;
                    persistence degrades much faster.
                  </Note>
                )}
              </CardBody>
            </Card>

            <div className="mt-6 grid gap-6 xl:grid-cols-2">
              <Card>
                <CardHeader title="By client group" meta={fin ? "MAE · final-model WAPE" : "MAE"} />
                <TableWrap label="Results by client group">
                  <thead>
                    <tr>
                      <th scope="col">Client group</th>
                      {shown.map((k) => (
                        <NumTh key={k} scope="col">
                          {k === "fedavg_geo" ? "Geo FedAvg" : m.arms[k].label.replace(" GRU", "")}
                        </NumTh>
                      ))}
                      {fin && <NumTh scope="col">WAPE</NumTh>}
                    </tr>
                  </thead>
                  <tbody>
                    {CLIENTS.map((c) => (
                      <tr key={c}>
                        <td>{CLIENT_NAME[c]}</td>
                        {shown.map((k) => (
                          <Num key={k} className={k === "local" ? "text-fg" : "text-fg-2"}>
                            {num(m.arms[k].client[c].mae, 2)}
                          </Num>
                        ))}
                        {fin && <Num>{pct(fin.by_client_wape_pct[c], 2)}</Num>}
                      </tr>
                    ))}
                  </tbody>
                </TableWrap>
              </Card>
              <Card>
                <CardHeader title="By period" meta="MAE" />
                <TableWrap label="Results by period">
                  <thead>
                    <tr>
                      <th scope="col">Period</th>
                      {shown.map((k) => (
                        <NumTh key={k} scope="col">
                          {k === "fedavg_geo" ? "Geo FedAvg" : m.arms[k].label.replace(" GRU", "")}
                        </NumTh>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {period.map(([label, get]) => (
                      <tr key={label}>
                        <td>{label}</td>
                        {shown.map((k) => (
                          <Num key={k} className={k === "local" ? "text-fg" : "text-fg-2"}>
                            {num(get(m.arms[k]))}
                          </Num>
                        ))}
                      </tr>
                    ))}
                    <tr className="[&>*]:!border-line-strong">
                      <td className="text-fg-2">Validation</td>
                      {shown.map((k) => (
                        <Num key={k} className="text-fg-3">
                          {num(m.arms[k].validation_mae)}
                        </Num>
                      ))}
                    </tr>
                  </tbody>
                </TableWrap>
              </Card>
            </div>

            <Card className="mt-6">
              <CardHeader title="Evaluation Methodology" />
              <CardBody>
                <p className="max-w-[80ch] text-sm leading-relaxed text-fg-2">
                  Before the test was opened, the study fixed its rule: the final model is the approach with the lowest validation MAE, and negative predictions
                  are clipped at 0 only if that lowers validation MAE. Local-only won on validation ({num(m.arms.local.validation_mae)}) and clipping helped, so the
                  test confirmed a choice already made. The app's original GRU is seed {m.deployed.final.seed}, the local-only seed with the lowest validation MAE; its
                  own test MAE is {num(m.deployed.final.test_mae_this_seed)}. The later optimized MLP is a separate, non-federated model chosen on validation data and
                  evaluated once on October–December 2023.
                </p>
              </CardBody>
            </Card>
          </>
        );
      }}
    </StudyGate>
  );
}
