import { BarCompare, HorizonChart, type BarDatum } from "../components/charts";
import { ForecastingPerformance, ResultsTable, StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, Legend, Note, Num, NumTh, PageHeader, SectionHeader, Stat, StatStrip, TableWrap, Tag } from "../components/ui";
import { CLIENT_LABELS, TEST, TEST_TARGETS, WIN_VS_BASE } from "../lib/federated3";
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
          has not been reached by any evaluated model, including the selected federated model ({pct(TEST[0].wape, 3)} on Jul–Dec 2024).
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
              { label: `Original GRU ${pct(fin.wape_pct, 2)}`, color: "var(--color-fg-3)" },
              { label: "Reference 6%", color: "var(--color-accent)" },
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

function FederatedResults() {
  const [iid, geo] = TEST;
  const persist = TEST[TEST.length - 1];
  const byClient = [TEST[0], TEST[1], TEST[2], persist];
  return (
    <>
      <StatStrip className="grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Selected model, IID clients" value={pct(iid.wape, 3)} unit="test WAPE" note={`± ${num(iid.sd!, 3)} over 3 seeds`} accent />
        <Stat label="Selected model, geographic clients" value={pct(geo.wape, 3)} unit="test WAPE" note={`± ${num(geo.sd!, 3)} over 3 seeds`} />
        <Stat label="Persistence" value={pct(persist.wape, 3)} unit="test WAPE" note="repeat the last reading" />
        <Stat label="Lower than persistence" value={`${num(persist.wape - iid.wape, 2)} pts`} note={`about ${num((100 * (persist.wape - iid.wape)) / persist.wape, 0)}% relative`} />
      </StatStrip>

      <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <Card>
          <CardHeader title="Test WAPE" meta="Jul–Dec 2024 · lower is better" />
          <CardBody>
            <BarCompare label="Test WAPE, July to December 2024" digits={3} suffix="%" data={TEST.map((t) => ({ label: t.short, value: t.wape, kind: t.kind }))} />
            <Legend
              items={[
                { label: "Selected federated model", color: "var(--color-accent)" },
                { label: "FedAvg baseline", color: "var(--color-ref)" },
                { label: "Non-federated and naive references", color: "var(--color-line-strong)" },
              ]}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="All test results" meta={`${TEST_TARGETS.toLocaleString("en-US")} targets, same for every row`} />
          <TableWrap label="Final test results, July to December 2024">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <NumTh scope="col">WAPE</NumTh>
                <NumTh scope="col">MAE</NumTh>
                <NumTh scope="col">RMSE</NumTh>
                <NumTh scope="col">Validation</NumTh>
              </tr>
            </thead>
            <tbody>
              {TEST.map((t) => (
                <tr key={t.key} className={t.kind === "final" ? "bg-accent/[0.04]" : undefined}>
                  <th scope="row" className="!bg-transparent !text-sm !font-normal !text-fg">
                    <span className="flex flex-col">
                      <span className="flex items-center gap-2">
                        {t.label}
                        {t.key === "win_iid" && <Tag tone="accent">Selected</Tag>}
                      </span>
                      <span className="text-xs text-fg-3">{t.note}</span>
                    </span>
                  </th>
                  <Num className={t.kind === "final" ? "font-semibold" : undefined}>
                    {pct(t.wape, 3)}
                    {t.sd !== undefined && <span className="ml-1 text-xs font-normal text-fg-3">± {num(t.sd, 3)}</span>}
                  </Num>
                  <Num>{num(t.mae, 2)}</Num>
                  <Num>{num(t.rmse, 2)}</Num>
                  <Num className="text-fg-3">{t.val !== undefined ? pct(t.val, 3) : "—"}</Num>
                </tr>
              ))}
            </tbody>
          </TableWrap>
          <Note className="px-5 py-4">
            MAE and RMSE are in vehicles per 5 minutes. Validation (January–June 2024) decided the selection; the test period was opened once, after the model was
            frozen.
          </Note>
        </Card>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader title="By client group" meta="test WAPE" />
          <TableWrap label="Test WAPE by client group">
            <thead>
              <tr>
                <th scope="col">Client group</th>
                {byClient.map((t) => (
                  <NumTh key={t.key} scope="col">
                    {t.short}
                  </NumTh>
                ))}
              </tr>
            </thead>
            <tbody>
              {CLIENT_LABELS.map((c, i) => (
                <tr key={c}>
                  <td className="whitespace-nowrap">{c}</td>
                  {byClient.map((t) => (
                    <Num key={t.key} className={t.kind === "final" ? "text-fg" : "text-fg-2"}>
                      {pct(t.clients[i], 2)}
                    </Num>
                  ))}
                </tr>
              ))}
            </tbody>
          </TableWrap>
          <Note className="px-5 py-4">LA / Ventura is the easiest group and Inland / Central the hardest, for every model.</Note>
        </Card>
        <Card>
          <CardHeader title="By forecast horizon" meta="test WAPE" />
          <TableWrap label="Test WAPE at 5 and 60 minutes ahead">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <NumTh scope="col">5 min ahead</NumTh>
                <NumTh scope="col">60 min ahead</NumTh>
              </tr>
            </thead>
            <tbody>
              {[TEST[0], TEST[1], TEST[2], TEST[4], persist].map((t) => (
                <tr key={t.key}>
                  <td>{t.short}</td>
                  <Num className={t.kind === "final" ? "text-fg" : "text-fg-2"}>{pct(t.h5, 2)}</Num>
                  <Num className={t.kind === "final" ? "text-fg" : "text-fg-2"}>{pct(t.h60, 2)}</Num>
                </tr>
              ))}
            </tbody>
          </TableWrap>
          <Note className="px-5 py-4">Error roughly doubles from 5 to 60 minutes ahead; persistence degrades faster, reaching {pct(persist.h60, 1)}.</Note>
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader title="What the test shows" />
        <CardBody>
          <ul className="flex max-w-[80ch] flex-col gap-2.5 text-sm leading-relaxed text-fg-2">
            <li>
              <b className="font-semibold text-fg">FedAvgM beats plain FedAvg.</b> Matched by seed, the selected model is {num(-WIN_VS_BASE.iid, 3)} WAPE points better
              on IID clients and {num(-WIN_VS_BASE.geo, 3)} on geographic clients, in all six seed pairs.
            </li>
            <li>
              <b className="font-semibold text-fg">The geographic split costs little.</b> Geographic (non-IID) clients add about 0.04 WAPE points over IID clients.
            </li>
            <li>
              <b className="font-semibold text-fg">Federated has not yet matched local training.</b> The local-only GRU and MLP and the centralised MLP score 0.08–0.25
              points better with half the training budget and one seed, so that comparison is not matched.
            </li>
            <li>
              <b className="font-semibold text-fg">The ranking held from validation to test.</b> Test WAPE is about 0.33 points lower than validation for every group:
              a difference between the periods, not a change in order.
            </li>
          </ul>
        </CardBody>
      </Card>
    </>
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
              How the selected federated model, MLP + FedAvgM, performed on July–December 2024, a period no model saw before it was frozen. The original 2023 GRU
              study follows for comparison. Lower is better everywhere on this page.
            </PageHeader>
            <SectionHeader title="Selected federated model" meta="2022–2024 study · held-out test, Jul–Dec 2024" />
            <FederatedResults />

            <SectionHeader title="Original 2023 GRU study" meta={`${m.test_windows.toLocaleString("en-US")} held-out windows · Aug 8 – Sep 30, 2023`} />
            <Note className="-mt-2 mb-6 max-w-[80ch]">
              A different test period, input features and target set from the 2022–2024 study, so these figures are not directly comparable with the results above.
            </Note>
            {m.optimized && (
              <div className="mb-10">
                <h3 className="mb-4 text-lg font-semibold">Optimized MLP vs original GRU</h3>
                <ForecastingPerformance o={m.optimized} />
              </div>
            )}
            <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-lg font-semibold">Four training approaches, 2023</h3>
              <p className="text-sm text-fg-3">GRU approaches and historical baselines</p>
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
              <CardHeader title="How the 2023 model was chosen" />
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
