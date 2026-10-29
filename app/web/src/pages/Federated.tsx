import { FedAvgDiagram, StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, Note, Num, NumTh, PageHeader, Stat, StatStrip, Steps, Swatch, TableWrap, Tag } from "../components/ui";
import { num, signed } from "../lib/format";
import { ARMS, CLIENT_NAME, CLIENTS, SERIES } from "../lib/study";
import type { ArmKey, Contrast, RunCost } from "../lib/types";

function Verdict({ c }: { c: Contrast }) {
  if (c.statistical && c.practical) return <Tag tone="accent">Detectable · meaningful</Tag>;
  if (c.statistical) return <Tag>Detectable · below threshold</Tag>;
  return <Tag>Not distinguishable</Tag>;
}

const ROUND = [
  ["Broadcast.", "The server sends the current global GRU weights to every client."],
  ["Train locally.", "Each client trains one epoch on its own sensors; raw data never leaves it."],
  ["Upload.", "Clients return updated weights only."],
  ["Average.", "The server averages them, weighted by each client's number of training windows."],
  ["Stop early.", "Training ends after 5 rounds without validation improvement; the best round is kept."],
];

export default function Federated() {
  return (
    <StudyGate>
      {({ metrics: m, sensors }) => {
        const c = m.contrasts;
        const rt = m.runtime;
        const runs = (arm: ArmKey) => [0, 1, 2].map((s) => rt[`${arm}_seed${s}`]).filter(Boolean) as RunCost[];
        const avg = (arm: ArmKey, g: (r: RunCost) => number) => runs(arm).reduce((t, r) => t + g(r), 0) / Math.max(1, runs(arm).length);
        const geo0 = rt.fedavg_geo_seed0;
        const perRound = geo0 && geo0.units_run ? geo0.communication_mb / geo0.units_run : NaN;
        return (
          <>
            <PageHeader title="Federated learning">
              Federated learning trains one shared model without moving raw data. Each client trains on its own sensors and sends only model weights to a server,
              which averages them (FedAvg). The study simulated four clients on one machine and compared this with pooling all data (centralised) and with each
              client training alone (local-only).
            </PageHeader>

            <Card>
              <CardHeader title="How FedAvg works here" meta={`${num(perRound, 2)} MB of weights per round, all four clients`} />
              <CardBody>
                <FedAvgDiagram sensors={sensors} />
              </CardBody>
            </Card>

            <StatStrip className="mt-6 grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">
              {CLIENTS.map((cl) => (
                <Stat
                  key={cl}
                  label={CLIENT_NAME[cl]}
                  value={num(m.arms.local.client[cl].mae, 2)}
                  unit="local-only MAE"
                  note={`Geographic FedAvg ${num(m.arms.fedavg_geo.client[cl].mae, 2)} · ${sensors.filter((s) => s.client === cl).length} sensors`}
                />
              ))}
            </StatStrip>

            <div className="mt-6 grid gap-6 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
              <Card>
                <CardHeader title="Accuracy trade-offs" meta="three matched seeds · MAE difference" />
                <TableWrap label="Pre-registered comparisons">
                  <thead>
                    <tr>
                      <th scope="col">Comparison</th>
                      <NumTh scope="col">Difference</NumTh>
                      <NumTh scope="col">95% interval</NumTh>
                      <th scope="col">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(c).map((x) => (
                      <tr key={`${x.a}-${x.b}`}>
                        <td>
                          {m.arms[x.a].label} vs {m.arms[x.b].label}
                        </td>
                        <Num>{signed(x.diff)}</Num>
                        <Num className="text-fg-2">
                          {num(x.reported_ci95[0])} to {num(x.reported_ci95[1])}
                        </Num>
                        <td>
                          <Verdict c={x} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </TableWrap>
                <Note className="px-5 py-4">
                  Positive means the first model has higher error. "Meaningful" uses the threshold fixed before the test, {num(m.sesoi_mae)} MAE. Intervals resample
                  whole sensors within client groups.
                </Note>
              </Card>
              <Card className="border-accent/30">
                <CardHeader title="Local-only training had the lowest mean error" />
                <CardBody className="flex flex-col gap-3 text-sm leading-relaxed text-fg-2">
                  <p>
                    Geographic FedAvg did not beat centralised or local-only training: it cost {num(c.C1.diff)} MAE against centralised and {num(c.C2.diff)} against
                    local-only, both above the practical threshold.
                  </p>
                  <p>
                    IID FedAvg was {num(-c.C3.diff)} better than geographic FedAvg, a detectable but small effect of the geographic split. Local-only and
                    centralised could not be told apart (Holm-adjusted p {num(c.C4.p_holm)}).
                  </p>
                </CardBody>
              </Card>
            </div>

            <div className="mt-6 grid gap-6 xl:grid-cols-2">
              <Card>
                <CardHeader title="One FedAvg round" />
                <CardBody>
                  <Steps items={ROUND} />
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Training cost" meta="mean per run, CPU" />
                <TableWrap label="Training cost">
                  <thead>
                    <tr>
                      <th scope="col">Approach</th>
                      <NumTh scope="col">Time</NumTh>
                      <NumTh scope="col">Model traffic</NumTh>
                      <NumTh scope="col">Test MAE</NumTh>
                    </tr>
                  </thead>
                  <tbody>
                    {ARMS.map((k) => (
                      <tr key={k}>
                        <td>
                          <span className="flex items-center gap-2.5">
                            <Swatch color={SERIES[k].color} dash={!!SERIES[k].dash} />
                            {m.arms[k].label}
                          </span>
                        </td>
                        <Num>{num(avg(k, (r) => r.minutes), 0)} min</Num>
                        <Num className="text-fg-2">{k.startsWith("fedavg") ? `${num(avg(k, (r) => r.communication_mb), 1)} MB` : "none"}</Num>
                        <Num>{num(m.arms[k].mae)}</Num>
                      </tr>
                    ))}
                  </tbody>
                </TableWrap>
                <Note className="px-5 py-4">Clients were simulated in one process, so these figures include no network latency.</Note>
              </Card>
            </div>
          </>
        );
      }}
    </StudyGate>
  );
}
