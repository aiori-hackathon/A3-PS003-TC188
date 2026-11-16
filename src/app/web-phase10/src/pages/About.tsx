import { ExternalLink } from "lucide-react";
import { StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, Num, NumTh, PageHeader, Swatch, TableWrap, Tag } from "../components/ui";
import { num } from "../lib/format";
import { MANIFEST_SHA, TEST } from "../lib/federated3";
import { REPO, SERIES } from "../lib/study";

const ENDPOINTS = [
  ["GET /api/health", "service status and device"],
  ["GET /api/models", "model provenance"],
  ["GET /api/sensors", "200 anonymised sensors"],
  ["GET /api/metrics", "verified aggregate results"],
  ["GET /api/demo", "historical examples"],
  ["POST /api/predict", "12-step forecast, historical or from 12 readings"],
  ["GET /api/replay/meta, sensors, day", "replay calendar and sensor positions"],
  ["GET /api/replay/frame, series, forecast", "map frames and frozen-model forecasts"],
  ["GET /api/replay/disruptions/*", "candidate anomaly catalog"],
];

export default function About() {
  return (
    <StudyGate>
      {({ health, models }) => {
        const pipeline = [
          ["Frozen research outputs", "Checkpoints from both studies, training-only scaling, the anomaly catalog and single test reports, all verified by hash."],
          ["Python + PyTorch", `Two local services on ${health.device.toUpperCase()}: the study API (2023 GRU) and the replay API (frozen MLP + FedAvgM).`],
          ["Validated JSON API", "Anonymised sensor ids, input validation, no paths or station identifiers; the website never trains."],
          ["React interface", "React, TypeScript and Vite with Tailwind CSS, Recharts and Lucide icons; no third-party scripts at runtime."],
        ];
        return (
          <>
            <PageHeader title="About">
              AIORI PS-003: federated traffic-flow prediction with communication-cost evaluation, built for AIORI Hackathon 3. AIORI forecasts freeway flow one hour
              ahead from three years of recorded XTraffic data (2022–2024, 200 California sensors) and tests whether the forecaster can be trained across regions
              without pooling their data. Everything shown is historical; nothing is live.
            </PageHeader>

            <ol className="grid gap-px overflow-hidden rounded-card border border-line bg-line sm:grid-cols-2 xl:grid-cols-4" aria-label="How the app works, from data to interface">
              {pipeline.map(([t, d], i) => (
                <li key={t} className="flex flex-col gap-1.5 bg-surface p-5">
                  <span className="tnum text-xs text-fg-3">Layer {i + 1}</span>
                  <h3 className="text-base font-semibold">{t}</h3>
                  <p className="text-sm leading-relaxed text-fg-2">{d}</p>
                </li>
              ))}
            </ol>

            <Card className="mt-6">
              <CardHeader title="Selected federated model" meta="2022–2024 study · served by Historical Traffic Replay" />
              <CardBody className="grid gap-x-8 gap-y-3 text-sm md:grid-cols-2">
                <dl className="grid grid-cols-[minmax(120px,auto)_1fr] gap-y-2">
                  <dt className="text-fg-3">Model</dt>
                  <dd>MLP + FedAvgM (server momentum 0.9), history A</dd>
                  <dt className="text-fg-3">Checkpoints</dt>
                  <dd>6: three seeds × IID and geographic clients</dd>
                  <dt className="text-fg-3">Test WAPE</dt>
                  <dd className="tnum">
                    {num(TEST[0].wape, 3)}% IID · {num(TEST[1].wape, 3)}% geographic (Jul–Dec 2024)
                  </dd>
                </dl>
                <dl className="grid grid-cols-[minmax(120px,auto)_1fr] gap-y-2">
                  <dt className="text-fg-3">Selection</dt>
                  <dd>validation only, Jan–Jun 2024; tested once</dd>
                  <dt className="text-fg-3">Manifest SHA-256</dt>
                  <dd className="min-w-0">
                    <code className="break-all">{MANIFEST_SHA}</code>
                  </dd>
                  <dt className="text-fg-3">Integrity</dt>
                  <dd>Replay refuses to start unless every checkpoint hash matches</dd>
                </dl>
              </CardBody>
            </Card>

            <Card className="mt-6">
              <CardHeader title="2023 study models" meta="served by the study API · selected on validation only" />
              <TableWrap label="2023 study models">
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <NumTh scope="col">Seed</NumTh>
                    <NumTh scope="col">Validation MAE</NumTh>
                    <NumTh scope="col">Test MAE (seed)</NumTh>
                    <th scope="col">Routing</th>
                    <th scope="col">Checkpoint SHA-256</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((x) => {
                    const hashes = Object.values(x.checkpoint_sha256 ?? {});
                    return (
                      <tr key={x.key}>
                        <td>
                          <span className="flex items-center gap-2.5 whitespace-nowrap">
                            <Swatch color={x.key === "centralised" ? SERIES.centralised.color : x.key === "final" ? "var(--color-fg-3)" : SERIES.local.color} />
                            {x.label}
                            {x.key === "candidate" && <Tag tone="accent">Optimized</Tag>}
                            {x.key === "final" && <Tag>Original GRU</Tag>}
                          </span>
                        </td>
                        <Num>{x.seed}</Num>
                        <Num>{num(x.validation_mae)}</Num>
                        <Num>{num(x.test_mae_this_seed)}</Num>
                        <td className="text-fg-2">{x.key === "centralised" ? "one model for all sensors" : "own client's model per sensor; clipped at 0"}</td>
                        <td className="whitespace-nowrap">
                          {hashes.length ? <code>{hashes[0]}</code> : <span className="text-fg-3">local candidate</span>}
                          {hashes.length > 1 && <span className="ml-1.5 text-xs text-fg-3">+{hashes.length - 1}</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </TableWrap>
            </Card>

            <div className="mt-6 grid gap-6 xl:grid-cols-2">
              <Card>
                <CardHeader title="API" />
                <CardBody className="py-2">
                  <dl className="text-sm">
                    {ENDPOINTS.map(([k, v], i) => (
                      <div key={k} className={`flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-2.5 ${i ? "border-t border-line" : ""}`}>
                        <dt>
                          <code>{k}</code>
                        </dt>
                        <dd className="text-fg-2">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </CardBody>
              </Card>
              <Card>
                <CardHeader title="Reproduce" />
                <CardBody className="flex flex-col gap-3 text-sm leading-relaxed text-fg-2">
                  <p>
                    From the repository root, run <code>.venv\Scripts\python.exe src\app\backend\replay_server.py</code> and open{" "}
                    <code>http://127.0.0.1:8010</code> for every page, including Historical Traffic Replay and the Disruption Explorer. The study API alone is{" "}
                    <code>src\app\backend\server.py</code> on <code>http://127.0.0.1:8000</code>. Both need the local data and checkpoints, which are not in the
                    repository.
                  </p>
                  <p>
                    Run the tests with <code>.venv\Scripts\python.exe -m pytest src\app\tests src\tests</code> and <code>npm test</code>. The federated study's code and its
                    verified results are in the repository's <code>src</code> folder.
                  </p>
                  <p>
                    Data: XTraffic 2022–2024, CC BY-NC 4.0 (the original GRU study used January–September 2023). Raw data, checkpoints and station identifiers are not
                    published.{" "}
                    <a href={REPO} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-accent underline-offset-4 hover:underline">
                      Source repository
                      <ExternalLink className="size-3.5" aria-hidden />
                    </a>
                  </p>
                </CardBody>
              </Card>
            </div>

            <Card className="mt-6">
              <CardHeader title="Acknowledgments" />
              <CardBody>
                <p className="max-w-[80ch] text-sm leading-relaxed text-fg-2">
                  Built on the XTraffic dataset and the AIORI Hackathon 3 problem statement PS-003. These are research findings from one historical dataset; no claim
                  of state-of-the-art accuracy is made, because no comparable published benchmark uses this exact protocol. Detected disruptions are statistical
                  candidates, not confirmed incidents.
                </p>
              </CardBody>
            </Card>
          </>
        );
      }}
    </StudyGate>
  );
}
