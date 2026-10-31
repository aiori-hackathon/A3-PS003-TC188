import { ExternalLink } from "lucide-react";
import { StudyGate } from "../components/study";
import { Card, CardBody, CardHeader, Num, NumTh, PageHeader, Swatch, TableWrap, Tag } from "../components/ui";
import { num } from "../lib/format";
import { REPO, SERIES } from "../lib/study";

const ENDPOINTS = [
  ["GET /api/health", "service status and device"],
  ["GET /api/models", "model provenance"],
  ["GET /api/sensors", "200 anonymised sensors"],
  ["GET /api/metrics", "verified aggregate results"],
  ["GET /api/demo", "historical examples"],
  ["POST /api/predict", "12-step forecast, historical or from 12 readings"],
];

export default function About() {
  return (
    <StudyGate>
      {({ health, models }) => {
        const pipeline = [
          ["Frozen research outputs", "Frozen checkpoints, training-only scaling and the single test report, verified by hash."],
          ["Python + PyTorch", `Rebuilds the four training features and runs the GRU on ${health.device.toUpperCase()} in a few milliseconds.`],
          ["Validated JSON API", "Anonymised sensor ids, input validation, no paths or station identifiers."],
          ["React interface", "React, TypeScript and Vite with Tailwind CSS, Recharts and Lucide icons; no third-party scripts at runtime."],
        ];
        return (
          <>
            <PageHeader title="About">AIORI PS-003: federated traffic-flow prediction with communication-cost evaluation, built for AIORI Hackathon 3.</PageHeader>

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
              <CardHeader title="Deployed models" meta="selected on validation only" />
              <TableWrap label="Deployed models">
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
                            <Swatch color={x.key === "centralised" ? SERIES.centralised.color : x.key === "final" ? "#8a8a8a" : SERIES.local.color} />
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
                    Start the service from the repository root with <code>.venv\Scripts\python.exe app\backend\server.py</code>, then open{" "}
                    <code>http://127.0.0.1:8000</code>. For development, run <code>npm run dev</code> in <code>app/web</code>.
                  </p>
                  <p>
                    Run the tests with <code>.venv\Scripts\python.exe -m pytest app\tests</code> and <code>npm test</code>. Percentage errors come from{" "}
                    <code>app/analysis/percentage_error.py</code>, read-only on the frozen test predictions.
                  </p>
                  <p>
                    Data: XTraffic 2023 (CC BY-NC 4.0); raw data, checkpoints and station identifiers are not published.{" "}
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
                  of state-of-the-art accuracy is made, because no comparable published benchmark uses this exact protocol.
                </p>
              </CardBody>
            </Card>
          </>
        );
      }}
    </StudyGate>
  );
}
