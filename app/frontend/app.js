/* AIORI Traffic Intelligence front end. Every number comes from the API (frozen Phase 9 artifacts);
   every forecast is computed by the frozen GRU on request. Nothing is invented. */
"use strict";

const ARMS = ["local", "centralised", "fedavg_iid", "fedavg_geo"];
const COLORS = { local: "#3bb27e", centralised: "#c8c8c8", fedavg_iid: "#8f8f8f", fedavg_geo: "#8f8f8f", persistence: "#666666",
  seasonal_naive_daily: "#555555", historical_average: "#4a4a4a", actual: "#d6a24a", history: "#ededed" };
const DASH = { local: null, centralised: null, fedavg_iid: "6 4", fedavg_geo: "2 3", persistence: "1 4" };
const BASE_LABEL = { persistence: "Persistence", seasonal_naive_daily: "Same time yesterday", historical_average: "Historical average" };
const CLIENT_ORDER = ["client_0_la_ventura", "client_1_bay_area", "client_2_coastal_south_central", "client_3_inland_central"];
const CLIENT_NAME = { client_0_la_ventura: "LA / Ventura", client_1_bay_area: "Bay Area", client_2_coastal_south_central: "Coastal / South Central", client_3_inland_central: "Inland / Central" };
const TITLES = { overview: "Overview", forecast: "Forecasting", performance: "Model performance", federated: "Federated learning", methodology: "Methodology", about: "About" };
const REPO = "https://github.com/aiori-hackathon/a3-ps003-tc188";
const state = { metrics: null, sensors: null, demo: null, models: null, health: null, rendered: {} };

/* ---------- utilities ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v, d = 1) => `${f(v, d)}%`;
const sign = (v, d = 3) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${f(Math.abs(v), d)}`;
const hhmm = (t) => t.slice(11, 16);
const niceDate = (t) => new Date(t + ":00").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric" });
const key = (k) => `<span class="key" style="background:${COLORS[k]}"></span>`;

async function api(path, body) {
  const opts = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  let res;
  try { res = await fetch(path, opts); } catch (e) { throw new Error("The model service is not reachable. Start it with .venv\\Scripts\\python.exe app\\backend\\server.py and reload."); }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data.error && data.error.message) || `Request failed (${res.status}).`);
  return data;
}
const alertBox = (msg, title = "This section could not load.") => `<div class="alert" role="alert"><strong>${esc(title)}</strong><span>${esc(msg)}</span></div>`;
const loading = (h = 260) => `<div class="skeleton" style="height:${h}px" aria-busy="true" aria-label="Loading"></div>`;

/* ---------- SVG charts ---------- */
function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const m = [1, 2, 2.5, 5, 10].find((k) => span / (k * step) <= n) * step;
  const out = [];
  for (let v = Math.floor(lo / m) * m; v <= hi + 1e-9; v += m) out.push(+v.toFixed(6));
  return out;
}
const CHARTS = new Map();
function remember(wrap, fn, opts) { CHARTS.set(wrap, () => fn(wrap, opts)); }
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => {
  for (const [w, draw] of CHARTS) { if (!w.isConnected) CHARTS.delete(w); else if (w.offsetParent !== null) draw(); }
}, 150); });

/** Line chart sized to its container. series: [{name, color, dash, width, values (null = gap), digits}] */
function lineChart(wrap, opts) {
  const { labels, series, yLabel, nowIndex, height = 300, xFormat = (l) => l, tipTitle = (l) => l } = opts;
  remember(wrap, lineChart, opts);
  if (wrap.offsetParent === null) return;                            // hidden: drawn when its view is shown
  wrap.innerHTML = "";
  wrap.classList.add("chart-wrap");
  const W = Math.max(300, Math.round(wrap.clientWidth || 800));
  const H = W < 560 ? Math.round(height * 0.78) : height, L = 44, R = 8, T = 18, B = 30;
  const all = series.flatMap((s) => s.values.filter((v) => v !== null && v !== undefined));
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.12 || 1; lo = Math.max(0, lo - pad); hi += pad;
  const yt = ticks(lo, hi, W < 560 ? 4 : 5);
  lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);
  const X = (i) => L + (W - L - R) * (labels.length === 1 ? 0.5 : i / (labels.length - 1));
  const Y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "chart", role: "img", "aria-label": `${yLabel} chart` });
  if (nowIndex !== undefined) {
    const x = (X(nowIndex - 1) + X(nowIndex)) / 2;
    svg.appendChild(svgEl("rect", { x, y: T, width: W - R - x, height: H - T - B, class: "now-band" }));
  }
  const g = svgEl("g", { class: "grid" });
  for (const v of yt) {
    g.appendChild(svgEl("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v) }));
    const t = svgEl("text", { x: L - 10, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = v.toLocaleString("en-US"); g.appendChild(t);
  }
  svg.appendChild(g);
  const every = Math.max(1, Math.ceil(labels.length / (W < 560 ? 4 : 8)));
  labels.forEach((l, i) => {
    if (i % every && i !== labels.length - 1) return;
    const t = svgEl("text", { x: X(i), y: H - 8, "text-anchor": i === 0 ? "start" : i === labels.length - 1 ? "end" : "middle" }); t.textContent = xFormat(l); svg.appendChild(t);
  });
  if (nowIndex !== undefined) {
    const x = (X(nowIndex - 1) + X(nowIndex)) / 2;
    svg.appendChild(svgEl("line", { x1: x, x2: x, y1: T - 6, y2: H - B, class: "now" }));
    const t = svgEl("text", { x: x + 8, y: T + 6, class: "now-label" }); t.textContent = "Forecast"; svg.appendChild(t);
    const t2 = svgEl("text", { x: x - 8, y: T + 6, "text-anchor": "end" }); t2.textContent = "Observed"; svg.appendChild(t2);
  }
  for (const s of series) {
    let d = "", pen = false;
    s.values.forEach((v, i) => {
      if (v === null || v === undefined) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true;
    });
    const p = svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 1.6, "stroke-linejoin": "round", "stroke-linecap": "round" });
    if (s.dash) p.setAttribute("stroke-dasharray", s.dash);
    svg.appendChild(p);
    if (s.dots) s.values.forEach((v, i) => { if (v !== null && v !== undefined) svg.appendChild(svgEl("circle", { cx: X(i), cy: Y(v), r: 2.4, fill: "#0e0e0e", stroke: s.color, "stroke-width": 1.4 })); });
  }
  const hover = svgEl("line", { class: "hover-line", y1: T, y2: H - B, x1: -10, x2: -10 });
  svg.appendChild(hover);
  const zone = svgEl("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent", tabindex: 0 });
  zone.setAttribute("aria-label", "Chart values. Use the left and right arrow keys to read each point.");
  svg.appendChild(zone);
  const tip = document.createElement("div"); tip.className = "tip"; tip.setAttribute("role", "status");
  wrap.append(svg, tip);
  let cur = -1;
  const show = (i) => {
    cur = Math.max(0, Math.min(labels.length - 1, i));
    const x = X(cur);
    hover.setAttribute("x1", x); hover.setAttribute("x2", x);
    tip.innerHTML = `<b>${esc(tipTitle(labels[cur]))}</b>` + series.filter((s) => s.values[cur] !== null && s.values[cur] !== undefined)
      .map((s) => `<div><span><span class="key" style="background:${s.color}"></span>${esc(s.name)}</span><b>${f(s.values[cur], s.digits ?? 1)}</b></div>`).join("");
    tip.style.left = `${Math.min(W - 196, Math.max(0, x + 14))}px`;
    tip.style.top = "6px"; tip.style.opacity = 1;
  };
  zone.addEventListener("mousemove", (e) => { const r = svg.getBoundingClientRect(); const x = ((e.clientX - r.left) / r.width) * W; show(Math.round(((x - L) / (W - L - R)) * (labels.length - 1))); });
  zone.addEventListener("mouseleave", () => { tip.style.opacity = 0; hover.setAttribute("x1", -10); hover.setAttribute("x2", -10); });
  zone.addEventListener("keydown", (e) => { if (e.key === "ArrowRight") { show(cur + 1); e.preventDefault(); } if (e.key === "ArrowLeft") { show(cur < 0 ? labels.length - 1 : cur - 1); e.preventDefault(); } });
  zone.addEventListener("blur", () => { tip.style.opacity = 0; });
}

/** CSS bar rows, lower is better. rows: [{label, value, kind: "final"|"arm"|"ref"}] */
function bars(rows, { digits = 3, suffix = "" } = {}) {
  const max = Math.max(...rows.map((r) => r.value)) * 1.04;
  return `<div class="bars">${rows.map((r, i) => `<div class="bar-row ${r.kind}"><span class="label" title="${esc(r.label)}">${esc(r.label)}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${(100 * r.value / max).toFixed(2)}%;animation-delay:${i * 40}ms"></div></div>
    <span class="value">${f(r.value, digits)}${suffix}</span></div>`).join("")}</div>`;
}

/* ---------- shared pieces ---------- */
function drawForecast(wrap, r, opts, height = 320) {
  const labels = [...r.history.timestamps, ...r.forecast_timestamps];
  const pad = (arr, before) => (before ? [...arr, ...Array(12).fill(null)] : [...Array(11).fill(null), r.history.values[11], ...arr]);
  const series = [{ name: "Observed", color: COLORS.history, values: pad(r.history.values, true), dots: true, width: 1.6 }];
  if (opts.showActual && r.actual) series.push({ name: "Observed afterwards", color: COLORS.actual, dash: "4 4", width: 1.5, values: [...Array(12).fill(null), ...r.actual.values] });
  if (opts.models.includes("centralised") && r.forecasts.centralised) series.push({ name: "Centralised GRU", color: COLORS.centralised, dash: "5 4", width: 1.5, values: pad(r.forecasts.centralised.values, false) });
  if (r.forecasts.final) series.push({ name: "Local-only GRU (final)", color: COLORS.local, width: 2.4, values: pad(r.forecasts.final.values, false) });
  lineChart(wrap, { labels, series, nowIndex: 12, yLabel: "Vehicles per 5 min", height, xFormat: hhmm, tipTitle: (l) => `${hhmm(l)} · ${niceDate(l)}` });
}

function forecastTicks(r, showActual) {
  const v = r.forecasts.final.values, a = showActual && r.actual ? r.actual.values : null;
  return `<div class="ticks">${r.forecast_timestamps.map((t, i) => `<div class="tick" style="animation-delay:${i * 25}ms">
    <span class="t">${esc(hhmm(t))}</span><span class="v">${Math.round(v[i])}</span>${a ? `<span class="o">obs <b>${a[i] === null ? "—" : Math.round(a[i])}</b></span>` : ""}</div>`).join("")}</div>`;
}

function resultsTable(m) {
  const pe = m.percentage_error;
  const wape = (k) => (pe && pe.arms[k] ? pct(pe.arms[k].raw.wape_pct, 2) : "—");
  const smape = (k) => (pe && pe.arms[k] ? pct(pe.arms[k].raw.smape_pct, 2) : "—");
  const rows = ARMS.map((k) => {
    const a = m.arms[k];
    return `<tr><td><span class="model">${key(k)}${esc(a.label)}${k === "local" ? ' <span class="tag accent">Final</span>' : ""}</span></td>
      <td class="num">${f(a.mae)}</td><td class="num">${f(a.rmse)}</td><td class="num">${wape(k)}</td><td class="num">${smape(k)}</td>
      <td class="num">${pct(a.improvement_over_baselines_pct.persistence)}</td></tr>`;
  }).join("");
  const refs = Object.entries(m.baselines).map(([k, b], i) => `<tr class="ref${i === 0 ? " first-ref" : ""}"><td><span class="model">${key(k)}${BASE_LABEL[k]}</span></td>
      <td class="num">${f(b.mae)}</td><td class="num">${f(b.rmse)}</td><td class="num">${pe && pe.baselines[k] ? pct(pe.baselines[k].wape_pct, 2) : "—"}</td><td class="num subtle">n/a</td><td class="num subtle">—</td></tr>`).join("");
  return `<div class="table-wrap"><table><caption class="sr-only">Held-out test results; lower is better</caption>
    <thead><tr><th>Model · mean of 3 seeds</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">WAPE</th><th class="num">sMAPE</th><th class="num">vs persistence</th></tr></thead>
    <tbody>${rows}${refs}</tbody></table></div>`;
}

/* ---------- views ---------- */
async function renderOverview(el) {
  const m = state.metrics, pe = m.percentage_error, c = m.contrasts, loc = m.arms.local;
  el.innerHTML = `
    <section class="hero">
      <h1>Forecast the next hour of freeway traffic with the study's frozen models.</h1>
      <p>AIORI PS-003 trained one compact GRU four ways (centralised, local-only and two federated variants) on nine months of California freeway flow. Every forecast here is computed live by a frozen checkpoint on historical data, and every metric comes from a single held-out test.</p>
      <div class="actions"><a class="btn primary" href="#forecast">Run a forecast</a><a class="btn" href="#performance">See the results</a></div>
    </section>
    <section class="grid cols-4 stat-grid" aria-label="Key results">
      <div class="card stat accent"><span class="stat-label">Final model MAE</span><span class="stat-value">${f(loc.clipped_mae, 2)}<small>veh / 5 min</small></span><span class="stat-note">Local-only GRU, clipped at 0, test</span></div>
      <div class="card stat"><span class="stat-label">WAPE</span><span class="stat-value">${pe ? pct(pe.target.final_model_wape_pct, 2) : "—"}</span><span class="stat-note">Target 6% · not yet reached</span></div>
      <div class="card stat"><span class="stat-label">Error vs persistence</span><span class="stat-value">−${pct(loc.improvement_over_baselines_pct.persistence)}</span><span class="stat-note">Lower MAE than the naive forecast</span></div>
      <div class="card stat"><span class="stat-label">Held-out windows</span><span class="stat-value">${(m.test_windows / 1e6).toFixed(2)}<small>M</small></span><span class="stat-note">${m.n_sensors} sensors · 8 Aug – 30 Sep 2023</span></div>
    </section>
    <section class="grid split">
      <div class="card">
        <div class="card-head"><div><h3 id="ov-title">Example forecast</h3><span class="meta" id="ov-sub">Loading a historical example</span></div><a class="btn ghost" href="#forecast">Open in Forecasting</a></div>
        <div class="card-body"><div id="ov-chart">${loading(260)}</div><div class="legend" style="margin-top:12px"><span><i style="border-color:${COLORS.history}"></i>Observed</span><span><i style="border-color:${COLORS.local}"></i>Local-only GRU forecast</span><span><i class="dash" style="border-color:${COLORS.actual}"></i>Observed afterwards</span></div></div>
      </div>
      <div class="card">
        <div class="card-head"><h3>Model insights</h3><span class="meta">MAE, held-out test</span></div>
        <div class="insight">
          <div class="insight-row"><div><h4>Local-only training was most accurate</h4><p>Lowest three-seed mean error of the four approaches</p></div><span class="v">${f(loc.mae)}</span></div>
          <div class="insight-row"><div><h4>Cost of geographic federation</h4><p>Geographic FedAvg vs centralised training</p></div><span class="v">${sign(c.C1.diff)}</span></div>
          <div class="insight-row"><div><h4>Federation vs training alone</h4><p>Geographic FedAvg vs local-only</p></div><span class="v">${sign(c.C2.diff)}</span></div>
          <div class="insight-row"><div><h4>Effect of the geographic split</h4><p>IID FedAvg vs geographic FedAvg</p></div><span class="v">${sign(c.C3.diff)}</span></div>
          <div class="insight-row"><div><h4>Practical threshold</h4><p>Smallest difference that counts, fixed before the test</p></div><span class="v">${f(m.sesoi_mae, 3)}</span></div>
        </div>
      </div>
    </section>
    <section class="section">
      <div class="section-head"><h2>Verified held-out results</h2><p>Vehicles per 5-minute interval · lower is better</p></div>
      <div class="card">${resultsTable(m)}</div>
    </section>`;
  try {
    const ex = state.demo.examples[0];
    const r = await api("/predict", { sensor: ex.sensor, time: ex.time, models: ["final"] });
    $("#ov-title").textContent = `${ex.label} · ${r.sensor.client_name}`;
    $("#ov-sub").textContent = `Sensor ${r.sensor.id} · ${niceDate(r.forecast_start)} · historical, held-out period`;
    drawForecast($("#ov-chart"), r, { showActual: true, models: ["final"] }, 260);
  } catch (e) { $("#ov-chart").innerHTML = alertBox(e.message, "The example forecast could not run."); }
}

function renderForecast(el) {
  const sensors = state.sensors;
  el.innerHTML = `
    <header class="page-head"><h1>Forecasting</h1>
      <p>Choose a sensor and a historical moment. The frozen GRU reads the previous 60 minutes and forecasts the next 60 in 5-minute steps, computed on demand each time you run it.</p></header>
    <div class="grid split-r">
      <div class="card">
        <div class="card-head"><h3>Inputs</h3><span class="meta">Jul 11 – Sep 30, 2023</span></div>
        <div class="card-body form">
          <label class="field"><span>Client group</span><select id="fc-client">${CLIENT_ORDER.map((c) => `<option value="${c}">${CLIENT_NAME[c]}</option>`).join("")}</select></label>
          <label class="field"><span>Sensor</span><select id="fc-sensor"></select></label>
          <div class="form-row">
            <label class="field"><span>Date</span><input type="date" id="fc-date" min="2023-07-11" max="2023-09-30" value="2023-09-12"></label>
            <label class="field"><span>Forecast from</span><input type="time" id="fc-time" step="300" value="07:30"></label>
          </div>
          <button class="btn primary block" id="fc-go">Run forecast</button>
          <div class="divider"></div>
          <label class="check"><span>Compare centralised GRU<small>Same input, pooled-data model</small></span><span class="switch"><input type="checkbox" id="fc-cmp" checked><span></span></span></label>
          <label class="check"><span>Show observed afterwards<small>Historical truth, for comparison</small></span><span class="switch"><input type="checkbox" id="fc-act" checked><span></span></span></label>
          <div class="divider"></div>
          <div class="field"><span>Historical examples</span><div class="examples" id="fc-examples"></div></div>
        </div>
        <details class="more"><summary>Forecast from your own readings</summary>
          <div class="more-body">
            <p class="xs subtle">Twelve 5-minute readings for the selected sensor, oldest first, ending just before the date and time above. Leave a value blank if it is missing; it is filled the way the model was trained.</p>
            <textarea id="fc-custom" rows="3" placeholder="410, 422, 418, , 430, 441, 437, 452, 460, 455, 470, 468" aria-label="Twelve readings, comma separated"></textarea>
            <button class="btn" id="fc-custom-go">Forecast from these readings</button>
          </div>
        </details>
      </div>
      <div class="grid" style="gap:16px">
        <div class="card" aria-live="polite">
          <div class="card-head"><div><h3 id="fc-title">Forecast</h3><span class="meta" id="fc-meta"></span></div><span class="tag" id="fc-mode" hidden></span></div>
          <div class="card-body"><div id="fc-chart">${loading(320)}</div><div class="legend" id="fc-legend" style="margin-top:12px"></div></div>
          <div id="fc-ticks"></div>
        </div>
        <div class="card" id="fc-table-card" hidden><div class="card-head"><h3>Forecast detail</h3><span class="meta" id="fc-units"></span></div><div class="table-wrap" id="fc-table"></div></div>
      </div>
    </div>`;
  const cl = $("#fc-client"), se = $("#fc-sensor"), ex = $("#fc-examples");
  const fill = () => {
    se.innerHTML = sensors.filter((s) => s.client === cl.value)
      .map((s) => `<option value="${s.id}">${s.id} · ${s.type_group} · median ${Math.round(s.median_test_flow)}</option>`).join("");
    const list = state.demo.examples.map((e, i) => ({ ...e, i })).filter((e) => sensors.find((s) => s.id === e.sensor).client === cl.value);
    ex.innerHTML = list.map((e) => `<button class="example" data-i="${e.i}" aria-pressed="false"><b>${esc(e.label)}</b><span>${esc(e.sensor)} · ${esc(niceDate(e.time))}, ${esc(hhmm(e.time))}</span></button>`).join("");
  };
  cl.addEventListener("change", fill); fill();
  ex.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    const d = state.demo.examples[+b.dataset.i];
    se.value = d.sensor; $("#fc-date").value = d.time.slice(0, 10); $("#fc-time").value = d.time.slice(11, 16);
    ex.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
    run();
  });
  let last = null;
  const go = $("#fc-go");
  $("#fc-cmp").addEventListener("change", () => last && show(last));
  $("#fc-act").addEventListener("change", () => last && show(last));
  function clear(html) {
    last = null;
    $("#fc-chart").innerHTML = html; $("#fc-legend").innerHTML = ""; $("#fc-ticks").innerHTML = "";
    $("#fc-table-card").hidden = true; $("#fc-title").textContent = "Forecast"; $("#fc-meta").textContent = ""; $("#fc-mode").hidden = true;
  }
  async function run(custom) {
    const body = { sensor: se.value, time: `${$("#fc-date").value}T${$("#fc-time").value}`, models: ["final", "centralised"] };
    if (custom) body.observations = custom;
    go.disabled = true; go.innerHTML = '<span class="spin" aria-hidden="true"></span>Running';
    if (!last) $("#fc-chart").innerHTML = loading(320);
    try { last = await api("/predict", body); show(last); }
    catch (e) { clear(alertBox(e.message, "No forecast for this choice.")); }
    finally { go.disabled = false; go.textContent = "Run forecast"; }
  }
  go.addEventListener("click", () => run());
  $("#fc-custom-go").addEventListener("click", () => {
    const parts = $("#fc-custom").value.split(",").map((s) => s.trim());
    if (parts.length !== 12) { clear(alertBox(`Enter exactly 12 readings separated by commas; you entered ${parts.length}.`, "Check your readings.")); return; }
    const vals = parts.map((s) => (s === "" ? null : Number(s)));
    if (vals.some((v) => v !== null && (!Number.isFinite(v) || v < 0))) { clear(alertBox("Readings must be non-negative numbers or blank.", "Check your readings.")); return; }
    run(vals);
  });
  function show(r) {
    const models = $("#fc-cmp").checked ? ["final", "centralised"] : ["final"];
    const showActual = $("#fc-act").checked;
    $("#fc-title").textContent = `${r.sensor.id} · ${r.sensor.client_name}`;
    $("#fc-meta").textContent = `${niceDate(r.forecast_start)}, from ${hhmm(r.forecast_start)} · computed in ${r.inference_ms} ms on ${r.device.toUpperCase()}`;
    const mode = $("#fc-mode"); mode.hidden = false;
    mode.className = r.mode === "historical" ? "tag warn" : "tag"; mode.textContent = r.mode === "historical" ? "Historical · held-out period" : "Your readings";
    drawForecast($("#fc-chart"), r, { showActual, models });
    $("#fc-legend").innerHTML = `<span><i style="border-color:${COLORS.history}"></i>Observed, previous hour</span><span><i style="border-color:${COLORS.local}"></i>Local-only GRU (final)</span>`
      + (models.includes("centralised") ? `<span><i class="dash" style="border-color:${COLORS.centralised}"></i>Centralised GRU</span>` : "")
      + (showActual && r.actual ? `<span><i class="dash" style="border-color:${COLORS.actual}"></i>Observed afterwards</span>` : "");
    $("#fc-ticks").innerHTML = forecastTicks(r, showActual);
    const v = r.forecasts.final.values, c = r.forecasts.centralised ? r.forecasts.centralised.values : null, a = r.actual ? r.actual.values : null;
    $("#fc-units").textContent = `${r.units} · ${r.data_label}`;
    $("#fc-table").innerHTML = `<table><thead><tr><th>Time</th><th class="num">Local-only GRU</th>${c ? '<th class="num">Centralised GRU</th>' : ""}${a ? '<th class="num">Observed afterwards</th><th class="num">Final model error</th>' : ""}</tr></thead><tbody>
      ${r.forecast_timestamps.map((t, i) => `<tr><td class="num" style="text-align:left">${hhmm(t)}</td><td class="num">${f(v[i], 1)}</td>${c ? `<td class="num">${f(c[i], 1)}</td>` : ""}${a ? `<td class="num">${f(a[i], 0)}</td><td class="num">${a[i] === null ? "—" : sign(v[i] - a[i], 1)}</td>` : ""}</tr>`).join("")}</tbody></table>`;
    $("#fc-table-card").hidden = false;
  }
  const d0 = state.demo.examples[0];
  cl.value = sensors.find((s) => s.id === d0.sensor).client; fill(); se.value = d0.sensor;
  $("#fc-date").value = d0.time.slice(0, 10); $("#fc-time").value = d0.time.slice(11, 16);
  ex.querySelector("button").setAttribute("aria-pressed", "true");
  run();
}

function renderPerformance(el) {
  const m = state.metrics, pe = m.percentage_error;
  const kind = (k) => (k === "local" ? "final" : BASE_LABEL[k] ? "ref" : "arm");
  const label = (k) => (m.arms[k] ? m.arms[k].label : BASE_LABEL[k]);
  const order = [...ARMS, ...Object.keys(m.baselines)];
  const val = (k, f2) => (m.arms[k] ? f2(m.arms[k]) : f2(m.baselines[k]));
  const fin = pe && pe.arms.local.clipped;
  const scale = 20;
  el.innerHTML = `
    <header class="page-head"><h1>Model performance</h1>
      <p>All four approaches and three simple reference forecasts on the held-out test: ${m.test_windows.toLocaleString("en-US")} windows from 8 August to 30 September 2023, evaluated once. Lower is better everywhere on this page.</p></header>
    <section class="grid cols-3">
      <div class="card"><div class="card-head"><h3>MAE</h3><span class="meta">vehicles / 5 min</span></div><div class="card-body">${bars(order.map((k) => ({ label: label(k), value: val(k, (a) => a.mae), kind: kind(k) })))}</div></div>
      <div class="card"><div class="card-head"><h3>RMSE</h3><span class="meta">penalises large misses</span></div><div class="card-body">${bars(order.map((k) => ({ label: label(k), value: val(k, (a) => a.rmse), kind: kind(k) })))}</div></div>
      <div class="card"><div class="card-head"><h3>WAPE</h3><span class="meta">error as % of traffic</span></div><div class="card-body">${pe ? bars(order.map((k) => ({ label: label(k), value: pe.arms[k] ? pe.arms[k].raw.wape_pct : pe.baselines[k].wape_pct, kind: kind(k) })), { digits: 2, suffix: "%" }) : '<div class="state">Not available.</div>'}</div></div>
    </section>
    ${pe ? `<section class="grid split">
      <div class="card"><div class="card-head"><h3>All results</h3><span class="meta">three-seed means · held-out test</span></div>${resultsTable(m)}</div>
      <div class="callout accent">
        <h3>WAPE target of 6%: not yet reached</h3>
        <p>The final model (local-only GRU, clipped at 0) has a verified WAPE of <b style="color:var(--text)">${pct(fin.wape_pct, 3)}</b>, ${f(pe.target.gap_pct_points, 2)} percentage points above the target. Even its 5-minute forecasts alone sit at ${pct(fin.by_horizon_wape_pct[0], 2)}.</p>
        <div class="meter" role="img" aria-label="WAPE ${f(fin.wape_pct, 2)} percent against a 6 percent target"><div class="fill" style="width:${(100 * fin.wape_pct / scale).toFixed(2)}%"></div><div class="mark" style="left:${(100 * 6 / scale).toFixed(2)}%"></div></div>
        <div class="meter-scale"><span>0%</span><span>10%</span><span>20%</span></div>
        <div class="meter-legend"><span><span class="key" style="background:var(--g2)"></span>Final model ${pct(fin.wape_pct, 2)}</span><span><span class="key" style="background:var(--accent)"></span>Target 6%</span></div>
        <div class="divider"></div>
        <p class="xs subtle">WAPE is total absolute error divided by total observed traffic over ${pe.cells.observed.toLocaleString("en-US")} test cells; sMAPE excludes cells where forecast and observation are both 0. MAPE is not shown because ${pct(100 * pe.cells.actual_zero / pe.cells.observed, 2)} of observations are zero. An MAE of ${f(m.arms.local.mae)} is in vehicles, not percent, and "${pct(m.arms.local.improvement_over_baselines_pct.persistence)} lower than persistence" is an improvement, not an error rate.</p>
      </div></section>` : ""}
    <section class="card">
      <div class="card-head"><div><h3>Error by forecast horizon</h3><span class="meta">MAE at each 5-minute step ahead</span></div></div>
      <div class="card-body"><div id="pf-horizon"></div>
        <div class="legend" style="margin-top:12px">${ARMS.map((k) => `<span><i class="${DASH[k] ? "dash" : ""}" style="border-color:${COLORS[k]}"></i>${esc(m.arms[k].label)}</span>`).join("")}<span><i class="dotted" style="border-color:${COLORS.persistence}"></i>Persistence</span></div>
        ${fin ? `<p class="xs subtle" style="margin-top:12px">Final-model WAPE rises from ${pct(fin.by_horizon_wape_pct[0], 1)} at 5 minutes to ${pct(fin.by_horizon_wape_pct[11], 1)} at 60 minutes; persistence degrades much faster.</p>` : ""}</div>
    </section>
    <section class="grid cols-2">
      <div class="card"><div class="card-head"><h3>By client group</h3><span class="meta">MAE${fin ? " · final-model WAPE" : ""}</span></div><div class="table-wrap"><table><thead><tr><th>Client group</th><th class="num">Local-only</th><th class="num">Centralised</th><th class="num">Geo FedAvg</th>${fin ? '<th class="num">WAPE</th>' : ""}</tr></thead><tbody>
        ${CLIENT_ORDER.map((c) => `<tr><td>${CLIENT_NAME[c]}</td><td class="num">${f(m.arms.local.client[c].mae, 2)}</td><td class="num">${f(m.arms.centralised.client[c].mae, 2)}</td><td class="num">${f(m.arms.fedavg_geo.client[c].mae, 2)}</td>${fin ? `<td class="num">${pct(fin.by_client_wape_pct[c], 2)}</td>` : ""}</tr>`).join("")}</tbody></table></div></div>
      <div class="card"><div class="card-head"><h3>By period</h3><span class="meta">MAE</span></div><div class="table-wrap"><table><thead><tr><th>Period</th><th class="num">Local-only</th><th class="num">Centralised</th><th class="num">Geo FedAvg</th></tr></thead><tbody>
        ${[["Aug 8–31", (a) => a.month["August (8-31)"]], ["September", (a) => a.month.September], ["Weekdays", (a) => a.weekday_weekend.weekday], ["Weekends", (a) => a.weekday_weekend.weekend], ["Validation", (a) => a.validation_mae]]
          .map(([l, g]) => `<tr${l === "Validation" ? ' class="ref first-ref"' : ""}><td>${l}</td><td class="num">${f(g(m.arms.local), 3)}</td><td class="num">${f(g(m.arms.centralised), 3)}</td><td class="num">${f(g(m.arms.fedavg_geo), 3)}</td></tr>`).join("")}</tbody></table></div></div>
    </section>
    <section class="callout">
      <h3>Why the local-only GRU is the final model</h3>
      <p>Before the test was opened, the study fixed its rule: the final model is the approach with the lowest validation MAE, and negative predictions are clipped at 0 only if that lowers validation MAE. Local-only won on validation (${f(m.arms.local.validation_mae)}) and clipping helped, so the test confirmed a choice already made. This app serves seed ${m.deployed.final.seed}, the local-only seed with the lowest validation MAE; its own test MAE is ${f(m.deployed.final.test_mae_this_seed)}.</p>
    </section>`;
  const labels = Array.from({ length: 12 }, (_, i) => `${(i + 1) * 5} min`);
  lineChart($("#pf-horizon"), { labels, yLabel: "MAE", height: 230,
    series: [{ name: "Persistence", color: COLORS.persistence, values: m.baselines.persistence.horizon, dash: DASH.persistence, width: 1.6, digits: 2 },
      ...ARMS.slice().reverse().map((k) => ({ name: m.arms[k].label, color: COLORS[k], values: m.arms[k].horizon, dash: DASH[k], width: k === "local" ? 2.4 : 1.5, digits: 2 }))] });
}

function fedDiagram() {
  const names = CLIENT_ORDER.map((c) => CLIENT_NAME[c]);
  const counts = CLIENT_ORDER.map((c) => state.sensors.filter((s) => s.client === c).length);
  const server = (cx, y, w) => `<rect class="node server" x="${cx - w / 2}" y="${y}" width="${w}" height="56" rx="10"/>
    <text class="title" x="${cx}" y="${y + 24}" text-anchor="middle">Server</text><text x="${cx}" y="${y + 43}" text-anchor="middle">averages weights · never sees data</text>`;
  const client = (x, y, w, i) => `<rect class="node" x="${x - w / 2}" y="${y}" width="${w}" height="62" rx="10"/>
    <text class="title" x="${x}" y="${y + 25}" text-anchor="middle">${names[i]}</text><text x="${x}" y="${y + 44}" text-anchor="middle">${counts[i]} sensors · data stays here</text>`;
  const link = (x1, y1, x2, y2) => `<path class="edge" d="M${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}"/><path class="flow" d="M${x2} ${y2} C ${x2} ${(y1 + y2) / 2}, ${x1} ${(y1 + y2) / 2}, ${x1} ${y1}"/>`;
  const label = "FedAvg: a server exchanges model weights with four clients; raw data stays with each client";
  const xs = [95, 285, 475, 665];
  const wide = `<svg class="diagram wide" viewBox="0 0 760 250" role="img" aria-label="${label}">${xs.map((x) => link(380, 74, x, 166)).join("")}${server(380, 18, 240)}${xs.map((x, i) => client(x, 166, 164, i)).join("")}</svg>`;
  const pos = [[105, 150], [315, 150], [105, 240], [315, 240]];
  const compact = `<svg class="diagram compact" viewBox="0 0 420 312" role="img" aria-label="${label}">${pos.map(([x, y]) => link(210, 70, x, y)).join("")}${server(210, 14, 250)}${pos.map(([x, y], i) => client(x, y, 196, i)).join("")}</svg>`;
  return wide + compact;
}

function renderFederated(el) {
  const m = state.metrics, c = m.contrasts, rt = m.runtime;
  const runs = (arm) => [0, 1, 2].map((s) => rt[`${arm}_seed${s}`]).filter(Boolean);
  const avg = (arm, g) => runs(arm).reduce((t, r) => t + g(r), 0) / Math.max(1, runs(arm).length);
  const geo0 = rt.fedavg_geo_seed0;
  const perRound = geo0 && geo0.units_run ? geo0.communication_mb / geo0.units_run : NaN;
  const verdict = (x) => (x.statistical && x.practical ? '<span class="tag accent">Detectable · meaningful</span>' : x.statistical ? '<span class="tag">Detectable · below threshold</span>' : '<span class="tag">Not distinguishable</span>');
  el.innerHTML = `
    <header class="page-head"><h1>Federated learning</h1>
      <p>Federated learning trains one shared model without moving raw data. Each client trains on its own sensors and sends only model weights to a server, which averages them (FedAvg). The study simulated four clients on one machine and compared this with pooling all data (centralised) and with each client training alone (local-only).</p></header>
    <section class="card">
      <div class="card-head"><h3>How FedAvg works here</h3><span class="meta">${f(perRound, 2)} MB of weights per round, all four clients</span></div>
      <div class="card-body">${fedDiagram()}</div>
    </section>
    <section class="grid cols-4">${CLIENT_ORDER.map((cl) => `<div class="card stat"><span class="stat-label">${CLIENT_NAME[cl]}</span><span class="stat-value">${f(m.arms.local.client[cl].mae, 2)}<small>local-only</small></span><span class="stat-note">Geographic FedAvg ${f(m.arms.fedavg_geo.client[cl].mae, 2)} · ${state.sensors.filter((s) => s.client === cl).length} sensors</span></div>`).join("")}</section>
    <section class="grid split">
      <div class="card"><div class="card-head"><h3>Accuracy trade-offs</h3><span class="meta">three matched seeds · MAE difference</span></div>
        <div class="table-wrap"><table><thead><tr><th>Comparison</th><th class="num">Difference</th><th class="num">95% interval</th><th>Result</th></tr></thead><tbody>
        ${Object.values(c).map((x) => `<tr><td>${esc(m.arms[x.a].label)} vs ${esc(m.arms[x.b].label)}</td><td class="num">${sign(x.diff)}</td><td class="num">${f(x.reported_ci95[0])} to ${f(x.reported_ci95[1])}</td><td>${verdict(x)}</td></tr>`).join("")}
        </tbody></table></div>
        <p class="xs subtle" style="padding:0 20px 16px">Positive means the first model has higher error. "Meaningful" uses the threshold fixed before the test, ${f(m.sesoi_mae, 3)} MAE. Intervals resample whole sensors within client groups.</p></div>
      <div class="callout accent"><h3>Local-only training had the lowest mean error</h3>
        <p>Geographic FedAvg did not beat centralised or local-only training: it cost ${f(c.C1.diff)} MAE against centralised and ${f(c.C2.diff)} against local-only, both above the practical threshold.</p>
        <p>IID FedAvg was ${f(-c.C3.diff)} better than geographic FedAvg, a detectable but small effect of the geographic split. Local-only and centralised could not be told apart (Holm-adjusted p ${f(c.C4.p_holm, 3)}).</p></div>
    </section>
    <section class="grid cols-2">
      <div class="card"><div class="card-head"><h3>One FedAvg round</h3></div><div class="card-body"><ol class="steps">
        <li><span><b>Broadcast.</b> The server sends the current global GRU weights to every client.</span></li>
        <li><span><b>Train locally.</b> Each client trains one epoch on its own sensors; raw data never leaves it.</span></li>
        <li><span><b>Upload.</b> Clients return updated weights only.</span></li>
        <li><span><b>Average.</b> The server averages them, weighted by each client's number of training windows.</span></li>
        <li><span><b>Stop early.</b> Training ends after 5 rounds without validation improvement; the best round is kept.</span></li></ol></div></div>
      <div class="card"><div class="card-head"><h3>Training cost</h3><span class="meta">mean per run, CPU</span></div><div class="table-wrap"><table><thead><tr><th>Approach</th><th class="num">Time</th><th class="num">Model traffic</th><th class="num">Test MAE</th></tr></thead><tbody>
        ${ARMS.map((k) => `<tr><td><span class="model">${key(k)}${esc(m.arms[k].label)}</span></td><td class="num">${f(avg(k, (r) => r.minutes), 0)} min</td><td class="num">${k.startsWith("fedavg") ? `${f(avg(k, (r) => r.communication_mb), 1)} MB` : "none"}</td><td class="num">${f(m.arms[k].mae)}</td></tr>`).join("")}</tbody></table></div>
        <p class="xs subtle" style="padding:0 20px 16px">Clients were simulated in one process, so these figures include no network latency.</p></div>
    </section>`;
}

function renderMethodology(el) {
  const m = state.metrics;
  el.innerHTML = `
    <header class="page-head"><h1>Methodology</h1>
      <p>A pre-registered, chronological evaluation. Models were chosen on validation data, frozen and hashed, then tested once on a later period none of them had seen.</p></header>
    <section class="card"><div class="card-head"><h3>Data and split</h3><span class="meta">XTraffic 2023 · January–September · 5-minute flow</span></div>
      <div class="card-body grid" style="gap:16px">
        <div class="timeline" role="img" aria-label="Train 191 days, validation 28 days, test 54 days">
          <div style="flex:191"><b>Train · 191 days</b>Jan 1 – Jul 10</div><div style="flex:28"><b>Validation</b>28 days</div><div class="test" style="flex:54"><b>Test · 54 days</b>Aug 8 – Sep 30</div></div>
        <p class="small muted">200 sensors, 50 per client group, chosen with training-period statistics only. Each forecast window uses 12 past steps (60 minutes) to predict 12 future steps, and inputs and targets always stay inside one split. October–December were never opened.</p></div></section>
    <section class="grid cols-2">
      <div class="card"><div class="card-head"><h3>Model</h3></div><div class="card-body"><dl class="kv">
        <dt>Architecture</dt><dd>GRU, 64 hidden units, linear head</dd><dt>Input</dt><dd>12 steps × 4 features</dd><dt>Features</dt><dd>scaled flow, observed flag, time-of-day sin/cos</dd><dt>Output</dt><dd>12 steps at once (60 min)</dd><dt>Scaling</dt><dd>per sensor, training period only</dd><dt>Missing data</dt><dd>inputs filled from the past; targets masked</dd></dl></div></div>
      <div class="card"><div class="card-head"><h3>Training and selection</h3></div><div class="card-body"><dl class="kv">
        <dt>Optimiser</dt><dd>Adam, learning rate 0.001</dd><dt>Batch</dt><dd>512, gradient clip 1.0</dd><dt>Budget</dt><dd>≤ 50 epochs or rounds, patience 5</dd><dt>Runs</dt><dd>4 approaches × 3 seeds = 12, all completed</dd><dt>Selection</dt><dd>lowest validation MAE, fixed before test</dd><dt>Test windows</dt><dd>${m.test_windows.toLocaleString("en-US")}</dd></dl></div></div>
    </section>
    <section class="grid cols-2">
      <div class="card"><div class="card-head"><h3>Evaluation protocol</h3></div><div class="card-body"><ol class="steps">
        <li><span><b>Freeze.</b> Every completed checkpoint, the data and the code were hashed.</span></li>
        <li><span><b>Pre-check.</b> Each checkpoint reproduced its saved validation predictions byte for byte.</span></li>
        <li><span><b>Test once.</b> All 12 runs and three reference forecasts on the held-out period.</span></li>
        <li><span><b>Compare.</b> Matched seeds, sensor-level bootstrap within clients, freeway-cluster check, Holm correction.</span></li></ol></div></div>
      <div class="card"><div class="card-head"><h3>Limitations</h3></div><div class="card-body"><ul class="bullets">
        <li><b>Provisional geography.</b> Client labels come from a sensor-ID mapping that is supported but not confirmed.</li>
        <li><b>Few ramps.</b> 10 of 200 sensors: few ramps meet the 99% coverage rule over six months.</li>
        <li><b>Historical only.</b> One dataset, evaluated offline; not a real-time deployment validation.</li>
        <li><b>Simulated federation.</b> Communication is counted, not measured on a network.</li>
        <li><b>One configuration.</b> The same training settings for every approach, without separate tuning.</li></ul></div></div>
    </section>`;
}

function renderAbout(el) {
  const h = state.health;
  const prov = state.models.map((x) => `<tr><td><span class="model">${key(x.key === "final" ? "local" : "centralised")}${esc(x.label)}${x.key === "final" ? ' <span class="tag accent">Final</span>' : ""}</span></td><td class="num">${x.seed}</td><td class="num">${f(x.validation_mae)}</td><td class="num">${f(x.test_mae_this_seed)}</td><td>${x.key === "final" ? "own client's model per sensor; clipped at 0" : "one model for all sensors"}</td><td><code>${esc(Object.values(x.checkpoint_sha256)[0])}</code>${Object.keys(x.checkpoint_sha256).length > 1 ? ` <span class="xs subtle">+${Object.keys(x.checkpoint_sha256).length - 1}</span>` : ""}</td></tr>`).join("");
  el.innerHTML = `
    <header class="page-head"><h1>About</h1>
      <p>AIORI PS-003: federated traffic-flow prediction with communication-cost evaluation, built for AIORI Hackathon 3.</p></header>
    <section class="pipeline" aria-label="How the app works">
      <div class="pipe"><span class="step-no">Artifacts</span><h3>Frozen research outputs</h3><p>Phase 9 checkpoints, training-only scaling and the single test report, verified by hash.</p></div>
      <div class="pipe"><span class="step-no">Inference</span><h3>Python + PyTorch</h3><p>Rebuilds the four training features and runs the GRU on ${esc((h.device || "cpu").toUpperCase())} in a few milliseconds.</p></div>
      <div class="pipe"><span class="step-no">API</span><h3>Validated JSON</h3><p>Anonymised sensor ids, input validation, no paths or station identifiers.</p></div>
      <div class="pipe"><span class="step-no">Interface</span><h3>This dashboard</h3><p>Plain HTML, CSS and JavaScript; hand-built SVG charts; no third-party scripts.</p></div>
    </section>
    <section class="card"><div class="card-head"><h3>Deployed models</h3><span class="meta">selected on validation only</span></div>
      <div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Seed</th><th class="num">Validation MAE</th><th class="num">Test MAE (seed)</th><th>Routing</th><th>Checkpoint SHA-256</th></tr></thead><tbody>${prov}</tbody></table></div></section>
    <section class="grid cols-2">
      <div class="card"><div class="card-head"><h3>API</h3></div><div class="card-body"><dl class="kv">
        <dt><code>GET /health</code></dt><dd>service status and device</dd><dt><code>GET /models</code></dt><dd>model provenance</dd><dt><code>GET /sensors</code></dt><dd>200 anonymised sensors</dd>
        <dt><code>GET /metrics</code></dt><dd>verified aggregate results</dd><dt><code>GET /demo</code></dt><dd>historical examples</dd><dt><code>POST /predict</code></dt><dd>12-step forecast</dd></dl></div></div>
      <div class="card"><div class="card-head"><h3>Reproduce</h3></div><div class="card-body prose">
        <p>Start the service from the repository root with <code>.venv\\Scripts\\python.exe app\\backend\\server.py</code> and open <code>http://127.0.0.1:8000</code>.</p>
        <p>Run the tests with <code>.venv\\Scripts\\python.exe -m pytest app\\tests</code>. Percentage errors come from <code>app/analysis/percentage_error.py</code>, read-only on the frozen test predictions.</p>
        <p>Source: <a href="${REPO}" target="_blank" rel="noopener">${REPO.replace("https://", "")}</a>. Data: XTraffic 2023 (CC BY-NC 4.0); raw data, checkpoints and station identifiers are not published.</p></div></div>
    </section>
    <section class="callout"><h3>Acknowledgments</h3><p>Built on the XTraffic dataset and the AIORI Hackathon 3 problem statement PS-003. These are research findings from one historical dataset; no claim of state-of-the-art accuracy is made, because no comparable published benchmark uses this exact protocol.</p></section>`;
}

/* ---------- routing ---------- */
const VIEWS = { overview: renderOverview, forecast: renderForecast, performance: renderPerformance, federated: renderFederated, methodology: renderMethodology, about: renderAbout };
function setDrawer(open) {
  $("#sidebar").classList.toggle("open", open); $("#scrim").classList.toggle("show", open);
  $("#menu-btn").setAttribute("aria-expanded", String(open));
}
async function route() {
  const name = (location.hash || "#overview").slice(1);
  const k = VIEWS[name] ? name : "overview";
  document.querySelectorAll(".nav-link").forEach((a) => { if (a.getAttribute("href") === `#${k}`) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== `view-${k}`; });
  $("#crumb").textContent = TITLES[k];
  $("#top-cta").hidden = k === "forecast" || k === "overview";
  document.title = `${TITLES[k]} · AIORI Traffic Intelligence`;
  setDrawer(false);
  window.scrollTo(0, 0);
  const el = $(`#view-${k}`);
  if (!state.metrics) { if (!el.innerHTML) el.innerHTML = loading(420); return; }
  if (!state.rendered[k]) {
    state.rendered[k] = true;
    try { await VIEWS[k](el); } catch (e) { el.innerHTML = alertBox(e.message); state.rendered[k] = false; }
  } else {
    for (const [w, draw] of CHARTS) if (el.contains(w)) draw();
  }
  const h = el.querySelector("h1"); if (h) { h.setAttribute("tabindex", "-1"); h.focus({ preventScroll: true }); }
}

async function boot() {
  $("#menu-btn").addEventListener("click", () => setDrawer(!$("#sidebar").classList.contains("open")));
  $("#scrim").addEventListener("click", () => setDrawer(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") setDrawer(false); });
  window.addEventListener("hashchange", route);
  route();
  try {
    const [health, metrics, sensors, demo, models] = await Promise.all([api("/health"), api("/metrics"), api("/sensors"), api("/demo"), api("/models")]);
    Object.assign(state, { health, metrics, sensors, demo, models });
    $("#api-status").innerHTML = `<span class="dot ok"></span>Model service ready · ${esc(health.device.toUpperCase())}`;
    document.querySelectorAll(".view").forEach((v) => { if (v.hidden) v.innerHTML = ""; });
    route();
  } catch (e) {
    $("#api-status").innerHTML = `<span class="dot bad"></span>Model service unavailable`;
    document.querySelectorAll(".view").forEach((v) => { v.innerHTML = alertBox(e.message, "The model service is not available."); });
  }
}
boot();
