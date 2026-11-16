import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./replay.css";
import { useEffect, useRef } from "react";
import { num } from "../../lib/format";
import { NO_DATA, relColor, STATUS_TEXT, type Frame, type ReplaySensor } from "../../lib/replay";
import { token, useTheme } from "../../lib/theme";

type Props = { sensors: ReplaySensor[]; frame: Frame | null; selected: string | null; onSelect: (id: string) => void };

/** Leaflet map of the study sensors that have usable coordinates. One canvas layer; markers are restyled per frame,
 * never recreated, so playback only changes colours. */
export function ReplayMap({ sensors, frame, selected, onSelect }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const markers = useRef(new Map<number, L.CircleMarker>());
  const latest = useRef({ frame, selected, onSelect });
  latest.current = { frame, selected, onSelect };
  const theme = useTheme();

  useEffect(() => {
    if (!box.current) return;
    const m = L.map(box.current, { preferCanvas: true, zoomControl: true, attributionControl: true });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      className: "replay-tiles",
    }).addTo(m);
    const renderer = L.canvas({ padding: 0.3 });
    const pts: L.LatLngExpression[] = [];
    sensors.forEach((s, i) => {
      if (s.lat === null || s.lng === null) return;                       // never placed without verified coordinates
      const mk = L.circleMarker([s.lat, s.lng], { renderer, radius: 5, weight: 1, color: token("marker"), fillOpacity: 0.95, fillColor: NO_DATA })
        .bindTooltip(() => tip(s, i, latest.current.frame), { direction: "top", offset: [0, -4], className: "replay-tip" })
        .on("click", () => latest.current.onSelect(s.id))
        .addTo(m);
      markers.current.set(i, mk);
      pts.push([s.lat, s.lng]);
    });
    if (pts.length) m.fitBounds(L.latLngBounds(pts), { padding: [24, 24] });
    map.current = m;
    const ro = new ResizeObserver(() => m.invalidateSize());
    ro.observe(box.current);
    return () => {
      ro.disconnect();
      m.remove();
      markers.current.clear();
      map.current = null;
    };
  }, [sensors]);

  useEffect(() => {
    // Canvas markers cannot read CSS variables, so outline colours come from the tokens here (and again on theme change).
    const [fg, marker, lineStrong, fg3] = ["fg", "marker", "line-strong", "fg-3"].map(token);
    markers.current.forEach((mk, i) => {
      const s = sensors[i];
      const st = frame?.status[i];
      const has = frame !== null && frame.values[i] !== null;
      const sel = s.id === selected;
      mk.setStyle({
        fillColor: has ? relColor(frame!.relative[i]) : NO_DATA,
        fillOpacity: has ? 0.95 : 0.15,
        color: sel ? fg : has ? marker : st === undefined ? lineStrong : fg3,
        weight: sel ? 2.5 : 1,
        dashArray: has || st === undefined ? undefined : "2 2",
      });
      mk.setRadius(sel ? 8 : 5);
      if (sel) mk.bringToFront();
    });
  }, [frame, selected, sensors, theme]);

  return <div ref={box} className="isolate h-[460px] w-full overflow-hidden rounded-card border border-line bg-bg sm:h-[540px]" role="region" aria-label="Map of study sensors (provisional locations)" />;
}

function tip(s: ReplaySensor, i: number, f: Frame | null): string {
  const head = `<b>${s.id}</b> · ${esc(s.region)} · ${esc(s.type_group)}${s.freeway ? ` · ${esc(s.freeway)}` : ""}`;
  if (!f) return head;
  const v = f.values[i];
  const what = f.layer === "forecast" ? `Forecast for ${f.target_time?.slice(11) ?? "—"}` : `Observed ${f.time.slice(11)}`;
  const body = v === null ? STATUS_TEXT[f.status[i]] : `${num(v, 0)} vehicles / 5 min${f.relative[i] !== null ? ` · ${num(f.relative[i]!, 2)}× typical` : ""}`;
  return `${head}<br/>${what}: ${body}${s.shares_location ? "<br/><i>Shares a location with another sensor</i>" : ""}`;
}

const esc = (t: string) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
