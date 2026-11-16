"""AIORI app with the Historical Traffic Replay feature (Phase 9A, experimental). Isolated from production:

    .venv\\Scripts\\python.exe src\\app\\backend\\replay_server.py        # http://127.0.0.1:8010/replay

It serves every original route unchanged (it reuses server.Engine and server.Handler) plus GET /api/replay/*, and the
front-end build in app/web/dist-replay (built with VITE_REPLAY=1). The production entry point (server.py on port 8000,
the keep-alive task and the tunnel) is not modified and does not expose the replay routes; to roll back, stop this
process (and optionally delete dist-replay).

Replay endpoints (GET, query parameters only):
  /api/replay/meta                                    coverage, dates, periods, model provenance, data notes
  /api/replay/sensors                                 200 sensors with provisional coordinates
  /api/replay/day?date=YYYY-MM-DD                     per-slot count of sensors with a valid reading
  /api/replay/frame?time=..&layer=observed|forecast[&partition=geo|iid&horizon=1..12]
  /api/replay/series?sensor=LA-01&date=YYYY-MM-DD     one sensor's readings for a day
  /api/replay/forecast?sensor=LA-01&time=..[&partition=geo|iid]

Historical Disruption Explorer (Phase 9B; offline catalog from disruption_detect.py, loaded once at start-up):
  /api/replay/disruptions/meta                        catalog version, methods, thresholds, summary, limitations
  /api/replay/disruptions?from=&to=&region=&category=&direction=&sensor=&min_sensors=&min_duration=&max_duration=
                         &min_peak_z=&confidence=&quality=&holiday=&sort=&page=&page_size=   (bounded, paginated)
  /api/replay/disruptions/<id>                        one candidate event
  /api/replay/disruptions/<id>/series?sensor=LA-01    observed flow, typical flow and thresholds around the event
  /api/replay/disruptions/<id>/compare                affected sensors on one normalised axis
  /api/replay/disruptions/<id>/robustness?sensor=LA-01[&partition=geo|iid]   exploratory forecast-error comparison
"""

import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
DIST = HERE.parent / "web" / "dist-replay"
if (DIST / "index.html").is_file():
    os.environ.setdefault("AIORI_FRONTEND_DIR", str(DIST))     # must be set before server.py reads it
sys.path.insert(0, str(HERE))

import server as S  # noqa: E402
from replay import Replay, ReplayError  # noqa: E402
from disruptions import ID_RE, Disruptions, Unavailable  # noqa: E402

REPLAY = None
DISRUPTIONS = None                                             # Disruptions instance, or the reason it is unavailable (str)
EVENT_SUB = {"": set(), "series": {"sensor"}, "compare": set(), "robustness": {"sensor", "partition"}}
ALLOWED = {"meta": set(), "sensors": set(), "day": {"date"}, "frame": {"time", "layer", "partition", "horizon"},
           "series": {"sensor", "date"}, "forecast": {"sensor", "time", "partition"}}


def _query(qs, name):
    q = parse_qs(qs, keep_blank_values=True, max_num_fields=10)
    extra = set(q) - ALLOWED[name]
    if extra:
        raise ReplayError(400, f"unknown parameter(s): {', '.join(sorted(extra))[:80]}")
    if any(len(v) > 1 for v in q.values()):
        raise ReplayError(400, "each parameter may appear once")
    return {k: v[0] for k, v in q.items()}


def disruption_route(rest, qs):
    if isinstance(DISRUPTIONS, str):
        raise ReplayError(503, DISRUPTIONS)
    q = parse_qs(qs, keep_blank_values=True, max_num_fields=20)
    if any(len(v) > 1 for v in q.values()):
        raise ReplayError(400, "each parameter may appear once")
    p = {k: v[0] for k, v in q.items()}
    if rest == "":
        return DISRUPTIONS.list(p)
    if rest == "meta":
        if p:
            raise ReplayError(400, "meta takes no parameters")
        return DISRUPTIONS.meta()
    eid, _, sub = rest.partition("/")
    if not ID_RE.fullmatch(eid) or sub not in EVENT_SUB:
        raise ReplayError(404, "Not found")
    extra = set(p) - EVENT_SUB[sub]
    if extra:
        raise ReplayError(400, f"unknown parameter(s): {', '.join(sorted(extra))[:80]}")
    if sub == "":
        return DISRUPTIONS.detail(eid)
    if sub == "compare":
        return DISRUPTIONS.compare(eid)
    if "sensor" not in p:
        raise ReplayError(400, "sensor is required")
    if sub == "series":
        return DISRUPTIONS.series(eid, p["sensor"])
    return DISRUPTIONS.robustness(eid, p["sensor"], p.get("partition", "geo"))


def route(name, qs):
    p = _query(qs, name)
    if name == "meta":
        return REPLAY.meta()
    if name == "sensors":
        return REPLAY.sensors
    if name == "day":
        return REPLAY.day(p.get("date"))
    if name == "series":
        return REPLAY.series(p.get("sensor"), p.get("date"))
    if name == "forecast":
        return REPLAY.forecast(p.get("sensor"), p.get("time"), p.get("partition", "geo"))
    h = p.get("horizon", "12")
    if not h.isdigit():
        raise ReplayError(400, "horizon must be 1 to 12 steps")
    return REPLAY.frame(p.get("time"), p.get("layer", "observed"), p.get("partition", "geo"), int(h))


class Handler(S.Handler):
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path.rstrip("/")
        if not path.startswith("/api/replay/"):
            return super().do_GET()
        name = path[len("/api/replay/"):]
        if name == "disruptions" or name.startswith("disruptions/"):
            try:
                return self._send(200, disruption_route(name[len("disruptions/"):] if "/" in name else "", u.query))
            except ReplayError as e:
                return self._err(e.status, e.message)
            except Exception:  # noqa: BLE001 - never leak internals
                return self._err(500, "Internal error")
        if name not in ALLOWED:
            return self._err(404, "Not found")
        try:
            return self._send(200, route(name, u.query))
        except ReplayError as e:
            return self._err(e.status, e.message)
        except Exception:  # noqa: BLE001 - never leak internals
            return self._err(500, "Internal error")


def make_server(host=None, port=None):
    global REPLAY, DISRUPTIONS
    S.ENGINE = S.ENGINE or S.Engine()
    REPLAY = REPLAY or Replay()
    if DISRUPTIONS is None:
        try:
            DISRUPTIONS = Disruptions(REPLAY)
        except Unavailable as e:                                 # replay keeps working; only the explorer reports 503
            DISRUPTIONS = str(e)
    host = host or os.environ.get("AIORI_HOST", "127.0.0.1")
    port = int(port if port is not None else os.environ.get("AIORI_REPLAY_PORT", "8010"))
    return ThreadingHTTPServer((host, port), Handler)


def main():
    srv = make_server()
    print(f"AIORI + Historical Traffic Replay on http://{srv.server_address[0]}:{srv.server_address[1]}/replay "
          f"(replay data loaded in {REPLAY.load_seconds}s, device {REPLAY.device}; frontend {S.FRONTEND}); disruption catalog: "
          f"{DISRUPTIONS if isinstance(DISRUPTIONS, str) else f'{len(DISRUPTIONS.events)} events, loaded in {DISRUPTIONS.load_seconds}s'}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
