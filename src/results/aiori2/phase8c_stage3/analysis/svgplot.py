"""Minimal SVG plotting (categorical x, linear y). Used only by the Phase 8C analysis scripts."""
from xml.sax.saxutils import escape


class Plot:
    W, H, L, R, T, B = 900, 420, 70, 20, 40, 110

    def __init__(self, title, ylabel, n, ymin, ymax, xlabels, legend=None):
        self.n, self.ymin, self.ymax, self.e = n, ymin, ymax, []
        pw, ph = self.W - self.L - self.R, self.H - self.T - self.B
        self.x = lambda i: self.L + (i + .5) * pw / n
        self.y = lambda v: self.T + ph * (1 - (v - ymin) / (ymax - ymin))
        e = self.e
        e.append(f'<text x="{self.W/2}" y="22" text-anchor="middle" font-size="14">{escape(title)}</text>')
        e.append(f'<text transform="translate(16,{self.T+ph/2}) rotate(-90)" text-anchor="middle" font-size="12">{escape(ylabel)}</text>')
        steps = 8
        for k in range(steps + 1):
            v = ymin + (ymax - ymin) * k / steps
            e.append(f'<line x1="{self.L}" x2="{self.W-self.R}" y1="{self.y(v):.1f}" y2="{self.y(v):.1f}" stroke="#ddd"/>')
            e.append(f'<text x="{self.L-6}" y="{self.y(v)+4:.1f}" text-anchor="end" font-size="10">{v:.2f}</text>')
        for i, t in enumerate(xlabels):
            e.append(f'<text transform="translate({self.x(i):.1f},{self.H-self.B+12}) rotate(35)" font-size="10">{escape(t)}</text>')
        for k, (name, col) in enumerate((legend or {}).items()):
            e.append(f'<rect x="{self.W-200}" y="{self.T+6+k*16}" width="10" height="10" fill="{col}"/>'
                     f'<text x="{self.W-185}" y="{self.T+15+k*16}" font-size="10">{escape(name)}</text>')

    def dot(self, i, v, col, r=3, op=1):
        self.e.append(f'<circle cx="{self.x(i):.1f}" cy="{self.y(v):.1f}" r="{r}" fill="{col}" fill-opacity="{op}"/>')

    def errbar(self, i, m, sd, col):
        x = self.x(i)
        self.e.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{self.y(m-sd):.1f}" y2="{self.y(m+sd):.1f}" stroke="{col}" stroke-width="2"/>')
        self.e.append(f'<rect x="{x-12:.1f}" y="{self.y(m)-1.5:.1f}" width="24" height="3" fill="{col}"/>')

    def bar(self, xi, v, w, col):
        px = self.x(0) - self.x(-1)
        x0, y0 = self.x(xi) - w * px / 2, self.y(max(v, self.ymin))
        self.e.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w*px:.1f}" height="{self.y(self.ymin)-y0:.1f}" fill="{col}"/>')

    def line(self, xs, ys, col):
        pts = " ".join(f"{self.x(i):.1f},{self.y(min(max(v, self.ymin), self.ymax)):.1f}" for i, v in zip(xs, ys))
        self.e.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.2" stroke-opacity=".75"/>')

    def vline(self, i, col):
        self.e.append(f'<line x1="{self.x(i):.1f}" x2="{self.x(i):.1f}" y1="{self.T}" y2="{self.H-self.B}" stroke="{col}" stroke-dasharray="3 3"/>')

    def save(self, path):
        body = "\n".join(self.e)
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.W} {self.H}" font-family="sans-serif">'
                        f'<rect width="100%" height="100%" fill="#fff"/>{body}</svg>', encoding="utf-8")
