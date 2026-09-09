"""Redraw the infographic's bottom-right panel with the real GEMINGA project content,
matching the sketch/graph-paper style of the rest of the figure."""
import math, random
random.seed(7)

W, H = 1310, 730
PAPER, GRID = "#FFFCF3", "#D9DFE7"
INK, RED, YEL, BLU = "#1A1A1A", "#E8412F", "#F5C518", "#2E77C4"
F = "DejaVu Sans"


def jline(x1, y1, x2, y2, amp=1.6, seg=9):
    """One hand-drawn stroke from (x1,y1) to (x2,y2)."""
    pts = []
    for i in range(seg + 1):
        t = i / seg
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if 0 < i < seg:
            x += random.uniform(-amp, amp)
            y += random.uniform(-amp, amp)
        pts.append((x, y))
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts)):
        px, py = pts[i - 1]
        cx, cy = pts[i]
        d += f" Q{(px+cx)/2:.1f},{(py+cy)/2:.1f} {cx:.1f},{cy:.1f}"
    return d


def jrect(x, y, w, h, sw=3.4, stroke=INK, fill="none", amp=1.6, passes=1, rx=0):
    out = []
    if fill != "none":
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>')
    for _ in range(passes):
        for a, b, c, d_ in ((x, y, x + w, y), (x + w, y, x + w, y + h),
                            (x + w, y + h, x, y + h), (x, y + h, x, y)):
            out.append(f'<path d="{jline(a,b,c,d_,amp)}" fill="none" stroke="{stroke}" '
                       f'stroke-width="{sw}" stroke-linecap="round"/>')
    return "".join(out)


def hatch(x, y, w, h, color=RED, step=15, sw=3.0):
    out = []
    for i in range(-int(h), int(w), step):
        x1, y1 = x + i, y + h
        x2, y2 = x + i + h, y
        x1c, x2c = max(x, x1), min(x + w, x2)
        if x2c <= x1c:
            continue
        t1 = (x1c - x1) / max(h, 1)
        t2 = (x2c - x1) / max(h, 1)
        out.append(f'<path d="{jline(x1c, y+h-t1*h, x2c, y+h-t2*h, 1.0, 3)}" fill="none" '
                   f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" opacity="0.9"/>')
    return "".join(out)


def text(x, y, s, size=30, weight="bold", fill=INK, anchor="start", tilt=None, ls=0):
    tilt = random.uniform(-0.5, 0.5) if tilt is None else tilt
    return (f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}" '
            f'transform="rotate({tilt:.2f} {x} {y})">{s}</text>')


p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
p.append(f'<rect width="{W}" height="{H}" fill="{PAPER}"/>')
g = 23.5
p.append('<g stroke="%s" stroke-width="1" opacity="0.75">' % GRID)
x = 0.0
while x < W:
    p.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{H}"/>')
    x += g
y = 0.0
while y < H:
    p.append(f'<line x1="0" y1="{y:.1f}" x2="{W}" y2="{y:.1f}"/>')
    y += g
p.append('</g>')

# outer panel frame
p.append(jrect(16, 54, W - 34, H - 74, sw=4.0, amp=2.0))

# header tab
p.append(jrect(34, 14, 322, 62, sw=3.6, fill=PAPER))
p.append(text(52, 58, "GITHUB PROJECT", 34, ls=1))

# title band
p.append(hatch(62, 96, 62, 92))
p.append(hatch(1186, 96, 62, 92))
p.append(jrect(132, 96, 1046, 92, sw=4.2, stroke=RED, fill=PAPER))
p.append(text(655, 166, "PROJECT GEMINGA", 62, anchor="middle", ls=2, tilt=-0.2))
p.append(text(655, 224,
              "GATED · EXTENSIBLE · MEMORY-BUDGETED INGESTION",
              27, weight="bold", fill=BLU, anchor="middle", ls=1.2, tilt=0.2))
p.append(text(655, 256, "for N-MODAL GENOMICS &amp; ARRAYS",
              25, weight="normal", fill=BLU, anchor="middle", ls=1, tilt=-0.15))

# ---------------- left column ----------------
rows = [
    ("CODE", YEL, ["budget.rs — the governor", "fastx.rs · reader.rs · hub.rs", "formats.py — h5ad, TIFF, IPC"]),
    ("DATA", BLU, [".fastq.gz  .h5ad  .parquet", ".arrow  .ome.tif  .zarr", "local · range · Hub"]),
    ("DOCS", RED, ["README · CONTRIBUTING", "CITATION.cff · examples/"]),
    ("PEOPLE", BLU, ["memory_bug.yml", "format_request.yml", "discussions"]),
]
y = 306
for label, col, items in rows:
    p.append(jrect(58, y - 34, 52, 52, sw=3.0, fill=col, amp=1.2, rx=6))
    if label == "CODE":
        p.append(text(84, y + 4, "&lt;/&gt;", 22, fill=INK, anchor="middle"))
    elif label == "DATA":
        for k in range(3):
            p.append(f'<rect x="70" y="{y-26+k*14}" width="28" height="9" fill="{PAPER}" opacity="0.9"/>')
    elif label == "DOCS":
        for k in range(3):
            p.append(f'<rect x="70" y="{y-24+k*13}" width="28" height="5" fill="{PAPER}" opacity="0.95"/>')
    else:
        for cx, cy, r in ((75, y-18, 7), (95, y-18, 7), (85, y+2, 8)):
            p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{PAPER}" opacity="0.97"/>')
    p.append(text(128, y + 8, label, 34, ls=1))
    ty = y - 14
    for it in items:
        p.append(f'<circle cx="318" cy="{ty-8}" r="4.5" fill="{INK}"/>')
        p.append(text(334, ty, it, 23, weight="normal"))
        ty += 32
    y += 108

# ---------------- right column: measured, not promised ----------------
RX, RY = 726, 300
p.append(text(RX, RY, "MEASURED, NOT PROMISED", 32, ls=0.5))
p.append(f'<path d="{jline(RX, RY+10, RX+466, RY+10, 1.4)}" fill="none" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')

bars = [("DATA STREAMED", 210, BLU, "210 MB"),
        ("BUDGET DECLARED", 256, YEL, "256 MB"),
        ("PEAK RESIDENT", 18, RED, "18 MB")]
scale = 430 / 256
by = RY + 52
for name, val, col, lab in bars:
    p.append(text(RX, by, name, 22, weight="bold"))
    w = max(val * scale, 10)
    p.append(f'<rect x="{RX}" y="{by+10}" width="{w:.0f}" height="30" fill="{col}" opacity="0.85"/>')
    p.append(jrect(RX, by + 10, w, 30, sw=2.6, amp=1.1))
    p.append(text(RX + w + 14, by + 34, lab, 23, weight="bold", fill=INK))
    by += 76

p.append(jrect(RX, by + 10, 466, 96, sw=3.0, fill=PAPER, amp=1.4))
p.append(text(RX + 20, by + 50, "5 formats · 1 process · 0 denials", 25, fill=INK))
p.append(text(RX + 20, by + 84, "8 GB laptop, same ceiling", 23, weight="normal", fill=BLU))

p.append('</svg>')
open("assets/panel.svg", "w").write("".join(p))
print("panel.svg written")

# Composite over the artwork:
#   python tools/build_panel.py
#   python -c "from PIL import Image; b=Image.open('assets/_source.jpg'); \
#     b.paste(Image.open('assets/panel.png'), (1470,770)); b.save('assets/infographic.jpg', quality=93)"
