"""Two images: where you are in a dataset, and what it is costing you.

    python examples/hud_figures.py --gef fixtures/chip.gef --out figures/
"""
import argparse
import os
import numpy as np
from PIL import Image
import geminga as g


def zoom_strip(gef, out, w=760, h=560, budget="64MB", points="8MB"):
    """Three zoom levels side by side: the minimap box shrinks, the ladder moves, the
    budget bar follows. Nothing about the layout is hand-placed per panel."""
    shots = []
    for frac, cap in ((1.0, "8MB"), (0.30, "8MB"), (0.06, "8MB")):
        b = g.Budget(ram=budget, vram="256MB", spill="1GB")
        st = g.Stage(w, h, point_budget=cap)
        v = g.open_gef(gef, budget=b).reset()
        ex = v.gef.extent
        cx, cy = (ex[0] + ex[1]) / 2, (ex[2] + ex[3]) / 2
        half = (ex[1] - ex[0]) * frac / 2
        win = (int(cx - half), int(cx + half), int(cy - half), int(cy + half))
        st.show(v, window=win)
        st.hud(v, budget=b)
        shots.append(np.asarray(st.render())[:, :, :3])
        v.gef.close()
    gap = 12
    strip = Image.new("RGB", (w * 3 + gap * 2, h), (240, 238, 232))
    for i, arr in enumerate(shots):
        strip.paste(Image.fromarray(arr), (i * (w + gap), 0))
    p = os.path.join(out, "hud_zoom.png")
    strip.save(p)
    return p


def workspace_shot(gef, h5, h5ad, out, w=880, h=620):
    """One stage, one workspace: every cell's partition and the peak it reached in it."""
    ws = g.workspace(ram="256MB", name="glioma session")
    for name, path, slice_ in (("stereo", gef, "96MB"), ("cite", h5, "64MB"),
                               ("atlas", h5ad, "48MB")):
        if path and os.path.exists(path):
            ws.add(name, path, ram=slice_, note=os.path.basename(path))
    for c in list(ws):
        try:
            if c.name == "stereo":
                c.view.reset().auto_bin()
            elif c.name == "cite":
                c.view.select_frac(0.0, 1.0)
            else:
                c.view.select_frac(0.0, 0.3)
            c.load()
        except Exception:
            pass
    st = g.Stage(w, h, point_budget="8MB")
    cell = ws["stereo"]
    v = cell.view.reset()
    ex = v.gef.extent
    cx, cy = (ex[0] + ex[1]) / 2, (ex[2] + ex[3]) / 2
    half = (ex[1] - ex[0]) * 0.22 / 2
    st.show(v, window=(int(cx - half), int(cx + half), int(cy - half), int(cy + half)))
    st.hud(v, workspace=ws)
    p = os.path.join(out, "hud_workspace.png")
    st.render_png(p)
    ws.close()
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gef", default="/tmp/gem/chip.gef")
    ap.add_argument("--h5", default="/tmp/gem/glioma/filtered_feature_bc_matrix.h5")
    ap.add_argument("--h5ad", default="/tmp/gem/sc.h5ad")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(zoom_strip(a.gef, a.out))
    print(workspace_shot(a.gef, a.h5, a.h5ad, a.out))
