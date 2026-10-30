"""A scripted navigation session, rendered headless: a filmstrip and an animated GIF.

    python examples/explorer_session.py --gef fixtures/chip.gef --out figures/
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import geminga as g
from geminga import Event


def session(gef, out, w=720, h=540, point_budget="8MB"):
    b = g.Budget(ram="96MB", vram="256MB", spill="1GB")
    ex = g.explore(gef, budget=b, width=w, height=h, point_budget=point_budget)
    f = ex.full
    hot = (f[0] + (f[1] - f[0]) * 0.62, f[2] + (f[3] - f[2]) * 0.66)

    def px(wx, wy):
        cam = ex.stage.camera; sx, sy = cam.scale
        return ((wx - cam.cx) * sx + 1) / 2 * w, (1 - (wy - cam.cy) * sy) / 2 * h

    top = ex.view.gene_profile(1)[0][0]
    steps, captions = [], []
    def add(ev, cap): steps.append(ev); captions.append(cap)

    frames = [(None, ex.frame())]; caps = ["whole chip, fitted"]
    x, y = px(*hot)
    for i in range(5):
        ex.handle(Event("scroll", x=x, y=y, steps=1)); frames.append((None, ex.frame()))
        caps.append(f"scroll in x{i + 1} toward the lobe")
    ex.bookmark("lobe")
    for i in range(6):
        ex.handle(Event("drag", dx=-22, dy=10)); frames.append((None, ex.frame()))
        caps.append(f"drag {i + 1}")
    for i in range(6):
        ex.handle(Event("scroll", x=w / 2, y=h / 2, steps=1)); frames.append((None, ex.frame()))
        caps.append(f"deeper x{i + 1}")
    ex.gene(top); frames.append((None, ex.frame())); caps.append(f"only {top}")
    ex.gene(None); ex.recall("lobe"); frames.append((None, ex.frame())); caps.append("recall 'lobe'")
    ex.fit(); frames.append((None, ex.frame())); caps.append("fit")

    # filmstrip: six moments that tell the story
    pick = [0, 5, 11, 17, 18, 19]
    tw, th = w // 2, h // 2
    gap = 10
    strip = Image.new("RGB", (3 * tw + 2 * gap, 2 * (th + 26) + gap), (240, 238, 232))
    d = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    for k, i in enumerate(pick):
        im = Image.fromarray(frames[i][1][:, :, :3]).resize((tw, th), Image.LANCZOS)
        cx, cy = (k % 3) * (tw + gap), (k // 3) * (th + 26 + gap)
        strip.paste(im, (cx, cy + 26))
        d.text((cx + 4, cy + 5), f"{k + 1}. {caps[i]}", fill=(11, 37, 69), font=font)
    p_strip = os.path.join(out, "explorer_session.png")
    strip.save(p_strip)

    gif = [Image.fromarray(fr[:, :, :3]).resize((w * 2 // 3, h * 2 // 3), Image.LANCZOS)
           .convert("P", palette=Image.ADAPTIVE, colors=128) for _, fr in frames]
    p_gif = os.path.join(out, "explorer_session.gif")
    gif[0].save(p_gif, save_all=True, append_images=gif[1:], duration=420, loop=0,
                optimize=True)
    return p_strip, p_gif, ex, b


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gef", default="/tmp/gem/chip.gef")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ps, pg, ex, b = session(a.gef, a.out)
    print(ps); print(pg)
    print(ex.telemetry.summary())
    print(f"rung changes {ex.telemetry.rung_changes}; budget peak {b.stats()['ram']['peak_human']}")
    for e in ex.telemetry.log:
        print(f"  reload @ frame {e['frame']:>2}: bin{e['rung']:<4} {e['points']:>9,} pts")
