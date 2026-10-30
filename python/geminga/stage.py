"""A small wgpu stage for looking at GEMINGA data.

PRISM's canvas is a general node-graph renderer: steps, lenses, grouping, promotion of nodes
until a RAM budget is spent. GEMINGA needs a fraction of that, so this is deliberately the
fraction — two pipelines, one draw call each, and nothing that knows what a pipeline step is.

What it does:

* **Points, instanced.** Every observation is one instance of a unit quad: position, size and
  a scalar that a colormap texture turns into colour. One draw call for the whole layer,
  which is what makes millions of spots viable. A point is a disc centred on its position; a
  *cell* (`points(..., cells=True)`, what `show()` uses for a Stereo-seq bin) is the square
  `[x, x + size) x [y, y + size)` anchored at its low corner, so a bin covers exactly its own
  footprint and neighbouring bins tile into a surface.
* **Text, from a glyph atlas.** A font is rasterised once into an R8 texture; each character is
  an instance sampling its own sub-rectangle. Labels live in *screen* space so they stay the
  same size as you zoom, which is what a label is for.
* **A camera.** Pan and zoom as a centre and a scale. Nothing else — no rotation, no
  projection matrix, because a chip is flat. `Camera.world_to_px` is the one mapping from world
  to pixels; tick labels, outlines and cells are all placed with it, so they agree.
* **Level of detail, from the cost model.** `show()` asks a GEMINGA view what a window would
  cost, coarsens the bin or stride until it fits the stage's own point budget, and only then
  loads. The renderer never decides how much data to pull; the budget does.

What it deliberately does not do: no widgets, no node graph, no scene tree, no layout engine,
no animation system. If you want those, the OME-Zarr export feeds napari and neuroglancer,
which have them.

Headless by default: it renders to a texture and hands back a numpy array, so it runs in CI on
a software adapter. A window, if `glfw` is present, is a convenience on top.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._core import human, parse_size

__all__ = ["Stage", "Camera", "TextAtlas", "available", "adapter_info"]

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
)

POINTS_WGSL = """
struct Camera {
    center : vec2<f32>,
    scale  : vec2<f32>,      // world units -> clip
    range  : vec2<f32>,      // value min, max
    flags  : vec2<f32>,      // x: 1 = log scale value, y: minimum point size in clip units
    style  : vec2<f32>,      // x: 1 = square cells (a bin's footprint), 0 = round points
};
@group(0) @binding(0) var<uniform> cam : Camera;
@group(0) @binding(1) var lut : texture_2d<f32>;
@group(0) @binding(2) var lut_s : sampler;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) t  : f32,
};

@vertex
fn vs(@location(0) corner : vec2<f32>,
      @location(1) inst_pos : vec2<f32>,
      @location(2) inst_size : f32,
      @location(3) inst_val : f32) -> VsOut {
    var out : VsOut;
    // Never narrower than a pixel or two on screen: at deep zoom a bin1 point is below a
    // pixel wide, and sub-pixel points leave gaps that read as speckle rather than tissue.
    let size = max(inst_size, cam.flags.y / cam.scale.x);
    let world = inst_pos + corner * size;
    let clip  = (world - cam.center) * cam.scale;
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.uv = corner;
    var v = inst_val;
    if (cam.flags.x > 0.5) { v = log(1.0 + max(v, 0.0)); }
    let lo = cam.range.x;
    let hi = max(cam.range.y, lo + 1e-9);
    out.t = clamp((v - lo) / (hi - lo), 0.0, 1.0);
    return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
    // round points: discard outside the inscribed circle. Cells keep their whole square, so
    // neighbouring bins tile into a surface and a bin covers exactly its footprint.
    if (cam.style.x < 0.5 && dot(in.uv, in.uv) > 0.25) { discard; }
    let c = textureSample(lut, lut_s, vec2<f32>(in.t, 0.5));
    return vec4<f32>(c.rgb, 1.0);
}
"""

RECT_WGSL = """
struct Screen { size : vec2<f32>, pad : vec2<f32>, };
@group(0) @binding(0) var<uniform> scr : Screen;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) col : vec4<f32>,
};

@vertex
fn vs(@location(0) corner : vec2<f32>,
      @location(1) rect : vec4<f32>,      // x, y, w, h in pixels, y down from top
      @location(2) col : vec4<f32>) -> VsOut {
    var out : VsOut;
    let c01 = corner + vec2<f32>(0.5, 0.5);
    let px = rect.xy + c01 * rect.zw;
    out.pos = vec4<f32>(px.x / scr.size.x * 2.0 - 1.0,
                        1.0 - px.y / scr.size.y * 2.0, 0.0, 1.0);
    out.col = col;
    return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> { return in.col; }
"""

TEXT_WGSL = """
struct Screen {
    size : vec2<f32>,        // viewport in pixels
    pad  : vec2<f32>,
};
@group(0) @binding(0) var<uniform> scr : Screen;
@group(0) @binding(1) var atlas : texture_2d<f32>;
@group(0) @binding(2) var atlas_s : sampler;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
    @location(1) col : vec4<f32>,
};

@vertex
fn vs(@location(0) corner : vec2<f32>,
      @location(1) rect : vec4<f32>,      // x, y, w, h in pixels, y down from top
      @location(2) uvr : vec4<f32>,       // u0, v0, u1, v1
      @location(3) col : vec4<f32>) -> VsOut {
    var out : VsOut;
    // The quad shared with the point pipeline is centred on the origin, but a glyph rect is
    // given as top-left plus size. Shift into [0,1] before using it for either position or
    // UV, or every character samples half of each neighbour.
    let c01 = corner + vec2<f32>(0.5, 0.5);
    let px = rect.xy + c01 * rect.zw;
    // pixels (y down) -> clip (y up)
    let clip = vec2<f32>(px.x / scr.size.x * 2.0 - 1.0,
                         1.0 - px.y / scr.size.y * 2.0);
    out.pos = vec4<f32>(clip, 0.0, 1.0);
    out.uv = mix(uvr.xy, uvr.zw, c01);
    out.col = col;
    return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
    let a = textureSample(atlas, atlas_s, in.uv).r;
    if (a < 0.02) { discard; }
    return vec4<f32>(in.col.rgb, in.col.a * a);
}
"""


def available() -> bool:
    """True when a wgpu adapter can be reached — software adapters count."""
    try:
        import wgpu
        wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        return True
    except Exception:
        return False


def adapter_info() -> dict:
    import wgpu
    a = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = dict(a.info)
    info["software"] = str(info.get("adapter_type", "")).lower() in ("cpu", "software")
    return info


@dataclass
class Camera:
    """Pan and zoom over a flat world. `scale` is derived, not set."""
    cx: float = 0.0
    cy: float = 0.0
    span: float = 1.0                 # world units across the shorter viewport axis
    width: int = 900
    height: int = 700

    @classmethod
    def fit(cls, extent: tuple, width: int, height: int, margin: float = 1.02) -> "Camera":
        x0, x1, y0, y1 = extent
        return cls((x0 + x1) / 2, (y0 + y1) / 2,
                   max(x1 - x0, y1 - y0) * margin, width, height)

    @property
    def scale(self) -> tuple:
        """World -> clip, preserving aspect so a square chip stays square."""
        a = self.width / max(self.height, 1)
        s = 2.0 / max(self.span, 1e-9)
        return (s / max(a, 1.0), s * min(a, 1.0))

    @property
    def visible(self) -> tuple:
        """(x0, x1, y0, y1) currently on screen."""
        sx, sy = self.scale
        hw, hh = 1.0 / sx, 1.0 / sy
        return (self.cx - hw, self.cx + hw, self.cy - hh, self.cy + hh)

    def world_to_px(self, wx, wy, width: Optional[int] = None, height: Optional[int] = None) -> tuple:
        """World -> pixels, y down from the top: the inverse of what the point shader does. A
        tick label, a selection outline and a bin all use this one mapping, so they agree."""
        sx, sy = self.scale
        w = self.width if width is None else width
        h = self.height if height is None else height
        return (((wx - self.cx) * sx + 1) / 2 * w, (1 - (wy - self.cy) * sy) / 2 * h)

    def px_to_world(self, px, py, width: Optional[int] = None, height: Optional[int] = None) -> tuple:
        """Pixels (y down) -> world."""
        sx, sy = self.scale
        w = self.width if width is None else width
        h = self.height if height is None else height
        return (self.cx + (px / w * 2 - 1) / sx, self.cy + (1 - py / h * 2) / sy)

    def zoom(self, factor: float, about: Optional[tuple] = None) -> "Camera":
        if about is not None:
            ax, ay = about
            self.cx = ax + (self.cx - ax) / factor
            self.cy = ay + (self.cy - ay) / factor
        self.span /= factor
        return self

    def pan(self, dx: float, dy: float) -> "Camera":
        self.cx += dx * self.span
        self.cy += dy * self.span
        return self


class TextAtlas:
    """One font rasterised into one R8 texture, plus where each glyph sits in it.

    ASCII only, single size, no kerning and no shaping. That is enough for axis ticks, gene
    names and a cost readout, and it keeps the whole thing to a texture and a lookup table.
    """

    #: ASCII, plus the handful of symbols GEMINGA's own labels use. Anything outside this
    #: set is dropped silently, which is how the "·" separators disappeared from headers
    #: the first time round.
    CHARS = "".join(chr(c) for c in range(32, 127)) + "·×–—±≤≥µÅ°"

    def __init__(self, size: int = 16, font: Optional[str] = None, pad: int = 2):
        from PIL import Image, ImageDraw, ImageFont
        path = font or next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
        self.font_path = path
        f = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        self.size = size
        metrics = {}
        x = 0
        heights = []
        for ch in self.CHARS:
            box = f.getbbox(ch)
            w = max(box[2] - box[0], 1) if box else 1
            h = max(box[3] - box[1], 1) if box else 1
            adv = f.getlength(ch)
            metrics[ch] = dict(w=w, h=h, ox=box[0] if box else 0, oy=box[1] if box else 0,
                               adv=adv, x=x)
            x += w + pad
            heights.append(h + (box[1] if box else 0))
        self.atlas_w, self.atlas_h = x, max(max(heights), size) + pad * 2
        img = Image.new("L", (self.atlas_w, self.atlas_h), 0)
        d = ImageDraw.Draw(img)
        for ch, m in metrics.items():
            d.text((m["x"] - m["ox"], -m["oy"] + pad), ch, fill=255, font=f)
        self.image = np.asarray(img, dtype=np.uint8)
        self.metrics = metrics
        self.pad = pad
        self.ascent = size

    def layout(self, text: str, x: float, y: float, scale: float = 1.0) -> list:
        """(rect, uv) per glyph, in pixels with y measured down from the top."""
        out = []
        cx = x
        for ch in text:
            m = self.metrics.get(ch)
            if m is None:
                cx += self.size * 0.5 * scale
                continue
            if ch != " ":
                w, h = m["w"] * scale, m["h"] * scale
                # `oy` is the glyph's ink top relative to the draw origin; the atlas pad is
                # an internal offset and belongs only in the UV, not in the layout.
                top = y + m["oy"] * scale
                out.append(((cx, top, w, h),
                            (m["x"] / self.atlas_w, self.pad / self.atlas_h,
                             (m["x"] + m["w"]) / self.atlas_w,
                             (self.pad + m["h"]) / self.atlas_h)))
            cx += m["adv"] * scale
        return out

    def measure(self, text: str, scale: float = 1.0) -> float:
        return sum(self.metrics.get(c, {"adv": self.size * 0.5})["adv"] for c in text) * scale


def _nice_ticks(lo: float, hi: float, max_ticks: int) -> tuple:
    """Round tick values in [lo, hi]: `(values, step)` with a step of 1, 2 or 5 x 10^k, the
    finest that keeps the count at or under `max_ticks`. A label shows its own value, and the
    stage places it with `world_to_px`, so the number printed is where the mark sits."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [], 1.0
    raw = (hi - lo) / max(max_ticks, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1, 2, 5, 10, 20, 50):
        step = mult * mag
        first = math.ceil(lo / step - 1e-9)
        last = math.floor(hi / step + 1e-9)
        if last - first + 1 <= max_ticks:
            return [i * step for i in range(first, last + 1)], step
    return [], step


def _fmt_tick(v: float, step: float) -> str:
    """`1,234` for whole steps, enough decimals for fractional ones; never `-0`."""
    digits = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    txt = f"{v:,.{digits}f}"
    return "0" if float(txt.replace(",", "")) == 0 else txt


def _lut(name: str = "magma", n: int = 256) -> np.ndarray:
    """A colormap as an RGBA8 row. Uses matplotlib if present, else a built-in ramp."""
    try:
        import matplotlib
        try:
            cmap = matplotlib.colormaps[name]                 # matplotlib >= 3.5
        except AttributeError:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(name)
        rgba = (np.asarray(cmap(np.linspace(0, 1, n))) * 255).astype(np.uint8)
        return rgba.reshape(1, n, 4)
    except Exception:
        t = np.linspace(0, 1, n)[:, None]
        stops = np.array([[0, 0, 4], [80, 18, 123], [182, 54, 121],
                          [251, 136, 97], [252, 253, 191]], np.float32) / 255.0
        idx = t.squeeze() * (len(stops) - 1)
        lo = np.floor(idx).astype(int).clip(0, len(stops) - 2)
        frac = (idx - lo)[:, None]
        rgb = stops[lo] * (1 - frac) + stops[lo + 1] * frac
        out = np.concatenate([rgb, np.ones((n, 1), np.float32)], 1)
        return (out * 255).astype(np.uint8).reshape(1, n, 4)


@dataclass
class _Label:
    text: str
    x: float
    y: float
    world: bool = False
    scale: float = 1.0
    color: tuple = (0.10, 0.10, 0.10, 1.0)


class Stage:
    """A flat wgpu canvas: points, labels, pan/zoom, PNG out.

        st = g.Stage(900, 700)
        st.show(view)                 # picks an LOD that fits, loads, uploads
        st.label("GBM chip", 16, 22, scale=1.4)
        st.render_png("chip.png")
    """

    MAX_POINTS = 4_000_000            # one upload; beyond this the LOD must coarsen

    def __init__(self, width: int = 900, height: int = 700, background=(0.996, 0.988, 0.953),
                 point_budget: str | int = "64MB", font_size: int = 15,
                 colormap: str = "magma", min_point_px: float = 1.8):
        import wgpu
        self.wgpu = wgpu
        self.width, self.height = int(width), int(height)
        self.background = background
        self.camera = Camera(width=self.width, height=self.height)
        self.point_budget = (int(parse_size(point_budget))
                             if isinstance(point_budget, str) else int(point_budget))
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self.format = "rgba8unorm"
        self.min_point_px = float(min_point_px)
        self.atlas = TextAtlas(font_size)
        self._labels: list = []
        self._rects: list = []
        self._reserved: list = []      # boxes the HUD occupies, so axes() can stay clear
        self._pts = None              # (N, 4) float32: x, y, size, value
        self._pts_buf = None          # its vertex buffer; None while there is nothing to draw
        self._cells = False           # the layer is bins (square, anchored at the origin corner)
        self._range = (0.0, 1.0)
        self._log = True
        self._lut_np = _lut(colormap)
        self._build()

    # ---- gpu setup ----
    def _build(self):
        d, wg = self.device, self.wgpu
        quad = np.array([[-.5, -.5], [.5, -.5], [.5, .5],
                         [-.5, -.5], [.5, .5], [-.5, .5]], np.float32)
        self._quad = d.create_buffer_with_data(data=quad, usage=wg.BufferUsage.VERTEX)

        self._cam_buf = d.create_buffer(size=48, usage=wg.BufferUsage.UNIFORM | wg.BufferUsage.COPY_DST)
        self._scr_buf = d.create_buffer(size=16, usage=wg.BufferUsage.UNIFORM | wg.BufferUsage.COPY_DST)

        self._lut_tex = self._texture(self._lut_np, "rgba8unorm")
        # Coverage goes in R because the fragment shader samples .r. Writing it into alpha
        # and leaving RGB at 255 makes every glyph sample as a filled white rectangle — the
        # text draws, in the right places, as solid blocks.
        atlas_rgba = np.zeros((*self.atlas.image.shape, 4), np.uint8)
        atlas_rgba[:, :, 0] = self.atlas.image
        atlas_rgba[:, :, 3] = 255
        self._atlas_tex = self._texture(atlas_rgba, "rgba8unorm")
        self._sampler = d.create_sampler(mag_filter="linear", min_filter="linear")

        self._pts_pipe = d.create_render_pipeline(
            layout="auto",
            vertex={"module": d.create_shader_module(code=POINTS_WGSL), "entry_point": "vs",
                    "buffers": [
                        {"array_stride": 8, "step_mode": "vertex",
                         "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 0}]},
                        {"array_stride": 16, "step_mode": "instance",
                         "attributes": [
                             {"format": "float32x2", "offset": 0, "shader_location": 1},
                             {"format": "float32", "offset": 8, "shader_location": 2},
                             {"format": "float32", "offset": 12, "shader_location": 3}]}]},
            fragment={"module": d.create_shader_module(code=POINTS_WGSL), "entry_point": "fs",
                      "targets": [{"format": self.format}]},
            primitive={"topology": "triangle-list"})

        self._rect_pipe = d.create_render_pipeline(
            layout="auto",
            vertex={"module": d.create_shader_module(code=RECT_WGSL), "entry_point": "vs",
                    "buffers": [
                        {"array_stride": 8, "step_mode": "vertex",
                         "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 0}]},
                        {"array_stride": 32, "step_mode": "instance",
                         "attributes": [
                             {"format": "float32x4", "offset": 0, "shader_location": 1},
                             {"format": "float32x4", "offset": 16, "shader_location": 2}]}]},
            fragment={"module": d.create_shader_module(code=RECT_WGSL), "entry_point": "fs",
                      "targets": [{"format": self.format, "blend": {
                          "color": {"src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha", "operation": "add"},
                          "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
                                    "operation": "add"}}}]},
            primitive={"topology": "triangle-list"})

        blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha",
                           "operation": "add"},
                 "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
                           "operation": "add"}}
        self._txt_pipe = d.create_render_pipeline(
            layout="auto",
            vertex={"module": d.create_shader_module(code=TEXT_WGSL), "entry_point": "vs",
                    "buffers": [
                        {"array_stride": 8, "step_mode": "vertex",
                         "attributes": [{"format": "float32x2", "offset": 0, "shader_location": 0}]},
                        {"array_stride": 48, "step_mode": "instance",
                         "attributes": [
                             {"format": "float32x4", "offset": 0, "shader_location": 1},
                             {"format": "float32x4", "offset": 16, "shader_location": 2},
                             {"format": "float32x4", "offset": 32, "shader_location": 3}]}]},
            fragment={"module": d.create_shader_module(code=TEXT_WGSL), "entry_point": "fs",
                      "targets": [{"format": self.format, "blend": blend}]},
            primitive={"topology": "triangle-list"})

        self._pts_bind = d.create_bind_group(
            layout=self._pts_pipe.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self._cam_buf, "offset": 0, "size": 48}},
                     {"binding": 1, "resource": self._lut_tex.create_view()},
                     {"binding": 2, "resource": self._sampler}])
        self._rect_bind = d.create_bind_group(
            layout=self._rect_pipe.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self._scr_buf, "offset": 0, "size": 16}}])
        self._txt_bind = d.create_bind_group(
            layout=self._txt_pipe.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": self._scr_buf, "offset": 0, "size": 16}},
                     {"binding": 1, "resource": self._atlas_tex.create_view()},
                     {"binding": 2, "resource": self._sampler}])

        self._target = d.create_texture(
            size=(self.width, self.height, 1), format=self.format,
            usage=self.wgpu.TextureUsage.RENDER_ATTACHMENT | self.wgpu.TextureUsage.COPY_SRC)

    def _texture(self, arr: np.ndarray, fmt: str):
        d, wg = self.device, self.wgpu
        h, w = arr.shape[:2]
        tex = d.create_texture(size=(w, h, 1), format=fmt,
                               usage=wg.TextureUsage.TEXTURE_BINDING | wg.TextureUsage.COPY_DST)
        d.queue.write_texture({"texture": tex}, np.ascontiguousarray(arr),
                              {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        return tex

    # ---- content ----
    @staticmethod
    def aggregate(x, y, v=None):
        """Sum values that share a location — one point per spot, not one per gene.

        A GEF holds one observation per gene per bin, so a single bin appears dozens of times.
        Drawn as-is, each copy is coloured by one gene's count and the last one drawn wins:
        the picture is noise even though the data are fine. Summing first makes a point's
        colour mean total MID at that spot, which is what a reader expects.

        Integer grid coordinates of either sign. Pairs are packed into one key relative to the
        coordinate minima — a plain `x << 32 | y` borrows from `x` whenever `y` is negative and
        decodes to the wrong location — and fall back to a row-wise unique where the packed key
        would not fit 63 bits.
        """
        x = np.asarray(x); y = np.asarray(y)
        if x.size == 0 or not (np.issubdtype(x.dtype, np.integer) and
                               np.issubdtype(y.dtype, np.integer)):
            return x, y, v
        x64, y64 = x.astype(np.int64), y.astype(np.int64)
        xmin, ymin = int(x64.min()), int(y64.min())
        ny = int(y64.max()) - ymin + 1
        if (int(x64.max()) - xmin + 1) * ny < (1 << 62):
            key = (x64 - xmin) * ny + (y64 - ymin)
            uniq, inv = np.unique(key, return_inverse=True)
            if uniq.size == key.size:
                return x, y, v
            ux, uy = uniq // ny + xmin, uniq % ny + ymin
        else:
            pairs, inv = np.unique(np.stack([x64, y64], 1), axis=0, return_inverse=True)
            if len(pairs) == x64.size:
                return x, y, v
            ux, uy = pairs[:, 0], pairs[:, 1]
        w = np.ones(x.size) if v is None else np.asarray(v, np.float64)
        return ux, uy, np.bincount(np.asarray(inv).reshape(-1), weights=w)

    def points(self, x, y, value=None, size: Optional[float] = None,
               log: bool = True, vrange: Optional[tuple] = None,
               aggregate: bool = True, cells: bool = False) -> "Stage":
        """Upload a point layer. Coordinates are world units; `size` too. Integer grid
        coordinates are aggregated per location unless `aggregate=False`.

        `cells=True` says each point is a grid cell: `(x, y)` is the cell's low corner and
        `size` its side, so the cell is drawn as the square `[x, x + size) x [y, y + size)`.
        That is what a Stereo-seq bin is — SAW names a bin by its origin — and drawing it as a
        round point centred on the origin put every bin half a cell off. Without `cells`, a
        point is a disc centred on its position, whatever `size` is.

        An empty layer is fine: nothing is uploaded and `render()` draws the background."""
        if aggregate:
            x, y, value = self.aggregate(x, y, value)
        x = np.asarray(x, np.float32).ravel()
        y = np.asarray(y, np.float32).ravel()
        n = x.size
        if n > self.MAX_POINTS:
            step = math.ceil(n / self.MAX_POINTS)
            x, y = x[::step], y[::step]
            value = None if value is None else np.asarray(value).ravel()[::step]
            n = x.size
        self._log = log
        self._cells = bool(cells)
        if n == 0:                       # a zero-length buffer cannot be created or drawn
            self._pts = np.empty((0, 4), np.float32)
            self._pts_buf = None
            self._range = (0.0, 1.0) if vrange is None else tuple(float(t) for t in vrange)
            return self
        v = (np.ones(n, np.float32) if value is None
             else np.asarray(value, np.float32).ravel()[:n])
        if size is None:
            span = max(float(x.max() - x.min()), float(y.max() - y.min()), 1e-6)
            size = span / max(math.sqrt(max(n, 1)), 1.0) * 1.6
        if cells:                        # the quad is centred on its position: move it to the middle
            x = x + np.float32(size / 2)
            y = y + np.float32(size / 2)
        arr = np.empty((n, 4), np.float32)
        arr[:, 0], arr[:, 1] = x, y
        arr[:, 2] = size
        arr[:, 3] = v
        self._pts = arr
        if vrange is None:
            vv = np.log1p(np.clip(v, 0, None)) if log else v
            self._range = (float(np.percentile(vv, 1)), float(np.percentile(vv, 99.5)))
        else:
            self._range = tuple(float(t) for t in vrange)
        self._pts_buf = self.device.create_buffer_with_data(
            data=arr, usage=self.wgpu.BufferUsage.VERTEX)
        return self

    def label(self, text: str, x: float, y: float, *, world: bool = False,
              scale: float = 1.0, color=(0.10, 0.10, 0.10, 1.0)) -> "Stage":
        """Screen-space text by default, so labels do not grow when you zoom."""
        self._labels.append(_Label(str(text), float(x), float(y), world, scale, color))
        return self

    def clear_labels(self) -> "Stage":
        self._labels.clear()
        return self

    # ---- screen-space rectangles ----
    def rect(self, x, y, w, h, color=(0.85, 0.88, 0.91, 1.0)) -> "Stage":
        """A filled rectangle in pixels, y measured down from the top."""
        self._rects.append((float(x), float(y), float(w), float(h), *color))
        return self

    def outline(self, x, y, w, h, color=(0.91, 0.25, 0.18, 1.0), width: float = 2.0) -> "Stage":
        """Four thin rectangles — an outline without a line pipeline."""
        x, y, w, h, t = float(x), float(y), float(w), float(h), float(width)
        self.rect(x, y, w, t, color); self.rect(x, y + h - t, w, t, color)
        self.rect(x, y, t, h, color); self.rect(x + w - t, y, t, h, color)
        return self

    def clear_rects(self) -> "Stage":
        self._rects.clear()
        self._reserved.clear()
        return self

    def _reserve(self, x, y, w, h, pad: int = 6):
        self._reserved.append((x - pad, y - pad, w + 2 * pad, h + 2 * pad))

    def _free(self, x, y, w, h) -> bool:
        for rx, ry, rw, rh in self._reserved:
            if x < rx + rw and x + w > rx and y < ry + rh and y + h > ry:
                return False
        return True

    def clear(self) -> "Stage":
        return self.clear_labels().clear_rects()

    def world_to_px(self, wx, wy) -> tuple:
        return self.camera.world_to_px(wx, wy, self.width, self.height)

    def draw_selections(self, sels, *, labels: bool = True, width: float = 2.0,
                        active: Optional[str] = None) -> "Stage":
        """Outline selections in world coordinates, coloured by intent, named if they fit.

        Rectangles and polygons are drawn as outlines built from thin screen-space rectangles,
        so they cost nothing beyond the rectangle draw call already in use. Anything off
        screen is skipped. The `active` selection is drawn thicker.
        """
        W, H = self.width, self.height
        for sel in sels:
            if sel.kind not in ("rect", "polygon"):
                continue
            col = tuple(sel.color)
            lw = width * (1.8 if active and sel.id == active else 1.0)
            if sel.kind == "rect":
                x0, x1, y0, y1 = sel.bounds
                px0, py1 = self.world_to_px(x0, y0)
                px1, py0 = self.world_to_px(x1, y1)
                if px1 < 0 or px0 > W or py1 < 0 or py0 > H:
                    continue
                self.outline(px0, py0, max(px1 - px0, 2), max(py1 - py0, 2), col, lw)
                tx, ty = px0 + 3, py0 + 3
            else:
                pts = [self.world_to_px(x, y) for x, y in sel.geometry["points"]]
                for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
                    n = max(int(math.hypot(bx - ax, by - ay) / 3), 1)
                    for i in range(n):                 # a dotted edge, one small square per step
                        t = i / n
                        self.rect(ax + (bx - ax) * t - lw / 2, ay + (by - ay) * t - lw / 2,
                                  lw, lw, col)
                tx, ty = min(p[0] for p in pts) + 3, min(p[1] for p in pts) + 3
            if labels and 0 <= tx < W - 40 and 0 <= ty < H - 20:
                txt = sel.name + (f"  {sel.score:.2f}" if sel.score is not None else "")
                w = self.atlas.measure(txt, 0.78) + 8
                self.rect(tx - 2, ty - 1, w, self.atlas.size * 1.15, (1.0, 0.99, 0.95, 0.85))
                self.label(txt, tx + 2, ty, scale=0.78, color=(0.04, 0.15, 0.27, 1.0))
        return self
    def minimap(self, view, *, size: int = 150, corner: str = "tr", margin: int = 14,
                cells: int = 44, label: bool = True, selections=None) -> "Stage":
        """A thumbnail of the whole dataset with the current window drawn on it.

        Built from the *coarsest* level available — the cheapest complete picture there is —
        so the minimap costs the same however deep you have zoomed. Drawn as flat rectangles,
        one per occupied cell, sharing the rectangle draw call with everything else.
        """
        full = view._full_window()
        if len(full) == 2:
            return self._minimap_1d(view, size, corner, margin, label)
        saved = view.window
        try:
            grid = self._thumbnail(view, full, cells)
        finally:
            view.select(*saved)
        x, y = self._corner(size, size, corner, margin)
        # The plate encloses the caption as well as the map. Grey text sitting directly on
        # the point cloud loses whichever characters land on dark points — it reads as
        # clipping, but it is contrast.
        cap_h = int(self.atlas.size * 1.5) if label else 0
        self._reserve(x - 3, y - 3, size + 6, size + 6 + cap_h)
        self.rect(x - 3, y - 3, size + 6, size + 6 + cap_h, (1.0, 0.99, 0.95, 0.94))
        self.outline(x - 3, y - 3, size + 6, size + 6 + cap_h, (0.72, 0.76, 0.80, 1.0), 1.0)
        if grid is not None and grid.size:
            gh, gw = grid.shape
            cw, ch = size / gw, size / gh
            lut = self._lut_np[0].astype(np.float32) / 255.0
            a = np.log1p(np.clip(grid, 0, None))
            hi = float(a.max()) or 1.0
            ys, xs = np.nonzero(a)
            for gy, gx in zip(ys, xs):
                c = lut[int(np.clip(a[gy, gx] / hi, 0, 1) * (len(lut) - 1))]
                # grid row 0 is the low-y edge; screen y grows downward
                self.rect(x + gx * cw, y + (gh - 1 - gy) * ch, cw + 0.6, ch + 0.6,
                          (float(c[0]), float(c[1]), float(c[2]), 1.0))
        fx0, fx1, fy0, fy1 = full[0], full[1], full[2], full[3]
        wx0, wx1, wy0, wy1 = saved[0], saved[1], saved[2], saved[3]
        sx = size / max(fx1 - fx0, 1e-9)
        sy = size / max(fy1 - fy0, 1e-9)
        bx = x + (wx0 - fx0) * sx
        bw = max((wx1 - wx0) * sx, 3)
        bh = max((wy1 - wy0) * sy, 3)
        by = y + size - (wy1 - fy0) * sy
        self.outline(bx, by, bw, bh, (0.91, 0.25, 0.18, 1.0), 2.0)
        if label:
            frac = ((wx1 - wx0) * (wy1 - wy0)) / max((fx1 - fx0) * (fy1 - fy0), 1e-9)
            # left-aligned to the minimap's own left edge: the caption is always narrower
            # than the map, so it fits in any corner without measuring against the viewport
            self.label(f"{frac:.2%} of extent", x + 2, y + size + 5, scale=0.8,
                       color=(0.35, 0.40, 0.46, 1.0))
        return self

    def _minimap_1d(self, view, size, corner, margin, label) -> "Stage":
        """A strip for row/record views: the whole extent, with the window marked."""
        h = 18
        x, y = self._corner(size, h, corner, margin)
        lo, hi = view._full_window()
        self.rect(x, y, size, h, (0.85, 0.88, 0.91, 1.0))
        w0, w1 = view.window
        s = size / max(hi - lo, 1e-9)
        self.rect(x + (w0 - lo) * s, y, max((w1 - w0) * s, 3), h, (0.91, 0.25, 0.18, 0.85))
        self.outline(x, y, size, h, (0.72, 0.76, 0.80, 1.0), 1.0)
        if label:
            self.label(f"{(w1 - w0) / max(hi - lo, 1):.2%} of extent", x, y + h + 8,
                       scale=0.8, color=(0.35, 0.40, 0.46, 1.0))
        return self

    def _thumbnail(self, view, full, cells):
        """Coarsest complete picture of the whole extent, as a small 2-D grid."""
        gef = getattr(view, "gef", None)
        if gef is not None:
            return gef.rasterize(full, gef.bins[-1], getattr(view, "genes", None),
                                 shape=min(cells, max(8, (full[1] - full[0]) // gef.bins[-1])))
        try:
            ov = view.select(*full).overview(cells)
            t = ov.series.get("thumbnail")
            return None if t is None else np.asarray(t)
        except Exception:
            return None

    @staticmethod
    def _stack(bars):
        return bars

    def budget_panel(self, budget, *, corner: str = "bl", margin: int = 14,
                     width: int = 260, title: str = "budget") -> "Stage":
        """Per-tier gauges: capacity, high-water mark, and what is held right now.

        The high-water mark is the number that matters — "it fits" is a claim about the peak,
        not about this instant — so it is drawn as a tick inside the capacity bar.
        """
        stats = budget.stats() if hasattr(budget, "stats") else budget
        tiers = [(k, v) for k, v in stats.items() if v.get("capacity")]
        row_h, gap = 15, 9
        h = 22 + len(tiers) * (row_h + gap)
        x, y = self._corner(width, h, corner, margin)
        self._reserve(x - 8, y - 8, width + 16, h + 16)
        self.rect(x - 8, y - 8, width + 16, h + 16, (1.0, 0.99, 0.95, 0.94))
        self.outline(x - 8, y - 8, width + 16, h + 16, (0.72, 0.76, 0.80, 1.0), 1.0)
        self.label(title, x, y, scale=0.9, color=(0.04, 0.15, 0.27, 1.0))
        yy = y + 20
        for name, st in tiers:
            cap = max(int(st["capacity"]), 1)
            used = int(st["used"]) / cap
            peak = int(st["peak"]) / cap
            over = peak > 1.0
            self.rect(x, yy, width, row_h, (0.87, 0.90, 0.93, 1.0))
            self.rect(x, yy, width * min(peak, 1.0), row_h,
                      (0.96, 0.77, 0.09, 0.55) if not over else (0.91, 0.25, 0.18, 0.75))
            self.rect(x, yy, width * min(used, 1.0), row_h, (0.18, 0.47, 0.77, 1.0))
            if 0 < peak <= 1.0:
                self.rect(x + width * peak - 1.5, yy - 2, 3, row_h + 4,
                          (0.04, 0.15, 0.27, 1.0))
            self.label(f"{name}", x + 4, yy + 1, scale=0.78, color=(1, 1, 1, 1)
                       if used > 0.12 else (0.25, 0.30, 0.36, 1.0))
            txt = f"{st['peak_human']} peak / {st['capacity_human']}"
            self.label(txt, x + width - self.atlas.measure(txt, 0.78) - 4, yy + 1,
                       scale=0.78, color=(0.25, 0.30, 0.36, 1.0))
            yy += row_h + gap
        return self

    def workspace_panel(self, ws, *, corner: str = "bl", margin: int = 14,
                        width: int = 300) -> "Stage":
        """One gauge per cell: its partition, and the peak it actually reached inside it."""
        rep = ws.report() if hasattr(ws, "report") else ws
        cells = rep["cells"]
        row_h, gap = 15, 9
        h = 22 + max(len(cells), 1) * (row_h + gap)
        x, y = self._corner(width, h, corner, margin)
        self._reserve(x - 8, y - 8, width + 16, h + 16)
        self.rect(x - 8, y - 8, width + 16, h + 16, (1.0, 0.99, 0.95, 0.94))
        self.outline(x - 8, y - 8, width + 16, h + 16, (0.72, 0.76, 0.80, 1.0), 1.0)
        total = max(rep["ram_cap"], 1)
        self.label(f"{rep['workspace']}  ·  {human(rep['ram_reserved'])} of "
                   f"{human(rep['ram_cap'])} reserved", x, y, scale=0.85,
                   color=(0.04, 0.15, 0.27, 1.0))
        yy = y + 20
        for c in cells:
            share = c["ram_cap"] / total
            self.rect(x, yy, width, row_h, (0.87, 0.90, 0.93, 1.0))
            self.rect(x, yy, width * share, row_h, (0.72, 0.79, 0.86, 1.0))
            if c["ram_cap"]:
                self.rect(x, yy, width * share * min(c["ram_peak"] / c["ram_cap"], 1.0),
                          row_h, (0.18, 0.47, 0.77, 1.0))
            self.label(c["name"], x + 4, yy + 1, scale=0.78, color=(0.25, 0.30, 0.36, 1.0))
            txt = f"{c['ram_peak_human']} / {c['ram_cap_human']}"
            self.label(txt, x + width - self.atlas.measure(txt, 0.78) - 4, yy + 1,
                       scale=0.78, color=(0.25, 0.30, 0.36, 1.0))
            yy += row_h + gap
        return self

    def lod_ladder(self, view, *, corner: str = "tl", margin: int = 14, top: int = 62,
                   width: int = 128) -> "Stage":
        """The pyramid as rungs, with the level currently on screen filled in."""
        gef = getattr(view, "gef", None)
        levels = list(gef.bins) if gef is not None else list(range(0, 8))
        current = view.bin_size if gef is not None else view.zoom
        row_h, gap = 13, 5
        h = 18 + len(levels) * (row_h + gap)
        x, y = self._corner(width, h, corner, margin)
        y = max(y, top)
        self._reserve(x - 8, y - 8, width + 16, h + 16)
        self.rect(x - 8, y - 8, width + 16, h + 16, (1.0, 0.99, 0.95, 0.94))
        self.outline(x - 8, y - 8, width + 16, h + 16, (0.72, 0.76, 0.80, 1.0), 1.0)
        self.label("detail", x, y, scale=0.8, color=(0.04, 0.15, 0.27, 1.0))
        yy = y + 16
        for lv in levels:
            on = (lv == current)
            self.rect(x, yy, width, row_h,
                      (0.18, 0.47, 0.77, 1.0) if on else (0.87, 0.90, 0.93, 1.0))
            name = f"bin{lv}" if gef is not None else f"zoom {lv}"
            self.label(name, x + 5, yy, scale=0.76,
                       color=(1, 1, 1, 1) if on else (0.35, 0.40, 0.46, 1.0))
            yy += row_h + gap
        return self

    def hud(self, view, budget=None, workspace=None, *, minimap: bool = True,
            ladder: bool = True, axes: bool = True) -> "Stage":
        """Everything at once: where you are, how deep, and what it is costing.

        Panels are laid out before the axes so the tick labels can be skipped where one would
        land underneath a panel — the rectangles are a single draw call ahead of all the text,
        so a panel can never paint over a label that was already queued.
        """
        if minimap:
            self.minimap(view)
        if ladder:
            self.lod_ladder(view)
        if workspace is not None:
            self.workspace_panel(workspace)
        elif budget is not None:
            self.budget_panel(budget)
        if axes:
            self.axes()
        return self

    def _corner(self, w, h, corner: str, margin: int) -> tuple:
        cx = margin if corner[1] == "l" else self.width - w - margin
        cy = margin if corner[0] == "t" else self.height - h - margin
        return cx, cy

    def axes(self, ticks: int = 5, color=(0.35, 0.40, 0.46, 1.0),
             left: int = 58, bottom: int = 22, top_margin: int = 62) -> "Stage":
        """Tick labels along the bottom and left edges, in world coordinates.

        Ticks are round numbers inside the part of the viewport the labels may use — `left`
        reserves room for the y labels so the x labels do not run under them, `bottom` for the
        x labels, and `top_margin` keeps the topmost y tick clear of whatever `show()` wrote in
        the header. At most `ticks + 1` per axis. Each label is centred on
        `Camera.world_to_px` of the value it prints, so the number and the mark agree; a label
        that would fall off screen or under a panel is skipped, not moved.
        """
        cam, scale = self.camera, 0.85
        W, H = self.width, self.height
        # ---- x: labels centred on their tick, along the bottom edge ----
        xs = [cam.px_to_world(px, 0, W, H)[0] for px in (left, W - 18)]
        values, step = _nice_ticks(xs[0], xs[1], ticks + 1)
        ly = H - self.atlas.size - bottom + 8
        right_edge = -1e9
        for v in values:
            txt = _fmt_tick(v, step)
            w = self.atlas.measure(txt, scale)
            lx = cam.world_to_px(v, 0, W, H)[0] - w / 2
            if lx < 0 or lx + w > W or lx < right_edge + 6:
                continue
            if self._free(lx, ly, w, self.atlas.size):
                self.label(txt, lx, ly, scale=scale, color=color)
                right_edge = lx + w
        # ---- y: labels centred (on the digits) on their tick, along the left edge ----
        # keep the lowest y label above the row the x labels sit in
        ys = [cam.px_to_world(0, py, W, H)[1] for py in (H - bottom - self.atlas.size * 0.8, top_margin)]
        values, step = _nice_ticks(ys[0], ys[1], ticks + 1)
        digit = self.atlas.metrics.get("0", {"oy": 0, "h": self.atlas.size * 0.7})
        centre = (digit["oy"] + digit["h"] / 2) * scale
        for v in values:
            txt = _fmt_tick(v, step)
            top = cam.world_to_px(0, v, W, H)[1] - centre
            if self._free(8, top, self.atlas.measure(txt, scale), self.atlas.size):
                self.label(txt, 8, top, scale=scale, color=color)
        return self

    # ---- the GEMINGA bridge ----
    def show(self, view, window: Optional[tuple] = None, *, value: str = "count",
             fit: bool = True, label: bool = True) -> dict:
        """Put a GEMINGA view on the stage, choosing a level of detail that fits.

        The stage never decides how much to read. It asks `cost()`, coarsens the view — a
        coarser GEF bin, a larger stride — until the estimate is under its point budget, and
        only then calls `load()`. Returns what it settled on.
        """
        if window:
            view.select(*window)
        chosen = self._fit_lod(view)
        from .journal import recording as _recording
        with _recording(via="stage"):
            data = view.load()
        x, y, v = self._xyv(data, value)
        if fit:
            ex = view.window if len(view.window) == 4 else None
            if ex is not None:
                self.camera = Camera.fit((ex[0], ex[1], ex[2], ex[3]), self.width, self.height)
            elif x.size:
                self.camera = Camera.fit((float(x.min()), float(x.max()),
                                          float(y.min()), float(y.max())),
                                         self.width, self.height)
        # A binned level is a grid, so a point should be a bin wide. Sizing from sqrt(n)
        # instead leaves a coarse level looking like scattered dust rather than a surface.
        size = float(getattr(view, "bin_size", 0)) if hasattr(view, "gef") else 0.0
        self.points(x, y, v, size=size or None, cells=bool(size))
        chosen["points"] = int(x.size)
        if label:
            c = view.cost()
            name = str(getattr(getattr(view, "gef", None), "chip", None)
                       or type(view).__name__)
            bits = [f"{chosen['points']:,} pts"]
            if "bin_size" in c:
                bits.append(f"bin{c['bin_size']}")
            elif c.get("stride", 1) > 1:
                bits.append(f"stride {c['stride']}")
            bits.append(c["resident_human"])
            sub = "  ·  ".join(bits)
            hx, hy = 150, 12
            # a plate behind the header: white text on a point cloud is unreadable, and the
            # rectangle pass runs before the text pass so this can never cover it
            w = max(self.atlas.measure(name, 1.25), self.atlas.measure(sub, 0.95)) + 18
            h = self.atlas.size * 2.9
            self._reserve(hx - 9, hy - 7, w, h)
            self.rect(hx - 9, hy - 7, w, h, (1.0, 0.99, 0.95, 0.90))
            self.outline(hx - 9, hy - 7, w, h, (0.72, 0.76, 0.80, 1.0), 1.0)
            self.label(name, hx, hy, scale=1.25)
            self.label(sub, hx, hy + self.atlas.size * 1.5, scale=0.95,
                       color=(0.18, 0.47, 0.77, 1.0))
        return chosen

    def _fit_lod(self, view) -> dict:
        """Coarsen until the estimate fits the point budget. Metadata only."""
        per_point = 16
        cap = max(self.point_budget // per_point, 1000)
        steps = 0
        if hasattr(view, "gef"):
            for b in view.gef.bins:
                view.use_bin(b)
                if view.cost()["units"] <= cap:
                    break
                steps += 1
            return {"mode": "bin", "bin_size": view.bin_size, "steps": steps,
                    "estimate": view.cost()["units"], "cap": cap}
        for z in range(0, 24):
            view.set_zoom(z)
            if view.cost()["units"] <= cap:
                break
            steps += 1
        return {"mode": "stride", "stride": view.stride, "steps": steps,
                "estimate": view.cost()["units"], "cap": cap}

    @staticmethod
    def _xyv(data, value: str):
        if isinstance(data, dict) and "x" in data:
            v = data.get(value, data.get("count"))
            return (np.asarray(data["x"]), np.asarray(data["y"]),
                    None if v is None else np.asarray(v))
        if hasattr(data, "schema"):
            names = data.schema.names
            xs = "x" if "x" in names else ("x_location" if "x_location" in names else None)
            ys = "y" if "y" in names else ("y_location" if "y_location" in names else None)
            if xs is None or ys is None:
                raise TypeError(f"no coordinate columns in {names}")
            v = data.column(value).to_numpy(zero_copy_only=False) if value in names else None
            return (data.column(xs).to_numpy(zero_copy_only=False),
                    data.column(ys).to_numpy(zero_copy_only=False), v)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            yy, xx = np.nonzero(data)
            return xx.astype(np.float32), yy.astype(np.float32), data[yy, xx]
        raise TypeError(f"cannot place a {type(data).__name__} on the stage")

    # ---- drawing ----
    def _upload_uniforms(self):
        sx, sy = self.camera.scale
        cam = np.array([self.camera.cx, self.camera.cy, sx, sy,
                        self._range[0], self._range[1],
                        1.0 if self._log else 0.0,
                        self.min_point_px * 2.0 / max(self.width, 1),
                        1.0 if self._cells else 0.0, 0.0, 0.0, 0.0], np.float32)
        self.device.queue.write_buffer(self._cam_buf, 0, cam)
        self.device.queue.write_buffer(
            self._scr_buf, 0, np.array([self.width, self.height, 0, 0], np.float32))

    def _text_instances(self) -> Optional[np.ndarray]:
        rows = []
        sx, sy = self.camera.scale
        for lb in self._labels:
            px, py = lb.x, lb.y
            if lb.world:                       # world anchor -> pixels, so it tracks the data
                cx = (lb.x - self.camera.cx) * sx
                cy = (lb.y - self.camera.cy) * sy
                px = (cx + 1) * 0.5 * self.width
                py = (1 - cy) * 0.5 * self.height
            for rect, uv in self.atlas.layout(lb.text, px, py, lb.scale):
                rows.append([*rect, *uv, *lb.color])
        if not rows:
            return None
        return np.asarray(rows, np.float32)

    def render(self) -> np.ndarray:
        """Draw once and return the frame as an (H, W, 4) uint8 array."""
        self._upload_uniforms()
        d, wg = self.device, self.wgpu
        enc = d.create_command_encoder()
        r, g_, b = self.background
        rp = enc.begin_render_pass(color_attachments=[{
            "view": self._target.create_view(),
            "clear_value": (r, g_, b, 1.0),
            "load_op": "clear", "store_op": "store"}])
        if self._pts_buf is not None and self._pts is not None and len(self._pts):
            rp.set_pipeline(self._pts_pipe)
            rp.set_bind_group(0, self._pts_bind)
            rp.set_vertex_buffer(0, self._quad)
            rp.set_vertex_buffer(1, self._pts_buf)
            rp.draw(6, len(self._pts))
        if self._rects:
            ri = np.asarray(self._rects, np.float32)
            rb = d.create_buffer_with_data(data=ri, usage=wg.BufferUsage.VERTEX)
            rp.set_pipeline(self._rect_pipe)
            rp.set_bind_group(0, self._rect_bind)
            rp.set_vertex_buffer(0, self._quad)
            rp.set_vertex_buffer(1, rb)
            rp.draw(6, len(ri))
        inst = self._text_instances()
        if inst is not None:
            buf = d.create_buffer_with_data(data=inst, usage=wg.BufferUsage.VERTEX)
            rp.set_pipeline(self._txt_pipe)
            rp.set_bind_group(0, self._txt_bind)
            rp.set_vertex_buffer(0, self._quad)
            rp.set_vertex_buffer(1, buf)
            rp.draw(6, len(inst))
        rp.end()
        d.queue.submit([enc.finish()])
        raw = d.queue.read_texture(
            {"texture": self._target, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": self.width * 4, "rows_per_image": self.height},
            (self.width, self.height, 1))
        return np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 4)

    def render_png(self, path: str) -> str:
        from PIL import Image
        Image.fromarray(self.render()[:, :, :3]).save(path)
        return path

    @property
    def draw_calls(self) -> int:
        return ((1 if self._pts is not None and len(self._pts) else 0)
                + (1 if self._rects else 0) + (1 if self._labels else 0))

    def stats(self) -> dict:
        n = 0 if self._pts is None else len(self._pts)
        return {"points": n, "rects": len(self._rects), "labels": len(self._labels),
                "glyphs": sum(len(l.text) for l in self._labels),
                "draw_calls": self.draw_calls,
                "vram_points": human(n * 16),
                "atlas": f"{self.atlas.atlas_w}x{self.atlas.atlas_h}",
                "adapter": str(dict(self.adapter.info).get("device", "?")),
                "viewport": f"{self.width}x{self.height}"}

    def __repr__(self) -> str:
        s = self.stats()
        return (f"Stage({s['viewport']}, {s['points']:,} points, {s['glyphs']} glyphs, "
                f"{s['draw_calls']} draw calls, {s['adapter']})")
