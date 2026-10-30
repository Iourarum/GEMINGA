"""Interactive navigation, with the budget still in charge.

The stage renders one frame. The explorer turns a stream of input — drags, scroll steps, key
presses — into a stream of frames, and its one real job is deciding **when a frame needs I/O**.

The naive loop reloads on every camera move. That makes a drag cost a disk read per mouse
event and turns a smooth pan into a slideshow, and it is wrong for a more basic reason: the
data was already in memory a moment ago. So the explorer loads a **margin** around what is on
screen and serves every frame it can from that, reloading only when

* the visible area leaves the loaded area, or
* the zoom moves far enough that a different rung of the pyramid would be chosen.

The second rule has **hysteresis**. Stepping *finer* waits until the finer rung's estimate is
comfortably under the cap, stepping *coarser* waits until the current rung is over it. Without
that band, a zoom that hovers near a boundary would thrash between two rungs, reloading on
every scroll tick.

Every reload goes through `view.load()`, so the previous region's lease is released before the
next is taken and the budget invariant holds throughout a session, not just per frame.

Scope, stated: this navigates **point-cloud views** — Stereo-seq GEF and GEM. Images already
have better interactive viewers; export them with `to_ome_zarr()` and open them in napari.

The window is a thin layer. All of the logic above runs headless, which is how it is tested:
`record()` plays a scripted session and returns the frames. `run()` opens a glfw window when a
display exists and feeds real mouse events into exactly the same `handle()` method.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .journal import recording as _recording
from ._core import Budget, human

__all__ = ["Explorer", "Event", "Telemetry", "explore"]


@dataclass
class Event:
    """One input. Coordinates are in window pixels, y measured down from the top."""
    kind: str                    # "drag" | "scroll" | "key" | "goto" | "gene" | "recall"
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    steps: float = 0.0           # scroll: positive zooms in
    key: str = ""
    arg: object = None


@dataclass
class Telemetry:
    frames: int = 0
    reloads: int = 0
    rung_changes: int = 0
    bytes_loaded: int = 0
    load_seconds: float = 0.0
    render_seconds: float = 0.0
    log: list = field(default_factory=list)

    @property
    def served_from_memory(self) -> float:
        """Fraction of frames that needed no I/O at all."""
        return 1.0 - self.reloads / max(self.frames, 1)

    def summary(self) -> str:
        return (f"{self.frames} frames, {self.reloads} reloads "
                f"({self.served_from_memory:.0%} served from memory), "
                f"{human(self.bytes_loaded)} read in {self.load_seconds:.2f}s")


class Explorer:
    """Pan, zoom, jump and bookmark over a point-cloud view, under a budget.

        ex = g.explore("chip.gef", budget=g.Budget(ram="2GB"))
        ex.run()                              # a window, if there is a display
        frames = ex.record(script)            # the same session, headless
    """

    ZOOM_STEP = 1.25                 # per scroll notch
    #: margins tried, widest first, per policy. "smooth" never shrinks the margin, "sharp"
    #: will go to zero for the finest rung, "balanced" keeps a floor so small drags stay free.
    POLICIES = {"smooth": (0.5,), "balanced": (0.5, 0.25, 0.125), "sharp": (0.5, 0.25, 0.125, 0.0)}
    FINER_BELOW = 0.85               # go finer only when the finer rung fits with 15% to spare
    MARGIN = 0.5                     # load this much extra on every side

    def __init__(self, view, budget: Optional[Budget] = None, width: int = 960,
                 height: int = 720, point_budget: str | int = "16MB", margin: float = MARGIN,
                 hud: bool = True, adaptive_margin: Optional[bool] = None,
                 policy: str = "balanced", selections=None):
        from .stage import Stage, Camera
        if len(view._full_window()) != 4:
            raise TypeError(
                f"{type(view).__name__} is not a 2-D point-cloud view. The explorer navigates "
                f"Stereo-seq GEF and GEM; for images, export with g.to_ome_zarr() and open in "
                f"napari or neuroglancer.")
        if not (hasattr(view, "gef") or hasattr(view, "index")):
            raise TypeError(f"{type(view).__name__} has no pyramid to navigate")
        self.view = view
        self.budget = budget or getattr(view, "budget", None)
        self.stage = Stage(width, height, point_budget=point_budget)
        self.width, self.height = self.stage.width, self.stage.height
        self.margin = float(margin)
        if adaptive_margin is not None:                       # older keyword, still honoured
            policy = "sharp" if adaptive_margin else "smooth"
        if policy not in self.POLICIES:
            raise ValueError(f"policy must be one of {list(self.POLICIES)}")
        self.policy = policy
        self.adaptive_margin = policy != "smooth"
        self.hud = hud
        self.full = tuple(view._full_window())
        self.stage.camera = Camera.fit(self.full, self.width, self.height)
        self.loaded_window: Optional[tuple] = None
        self.loaded_rung = None
        self.loaded_points = 0
        self.bookmarks: dict[str, tuple] = {}
        self.book = selections                # a SelectionBook, or None to keep them in memory
        self._sels: list = []                 # this session's selections and candidates
        self.active = None                    # the selection the last action concerned
        self.telemetry = Telemetry()
        self._cap = max(self.stage.point_budget // 16, 1000)
        self._genes = None

    # ---- rungs: GEF bin levels, or GEM zoom levels -------------------------
    @property
    def rungs(self) -> list:
        if hasattr(self.view, "gef"):
            return list(self.view.gef.bins)                 # fine -> coarse
        return list(range(0, 12))                            # zoom 0 = finest

    def _set_rung(self, r):
        if hasattr(self.view, "gef"):
            self.view.use_bin(r)
        else:
            self.view.set_zoom(r)

    def _estimate(self, window: tuple, rung) -> int:
        """Points a rung would put on screen for a window. Metadata only."""
        self.view.select(*window)
        self._set_rung(rung)
        return int(self.view.cost()["units"])

    def _choose_rung(self, window: tuple):
        """Finest rung that fits the cap, with hysteresis against the rung already loaded."""
        rungs = self.rungs
        best = rungs[-1]
        for r in rungs:
            if self._estimate(window, r) <= self._cap:
                best = r
                break
        cur = self.loaded_rung
        if cur is None or best == cur:
            return best
        i_best, i_cur = rungs.index(best), rungs.index(cur)
        if i_best < i_cur:
            # finer is available — take it only with room to spare
            return best if self._estimate(window, best) <= self._cap * self.FINER_BELOW else cur
        # coarser is being suggested — only leave the current rung once it no longer fits
        return best if self._estimate(window, cur) > self._cap else cur

    # ---- geometry -----------------------------------------------------------
    def visible(self) -> tuple:
        x0, x1, y0, y1 = self.stage.camera.visible
        f = self.full
        return (max(x0, f[0]), min(x1, f[1]), max(y0, f[2]), min(y1, f[3]))

    def _expand(self, w: tuple, margin: Optional[float] = None) -> tuple:
        m = self.margin if margin is None else margin
        x0, x1, y0, y1 = w
        mx, my = (x1 - x0) * m, (y1 - y0) * m
        f = self.full
        return (max(f[0], int(x0 - mx)), min(f[1], int(math.ceil(x1 + mx))),
                max(f[2], int(y0 - my)), min(f[3], int(math.ceil(y1 + my))))

    @staticmethod
    def _inside(inner: tuple, outer: tuple) -> bool:
        return (inner[0] >= outer[0] and inner[1] <= outer[1]
                and inner[2] >= outer[2] and inner[3] <= outer[3])

    def px_to_world(self, px: float, py: float) -> tuple:
        return self.stage.camera.px_to_world(px, py, self.width, self.height)

    # ---- loading, only when needed -----------------------------------------
    def _plan(self, vis: tuple) -> tuple:
        """Pick the loaded region: the finest rung first, then the most pan slack it allows.

        A margin buys smooth panning, but it multiplies the loaded area — a 0.5 margin on
        every side is four times the visible area — and a larger area forces a coarser rung.
        Left fixed, the margin silently trades resolution for slack. So try shrinking it, and
        keep whichever margin permits the finest rung, preferring the widest among ties.
        """
        rungs = self.rungs
        best = None
        scale = self.margin / 0.5
        for m in (f * scale for f in self.POLICIES[self.policy]):
            want = self._expand(vis, m)
            rung = self._choose_rung(want)
            key = (rungs.index(rung), -m)
            if best is None or key < best[0]:
                best = (key, want, rung)
        return best[1], best[2]

    def _ensure_loaded(self) -> bool:
        vis = self.visible()
        if vis[1] <= vis[0] or vis[3] <= vis[2]:
            return False
        want, rung = self._plan(vis)
        covered = self.loaded_window is not None and self._inside(vis, self.loaded_window)
        if covered and rung == self.loaded_rung:
            return False                                         # served from memory
        t0 = time.perf_counter()
        self.view.select(*want)
        self._set_rung(rung)
        with _recording(via="explorer", frame=self.telemetry.frames):
            data = self.view.load()
        x, y, v = self.stage._xyv(data, "count")
        size = float(rung) if hasattr(self.view, "gef") else None
        # A GEF rung is a bin size and each observation sits at its bin's origin: draw it as
        # that square cell, not as a disc centred on the corner. An empty window is fine too.
        self.stage.points(x, y, v, size=size or None, cells=size is not None)
        dt = time.perf_counter() - t0
        if self.loaded_rung is not None and rung != self.loaded_rung:
            self.telemetry.rung_changes += 1
        nbytes = sum(int(np.asarray(a).nbytes) for a in (x, y) + ((v,) if v is not None else ()))
        self.telemetry.reloads += 1
        self.telemetry.bytes_loaded += nbytes
        self.telemetry.load_seconds += dt
        self.telemetry.log.append({"frame": self.telemetry.frames, "rung": rung,
                                   "points": int(x.size), "window": want, "seconds": dt})
        self.loaded_window, self.loaded_rung, self.loaded_points = want, rung, int(x.size)
        return True

    # ---- input --------------------------------------------------------------
    def handle(self, ev: Event) -> "Explorer":
        cam = self.stage.camera
        if ev.kind == "drag":
            sx, sy = cam.scale
            cam.cx -= ev.dx / self.width * 2 / sx
            cam.cy += ev.dy / self.height * 2 / sy
        elif ev.kind == "scroll":
            about = self.px_to_world(ev.x, ev.y)
            cam.zoom(self.ZOOM_STEP ** ev.steps, about=about)
        elif ev.kind == "key":
            k = ev.key.lower()
            if k in ("0", "home", "f"):
                self.fit()
            elif k in ("+", "="):
                cam.zoom(self.ZOOM_STEP)
            elif k in ("-", "_"):
                cam.zoom(1 / self.ZOOM_STEP)
            elif k == "s":
                self.select_view(intent="need-data")
            elif k == "m" and self.active is not None:
                self.describe_active(); self.more_like_this()
            elif k == "n":
                sl = [x for x in self.selections() if x.kind in ("rect", "polygon")]
                if sl:
                    ids = [x.id for x in sl]
                    cur = getattr(self.active, "id", None)
                    i = (ids.index(cur) + 1) % len(sl) if cur in ids else 0
                    self.goto_selection(sl[i])
            elif k in ("left", "right", "up", "down"):
                step = 0.25
                d = {"left": (-step, 0), "right": (step, 0),
                     "up": (0, step), "down": (0, -step)}[k]
                cam.pan(*d)
        elif ev.kind == "goto":
            self.goto(ev.arg)
        elif ev.kind == "gene":
            self.gene(ev.arg)
        elif ev.kind == "recall":
            self.recall(ev.arg)
        self._clamp()
        return self

    def _clamp(self):
        """Keep the camera over the data, and never zoom out past the whole extent."""
        cam, f = self.stage.camera, self.full
        span_full = max(f[1] - f[0], f[3] - f[2]) * 1.05
        cam.span = min(max(cam.span, 1.0), span_full)
        cam.cx = min(max(cam.cx, f[0]), f[1])
        cam.cy = min(max(cam.cy, f[2]), f[3])

    def fit(self) -> "Explorer":
        from .stage import Camera
        self.stage.camera = Camera.fit(self.full, self.width, self.height)
        return self

    def goto(self, window: tuple) -> "Explorer":
        """Jump to an explicit (x0, x1, y0, y1)."""
        from .stage import Camera
        self.stage.camera = Camera.fit(tuple(window), self.width, self.height)
        return self

    def gene(self, name: Optional[str]) -> "Explorer":
        """Show only one gene (or all, with None). A contiguous slice, so the reload is cheap."""
        if not hasattr(self.view, "select_genes"):
            raise TypeError("gene filtering needs a GEF view")
        self.view.select_genes([name] if name else None)
        self._genes = name
        self.loaded_window = None                           # force the next frame to reload
        return self

    # ---- selections -------------------------------------------------------
    def selections(self) -> list:
        """Selections on this dataset: from the book if there is one, plus this session's."""
        seen = {s.id for s in self._sels}
        stored = self.book.find(dataset=self.view) if self.book is not None else []
        return self._sels + [s for s in stored if s.id not in seen]

    def select_view(self, name: Optional[str] = None, *, intent: str = "interesting",
                    note: str = "", bounds: Optional[tuple] = None):
        """Mark what is on screen (or `bounds`) as a selection, at the rung currently loaded."""
        from .selections import select
        if self.loaded_rung is not None:
            self._set_rung(self.loaded_rung)
        b = tuple(bounds) if bounds is not None else self.visible()
        sel = select(self.view, name or f"sel-{len(self._sels) + 1}", b, intent=intent,
                     note=note, book=self.book)
        self._sels.append(sel)
        self.active = sel
        return sel

    def describe_active(self):
        from .selections import describe
        if self.active is None:
            raise RuntimeError("no active selection; mark one with select_view()")
        return describe(self.active, self.view, book=self.book)

    def more_like_this(self, k: int = 6) -> list:
        """Candidates resembling the active selection, drawn in yellow on the next frame."""
        from .selections import similar
        if self.active is None:
            raise RuntimeError("no active selection")
        c = similar(self.active, self.view, k=k, book=self.book)
        self._sels.extend(c)
        return c

    def goto_selection(self, sel, pad: float = 0.25) -> "Explorer":
        x0, x1, y0, y1 = sel.bounds
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        self.active = sel
        return self.goto((x0 - px, x1 + px, y0 - py, y1 + py))

    def bookmark(self, name: str) -> "Explorer":
        cam = self.stage.camera
        self.bookmarks[name] = (cam.cx, cam.cy, cam.span, self._genes)
        return self

    def recall(self, name: str) -> "Explorer":
        cx, cy, span, genes = self.bookmarks[name]
        cam = self.stage.camera
        cam.cx, cam.cy, cam.span = cx, cy, span
        if genes != self._genes:
            self.gene(genes)
        return self

    # ---- frames -------------------------------------------------------------
    def frame(self) -> np.ndarray:
        reloaded = self._ensure_loaded()
        st = self.stage
        st.clear()
        vis = self.visible()
        # Planning probes several rungs through cost(), and each probe leaves the view on the
        # rung it tried last. Put it back on the rung actually loaded, or the ladder reports a
        # rung that was merely considered — header saying bin50 while the ladder lit bin20.
        if self.loaded_rung is not None:
            self._set_rung(self.loaded_rung)
        # the HUD describes what is on screen, not the margin that was loaded around it
        self.view.select(*vis)
        sels = self.selections()
        if sels:
            st.draw_selections(sels, active=getattr(self.active, "id", None))
        if self.hud:
            st.minimap(self.view, selections=sels)
            st.lod_ladder(self.view)
            if self.budget is not None:
                st.budget_panel(self.budget)
        self._header(reloaded)
        if self.hud:
            st.axes()
        t0 = time.perf_counter()
        img = st.render()
        self.telemetry.render_seconds += time.perf_counter() - t0
        self.telemetry.frames += 1
        return img

    def _header(self, reloaded: bool):
        st, t = self.stage, self.telemetry
        name = str(getattr(getattr(self.view, "gef", None), "chip", None) or "dataset")
        if self._genes:
            name += f"  ·  {self._genes}"
        rung = (f"bin{self.loaded_rung}" if hasattr(self.view, "gef")
                else f"zoom {self.loaded_rung}")
        sub = f"{self.loaded_points:,} pts loaded  ·  {rung}"
        state = ("reloaded this frame" if reloaded
                 else f"from memory  ·  {t.served_from_memory:.0%} of {t.frames + 1} frames")
        hx, hy = 150, 12
        w = max(st.atlas.measure(name, 1.25), st.atlas.measure(sub, 0.95),
                st.atlas.measure(state, 0.85)) + 18
        h = st.atlas.size * 4.1
        st._reserve(hx - 9, hy - 7, w, h)
        st.rect(hx - 9, hy - 7, w, h, (1.0, 0.99, 0.95, 0.92))
        st.outline(hx - 9, hy - 7, w, h, (0.72, 0.76, 0.80, 1.0), 1.0)
        st.label(name, hx, hy, scale=1.25)
        st.label(sub, hx, hy + st.atlas.size * 1.5, scale=0.95, color=(0.18, 0.47, 0.77, 1.0))
        st.label(state, hx, hy + st.atlas.size * 2.7, scale=0.85,
                 color=(0.91, 0.25, 0.18, 1.0) if reloaded else (0.35, 0.55, 0.30, 1.0))

    # ---- sessions -----------------------------------------------------------
    def record(self, events: Iterable, every: int = 1) -> list:
        """Play a scripted session headless and return (event, frame) pairs.

        This is how the explorer is tested, and how a session is shown in a paper: the same
        `handle()` the window uses, driven by a list instead of a mouse.
        """
        out = [(None, self.frame())]
        for i, ev in enumerate(events):
            ev = ev if isinstance(ev, Event) else Event(**ev)
            self.handle(ev)
            img = self.frame()
            if (i + 1) % every == 0:
                out.append((ev, img))
        return out

    def run(self, title: str = "GEMINGA") -> "Explorer":
        """Open a window and navigate with the mouse and keyboard.

        Drag to pan, scroll to zoom about the cursor, `0`/`f` to fit, arrow keys to step,
        `+`/`-` to zoom, `g` to cycle the top genes, `b` to bookmark, `r` to recall, `s` to mark
        the view as a selection, `m` for more like it, `n` to step through selections, `q` quits.
        Frames are rendered off-screen by the same code the headless path uses and blitted, so
        the window adds nothing the tests do not already cover.
        """
        try:
            import glfw
        except ImportError:
            raise RuntimeError("pip install glfw for a window; record() works without one")
        if not glfw.init():
            raise RuntimeError("no display available (glfw could not initialise); use record() "
                               "to produce frames headless")
        from OpenGL import GL                                  # blit only
        win = glfw.create_window(self.width, self.height, title, None, None)
        glfw.make_context_current(win)
        state = {"drag": None, "dirty": True}
        top = [n for n, _ in self.view.gene_profile(8)] if hasattr(self.view, "gene_profile") else []
        gi = {"i": -1}

        def on_button(w, b, action, mods):
            if b == glfw.MOUSE_BUTTON_LEFT:
                state["drag"] = glfw.get_cursor_pos(w) if action == glfw.PRESS else None

        def on_move(w, x, y):
            if state["drag"] is not None:
                px, py = state["drag"]
                self.handle(Event("drag", dx=x - px, dy=y - py))
                state["drag"] = (x, y); state["dirty"] = True

        def on_scroll(w, sx, sy):
            x, y = glfw.get_cursor_pos(w)
            self.handle(Event("scroll", x=x, y=y, steps=sy)); state["dirty"] = True

        def on_key(w, key, sc, action, mods):
            if action != glfw.PRESS:
                return
            name = glfw.get_key_name(key, sc) or {glfw.KEY_LEFT: "left", glfw.KEY_RIGHT: "right",
                                                  glfw.KEY_UP: "up", glfw.KEY_DOWN: "down",
                                                  glfw.KEY_HOME: "home"}.get(key, "")
            if name == "q":
                glfw.set_window_should_close(w, True)
            elif name == "g" and top:
                gi["i"] = (gi["i"] + 1) % (len(top) + 1)
                self.gene(top[gi["i"]] if gi["i"] < len(top) else None)
            elif name == "b":
                self.bookmark("b")
            elif name == "r" and "b" in self.bookmarks:
                self.recall("b")
            else:
                self.handle(Event("key", key=name))
            state["dirty"] = True

        glfw.set_mouse_button_callback(win, on_button)
        glfw.set_cursor_pos_callback(win, on_move)
        glfw.set_scroll_callback(win, on_scroll)
        glfw.set_key_callback(win, on_key)
        while not glfw.window_should_close(win):
            if state["dirty"]:
                img = np.flipud(self.frame())
                GL.glDrawPixels(self.width, self.height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img)
                glfw.swap_buffers(win)
                state["dirty"] = False
            glfw.wait_events_timeout(0.05)
        glfw.terminate()
        return self

    def __repr__(self) -> str:
        return (f"Explorer({type(self.view).__name__}, {self.width}x{self.height}, "
                f"{self.telemetry.summary()})")


def explore(target, budget: Optional[Budget] = None, **kw) -> Explorer:
    """An explorer over a GEF or GEM path (a `str` or any `os.PathLike`, such as a
    `pathlib.Path`), or over a view you already have."""
    import geminga as g
    if isinstance(target, (str, os.PathLike)):
        view = g.view(os.fspath(target), budget=budget)
    else:
        view = target
    return Explorer(view, budget=budget, **kw)
