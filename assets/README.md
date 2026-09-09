# Assets

| File | Use | Notes |
|---|---|---|
| `logo.svg`, `logo-512.png`, `logo-128.png` | app icon, favicon, badges | Clean vector. Legible down to 32 px — the sketch is not. |
| `logo-wordmark.svg`, `logo-wordmark.png` | README header, slides | Mark + wordmark + tagline lockup. |
| `infographic.jpg` | README hero, posters, grant decks | The pulsar and the project on one sheet. |
| `hero-pulsar.jpg` | talks, blog posts | The pulsar panel alone. |
| `social-preview.jpg` | GitHub Settings → Social preview | 1280×640, the size GitHub asks for. |

Palette: ink `#1A1A1A`, red `#E8412F`, yellow `#F5C518`, blue `#2E77C4`, deep navy `#0B2545`,
paper `#FFFCF3`. The three bars under the wordmark are red / yellow / blue in that order.

The bottom-right panel of `infographic.jpg` is regenerated from `tools/build_panel.py`, so
project facts in the figure can be updated without redrawing the whole sheet. The other three
panels are hand-drawn artwork.

Before shipping the wordmark anywhere that matters, convert its text to paths — it currently
depends on a system sans-serif being available.
