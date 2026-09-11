# Submission deck

Five 16:9 slides, PDF, matching the required section order: title, project
objective, proposed solution, solution validation, results and conclusions.

**The built deck is committed at `report/slides.pdf`.** Nothing needs to be run
to submit it — download that file. The rest of this page is only for
regenerating it after an edit.

```bash
bash report/build.sh            # slides.pdf and report.pdf, with the limit check
bash report/build.sh slides     # just the deck
```

`build.sh` fails rather than warns if either output exceeds 5 pages or 20 MiB,
so a paragraph added to a slide cannot quietly push the submission over.

`report.pdf` is the same material as a long-form A4 document. It is not the
submission; it is kept because it has room for detail the slides cannot hold.

## Fonts

| Role | Family | Where it is used |
|---|---|---|
| Display | Artifakt, falling back to Fira Sans | Slide titles, headings, card titles |
| Body | Aptos | All running text |
| Mono | Cascadia Mono | Numbers, code, flags, reason codes |

Artifakt is Autodesk's corporate typeface and is licensed only for Autodesk
work, so it cannot be redistributed or installed here. The CSS asks for it
first and falls back to **Fira Sans**, which Erik Spiekermann and Ralph du
Carrois also designed and which is openly licensed (OFL) — the closest
legitimate match. On a machine that has Artifakt installed, the deck picks it
up with no change.

Aptos is free from the Microsoft Download Center. Cascadia Mono is OFL and ships
with Windows Terminal, VS Code and most Linux distributions. If a family is
missing the deck still builds, it just substitutes.

## Rebuilding in Colab

Colab has matplotlib but no browser, and `apt install chromium-browser` there
resolves to a snap stub that cannot run. Install Chrome from Google's `.deb`
instead:

```python
!wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
!apt-get install -y -q ./google-chrome-stable_current_amd64.deb
!bash report/build.sh
```

Chrome commonly lingers for a minute or two after writing the file, which is
why `build.sh` caps it with `timeout` and judges the result by the PDF on disk
rather than by the exit status.

## Rebuilding without a command line

`slides.html` is self-contained apart from `figures/`, so opening it in any
browser and printing to PDF gives the same output — it is the same rendering
engine `build.sh` drives. Turn off headers and footers and set margins to
"None", or the slide size is wrong and the page count grows past five.

## Files

| File | Purpose |
|---|---|
| `slides.html` | The deck. `@page size: 338.667mm 190.5mm` is 13.333in x 7.5in, the standard 16:9 slide. One `<section class="slide">` per slide. |
| `report.html` | The same material as an A4 document, for detail that does not fit on a slide. |
| `data/t4_runs.json` | Every measured number either document quotes, with the notebook commit and cell each came from. |
| `make_figures.py` | Generates the three figures, in a light theme for the report and a dark theme for the deck. No number is introduced here. |
| `build.sh` | Figures, then HTML to PDF via headless Chrome, then the limit check. |
| `figures/`, `figures/dark/` | Generated output; regenerate rather than edit. |

## Where the numbers come from

`data/t4_runs.json` is transcribed from the printed output of
`scripts/smoke_load.py` in `integration.ipynb` at commit `3a6690e`, where the
two arms were run against live vLLM on a Tesla T4. Each run records its
notebook cell and execution count so any figure can be traced back to the cell
that produced it.

One run is listed under `excluded_runs` rather than deleted: a pass in which
every request returned `ConnectError` because no gateway was listening on the
target port. Its 35,128 attacker "requests" are refused connections returning
instantly, not load. Keeping the exclusion visible is the point.

## Editing

Changing a measured value means changing `data/t4_runs.json` and re-running
`build.sh`, not editing a figure or a table cell. The results table is written
out in full for typesetting reasons, so if you change the data file, check the
table against it.

Team and member names are on the title slide, in the `.byline` block.
