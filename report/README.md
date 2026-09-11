# Submission deck

Five 16:9 slides, PDF, matching the required section order: title, project
objective, proposed solution, solution validation, results and conclusions.

**The built deck is committed at `report/slides.pdf`.** Nothing needs to be run
to submit it — download that file. The rest of this page is only for
regenerating it after an edit.

```bash
pip install -e ".[report]"      # matplotlib; not needed to run the gateway
bash report/build.sh            # regenerates figures, prints slides.pdf, checks limits
```

`build.sh` fails rather than warns if the output exceeds 5 pages or 20 MiB, so
a paragraph added to a slide cannot quietly push the submission over.

There is deliberately only one document. An earlier long-form A4 `report.pdf`
was removed: it duplicated the deck's material and its results section still
described the single-attacker smoke run as the headline, which the repository
README correctly flagged as contradicting the primary Locust result. Keeping
one artifact means there is nothing that can disagree with itself. It is in
git history if it is ever wanted back.

## What the deck reports

The primary result is the paired Locust `flood` run
`demo-20260911-180828-5190b1`, transcribed into `data/locust_flood.json` from
the raw per-request logs in `results/demo_evidence.zip` — not from console
output. It is not a flattering result: the queue cap bounded the backend queue
to zero and the gateway still refused 92% of legitimate requests, because
`cheap_cost_threshold` was 512 while legitimate requests cost 48–49 and
attacker requests cost 320–322, so both classes counted as cheap and the
reserved lane partitioned nothing.

The earlier single-attacker smoke run is in `data/t4_runs.json` and is
referenced as a small-sample secondary, never mixed with the flood numbers.
It has 3 legitimate requests in one arm and 15 in the other.

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

Colab already has matplotlib, so `.[report]` is not needed there.

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
| `data/locust_flood.json` | The primary result: the paired Locust flood run, with its provenance, the cost distributions that explain it, and the config in effect. |
| `data/t4_runs.json` | The earlier single-attacker smoke run, with the notebook cell each number came from, plus the backend calibration. |
| `make_figures.py` | Generates the figures, in a light theme and a dark theme, from those two files. No number is introduced here. |
| `build.sh` | Figures, then HTML to PDF via headless Chrome, then the limit check. |
| `figures/`, `figures/dark/` | Generated output; regenerate rather than edit. |

Figures 4 and 5 come from the flood data and are the ones the deck uses.
Figures 1 to 3 come from the smoke data and are kept because they are what the
earlier arm measured.

## Editing

Changing a measured value means changing the data file and re-running
`build.sh`, not editing a figure or a table cell. The results table on slide 5
is written out in full for typesetting reasons, so if you change
`data/locust_flood.json`, check the table against it.

Team and member names are on the title slide, in the `.byline` block.
