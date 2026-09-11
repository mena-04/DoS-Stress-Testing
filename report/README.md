# Submission report

Five pages, PDF, matching the required section order: title, project objective,
proposed solution, solution validation, results and conclusions.

**The built PDF is committed at `report/report.pdf`.** Nothing needs to be run
to submit it — download that file. The rest of this section is only for
regenerating it after an edit.

```bash
bash report/build.sh     # regenerates figures, prints report.pdf, checks the limits
```

`build.sh` fails rather than warns if the output exceeds 5 pages or 20 MiB, so a
paragraph added to `report.html` cannot quietly push the submission over.

### Rebuilding in Colab

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

### Rebuilding without a command line

`report.html` is self-contained apart from `figures/`, so opening it in any
browser and printing to PDF gives the same output — it is the same rendering
engine `build.sh` drives. Choose A4, margins "Default", and turn off headers
and footers, or the page count grows past five.

## Files

| File | Purpose |
|---|---|
| `report.html` | The report. Print CSS (`@page size: A4`) drives pagination; one `<div class="page">` per printed page. |
| `data/t4_runs.json` | Every measured number the report quotes, with the notebook commit and cell each came from. |
| `make_figures.py` | Generates the three figures from that file. No number is introduced here. |
| `build.sh` | Figures, then HTML to PDF via headless Chrome, then the limit check. |
| `figures/` | Generated output; regenerate rather than edit. |

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
`build.sh`, not editing a figure or a table cell. The table in `report.html` is
written out in full for typesetting reasons, so if you change the data file,
check the table against it.

Team and member names are in the `.meta` block on the title page.
