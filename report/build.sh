#!/usr/bin/env bash
# Build the submission PDFs.
#
#   bash report/build.sh            # slides.pdf and report.pdf
#   bash report/build.sh slides     # just the deck
#
# slides.pdf is the submission: five 16:9 slides. report.pdf is the same
# material as a long-form A4 document, kept as backing detail.
#
# Headless Chrome rather than a LaTeX toolchain: it is already present
# wherever a browser is, and it honours the @page rules that the page limit
# depends on.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

targets=("${@:-slides report}")
# Unquoted on purpose: the default above is one word-splittable string.
# shellcheck disable=SC2206
targets=(${targets[@]})

python3 "$here/make_figures.py" --theme both

chrome=""
for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$candidate" >/dev/null 2>&1; then chrome="$candidate"; break; fi
done
if [ -z "$chrome" ]; then
  echo "no Chrome or Chromium found; open the .html and print to PDF" >&2
  exit 1
fi

status=0
for name in "${targets[@]}"; do
  html="$here/$name.html"
  pdf="$here/$name.pdf"
  if [ ! -f "$html" ]; then
    echo "no such document: $html" >&2
    status=1
    continue
  fi

  rm -f "$pdf"
  # Chrome often lingers after the file is written, so it is capped and the
  # result is judged by the PDF on disk rather than by the exit status.
  timeout 240 "$chrome" \
    --headless=new --disable-gpu --no-sandbox --disable-dev-shm-usage \
    --user-data-dir="$(mktemp -d)" \
    --no-pdf-header-footer \
    --print-to-pdf="$pdf" \
    "file://$html" >/dev/null 2>&1

  if [ ! -s "$pdf" ]; then
    echo "chrome produced no PDF for $name" >&2
    status=1
    continue
  fi

  python3 - "$pdf" <<'PY' || status=1
import re
import sys

data = open(sys.argv[1], "rb").read()
pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
size_mib = len(data) / 1024 / 1024
print(f"{sys.argv[1]}: {pages} pages, {size_mib:.2f} MiB")

# The brief caps the submission at 5 pages and 20 MiB. Failing the build is
# better than discovering it at upload time.
if pages > 5:
    sys.exit(f"FAIL: {pages} pages exceeds the 5-page limit")
if size_mib > 20:
    sys.exit(f"FAIL: {size_mib:.1f} MiB exceeds the 20 MiB limit")
print("within submission limits")
PY
done

exit "$status"
