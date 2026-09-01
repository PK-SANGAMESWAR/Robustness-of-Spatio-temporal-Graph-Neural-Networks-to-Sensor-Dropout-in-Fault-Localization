#!/usr/bin/env bash
# Compile paper/overleaf.tex locally and report the page count.
#
# WHY THIS EXISTS
# ---------------
# The project notes recorded that the paper "does not compile locally" because
# ecrc.sty (Elsevier CRC / Procedia) is not on CTAN, and that the page count was
# therefore checkable only on Overleaf.  The consequence was that the standing
# 12-page task was being worked on against a stale, unverifiable figure: the
# reference paper/preview-local.pdf no longer exists on disk, and the "13 pages"
# it was said to show is 2 pages under what the source actually produces.
#
# ecrc.sty is redistributable with the Elsevier template bundle and is kept in
# paper/texlib/ alongside two placeholder logo PDFs.  The logos are BLANK
# stand-ins for Elsevier's own artwork, which is not redistributable; they
# occupy a fixed-size box on page 1, so the layout and the page count match the
# real build, but the produced PDF is a PREVIEW and must not be submitted.
# Submit from Overleaf, where the real logos are present.
#
# USAGE
#   bash scripts/build_paper_preview.sh          # build + page count
#   bash scripts/build_paper_preview.sh --keep   # also leave the PDF in place
#
# The build runs in a temporary directory so no .aux/.log/.pdf litter lands in
# paper/.  Two passes are run so cross-references and citations resolve.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="$REPO/paper"
TEXLIB="$PAPER_DIR/texlib"
BUILD="$(mktemp -d)"
KEEP="${1:-}"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found; install TeX Live to use this script." >&2
  exit 1
fi
for f in ecrc.sty elsevier-logo-3p.pdf SDlogo-3p.pdf; do
  if [ ! -f "$TEXLIB/$f" ]; then
    echo "Missing $TEXLIB/$f — cannot build. See the header of this script." >&2
    exit 1
  fi
done

cd "$PAPER_DIR"
for pass in 1 2; do
  TEXINPUTS="$TEXLIB:./:" pdflatex -interaction=nonstopmode \
    -output-directory="$BUILD" overleaf.tex >/dev/null 2>&1 || true
done

if [ ! -f "$BUILD/overleaf.pdf" ]; then
  echo "Build produced no PDF. Errors:" >&2
  grep -E "^!" "$BUILD/overleaf.log" | head -20 >&2
  exit 1
fi

PAGES="$(pdfinfo "$BUILD/overleaf.pdf" | awk '/^Pages:/ {print $2}')"
echo "=== paper/overleaf.tex preview build"
echo "Pages: $PAGES"

# Undefined references/citations silently change the page count, so surface them.
if grep -q "undefined" "$BUILD/overleaf.log"; then
  echo "--- undefined references/citations (page count may shift once fixed):"
  grep -oE "(Reference|Citation) \`[^']*'" "$BUILD/overleaf.log" | sort -u | head
fi

BAD="$(grep -cE "^! " "$BUILD/overleaf.log" || true)"
[ "$BAD" -gt 0 ] && { echo "--- LaTeX errors:"; grep -E "^! " "$BUILD/overleaf.log" | sort -u | head; }

if [ "$KEEP" = "--keep" ]; then
  cp "$BUILD/overleaf.pdf" "$PAPER_DIR/preview-local.pdf"
  echo "Wrote $PAPER_DIR/preview-local.pdf (PREVIEW ONLY — placeholder logos)"
fi
rm -rf "$BUILD"
