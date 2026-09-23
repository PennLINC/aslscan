#!/usr/bin/env bash
# Fetch real BIDS ASL sidecars from bids-examples into tests/fixtures/protocols/<example>/.
# Records the commit SHA fetched in tests/fixtures/protocols/SOURCES.md. Re-run to refresh.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HERE/tests/fixtures/protocols"
REPO="bids-standard/bids-examples"
SHA="$(curl -sf "https://api.github.com/repos/$REPO/commits/master" | grep -o '"sha": *"[0-9a-f]\{40\}"' | head -1 | grep -o '[0-9a-f]\{40\}')"
RAW="https://raw.githubusercontent.com/$REPO/$SHA"

# example subject
declare -A SUBJ=( [asl001]=sub-Sub103 [asl002]=sub-Sub103 [asl003]=sub-Sub1 [asl004]=sub-Sub1 [asl005]=sub-Sub103 )

{
  echo "# Protocol fixture sources"
  echo
  echo "Fetched from https://github.com/$REPO at commit \`$SHA\` by tools/fetch_protocol_fixtures.sh."
  echo "Only the JSON sidecar and aslcontext.tsv are kept; no image data."
  echo
} > "$OUT/SOURCES.md"

for ex in "${!SUBJ[@]}"; do
  s="${SUBJ[$ex]}"
  mkdir -p "$OUT/$ex"
  curl -sf "$RAW/$ex/$s/perf/${s}_asl.json" -o "$OUT/$ex/asl.json"
  curl -sf "$RAW/$ex/$s/perf/${s}_aslcontext.tsv" -o "$OUT/$ex/aslcontext.tsv"
  echo "- \`$ex/\`: \`$ex/$s/perf/${s}_asl.json\` and \`..._aslcontext.tsv\`" >> "$OUT/SOURCES.md"
  echo "fetched $ex ($s)"
done
