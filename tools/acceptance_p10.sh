#!/usr/bin/env bash
# P10 acceptance (P10 plan, Tasks 5 and 7): timed motion traces on the P9 ASL figure protocol
# (tests/fixtures/protocols/p10_mb2: 2D PCASL, multiband 2 interleaved, 20 slices in 10 shots 40 ms
# apart, TR 4.6 s) on work/phantom-3t-z100; every run validated; tools/acceptance_p10_check.py
# checks the numbers.
#
#     tools/acceptance_p10.sh <out dir>
#
# Needs a release build with `cli,kspace,par`, python3, the rust-trx micromamba environment and deno
# (the bids-validator, under ~/.nvm/versions/node/v22.22.2/bin). Exits nonzero on any failure.
set -euo pipefail
OUT=${1:?usage: acceptance_p10.sh <out dir>}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export PATH=$HOME/.nvm/versions/node/v22.22.2/bin:$PATH
BIN=./target/release/aslscan
FX=tests/fixtures/protocols/p10_mb2
PH=work/phantom-3t-z100
FAILED=0

run() { # name overlay
  rm -rf "${OUT:?}/$1"
  local s rc=0
  s=$(date +%s.%N)
  $BIN --asl-json $FX/asl.json --aslcontext $FX/aslcontext.tsv --overlay "$2" --phantom $PH --out "$OUT/$1" \
      > "$OUT/$1.log" 2>&1 || rc=$?
  echo "== $1: exit $rc, $(echo "$(date +%s.%N) - $s" | bc) s"
  grep -E "^(contrast|spin history|note)" "$OUT/$1.log" || true
  if [ $rc -ne 0 ]; then echo "FAIL: $1 exited $rc"; tail -3 "$OUT/$1.log"; FAILED=1; return; fi
  if ! deno run -A jsr:@bids/validator "$OUT/$1" > "$OUT/$1.validator.txt" 2>&1; then
    echo "FAIL: $1 does not validate"; grep -iE "error" "$OUT/$1.validator.txt" | head -5; FAILED=1
  fi
}
overlay() { # name motion-lines
  { cat $FX/overlay.toml; printf '%b' "$2"; } > "$OUT/$1.toml"
  echo "$OUT/$1.toml"
}

# the nod: 2 mm forward, 3 mm up, 0.06 rad pitch, rising across volume 12's ten shots (58.80 to
# 59.16 s on the series clock) and falling across volume 13's (63.40 to 63.76 s)
NOD="start=58.8,rise=0.36,hold=4.24,fall=0.36,trans=0:2:3,rot=0.06:0:0"
python3 tools/motion_trace.py --rate 1000 --duration 190 --out "$OUT/nod-linear.tsv" --nod "$NOD,shape=linear"
python3 tools/motion_trace.py --rate 100 --duration 190 --out "$OUT/nod-cosine.tsv" --nod "$NOD"
cp $FX/nod_events.tsv "$OUT/nod_events.tsv"

run still "$(overlay still '')"
# criterion 2: the timed linear nod against P9's per-shot events, spin history on
run events "$(overlay events '[motion]\nspin_history = true\n[motion.within_volume]\nevents = "nod_events.tsv"\n')"
run timed-linear "$(overlay timed-linear '[motion]\nmode = "trajectory"\ntrajectory = "nod-linear.tsv"\nwithin_volume_sampling = "shot"\nspin_history = true\n')"
# criterion 3: the cosine nod, one pose per volume against one per shot, spin history on
run timed-volume "$(overlay timed-volume '[motion]\nmode = "trajectory"\ntrajectory = "nod-cosine.tsv"\nspin_history = true\n')"
run timed-shot "$(overlay timed-shot '[motion]\nmode = "trajectory"\ntrajectory = "nod-cosine.tsv"\nwithin_volume_sampling = "shot"\nspin_history = true\n')"

"$HOME/.local/bin/micromamba" run -n rust-trx python tools/acceptance_p10_check.py "$OUT" || FAILED=1
[ $FAILED = 0 ] && echo "P10 acceptance: all passed" || { echo "P10 acceptance: FAILED"; exit 1; }
