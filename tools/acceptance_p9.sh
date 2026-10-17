#!/usr/bin/env bash
# P9 acceptance (P9 plan, Tasks 5 and 10): the motivating single-band protocol
# (tests/fixtures/protocols/p9_single_band: single-PLD PCASL, 20 pairs, 20 slices of 5 mm ascending,
# TR 4.6 s) on work/phantom-3t-z100, with each P9 overlay; every run validated.
#
#     tools/acceptance_p9.sh <out dir>
#
# Needs a release build with `cli,kspace,par`, python3 and deno (the bids-validator, under
# ~/.nvm/versions/node/v22.22.2/bin). Exits nonzero on any failure.
set -euo pipefail
OUT=${1:?usage: acceptance_p9.sh <out dir>}
cd "$(dirname "$0")/.."
export PATH=$HOME/.nvm/versions/node/v22.22.2/bin:$PATH
BIN=./target/release/aslscan
FX=tests/fixtures/protocols/p9_single_band
PH=work/phantom-3t-z100
mkdir -p "$OUT"
FAILED=0

run() { # name overlay
  rm -rf "${OUT:?}/$1"
  local s rc=0
  s=$(date +%s.%N)
  $BIN --asl-json $FX/asl.json --aslcontext $FX/aslcontext.tsv --overlay "$FX/$2" --phantom $PH --out "$OUT/$1" \
      > "$OUT/$1.log" 2>&1 || rc=$?
  echo "== $1: exit $rc, $(echo "$(date +%s.%N) - $s" | bc) s"
  grep -E "^(contrast|simulated)" "$OUT/$1.log" || true
  if [ $rc -ne 0 ]; then echo "FAIL: $1 exited $rc"; tail -3 "$OUT/$1.log"; FAILED=1; return; fi
  if ! deno run -A jsr:@bids/validator "$OUT/$1" > "$OUT/$1.validator.txt" 2>&1; then
    echo "FAIL: $1 does not validate"; grep -iE "error" "$OUT/$1.validator.txt" | head -5; FAILED=1
  fi
}

# criterion 1: single-band within-volume events, one slice per event
run events-random overlay.toml
run drift-slice overlay-slice.toml
run events-file overlay-events.toml
python3 - "$OUT" <<'EOF' || FAILED=1
import json, sys
out = sys.argv[1]
gt = lambda run, f: f"{out}/{run}/sub-01/perf/ground-truth/sub-01_desc-{f}_gt.tsv"
ok = True
def check(cond, msg):
    global ok
    print(("ok   " if cond else "FAIL ") + msg)
    ok &= cond
rows = [l.split("\t") for l in open(gt("events-random", "motionEvents")).read().splitlines()[1:]]
check(len(rows) > 0 and all(len(r[2].split(",")) == 1 for r in rows), f"criterion 1: {len(rows)} events, one slice each")
for run in ["events-random", "drift-slice", "events-file"]:
    shots = open(gt(run, "motionShots")).read().splitlines()[1:]
    check(len(shots) == 40 * 20, f"{run}: {len(shots)} shot rows (40 volumes x 20 shots)")
    m = json.load(open(f"{out}/{run}/sub-01/perf/sub-01_part-mag_asl.json"))["AslscanSimulation"]["Motion"]
    check(m.get("Path") == "shot", f"{run}: sidecar Motion.Path = {m.get('Path')}")
# the drift run: shot 19 of volume 0 lies on the way to volume 1's pose (0.5 / 39 mm per volume)
shots = [l.split("\t") for l in open(gt("drift-slice", "motionShots")).read().splitlines()[1:]]
z19 = float(shots[19][6])
check(0.0 < z19 < 0.5 / 39, f"drift: volume 0, shot 19 trans_z {z19:.6f} mm, between 0 and {0.5 / 39:.6f}")
ev = [l.split("\t") for l in open(gt("events-file", "motionEvents")).read().splitlines()[1:]]
check(len(ev) == 1 and ev[0][:3] == ["0", "10", "10"], f"events file: {ev}")
sys.exit(0 if ok else 1)
EOF
[ $FAILED = 0 ] && echo "P9 acceptance: all passed" || { echo "P9 acceptance: FAILED"; exit 1; }
