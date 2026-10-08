#!/bin/bash
# P7 milestone C acceptance (P7 plan, Task 19): the addendum's four validator series on the crop (QUASAR,
# Look-Locker Hadamard, 3D Look-Locker, multi-TE 3D), then with --large the 3D trains on the full 3T
# phantom (work/phantom-3t, from tools/hrgt_to_bids.py), each with its run time, peak memory and the
# sidecar's interpolation nodes and memory estimate.
#
#     tools/acceptance_p7.sh <out dir> [--large]
#
# Needs a release build with `cli,kspace,par`, GNU time, python3 and deno (the bids-validator). The
# full phantom's field of view (233 mm, 117 lines at 2 mm) makes the fixtures' 0.5 ms effective echo
# spacing a 58.5 ms block per echo whatever the segmentation, and its 63 partitions at 3 mm make one
# train longer than the fixtures' repetition: the large runs shorten the effective spacing and segment
# the train (kz x ky shots), and one long train (21 excitations, TR 5 s) shows the interpolation cost.
set -u
OUT=${1:?usage: acceptance_p7.sh <out dir> [--large]}
LARGE=${2:-}
cd "$(dirname "$0")/.."
BIN=./target/release/aslscan
FX=tests/fixtures/protocols
mkdir -p "$OUT"
FAILED=0

run() { # name fixture-dir phantom [expected refusal text]
  local f=$2 jsons=""
  if [ -f "$f/asl.json" ]; then jsons="--asl-json $f/asl.json"; else for j in "$f"/asl-echo-*.json; do jsons="$jsons --asl-json $j"; done; fi
  rm -rf "$OUT/$1"
  /usr/bin/time -f "%e s, peak RSS %M KiB" -o "$OUT/$1.time" $BIN $jsons --aslcontext "$f/aslcontext.tsv" \
      --overlay "$f/overlay.toml" --phantom "$3" --out "$OUT/$1" > "$OUT/$1.log" 2>&1
  local rc=$?
  echo "== $1: exit $rc, $(tail -1 "$OUT/$1.time")"
  grep "^aslscan:" "$OUT/$1.log"
  if [ -n "${4:-}" ]; then
    if [ $rc -eq 0 ] || ! grep -q "$4" "$OUT/$1.log"; then echo "FAIL: $1 should be refused ($4)"; FAILED=1; fi
    return
  fi
  [ $rc -ne 0 ] && { echo "FAIL: $1 exited $rc"; FAILED=1; }
  local s
  s=$(find "$OUT/$1/sub-01/perf" -maxdepth 1 -name '*part-mag_asl.json' 2>/dev/null | sort | head -1)
  [ -n "$s" ] && python3 - "$s" <<'EOF'
import json, sys
ro = json.load(open(sys.argv[1]))["AslscanSimulation"].get("Readout", {})
li = ro.get("LabelInterpolation")
if li:
    print("   nodes K", li["NodeSlots"], "| achieved", li["AchievedRelativeError"], "| image memory estimate GiB",
          f'{ro["MemoryGiB"]["Estimate"]:.3g}')
EOF
}

validate() { # name
  echo "-- bids-validator $1"
  local log="$OUT/$1.validator.log"
  deno run -A jsr:@bids/validator "$OUT/$1" > "$log" 2>&1
  local rc=$?
  grep -E "\[ERROR\]|\[WARNING\]" "$log" | sort | uniq -c
  # the validator exits 1 on errors; anything else it prints as an error is a failure too
  if [ $rc -ne 0 ] || grep -qE "\[ERROR\]|^error:" "$log"; then echo "FAIL: $1 does not validate (exit $rc)"; FAILED=1; fi
}

variant() { # fixture kz ky effective-echo-spacing-s [repetition-s] [overlay lines]
  local v="$OUT/fx-$1-$2x$3${5:+-tr$5}"
  rm -rf "$v"
  cp -r "$FX/$1" "$v"
  python3 - "$v" "$2" "$3" "$4" "${5:-}" "${6:-}" <<'EOF'
import json, pathlib, sys
v, kz, ky, esp, tr, extra = pathlib.Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], sys.argv[6]
for j in v.glob("asl*.json"):
    s = json.loads(j.read_text())
    s["NumberShots"], s["EffectiveEchoSpacing"] = kz * ky, esp
    if tr:
        s["RepetitionTimePreparation"] = float(tr)
    j.write_text(json.dumps(s, indent=1) + "\n")
ov = (v / "overlay.toml").read_text()
assert "[readout]\n" in ov, "the fixture has a [readout] table"
(v / "overlay.toml").write_text(ov.replace("[readout]\n", f"[readout]\nkz_segments = {kz}\n", 1) + extra.replace("\\n", "\n"))
EOF
  echo "$v"
}

CROP=tests/fixtures/phantom-crop
run quasar $FX/p7_quasar $CROP
run llh $FX/p7_ll_hadamard $CROP
run ll3d $FX/p7_ll3d $CROP
run multite3d $FX/p7_multite3d $CROP
for n in quasar llh ll3d multite3d; do validate $n; done

if [ "$LARGE" = "--large" ]; then
  P3T=work/phantom-3t
  run ll3d-3t "$(variant p7_ll3d 9 3 0.0001)" $P3T
  run ge3d-3t "$(variant p7_ge3d 9 3 0.0001)" $P3T
  run multite3d-3t "$(variant p7_multite3d 9 13 0.00008)" $P3T
  validate ll3d-3t
  validate multite3d-3t
  # one long train: refused at the default image-memory limit, then run with a higher one
  run ge3d-3t-long "$(variant p7_ge3d 3 3 0.0001 5.0)" $P3T "max_memory_gib"
  run ge3d-3t-long-8gib "$(variant p7_ge3d 3 3 0.0001 5.0 '\n[images]\nmax_memory_gib = 8.0\n')" $P3T
fi
exit $FAILED
