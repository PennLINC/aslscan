#!/usr/bin/env bash
# Byte identity of this build's output against a base revision, for protocols that use none of
# the features added since (the P2/P4 regression check, made a tool):
#
#     tools/regress_identity.sh p2-complete
#
# Builds the base in a sibling worktree (../aslscan-base-<rev>, its own target dir) and this
# tree, runs both binaries on the same protocols, and compares every NIfTI (decompressed) and
# every sidecar byte for byte. Needs work/phantom-3t and work/phantom-3t-z100 (see
# tools/hrgt_to_bids.py and the P3 plan). Exits nonzero on any difference or failed run.
set -euo pipefail
BASE=${1:?usage: tools/regress_identity.sh <base-rev>}
export PATH=$HOME/.cargo/bin:$PATH
HERE=$(cd "$(dirname "$0")/.." && pwd)
WT=$(cd "$HERE/.." && pwd)/aslscan-base-${BASE//\//-}
OUT=$HERE/work/regress
rm -rf "$OUT"; mkdir -p "$OUT/inputs"
if [ ! -d "$WT" ]; then
  git -C "$HERE" worktree add -q "$WT" "$BASE"
fi
(cd "$WT" && git checkout -q "$BASE" && CARGO_TARGET_DIR="$WT/target" cargo build -q --release --features cli,kspace,par)
(cd "$HERE" && cargo build -q --release --features cli,kspace,par)
OLD=$WT/target/release/aslscan
NEW=$HERE/target/release/aslscan
P=$HERE/tests/fixtures/protocols
FULL=$HERE/work/phantom-3t
Z100=$HERE/work/phantom-3t-z100
CROP=$HERE/tests/fixtures/phantom-crop
A=$OUT/inputs
printf '[m0]\nrepetition_time = 8.0\n' > $A/ov.toml
printf 'seed = 5\n[m0]\nrepetition_time = 8.0\n[motion]\nmode = "random"\ntrans_mm = [2.0, 2.0, 1.0]\nrot_deg = [1.0, 1.0, 2.0]\nvolumes = [5, 20, 40]\n' > $A/ov-motion.toml
printf '[m0]\nrepetition_time = 8.0\n[acquisition]\nnoise_variance = 4.0\n' > $A/ov-noise.toml
printf '[m0]\nrepetition_time = 8.0\n[signal]\nacq_contrast = "ir"\n' > $A/ov-ir.toml
sed 's/"BackgroundSuppression":true/"BackgroundSuppression":false/' $P/asl002/asl.json > $A/asl002-off.json
# a second geometry for the P3 paths: the crop at 2 x 2 x 3 mm
crop_json() {
  printf '{"ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012, %s}\n' "$1"
}
crop_json '"BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2, "BackgroundSuppressionPulseTime": [2.0, 3.2]' > $A/crop-bs.json
crop_json '"BackgroundSuppression": false' > $A/crop-plain.json
printf 'control\nlabel\ncontrol\nlabel\n' | sed '1i volume_type' > $A/crop-ctx.tsv
printf 'seed = 3\n[signal]\nacq_contrast = "ir"\n' > $A/crop-ir.toml
printf 'seed = 3\n[motion]\nmode = "random"\ntrans_mm = [1.0, 1.0, 0.0]\nrot_deg = [0.0, 0.0, 2.0]\nvolumes = [1, 3]\n' > $A/crop-motion.toml
printf 'seed = 3\n' > $A/crop-seed.toml
cases=(
  "pasl_cutoff|$P/pasl_cutoff/asl.json|$P/pasl_cutoff/aslcontext.tsv|$P/pasl_cutoff/overlay.toml|$FULL"
  "crop_pcasl|$P/crop_pcasl/asl.json|$P/crop_pcasl/aslcontext.tsv|$P/crop_pcasl/overlay.toml|$CROP"
  "asl002_bs|$P/asl002/asl.json|$P/asl002/aslcontext.tsv|$A/ov.toml|$Z100"
  "asl002_motion|$P/asl002/asl.json|$P/asl002/aslcontext.tsv|$A/ov-motion.toml|$Z100"
  "asl002_noise|$A/asl002-off.json|$P/asl002/aslcontext.tsv|$A/ov-noise.toml|$Z100"
  "asl002_ir|$A/asl002-off.json|$P/asl002/aslcontext.tsv|$A/ov-ir.toml|$Z100"
  "crop_bs|$A/crop-bs.json|$A/crop-ctx.tsv|$A/crop-seed.toml|$CROP"
  "crop_ir|$A/crop-plain.json|$A/crop-ctx.tsv|$A/crop-ir.toml|$CROP"
  "crop_motion|$A/crop-plain.json|$A/crop-ctx.tsv|$A/crop-motion.toml|$CROP"
)
fail=0
for c in "${cases[@]}"; do
  IFS='|' read -r name json ctx ov ph <<< "$c"
  for side in old new; do
    bin=$OLD; [ $side = new ] && bin=$NEW
    if ! $bin --asl-json $json --aslcontext $ctx --overlay $ov --phantom $ph --out $OUT/$side-$name > $OUT/log-$side-$name.txt 2>&1; then
      echo "$name: $side run FAILED"; tail -2 $OUT/log-$side-$name.txt; fail=1; continue 2
    fi
  done
  n=0; bad=0
  while IFS= read -r f; do
    rel=${f#$OUT/old-$name/}
    g=$OUT/new-$name/$rel
    n=$((n+1))
    if [ ! -f "$g" ]; then echo "  MISSING $name $rel"; bad=1; continue; fi
    case $f in
      *.nii.gz) cmp -s <(zcat "$f") <(zcat "$g") || { echo "  DIFF $name $rel"; bad=1; } ;;
      *) cmp -s "$f" "$g" || { echo "  DIFF $name $rel"; bad=1; } ;;
    esac
  done < <(find $OUT/old-$name -type f)
  extra=$(find $OUT/new-$name -type f | wc -l)
  [ "$extra" = "$n" ] || { echo "  file count $n vs $extra"; bad=1; }
  echo "$name: $n files, $([ $bad = 0 ] && echo identical || echo DIFFERENT)"
  [ $bad = 0 ] || fail=1
done
exit $fail
