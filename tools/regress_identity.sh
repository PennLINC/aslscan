#!/usr/bin/env bash
# Byte identity of this build's output against a base, for protocols that use none of the
# features added since (the P2/P4 regression check, made a tool; reworked for P5):
#
#     tools/regress_identity.sh <aslscan-rev> <mrsim-acq-rev>            # e.g. p4-complete 054bfdf
#     tools/regress_identity.sh <aslscan-rev> <mrsim-acq-rev> --self-test
#
# aslscan depends on mrsim-acq by `path = "../mrsim-acq"`, so a base aslscan built in a lone
# worktree links the LIVE mrsim-acq and a change to the acquisition stage is in both binaries.
# The base is therefore a pair of side-by-side worktrees, ../p5-base/{aslscan,mrsim-acq}, each
# clean and at its revision, and `cargo metadata` confirms which mrsim-acq every build resolved.
# Both feature sets are built (cli,kspace,par and cli: their bits differ), every NIfTI is compared
# decompressed and every sidecar byte for byte.
#
# The new side is a snapshot of the live sources, ../p5-new/{aslscan,mrsim-acq}, taken at the start.
# --self-test shows the comparison can fail: the "new" side is ../p5-selftest/{aslscan,mrsim-acq},
# copies of the live sources with mrsim-acq's forward signal scale multiplied by 1 + 1e-4 (large
# enough to survive reconstruction and the f32 cast; never committed). It passes only if every run
# succeeds AND at least one NIfTI differs in content.
#
# Needs work/phantom-3t, work/phantom-3t-z100 and work/phantom-3t-z97 (see tools/hrgt_to_bids.py:
# z97 is --crop 0:197 0:233 46:143, which fits asl004's 24 slices of 4.05 mm).
# Exits nonzero on any difference (or, under --self-test, on no difference) or failed run.
set -euo pipefail
A_REV=${1:?usage: tools/regress_identity.sh <aslscan-rev> <mrsim-acq-rev> [--self-test]}
M_REV=${2:?usage: tools/regress_identity.sh <aslscan-rev> <mrsim-acq-rev> [--self-test]}
SELF=${3:-}
export PATH=$HOME/.cargo/bin:$PATH
HERE=$(cd "$(dirname "$0")/.." && pwd)
PARENT=$(cd "$HERE/.." && pwd)
LIVE_M=$PARENT/mrsim-acq
BASE=$PARENT/p5-base
OUT=$HERE/work/regress
rm -rf "$OUT"; mkdir -p "$OUT/inputs"

common() { realpath "$(git -C "$1" rev-parse --path-format=absolute --git-common-dir)"; }
# A worktree of repository $1 at $3, at path $2: created if absent; a reused one must be that
# repository's, clean (untracked files included: a stray source file would be built), and at the
# commit, or the "base" binary is not the base.
pin() {
  local repo=$1 wt=$2 rev=$3 sha
  sha=$(git -C "$repo" rev-parse --verify "$rev^{commit}")
  [ -d "$wt" ] || git -C "$repo" worktree add -q --detach "$wt" "$sha"
  [ "$(common "$wt")" = "$(common "$repo")" ] || { echo "$wt is not a worktree of $repo"; exit 1; }
  git -C "$wt" checkout -q --detach "$sha"
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=all)" ] \
    || { echo "$wt has local changes; remove it or clean it"; git -C "$wt" status --short; exit 1; }
  [ "$(git -C "$wt" rev-parse HEAD)" = "$sha" ] || { echo "$wt is not at $sha"; exit 1; }
}
mkdir -p "$BASE"
pin "$HERE" "$BASE/aslscan" "$A_REV"
pin "$LIVE_M" "$BASE/mrsim-acq" "$M_REV"

# The new side builds from a snapshot of the live sources taken now, so an edit made while the
# (long) comparison runs cannot change what is being compared. rsync --delete keeps unchanged
# files' times, so the snapshot's own target directories stay incremental between runs.
NEWROOT=$PARENT/p5-new
[ "$SELF" = "--self-test" ] && NEWROOT=$PARENT/p5-selftest
mkdir -p "$NEWROOT"
rsync -a --delete --exclude target --exclude 'target-*' --exclude work --exclude .git "$HERE/" "$NEWROOT/aslscan/"
rsync -a --delete --exclude target --exclude 'target-*' --exclude .git "$LIVE_M/" "$NEWROOT/mrsim-acq/"
echo "new side: aslscan $(git -C "$HERE" rev-parse --short HEAD)$([ -z "$(git -C "$HERE" status --porcelain -- src Cargo.toml)" ] || echo +local), mrsim-acq $(git -C "$LIVE_M" rev-parse --short HEAD)$([ -z "$(git -C "$LIVE_M" status --porcelain -- src Cargo.toml)" ] || echo +local)"
if [ "$SELF" = "--self-test" ]; then
  K=$NEWROOT/mrsim-acq/src/kspace.rs
  sed -i 's/amp\[i\] = acq\.signal_scale \* coil_sensitivity(/amp[i] = acq.signal_scale * 1.0001 * coil_sensitivity(/' "$K"
  grep -q 'acq.signal_scale \* 1.0001 \* coil_sensitivity(' "$K" || { echo "self-test patch did not apply"; exit 1; }
elif [ -n "$SELF" ]; then
  echo "unknown option $SELF"; exit 1
fi

# The mrsim-acq a build resolved, from cargo metadata, must be the one expected.
resolved() {
  (cd "$1" && cargo metadata -q --format-version 1) \
    | grep -o '"manifest_path":"[^"]*mrsim-acq/Cargo.toml"' | head -1 | sed 's/.*:"\(.*\)\/Cargo.toml"/\1/'
}
check_dep() {
  local got; got=$(realpath "$(resolved "$1")")
  [ "$got" = "$(realpath "$2")" ] || { echo "$1 resolves mrsim-acq at $got, expected $2"; exit 1; }
}
check_dep "$BASE/aslscan" "$BASE/mrsim-acq"
check_dep "$NEWROOT/aslscan" "$NEWROOT/mrsim-acq"

P=$HERE/tests/fixtures/protocols
FULL=$HERE/work/phantom-3t
Z100=$HERE/work/phantom-3t-z100
Z97=$HERE/work/phantom-3t-z97
CROP=$HERE/tests/fixtures/phantom-crop
A=$OUT/inputs
printf '[m0]\nrepetition_time = 8.0\n' > $A/ov.toml
printf 'seed = 5\n[m0]\nrepetition_time = 8.0\n[motion]\nmode = "random"\ntrans_mm = [2.0, 2.0, 1.0]\nrot_deg = [1.0, 1.0, 2.0]\nvolumes = [5, 20, 40]\n' > $A/ov-motion.toml
printf '[m0]\nrepetition_time = 8.0\n[acquisition]\nnoise_variance = 4.0\n' > $A/ov-noise.toml
printf '[m0]\nrepetition_time = 8.0\n[signal]\nacq_contrast = "ir"\n' > $A/ov-ir.toml
sed 's/"BackgroundSuppression":true/"BackgroundSuppression":false/' $P/asl002/asl.json > $A/asl002-off.json
# asl004 as published reads its first line before the excitation in this model (the main spec's
# deferred pre-echo line timing), so it runs with the readout shortened to 0.025 s
sed 's/"TotalReadoutTime":0.06/"TotalReadoutTime":0.025/' $P/asl004/asl.json > $A/asl004.json
# a second geometry for the P3 paths: the crop at 2 x 2 x 3 mm
crop_json() {
  printf '{"ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012, %s}\n' "$1"
}
crop_json '"BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2, "BackgroundSuppressionPulseTime": [2.0, 3.2]' > $A/crop-bs.json
crop_json '"BackgroundSuppression": false' > $A/crop-plain.json
crop_json '"BackgroundSuppression": false, "VascularCrushing": true, "VascularCrushingVENC": [0.0, 4.0, 0.0, 4.0]' > $A/crop-crush.json
printf 'control\nlabel\ncontrol\nlabel\n' | sed '1i volume_type' > $A/crop-ctx.tsv
printf 'seed = 3\n[signal]\nacq_contrast = "ir"\n' > $A/crop-ir.toml
printf 'seed = 3\n[motion]\nmode = "random"\ntrans_mm = [1.0, 1.0, 0.0]\nrot_deg = [0.0, 0.0, 2.0]\nvolumes = [1, 3]\n' > $A/crop-motion.toml
printf 'seed = 3\n' > $A/crop-seed.toml
# P4 cases on the crop: physiological noise alone; the arterial compartment with crushing and
# exchange; bolus-position suppression with a slab pulse during labeling
printf 'seed = 3\n[physio]\ntissue_cardiac = 0.02\ntissue_drift = 0.01\nlabel_respiratory = 0.03\n' > $A/crop-physio.toml
printf 'seed = 3\n[kinetic]\nexchange_time = 0.5\n[macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\narterial_transit_time = { grey_matter = 3.0, white_matter = 3.2, csf = 0.0 }\n[vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n' > $A/crop-macro.toml
printf 'seed = 3\n[background_suppression]\nmodel = "bolus-position"\npulse_region = "slab"\nslab_entry_time = 0.3\n' > $A/crop-bolus.toml
crop_json '"BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2, "BackgroundSuppressionPulseTime": [1.5, 2.7]' > $A/crop-bs-early.json
# PASL on the crop: cutoff 0.7 s, PLD 1.8 s; suppression after the cutoff, and motion
pasl_json() {
  printf '{"ArterialSpinLabelingType": "PASL", "BolusCutOffFlag": true, "BolusCutOffTechnique": "Q2TIPS", "BolusCutOffDelayTime": 0.7, "PostLabelingDelay": 1.8, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012, %s}\n' "$1"
}
pasl_json '"BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 2, "BackgroundSuppressionPulseTime": [0.9, 1.5]' > $A/pasl-bs.json
pasl_json '"BackgroundSuppression": false' > $A/pasl-plain.json
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
  "crop_voxel|$P/crop_pcasl/asl.json|$P/crop_pcasl/aslcontext.tsv|$P/crop_pcasl/overlay.toml|$CROP|--t2-mode voxel"
  "crop_pasl_bs|$A/pasl-bs.json|$A/crop-ctx.tsv|$A/crop-seed.toml|$CROP"
  "crop_pasl_motion|$A/pasl-plain.json|$A/crop-ctx.tsv|$A/crop-motion.toml|$CROP"
  "asl004_bs|$A/asl004.json|$P/asl004/aslcontext.tsv|$A/ov.toml|$Z97"
  "asl004_voxel|$A/asl004.json|$P/asl004/aslcontext.tsv|$A/ov.toml|$Z97|--t2-mode voxel"
  "p4_all|$P/p4_all/asl.json|$P/p4_all/aslcontext.tsv|$P/p4_all/overlay.toml|$CROP"
  "p4_physio|$A/crop-plain.json|$A/crop-ctx.tsv|$A/crop-physio.toml|$CROP"
  "p4_macro_crush|$A/crop-crush.json|$A/crop-ctx.tsv|$A/crop-macro.toml|$CROP"
  "p4_macro_voxel|$A/crop-crush.json|$A/crop-ctx.tsv|$A/crop-macro.toml|$CROP|--t2-mode voxel"
  "p4_bolus|$A/crop-bs-early.json|$A/crop-ctx.tsv|$A/crop-bolus.toml|$CROP"
)

fail=0; differed=0
for feats in cli,kspace,par cli; do
  tag=${feats//,/-}
  OLD_T=$BASE/target-$tag
  NEW_T=$NEWROOT-target-$tag
  (cd "$BASE/aslscan" && CARGO_TARGET_DIR="$OLD_T" cargo build -q --release --features $feats)
  (cd "$NEWROOT/aslscan" && CARGO_TARGET_DIR="$NEW_T" cargo build -q --release --features $feats)
  OLD=$OLD_T/release/aslscan
  NEW=$NEW_T/release/aslscan
  for c in "${cases[@]}"; do
    IFS='|' read -r name json ctx ov ph extra <<< "$c"
    name=$name.$tag
    for side in old new; do
      bin=$OLD; [ $side = new ] && bin=$NEW
      if ! $bin --asl-json $json --aslcontext $ctx --overlay $ov --phantom $ph $extra --out $OUT/$side-$name > $OUT/log-$side-$name.txt 2>&1; then
        echo "$name: $side run FAILED"; tail -2 $OUT/log-$side-$name.txt; fail=1; continue 2
      fi
    done
    n=0; bad=0
    # the file list and the decompressions are checked for failure, not left inside process
    # substitutions, whose status set -e and pipefail never see (two unreadable gzips would
    # otherwise compare as two empty streams)
    files=$(find "$OUT/old-$name" -type f) || { echo "$name: listing failed"; fail=1; continue; }
    [ -n "$files" ] || { echo "$name: no output files"; fail=1; continue; }
    while IFS= read -r f; do
      rel=${f#$OUT/old-$name/}
      g=$OUT/new-$name/$rel
      n=$((n+1))
      if [ ! -f "$g" ]; then echo "  MISSING $name $rel"; bad=1; continue; fi
      case $f in
        *.nii.gz)
          if ! gzip -dc "$f" > "$OUT/a.nii" || ! gzip -dc "$g" > "$OUT/b.nii"; then
            echo "  UNREADABLE $name $rel"; fail=1
          elif ! cmp -s "$OUT/a.nii" "$OUT/b.nii"; then
            echo "  DIFF $name $rel"; bad=1; differed=1
          fi ;;
        *) cmp -s "$f" "$g" || { echo "  DIFF $name $rel"; bad=1; } ;;
      esac
    done <<< "$files"
    rm -f "$OUT/a.nii" "$OUT/b.nii"
    nnew=$(find $OUT/new-$name -type f | wc -l)
    [ "$nnew" = "$n" ] || { echo "  file count $n vs $nnew"; bad=1; }
    echo "$name: $n files, $([ $bad = 0 ] && echo identical || echo DIFFERENT)"
    [ $bad = 0 ] || [ "$SELF" = "--self-test" ] || fail=1
  done
done
if [ "$SELF" = "--self-test" ]; then
  [ $fail = 0 ] || { echo "self-test: a build or run failed, which is not a detected difference"; exit 1; }
  [ $differed = 1 ] || { echo "self-test FAILED: the perturbed acquisition stage produced identical images"; exit 1; }
  echo "self-test passed: the perturbation was detected"
  exit 0
fi
exit $fail
