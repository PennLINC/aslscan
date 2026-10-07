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
# Needs work/phantom-3t, work/phantom-3t-z100, work/phantom-3t-z97, work/phantom-3t-z120 and
# work/phantom-3t-asl001 (see tools/hrgt_to_bids.py: z97 is --crop 0:197 0:233 46:143, which fits
# asl004's 24 slices of 4.05 mm; z120 and asl001 are the P5 acceptance crops named in the
# asl005_p5 and asl001_p5 overlays).
#
# A case may name a required feature as its seventh field (kspace): a build without it skips the
# case, and the run checks the executed and skipped counts per build. Under --self-test every
# executed case must show a NIfTI difference, not just one of them.
#
# A case's sidecar field may list several sidecars separated by `+` (multi-TE, P6): each becomes
# its own --asl-json, in order. The P6 cases (P7 plan, Task 0) run from the committed fixtures and
# two variants of p6_ll that tools/regress_p6_inputs.py generates and checks.
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
# one run at a time: runs share work/regress and the base and snapshot target directories, and a
# second run clears the first's outputs mid-comparison
mkdir -p "$HERE/work"
exec 9>"$HERE/work/regress.lock"
flock -n 9 || { echo "another regress_identity.sh run holds $HERE/work/regress.lock"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT/inputs"

# --asl-json once per `+`-separated sidecar of a case's sidecar field, into the array JARGS
json_args() {
  local js j; JARGS=()
  IFS='+' read -ra js <<< "$1"
  for j in "${js[@]}"; do JARGS+=(--asl-json "$j"); done
}
json_args "a.json"; [ "${#JARGS[@]}" = 2 ] && [ "${JARGS[1]}" = a.json ] || { echo "json_args: one sidecar"; exit 1; }
json_args "a.json+b.json+c.json"; [ "${#JARGS[@]}" = 6 ] && [ "${JARGS[5]}" = c.json ] && [ "${JARGS[2]}" = --asl-json ] \
  || { echo "json_args: three sidecars"; exit 1; }

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
# (long) comparison runs cannot change what is being compared. Unchanged files keep their times, so
# the snapshot's own target directories stay incremental between runs; but a changed file must be
# NEWER than the last build, and rsync -a copies the source's time, which can be older than that
# build (cargo then reuses stale code: P7, Task 15). So files are compared by content, and every file
# rewritten gets the time of the copy.
NEWROOT=$PARENT/p5-new
[ "$SELF" = "--self-test" ] && NEWROOT=$PARENT/p5-selftest
mkdir -p "$NEWROOT"
snapshot() {
  local src=$1 dst=$2; shift 2
  # -a without -t: times are not copied, so a file rewritten for a content change gets the time of
  # the copy, and --checksum leaves files of equal content (and their times) alone
  rsync -rlpgoD --checksum --delete "$@" "$src/" "$dst/"
}
snapshot "$HERE" "$NEWROOT/aslscan" --exclude target --exclude 'target-*' --exclude work --exclude .git
snapshot "$LIVE_M" "$NEWROOT/mrsim-acq" --exclude target --exclude 'target-*' --exclude .git
echo "new side: aslscan $(git -C "$HERE" rev-parse --short HEAD)$([ -z "$(git -C "$HERE" status --porcelain -- src Cargo.toml)" ] || echo +local), mrsim-acq $(git -C "$LIVE_M" rev-parse --short HEAD)$([ -z "$(git -C "$LIVE_M" status --porcelain -- src Cargo.toml)" ] || echo +local)"
if [ "$SELF" = "--self-test" ]; then
  K=$NEWROOT/mrsim-acq/src/kspace.rs
  sed -i 's/amp\[i\] = acq\.signal_scale \* coil_sensitivity(/amp[i] = acq.signal_scale * 1.0001 * coil_sensitivity(/' "$K"
  grep -q 'acq.signal_scale \* 1.0001 \* coil_sensitivity(' "$K" || { echo "self-test patch did not apply"; exit 1; }
  # the spiral forward (P5 part C) forms its own amplitude
  S=$NEWROOT/mrsim-acq/src/spiral.rs
  sed -i 's/s\.amp\[i\] = acq\.signal_scale \* coil_sensitivity(/s.amp[i] = acq.signal_scale * 1.0001 * coil_sensitivity(/' "$S"
  grep -q 'acq.signal_scale \* 1.0001 \* coil_sensitivity(' "$S" || { echo "self-test patch did not apply to the spiral"; exit 1; }
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
# P5 cases. Gradient echo with an included M0 and suppression: 90 degrees is the closed-form branch
# (the longitudinal state is not propagated), 35 degrees the propagation branch with mixed
# preparations; the fixture itself (60 degrees, separate M0) as is. The m0scan row comes first and
# every per-row array gains its 0 entry.
ge_json() {
  sed -e "s/\"FlipAngle\": 60/\"FlipAngle\": $1/" -e 's/"M0Type": "Separate"/"M0Type": "Included"/' \
      -e 's/"PostLabelingDelay": \[1.8, 1.8, 1.0, 1.0\]/"PostLabelingDelay": [0.0, 1.8, 1.8, 1.0, 1.0]/' $P/p5_ge/asl.json
}
ge_json 90 > $A/p5-ge90.json
ge_json 35 > $A/p5-ge35.json
for f in $A/p5-ge90.json $A/p5-ge35.json; do
  grep -q '"M0Type": "Included"' $f && grep -q '\[0.0, 1.8' $f && ! grep -q '"FlipAngle": 60' $f \
    || { echo "p5_ge variant $f did not apply"; exit 1; }
done
sed '1a m0scan' $P/p5_ge/aslcontext.tsv > $A/p5-ge-m0.tsv
# the included M0 takes no [m0] table
sed '/^\[m0\]/,$d' $P/p5_ge/overlay.toml > $A/p5-ge-incl.toml
# segmented GRASE with a shot event in every volume between its two shots
{ cat $P/p5_grase/overlay.toml; printf '\n[motion.within_volume]\ndropout_rate = 1.0\nseverity = 0.3\njump_mm = [0.5, 0.0, 0.0]\njump_deg = [0.0, 0.0, 1.0]\n'; } > $A/p5-grase-seg.toml
Z120=$HERE/work/phantom-3t-z120
P001=$HERE/work/phantom-3t-asl001
# P6 cases: the two generated p6_ll variants (one readout per cycle, the legacy dispatch; an
# included m0scan between the cycles); the multi-TE fixtures pass one sidecar per echo
python3 "$HERE/tools/regress_p6_inputs.py" "$P/p6_ll" "$A"
mte() { echo "$P/$1/asl-echo-1.json+$P/$1/asl-echo-2.json+$P/$1/asl-echo-3.json"; }
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
  "p5_ge90|$A/p5-ge90.json|$A/p5-ge-m0.tsv|$A/p5-ge-incl.toml|$CROP"
  "p5_ge35|$A/p5-ge35.json|$A/p5-ge-m0.tsv|$A/p5-ge-incl.toml|$CROP"
  "p5_ge_m0sep|$P/p5_ge/asl.json|$P/p5_ge/aslcontext.tsv|$P/p5_ge/overlay.toml|$CROP"
  "p5_grase_seg|$P/p5_grase/asl.json|$P/p5_grase/aslcontext.tsv|$A/p5-grase-seg.toml|$CROP"
  "asl005_p5|$P/asl005/asl.json|$P/asl005/aslcontext.tsv|$P/asl005_p5/overlay.toml|$Z120"
  "asl001_p5|$P/asl001/asl.json|$P/asl001/aslcontext.tsv|$P/asl001_p5/overlay.toml|$P001||kspace"
  "p6_hadamard|$P/p6_hadamard/asl.json|$P/p6_hadamard/aslcontext.tsv|$P/p6_hadamard/overlay.toml|$CROP"
  "p6_hadamard_3t|$P/p6_hadamard_3t/asl.json|$P/p6_hadamard_3t/aslcontext.tsv|$P/p6_hadamard_3t/overlay.toml|$Z97"
  "p6_hadamard_grase|$P/p6_hadamard_grase/asl.json|$P/p6_hadamard_grase/aslcontext.tsv|$P/p6_hadamard_grase/overlay.toml|$Z120"
  "p6_hadamard_multite|$(mte p6_hadamard_multite)|$P/p6_hadamard_multite/aslcontext.tsv|$P/p6_hadamard_multite/overlay.toml|$CROP"
  "p6_multite|$(mte p6_multite)|$P/p6_multite/aslcontext.tsv|$P/p6_multite/overlay.toml|$CROP"
  "p6_multite_se|$(mte p6_multite_se)|$P/p6_multite_se/aslcontext.tsv|$P/p6_multite_se/overlay.toml|$CROP"
  "p6_ll|$P/p6_ll/asl.json|$P/p6_ll/aslcontext.tsv|$P/p6_ll/overlay.toml|$CROP"
  "p6_ll_legacy|$A/p6-ll-legacy.json|$A/p6-ll-legacy.tsv|$A/p6-ll-legacy.toml|$CROP"
  "p6_ll_m0|$A/p6-ll-m0.json|$A/p6-ll-m0.tsv|$A/p6-ll-m0.toml|$CROP"
)
# executed cases per build (the rest are skipped for a missing feature)
declare -A expect=([cli-kspace-par]=34 [cli]=33)

fail=0; undetected=0
for feats in cli,kspace,par cli; do
  tag=${feats//,/-}
  OLD_T=$BASE/target-$tag
  NEW_T=$NEWROOT-target-$tag
  (cd "$BASE/aslscan" && CARGO_TARGET_DIR="$OLD_T" cargo build -q --release --features $feats)
  (cd "$NEWROOT/aslscan" && CARGO_TARGET_DIR="$NEW_T" cargo build -q --release --features $feats)
  OLD=$OLD_T/release/aslscan
  NEW=$NEW_T/release/aslscan
  ran=0; skipped=0
  for c in "${cases[@]}"; do
    IFS='|' read -r name json ctx ov ph extra need <<< "$c"
    name=$name.$tag
    if [ -n "$need" ] && [[ ",$feats," != *",$need,"* ]]; then
      echo "$name: skipped (needs $need)"; skipped=$((skipped+1)); continue
    fi
    ran=$((ran+1))
    json_args "$json"
    for side in old new; do
      bin=$OLD; [ $side = new ] && bin=$NEW
      if ! $bin "${JARGS[@]}" --aslcontext $ctx --overlay $ov --phantom $ph $extra --out $OUT/$side-$name > $OUT/log-$side-$name.txt 2>&1; then
        echo "$name: $side run FAILED"; tail -2 $OUT/log-$side-$name.txt; fail=1; continue 2
      fi
    done
    n=0; bad=0; differed=0
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
    if [ "$SELF" = "--self-test" ] && [ $differed = 0 ]; then
      echo "  self-test FAILED for $name: the perturbed acquisition stage produced identical images"; undetected=1
    fi
  done
  echo "$tag: $ran executed, $skipped skipped"
  [ "$ran" = "${expect[$tag]}" ] && [ $((ran+skipped)) = ${#cases[@]} ] \
    || { echo "$tag: expected ${expect[$tag]} executed of ${#cases[@]}"; fail=1; }
done
if [ "$SELF" = "--self-test" ]; then
  [ $fail = 0 ] || { echo "self-test: a build or run failed, which is not a detected difference"; exit 1; }
  [ $undetected = 0 ] || { echo "self-test FAILED: some cases did not detect the perturbation"; exit 1; }
  echo "self-test passed: every executed case detected the perturbation"
  exit 0
fi
exit $fail
