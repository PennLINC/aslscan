#!/usr/bin/env python3
"""P10 acceptance checks (P10 plan, Tasks 5 and 7) on the runs tools/acceptance_p10.sh writes.

    python tools/acceptance_p10_check.py <out dir>

Needs numpy and nibabel (the rust-trx micromamba environment). A slice's ratio is its mean
magnitude over the still run's tissue (above 0.2 of that slice's still maximum) divided by the
still run's mean there.

Criteria (spec, "P10 acceptance criteria"):
  2. the timed linear nod (1 kHz, "shot" sampling, spin history) reproduces P9's per-shot events
     run to 1e-5 of the peak; volume 12's odd slices at 0.65-0.85 of the static, even ones at
     0.97-1.03;
  3. the cosine nod (100 Hz, spin history): under "volume" sampling volume 12 is the still run's to
     1e-4 of the peak and volume 13 is moved; under "shot", volume 12's even slices exceed its odd
     ones by at least 0.1 of the static (slices 1-18).
"""
import sys

import nibabel as nib
import numpy as np

out = sys.argv[1]
ok = True


def check(cond, msg):
    global ok
    print(("ok   " if cond else "FAIL ") + msg)
    ok &= bool(cond)


def load(run):
    return np.asanyarray(nib.load(f"{out}/{run}/sub-01/perf/sub-01_part-mag_asl.nii.gz").dataobj, dtype=np.float64)


still = load("still")
peak = np.abs(still).max()


def ratios(img, v):
    s = still[..., v]
    r = []
    for z in range(s.shape[2]):
        m = s[:, :, z] > 0.2 * s[:, :, z].max() if s[:, :, z].max() > 0 else np.zeros_like(s[:, :, z], bool)
        r.append(img[:, :, z, v][m].mean() / s[:, :, z][m].mean() if m.any() else np.nan)
    return np.array(r)


# criterion 2
events, linear = load("events"), load("timed-linear")
d = np.abs(events - linear).max() / np.abs(events).max()
check(d <= 1e-5, f"criterion 2: the timed linear nod against P9's events, largest difference {d:.2e} of the peak")
r = ratios(linear, 12)
print("      volume 12 slice ratios (timed linear):", " ".join(f"{x:.2f}" for x in r))
odd, even = r[1::2], r[0::2]
check(np.all((odd >= 0.65) & (odd <= 0.85)), f"criterion 2: odd slices {odd.min():.3f}-{odd.max():.3f} (0.65-0.85)")
check(np.all((even >= 0.97) & (even <= 1.03)), f"criterion 2: even slices {even.min():.3f}-{even.max():.3f} (0.97-1.03)")

# criterion 3
vol, shot = load("timed-volume"), load("timed-shot")
d12 = np.abs(vol[..., 12] - still[..., 12]).max() / peak
check(d12 <= 1e-4, f"criterion 3: \"volume\" sampling, volume 12 against the still run {d12:.2e} of the peak")
d13 = np.abs(vol[..., 13] - still[..., 13]).max() / peak
check(d13 > 1e-2, f"criterion 3: \"volume\" sampling, volume 13 is moved ({d13:.3f} of the peak)")
r = ratios(shot, 12)
print("      volume 12 slice ratios (timed cosine, shot):", " ".join(f"{x:.2f}" for x in r))
inner = r[1:19]
gap = np.nanmean(inner[1::2]) - np.nanmean(inner[0::2])  # inner[0] is slice 1 (odd)
check(gap >= 0.1, f"criterion 3: \"shot\" sampling, even slices above odd ones by {gap:.3f} of the static (>= 0.1)")
rv = ratios(vol, 12)
print("      volume 12 slice ratios (timed cosine, volume):", " ".join(f"{x:.2f}" for x in rv))

# criterion 4: 3D GRASE (asl005, four segments, 30 partitions), a step between segments 1 and 2 of
# volume 1
import json, os
if os.path.isdir(f"{out}/grase-step"):
    gs, gm = load("grase-still"), load("grase-step")
    diff = np.abs(gm - gs)
    per_vol = [diff[..., v].max() for v in range(gs.shape[3])]
    gpeak = np.abs(gs).max()
    print("      per-volume largest difference / peak:", " ".join(f"{x / gpeak:.2e}" for x in per_vol))
    check(all(per_vol[v] == 0.0 for v in range(gs.shape[3]) if v != 1), "criterion 4: only volume 1 changes")
    check(per_vol[1] > 1e-3 * gpeak, f"criterion 4: volume 1 changes ({per_vol[1] / gpeak:.3f} of the peak)")
    e = (diff[..., 1] ** 2).sum(axis=(0, 1))
    frac = e.max() / e.sum()
    print("      volume 1 difference energy per partition:", " ".join(f"{x / e.sum():.2f}" for x in e))
    check(frac <= 0.5, f"criterion 4: spread over the slab, the largest partition holds {frac:.2f} of the energy (<= 0.5)")
    m = json.load(open(f"{out}/grase-step/sub-01/perf/sub-01_part-mag_asl.json"))["AslscanSimulation"]["Motion"]
    check(any("one pose per readout segment" in a for a in m["Approximations"]), "criterion 4: the sidecar names the one-pose-per-segment approximation")

sys.exit(0 if ok else 1)
