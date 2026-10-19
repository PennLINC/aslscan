#!/usr/bin/env python3
"""P9 part B acceptance checks (P9 plan, Task 10) on the runs tools/acceptance_p9.sh writes.

    python tools/acceptance_p9_check.py <out dir>

Needs numpy and nibabel (the rust-trx micromamba environment). The tissue's static signal in a voxel
is M0 x (1 - exp(-TR/T1)) without presaturation and M0 x (1 - exp(-t_read/T1)) with it, from the
ground-truth M0 and T1 maps; the spin-history ground truth is the correction added to it.
Criteria (spec, "P9 acceptance criteria"):
  2. the slow drift: the correction is below 1e-3 of the tissue in every slice but the stack's ends;
  3. one 0.5 mm jump at shot 10 of volume 1: grey matter of slice 10 at 0.903 of the static within
     5e-3, nothing in slices 0-9 of that volume, a hair in the slices after;
  4. the same with presaturation: the dip against the presaturated reference, and below 1e-3 in
     the volume after.
"""
import sys

import nibabel as nib
import numpy as np

TR, SIG, DT = 4.6, 3.6, 0.04
out = sys.argv[1]
ok = True


def check(cond, msg):
    global ok
    print(("ok   " if cond else "FAIL ") + msg)
    ok &= bool(cond)


def load(run):
    gt = f"{out}/{run}/sub-01/perf/ground-truth/sub-01_desc-"
    get = lambda d: np.asanyarray(nib.load(f"{gt}{d}_gt.nii.gz").dataobj, dtype=np.float64)
    return get("spinHistory"), get("M0map"), get("T1map"), get("dseg")


def tissue(m0, t1, t_rec):
    with np.errstate(divide="ignore", invalid="ignore"):
        return m0 * (1.0 - np.exp(-t_rec / np.where(t1 > 0, t1, np.inf)))


# criterion 2
sh, m0, t1, dseg = load("sh-drift")
ref = tissue(m0, t1, TR)[..., None]
brain = (m0 > 0.2 * m0.max())[..., None] & np.ones(sh.shape, bool)
inner = np.zeros(sh.shape, bool)
inner[:, :, 1:-1, :] = True
rel = np.abs(sh) / np.where(ref > 0, ref, np.inf)
worst = rel[brain & inner].max()
check(worst < 1e-3, f"criterion 2: slow drift, largest correction {worst:.2e} of the tissue (slices 1-18)")


def dip(run, presat):
    sh, m0, t1, dseg = load(run)
    gm = (dseg == 1) & (m0 > 0.5 * m0.max())
    t_read = SIG + 10 * DT
    r = tissue(m0, t1, t_read if presat else TR)
    sl = gm[:, :, 10]
    ratio = 1.0 + sh[:, :, 10, 1][sl] / r[:, :, 10][sl]
    t1g = np.median(t1[:, :, 10][sl])
    big_r = (1 - np.exp(-t_read / t1g)) if presat else (1 - np.exp(-TR / t1g))
    want = 1 - 0.1 * (1 - (1 - np.exp(-DT / t1g)) / big_r)
    got = np.median(ratio)
    before = np.abs(sh[:, :, :10, 1]).max() / m0.max()
    after = (np.abs(sh[:, :, 11:, 1]) / np.where(r[:, :, 11:] > 0, r[:, :, 11:], np.inf))[m0[:, :, 11:] > 0.2 * m0.max()].max()
    return got, want, before, after, sh, m0, t1, sl.sum()


got, want, before, after, *_ , n = dip("sh-event", False)
check(abs(got - want) < 5e-3, f"criterion 3: slice 10 grey matter at {got:.4f} of the static, closed form {want:.4f} ({n} voxels)")
check(abs(want - 0.903) < 2e-3, f"criterion 3: the closed form at grey matter's T1 is {want:.4f} (0.903 at T1 1.33 s)")
check(before < 1e-6, f"criterion 3: slices 0-9 of volume 1, largest correction {before:.1e} of the largest M0")
check(after < 1e-3, f"criterion 3: slices 11-19 of volume 1, largest correction {after:.1e} of the tissue")
got, want, before, after, sh, m0, t1, n = dip("sh-presat", True)
check(abs(got - want) < 5e-3, f"criterion 4: presaturated, slice 10 at {got:.4f}, closed form {want:.4f}")
r = tissue(m0, t1, SIG)[..., None]
nxt = (np.abs(sh[..., 2]) / np.where(r[..., 0] > 0, r[..., 0], np.inf))[m0 > 0.2 * m0.max()].max()
check(nxt < 1e-3, f"criterion 4: volume 2 after presaturation, largest correction {nxt:.1e} of the tissue")
sys.exit(0 if ok else 1)
