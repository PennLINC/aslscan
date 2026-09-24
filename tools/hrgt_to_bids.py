#!/usr/bin/env python
"""Convert an ASLDRO packed ground truth (5D NIfTI + JSON) into the aslscan phantom layout.

    micromamba run -n simasl python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t --out work/phantom-3t
    micromamba run -n simasl python tools/hrgt_to_bids.py --name hrgt_icbm_2009a_nls_3t \\
        --out tests/fixtures/phantom-crop --crop 88:112 104:128 90:96

One NIfTI per quantity, each with a JSON sidecar carrying Units; dseg.json carries the label
names; phantom.json carries the kinetic constants the oracle was built with. Everything is
written float32 except dseg (int16). The 5D file's 4th axis is a singleton and is dropped.

Units are read from the source JSON's `units` list and checked against what this converter
expects for each quantity; a mismatch is an error, never a silent conversion.
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

from asldro.data.filepaths import GROUND_TRUTH_DATA

# source quantity -> (output stem, expected source unit string, output Units string, dtype)
MAPPING = {
    "perfusion_rate": ("perfusion", "ml/100g/min", "ml/100g/min", np.float32),
    "transit_time": ("att", "s", "s", np.float32),
    "t1": ("T1map", "s", "s", np.float32),
    "t2": ("T2map", "s", "s", np.float32),
    "t2_star": ("T2starmap", "s", "s", np.float32),
    "m0": ("M0map", "", "arbitrary", np.float32),
    "seg_label": ("dseg", "", "label indices", np.int16),
}


def parse_crop(spec):
    """Three `start:stop` slices, half-open, in voxel indices."""
    if spec is None:
        return None
    out = []
    for s in spec:
        a, b = s.split(":")
        a, b = int(a), int(b)
        if b <= a:
            sys.exit(f"bad crop range {s!r}")
        out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, choices=sorted(GROUND_TRUTH_DATA.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", nargs=3, metavar="START:STOP", help="x y z voxel ranges, half-open")
    a = ap.parse_args()

    paths = GROUND_TRUTH_DATA[a.name]
    meta = json.load(open(paths["json"]))
    img = nib.load(paths["nii"])
    data = np.asanyarray(img.dataobj)
    if data.ndim != 5 or data.shape[3] != 1:
        sys.exit(f"expected a 5D volume with a singleton 4th axis, got {data.shape}")
    quantities, units = meta["quantities"], meta["units"]
    if len(quantities) != data.shape[4] or len(units) != len(quantities):
        sys.exit("quantities/units/5th-axis length disagree")

    crop = parse_crop(a.crop)
    affine = img.affine.copy()
    if crop:
        # Explicit bounds only: a negative start would be normalised by NumPy for the slice but
        # not for the affine shift below, misregistering the crop by a whole axis.
        for ax, (lo, hi) in enumerate(crop):
            if not (0 <= lo < hi <= data.shape[ax]):
                sys.exit(f"crop axis {ax}: {lo}:{hi} is outside 0..{data.shape[ax]}")
        (x0, x1), (y0, y1), (z0, z1) = crop
        data = data[x0:x1, y0:y1, z0:z1]
        # the new voxel (0,0,0) is the old (x0,y0,z0): shift the origin by the linear part
        affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ np.array([x0, y0, z0], dtype=float)

    os.makedirs(a.out, exist_ok=True)
    label_names = {int(v): k for k, v in meta["segmentation"].items()}
    counts = None
    for qi, q in enumerate(quantities):
        if q not in MAPPING:
            sys.exit(f"unexpected quantity {q!r} in {paths['json']}")
        stem, expect_unit, out_unit, dtype = MAPPING[q]
        if units[qi] != expect_unit:
            sys.exit(f"{q}: source unit {units[qi]!r}, expected {expect_unit!r}")
        vol = data[..., 0, qi].astype(dtype)
        nib.Nifti1Image(np.ascontiguousarray(vol), affine).to_filename(os.path.join(a.out, stem + ".nii.gz"))
        side = {"Units": out_unit, "SourceQuantity": q, "Source": a.name}
        if q == "seg_label":
            side["LabelMap"] = {str(k): v for k, v in sorted(label_names.items())}
            counts = {k: int((vol == k).sum()) for k in sorted(label_names)}
        with open(os.path.join(a.out, stem + ".json"), "w") as f:
            json.dump(side, f, indent=2)
            f.write("\n")

    params = meta.get("parameters", {})
    phantom = {
        "LambdaBloodBrain": params.get("lambda_blood_brain"),
        "T1ArterialBlood": params.get("t1_arterial_blood"),
        "MagneticFieldStrength": params.get("magnetic_field_strength"),
        "Source": a.name,
        "Converter": "hrgt_to_bids.py",
        "Crop": [list(c) for c in crop] if crop else None,
        "VoxelSize": [float(v) for v in img.header.get_zooms()[:3]],
    }
    with open(os.path.join(a.out, "phantom.json"), "w") as f:
        json.dump(phantom, f, indent=2)
        f.write("\n")

    print(f"wrote {a.out}: dims {tuple(data.shape[:3])}, labels {counts}")
    if counts is not None and any(v == 0 for v in counts.values()):
        sys.exit("a label has no voxels; move the crop window")


if __name__ == "__main__":
    main()
