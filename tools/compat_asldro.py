#!/usr/bin/env python
"""Benchmark aslscan's ASLDRO compatibility mode against simasl (ASLDRO v2.2.0), voxel by voxel.

Run in the `simasl` environment (it imports `asldro`), from the aslscan repository root, after
`cargo build --release --features cli,kspace,par`:

    micromamba run -n simasl python tools/compat_asldro.py all --phantom 3t
    micromamba run -n simasl python tools/compat_asldro.py A --phantom 1.5t
    micromamba run -n simasl python tools/compat_asldro.py A --crop       # the cargo-test case

Each benchmark runs simasl's real `run_full_pipeline` (with `AcquireMriImageFilter` recorded, so
each volume's complex image is read from the run itself), translates the same protocol into an
aslscan `asl.json` + `aslcontext.tsv` + overlay, runs `aslscan --compat-asldro`, and compares
the two outputs on the same grid; it never resamples either. Reports go to
`work/compat/<bench>-<phantom>.{json,md}`; the exit status is nonzero when a criterion fails.
The spec is `mrsim-acq/docs/specs/2026-09-24-p2-asldro-compat-design.md` (parts B and C).
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import nibabel as nib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(REPO, "work", "compat")
PHANTOMS = {"3t": "hrgt_icbm_2009a_nls_3t", "1.5t": "hrgt_icbm_2009a_nls_1.5t"}
# The window of tests/fixtures/phantom-crop (half-open voxel ranges of the 3 T ground truth).
CROP = ((88, 112), (104, 128), (90, 96))
# Half-width (phantom voxels) of the spline's reach about a sample point: the cubic's support
# of 2 plus 6 for the prefiltered coefficients to settle (0.268^6 < 4e-4 of a step).
SPLINE_REACH = 8
MIN_PURE = 50
C_SEEDS = [2, 4, 6, 8, 10, 12, 14, 16]


# ------------------------------------------------------------------------------- rotations

def rot_x(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])


def simasl_matrix(rot_deg, transl, origin=(0.0, 0.0, 0.0)):
    """simasl's image-path motion (`utils/resampling.py:82-100`): p' = R (p - o) + o + t with
    R = Rx Ry Rz, as a 4x4 acting on world points."""
    r = rot_x(rot_deg[0]) @ rot_y(rot_deg[1]) @ rot_z(rot_deg[2])
    o, t = np.asarray(origin, float), np.asarray(transl, float)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = o - r @ o + t
    return m


def mrsim_matrix(rot_deg, trans_mm, centre):
    """`mrsim_acq::motion::Pose::to_matrix`: p' = R (p - c) + c + t with R = Rz Ry Rx
    (`mat::rotation_zyx_deg`)."""
    r = rot_z(rot_deg[2]) @ rot_y(rot_deg[1]) @ rot_x(rot_deg[0])
    c = np.asarray(centre, float)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = c - r @ c + np.asarray(trans_mm, float)
    return m


def zyx_angles(r):
    """(rx, ry, rz) degrees with r = Rz(rz) Ry(ry) Rx(rx); the gimbal case |r[2,0]| = 1 puts the
    whole in-plane angle on rx (rz = 0)."""
    sb = -r[2, 0]
    if abs(sb) < 1.0 - 1e-12:
        b = np.arcsin(sb)
        a = np.arctan2(r[2, 1], r[2, 2])
        c = np.arctan2(r[1, 0], r[0, 0])
    else:
        b = np.copysign(np.pi / 2, sb)
        c = 0.0
        # Ry(+-90) Rx(a): row 0 is (0, +-sin a, +-cos a), row 1 is (0, cos a, -sin a)
        a = np.arctan2(np.sign(sb) * r[0, 1], r[1, 1])
    return np.degrees([a, b, c])


def pose_to_mrsim(rot_deg, transl, fov_centre, origin=(0.0, 0.0, 0.0), wrong=None):
    """simasl's (rot_x/y/z degrees, transl_x/y/z mm) about `origin` -> mrsim-acq's Pose
    (rot_deg in Rz Ry Rx order, trans_mm) about the FOV centre: R the same matrix,
    t' = t + (R - I)(c - o). `wrong` builds benchmark D's negative controls: "order" passes
    simasl's angles straight through as Rz Ry Rx angles (translation converted for them),
    "centre" keeps the right angles with t' = t."""
    c, o, t = (np.asarray(v, float) for v in (fov_centre, origin, transl))
    r = rot_x(rot_deg[0]) @ rot_y(rot_deg[1]) @ rot_z(rot_deg[2])
    if wrong == "order":
        ang = np.asarray(rot_deg, float)
        r = rot_z(ang[2]) @ rot_y(ang[1]) @ rot_x(ang[0])
    else:
        ang = zyx_angles(r)
    if wrong == "centre":
        return ang, t
    return ang, t + (r - np.eye(3)) @ (c - o)


# ------------------------------------------------------------------------------- ground truth

def ground_truth(name):
    from asldro.data.filepaths import GROUND_TRUTH_DATA
    return dict(GROUND_TRUTH_DATA[name])


def check_affine(affine):
    """simasl's target axes are positive whatever the source's; aslscan keeps the source's
    signs. The two grids agree only for a positive identity linear part."""
    if not np.allclose(affine[:3, :3], np.eye(3), atol=1e-9):
        raise SystemExit(f"the ground truth's linear affine must be the positive identity, got\n{affine[:3, :3]}")


def packed_crop(name, crop, out_dir):
    """Write the crop window of a packed ASLDRO ground truth as a packed ground truth (5D NIfTI
    with the shifted affine, and the JSON), for simasl."""
    src = ground_truth(name)
    os.makedirs(out_dir, exist_ok=True)
    nii, js = os.path.join(out_dir, "gt.nii.gz"), os.path.join(out_dir, "gt.json")
    img = nib.load(src["nii"])
    (x0, x1), (y0, y1), (z0, z1) = crop
    data = np.asanyarray(img.dataobj)[x0:x1, y0:y1, z0:z1]
    aff = img.affine.copy()
    aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array([x0, y0, z0], float)
    out = nib.Nifti1Image(np.ascontiguousarray(data), aff, header=img.header.copy())
    out.set_qform(aff)
    out.set_sform(aff)
    out.to_filename(nii)
    shutil.copyfile(src["json"], js)
    return {"nii": nii, "json": js}


SYNTH = "hrgt_synth"


def synthetic_ground_truth(out_dir):
    """A packed ground truth on the 3 T phantom's grid, affine, quantities and per-label values,
    whose anatomy is three large blocks: WM x 30:167, y 30:203, z 30:159 with GM in its x < 98
    half and a CSF block x 110:140, y 90:140, z 60:100 inside the WM. The ICBM anatomy has no
    17-voxel cube of one tissue, so the pure-mask gates of benchmarks B and D need this one; it
    keeps every boundary kind (tissue to background, GM to WM, WM to CSF)."""
    os.makedirs(out_dir, exist_ok=True)
    nii = os.path.join(out_dir, SYNTH + ".nii.gz")
    js = os.path.join(out_dir, SYNTH + ".json")
    # Regenerated on every run, never reused: a cached file would keep testing an old generator.
    src = ground_truth(PHANTOMS["3t"])
    img = nib.load(src["nii"])
    meta = json.load(open(src["json"]))
    data = np.asanyarray(img.dataobj)
    qs = meta["quantities"]
    seg_i = qs.index("seg_label")
    seg = data[..., 0, seg_i]
    new_seg = np.zeros(seg.shape, data.dtype)
    new_seg[30:167, 30:203, 30:159] = 2
    new_seg[30:98, 30:203, 30:159] = 1
    new_seg[110:140, 90:140, 60:100] = 3
    out = np.zeros_like(data)
    for qi, q in enumerate(qs):
        if qi == seg_i:
            out[..., 0, qi] = new_seg
            continue
        vol = data[..., 0, qi]
        for lab in (1, 2, 3):
            vals = np.unique(vol[seg == lab])
            if vals.size != 1:
                raise SystemExit(f"{q} is not constant in label {lab}: {vals[:5]}")
            out[..., 0, qi][new_seg == lab] = vals[0]
        if q == "m0":
            # A linear ramp in M0 (0.05..1.95 of the label's value): both kernels reproduce a
            # linear field exactly, so the right pose still agrees inside the pure mask, while a
            # pose that misplaces the object by a fraction of a millimetre reads another value.
            x, y, z = np.indices(seg.shape, dtype=float)
            ramp = 1 + 0.6 * (x - 98) / 68 + 0.15 * (y - 116) / 86 + 0.2 * (z - 94) / 64
            out[..., 0, qi] = np.where(new_seg > 0, out[..., 0, qi] * ramp, 0).astype(out.dtype)
    nib.Nifti1Image(out, img.affine, header=img.header.copy()).to_filename(nii)
    shutil.copyfile(src["json"], js)
    return {"nii": nii, "json": js}


def fingerprint(gt, crop=None):
    """sha256 over the packed ground truth's NIfTI and JSON bytes, the converter's source and the
    crop: what a converted phantom was made from."""
    import hashlib
    h = hashlib.sha256()
    for path in (gt["nii"], gt["json"], os.path.join(REPO, "tools", "hrgt_to_bids.py")):
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    h.update(repr(crop).encode())
    return h.hexdigest()


def convert_phantom(name, out, crop=None, source=None):
    """tools/hrgt_to_bids.py; reused only when its fingerprint (see `fingerprint`) matches the
    inputs it would be converted from now. `source` converts a packed file instead of a name.
    Returns (directory, fingerprint)."""
    gt = {"nii": source, "json": source[:-7] + ".json"} if source else ground_truth(name)
    fp = fingerprint(gt, crop)
    stamp = os.path.join(out, ".fingerprint")
    if os.path.exists(stamp) and open(stamp).read().strip() == fp:
        return out, fp
    if os.path.exists(out):
        shutil.rmtree(out)
    which = ["--source", source] if source else ["--name", name]
    cmd = [sys.executable, os.path.join(REPO, "tools", "hrgt_to_bids.py"), *which, "--out", out]
    if crop:
        cmd += ["--crop"] + [f"{a}:{b}" for a, b in crop]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    open(stamp, "w").write(fp + "\n")
    return out, fp


def load_gt_arrays(gt):
    img = nib.load(gt["nii"])
    meta = json.load(open(gt["json"]))
    data = np.asanyarray(img.dataobj)[..., 0, :]
    maps = {q: np.asarray(data[..., i], float) for i, q in enumerate(meta["quantities"])}
    return img.affine, maps, meta


def constant_per_label(maps):
    """{quantity: worst within-label spread} over the foreground labels: the pure-mask criteria
    assume every map is constant per label."""
    seg = maps["seg_label"].astype(int)
    out = {}
    for q, m in maps.items():
        if q == "seg_label":
            continue
        worst = 0.0
        for lab in np.unique(seg[seg > 0]):
            v = m[seg == lab]
            worst = max(worst, float(v.max() - v.min()) / max(1e-30, float(np.abs(m).max())))
        out[q] = worst
    return out


# ------------------------------------------------------------------------------- simasl

def asl_series(acq_matrix, context="m0scan control label", **over):
    """simasl's ASL series parameters with the defaults spelled out where the benchmarks rely
    on them (`validators/user_parameter_input.py:174-280`); SNR 0 unless asked."""
    n = len(context.split())
    p = {
        "label_type": "pcasl", "label_duration": 1.8, "signal_time": 3.6, "label_efficiency": 0.85,
        "asl_context": context, "echo_time": [0.01] * n,
        "repetition_time": ([10.0] + [5.0] * (n - 1)) if context.split()[0] == "m0scan" else [5.0] * n,
        "acq_contrast": "se", "desired_snr": 0.0, "random_seed": 0, "acq_matrix": list(acq_matrix),
        "rot_x": [0.0] * n, "rot_y": [0.0] * n, "rot_z": [0.0] * n,
        "transl_x": [0.0] * n, "transl_y": [0.0] * n, "transl_z": [0.0] * n,
    }
    p.update(over)
    return p


def run_simasl(gt, series, parameter_override, zip_path, with_ground_truth=False):
    """simasl's own `run_full_pipeline`, with `AcquireMriImageFilter` and
    `TransformResampleImageFilter` replaced (in the `asldro.examples` namespace only) by
    subclasses that record their instances. Returns each ASL volume's complex image (from the
    recorded filters), the noise reference (the M0 resample, the first transform-resample of
    the ASL branch), the archive's ASL magnitude and affine, and its ground-truth maps."""
    import asldro.examples as ex
    acquired, resampled = [], []

    class RecAcquire(ex.AcquireMriImageFilter):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            acquired.append(self)

    class RecResample(ex.TransformResampleImageFilter):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            resampled.append(self)

    orig = (ex.AcquireMriImageFilter, ex.TransformResampleImageFilter)
    ex.AcquireMriImageFilter, ex.TransformResampleImageFilter = RecAcquire, RecResample
    image_series = [{"series_type": "asl", "series_description": "compat", "series_parameters": copy.deepcopy(series)}]
    if with_ground_truth:
        image_series.append({"series_type": "ground_truth", "series_description": "gt",
                             "series_parameters": {"acq_matrix": list(series["acq_matrix"])}})
    params = {
        "global_configuration": {"ground_truth": dict(gt), "image_override": {},
                                 "parameter_override": dict(parameter_override)},
        "image_series": image_series,
    }
    try:
        ex.run_full_pipeline(params, zip_path)
    finally:
        ex.AcquireMriImageFilter, ex.TransformResampleImageFilter = orig
    n = len(series["asl_context"].split())
    if len(acquired) != n:
        raise SystemExit(f"recorded {len(acquired)} acquisitions for {n} volumes")
    vols = [np.asarray(f.outputs["image"].image) for f in acquired]
    m0_ref = np.asarray(resampled[0].outputs["image"].image, float)
    out = {"volumes": vols, "m0_reference": m0_ref, "ground_truth": {}}
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as td:
        z.extractall(td)
        asl = [os.path.join(dp, f) for dp, _, fs in os.walk(td) for f in fs if f.endswith("_asl.nii.gz")]
        if len(asl) != 1:
            raise SystemExit(f"expected one ASL series in the archive, found {asl}")
        img = nib.load(asl[0])
        out["archive_magnitude"] = np.asarray(img.dataobj, float)
        out["affine"] = img.affine
        for dp, _, fs in os.walk(td):
            for f in fs:
                if "_ground_truth_" in f and f.endswith(".nii.gz"):
                    q = f[f.index("_ground_truth_") + len("_ground_truth_"):-len(".nii.gz")]
                    out["ground_truth"][q] = np.asarray(nib.load(os.path.join(dp, f)).dataobj, float)
    mag = out["archive_magnitude"]
    if mag.ndim == 3:
        mag = mag[..., None]
    for i, v in enumerate(vols):
        peak = max(1e-30, float(np.abs(v).max()))
        dev = float(np.abs(np.abs(v) - mag[..., i]).max()) / peak
        if not (np.isfinite(dev) and np.all(np.isfinite(v)) and dev <= 1e-6):
            raise SystemExit(f"the recorded volume {i} is not the archive's ({dev:.2e} of peak): the recording is not the run")
    return out


# ------------------------------------------------------------------------------- aslscan

def resolved_kinetics(gt_meta, parameter_override):
    """simasl's lambda and T1 of arterial blood: the ground truth's JSON parameters merged with
    `global_configuration.parameter_override` (`ground_truth_loader.py:187`). Series-level keys
    of those names are not overrides in simasl (`basefilter.py:192` raises), so they are never
    read from the series."""
    p = {**gt_meta.get("parameters", {}), **parameter_override}
    return p["lambda_blood_brain"], p["t1_arterial_blood"]


def translate(series, gt_shape, gt_affine, gt_meta, parameter_override=None, trajectory=None):
    """simasl's ASL series parameters -> (asl.json dict, aslcontext.tsv text, overlay TOML text)
    per the addendum's parameter table. `trajectory` is a TSV path for the motion overlay."""
    check_affine(gt_affine)
    parameter_override = parameter_override or {}
    ctx = series["asl_context"].split()
    n = len(ctx)
    for key in ("echo_time", "repetition_time", "rot_x", "rot_y", "rot_z", "transl_x", "transl_y", "transl_z"):
        if len(series[key]) != n:
            raise SystemExit(f"{key} has {len(series[key])} entries for {n} volumes")
    te = series["echo_time"]
    if any(t != te[0] for t in te):
        raise SystemExit("echo_time varies per volume; multi-TE ASL arrives with P6")
    lt = series["label_type"].upper()
    side = {
        "ArterialSpinLabelingType": lt,
        "LabelingEfficiency": series["label_efficiency"],
        "M0Type": "Included" if "m0scan" in ctx else "Absent",
        "BackgroundSuppression": False,
        "RepetitionTimePreparation": list(series["repetition_time"]),
        "EchoTime": te[0],
        "MagneticFieldStrength": gt_meta["parameters"]["magnetic_field_strength"],
        "AcquisitionVoxelSize": [s / a for s, a in zip(gt_shape, series["acq_matrix"])],
        "MRAcquisitionType": "2D",
        "SliceTiming": [0.0] * series["acq_matrix"][2],
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": 0.001,
    }
    if lt == "PASL":
        side.update({"BolusCutOffFlag": True, "BolusCutOffTechnique": "Q2TIPS",
                     "BolusCutOffDelayTime": series["label_duration"], "PostLabelingDelay": series["signal_time"]})
    else:
        side.update({"LabelingDuration": series["label_duration"],
                     "PostLabelingDelay": series["signal_time"] - series["label_duration"]})
    lam, t1b = resolved_kinetics(gt_meta, parameter_override)
    ov = [f"seed = {int(series['random_seed'])}", "[compat]", "asldro = true",
          f"desired_snr = {float(series['desired_snr'])}",
          "[kinetic]", f"lambda_blood_brain = {float(lam)}", f"t1_arterial_blood = {float(t1b)}"]
    contrast = series["acq_contrast"].lower()
    if contrast == "ir":
        ov += ["[signal]", 'acq_contrast = "ir"', f"inversion_time = {float(series['inversion_time'])}",
               f"excitation_flip_angle = {float(series['excitation_flip_angle'])}",
               f"inversion_flip_angle = {float(series['inversion_flip_angle'])}"]
    elif contrast != "se":
        raise SystemExit(f"acq_contrast {contrast!r}: aslscan simulates se and ir (gradient echo is P5)")
    if trajectory:
        ov += ["[motion]", 'mode = "trajectory"', f'trajectory = "{trajectory}"']
    return side, "volume_type\n" + "\n".join(ctx) + "\n", "\n".join(ov) + "\n"


def write_trajectory(path, series, fov_centre, wrong=None):
    """The aslscan trajectory TSV (mm, radians) for simasl's per-volume poses."""
    rows = ["trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z"]
    for i in range(len(series["asl_context"].split())):
        rot = [series[k][i] for k in ("rot_x", "rot_y", "rot_z")]
        tr = [series[k][i] for k in ("transl_x", "transl_y", "transl_z")]
        ang, t = pose_to_mrsim(rot, tr, fov_centre, wrong=wrong)
        r = np.radians(ang)
        rows.append("\t".join(repr(float(x)) for x in [*t, *r]))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def aslscan_binary(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: cargo build --release --features cli,kspace,par")
    return path


def run_aslscan(binary, phantom_dir, side, ctx, overlay, run_dir):
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    inp = os.path.join(run_dir, "inputs")
    os.makedirs(inp)
    paths = {k: os.path.join(inp, f) for k, f in (("j", "asl.json"), ("c", "aslcontext.tsv"), ("o", "overlay.toml"))}
    json.dump(side, open(paths["j"], "w"), indent=2)
    open(paths["c"], "w").write(ctx)
    open(paths["o"], "w").write(overlay)
    out = os.path.join(run_dir, "out")
    cmd = [binary, "--asl-json", paths["j"], "--aslcontext", paths["c"], "--overlay", paths["o"],
           "--phantom", phantom_dir, "--out", out, "--compat-asldro", "--t2-mode", "voxel"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"aslscan failed:\n{r.stdout}\n{r.stderr}")
    perf = os.path.join(out, "sub-01", "perf")
    mag = nib.load(os.path.join(perf, "sub-01_part-mag_asl.nii.gz"))
    ph = nib.load(os.path.join(perf, "sub-01_part-phase_asl.nii.gz"))
    m, p = np.asarray(mag.dataobj, float), np.asarray(ph.dataobj, float)
    if m.ndim == 3:
        m, p = m[..., None], p[..., None]
    gtd = os.path.join(perf, "ground-truth")
    gt = {}
    for desc in ("perfusion", "att", "T1map", "T2map", "M0map", "dseg"):
        f = os.path.join(gtd, f"sub-01_desc-{desc}_gt.nii.gz")
        gt[desc] = np.asarray(nib.load(f).dataobj, float)
    sidecar = json.load(open(os.path.join(perf, "sub-01_part-mag_asl.json")))
    return {"complex": m * np.exp(1j * p), "affine": mag.affine, "ground_truth": gt, "sidecar": sidecar}


# ------------------------------------------------------------------------------- comparison

def pure_mask(seg, acq_affine, acq_dims, gt_affine, voxel, motion=None):
    """The addendum's pure mask: acquisition voxels whose sample point q (the voxel centre, or
    its pre-motion position M^-1 p) has one foreground label over |i - q| <= max(8, v/2 + 0.5)
    phantom voxels per axis: the box footprint and the spline's reach together."""
    dims = seg.shape
    labels = [int(l) for l in np.unique(seg) if l > 0]
    half = np.array([max(SPLINE_REACH, v / 2.0 + 0.5) for v in voxel])
    idx = np.indices(acq_dims).reshape(3, -1).T.astype(float)
    world = idx @ acq_affine[:3, :3].T + acq_affine[:3, 3]
    if motion is not None:
        inv = np.linalg.inv(motion)
        world = world @ inv[:3, :3].T + inv[:3, 3]
    g2v = np.linalg.inv(gt_affine)
    q = world @ g2v[:3, :3].T + g2v[:3, 3]
    lo = np.ceil(q - half - 1e-9).astype(int)
    hi = np.floor(q + half + 1e-9).astype(int)
    inside = np.all(lo >= 0, axis=1) & np.all(hi < np.array(dims), axis=1)
    lo, hi = np.clip(lo, 0, np.array(dims) - 1), np.clip(hi, 0, np.array(dims) - 1)
    vol = np.prod(hi - lo + 1, axis=1)
    pure = np.zeros(len(q), bool)
    for lab in labels:
        s = np.zeros(tuple(d + 1 for d in dims), np.int64)
        s[1:, 1:, 1:] = (seg == lab).astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
        a0, b0, c0 = lo.T
        a1, b1, c1 = (hi + 1).T
        cnt = (s[a1, b1, c1] - s[a0, b1, c1] - s[a1, b0, c1] - s[a1, b1, c0]
               + s[a0, b0, c1] + s[a0, b1, c0] + s[a1, b0, c0] - s[a0, b0, c0])
        pure |= cnt == vol
    return (pure & inside).reshape(acq_dims)


def compare(a, b, mask=None):
    """max |a - b| / peak(b) and RMS(a - b) / RMS(b), over all voxels or a mask."""
    if mask is not None:
        a, b = a[mask], b[mask]
    peak = float(np.abs(b).max()) if b.size else 0.0
    rms = float(np.sqrt(np.mean(b ** 2))) if b.size else 0.0
    d = a - b
    return {"max_rel": float(np.abs(d).max()) / peak if peak > 0 else float("nan"),
            "rms_rel": float(np.sqrt(np.mean(d ** 2))) / rms if rms > 0 else float("nan"),
            "voxels": int(b.size)}


def check_affines(a, s):
    if not np.allclose(a["affine"], s["affine"], atol=1e-6):
        raise SystemExit(f"the outputs are on different grids:\naslscan\n{a['affine']}\nsimasl\n{s['affine']}")
    if a["complex"].shape[:3] != s["volumes"][0].shape:
        raise SystemExit(f"shapes differ: {a['complex'].shape[:3]} vs {s['volumes'][0].shape}")


def per_volume(a, s, ctx, mask=None):
    """Per-volume comparison on the signed real part, plus control - label (first pair)."""
    re_a = np.real(a["complex"])
    rows = []
    for i, kind in enumerate(ctx):
        rows.append({"volume": i, "kind": kind, "all": compare(re_a[..., i], np.real(s["volumes"][i])),
                     "pure": compare(re_a[..., i], np.real(s["volumes"][i]), mask) if mask is not None else None})
    if "control" in ctx and "label" in ctx:
        c, l = ctx.index("control"), ctx.index("label")
        da = re_a[..., c] - re_a[..., l]
        ds = np.real(s["volumes"][c]) - np.real(s["volumes"][l])
        rows.append({"volume": "control-label", "kind": "difference", "all": compare(da, ds),
                     "pure": compare(da, ds, mask) if mask is not None else None})
    return rows


# ------------------------------------------------------------------------------- the benchmarks

class Bench:
    def __init__(self, args, phantom_key, crop=False):
        self.args = args
        self.binary = aslscan_binary(args.aslscan)
        self.name = PHANTOMS.get(phantom_key, SYNTH)
        self.tag = phantom_key + ("-crop" if crop else "")
        # The pure-mask criteria gate only where the mask can exist (the synthetic blocks); on
        # the ICBM anatomy B and D are reported with the mask size.
        self.gated = phantom_key == "synth"
        os.makedirs(WORK, exist_ok=True)
        if phantom_key == "synth":
            self.gt = synthetic_ground_truth(os.path.join(WORK, "gt-synth"))
            self.phantom, self.fingerprint = convert_phantom(SYNTH, os.path.join(WORK, "phantom-synth"), source=self.gt["nii"])
        elif crop:
            self.gt = packed_crop(self.name, CROP, os.path.join(WORK, "gt-crop"))
            self.phantom, self.fingerprint = convert_phantom(self.name, os.path.join(WORK, "phantom-crop"), CROP)
        else:
            self.gt = ground_truth(self.name)
            self.phantom, self.fingerprint = convert_phantom(self.name, os.path.join(REPO, "work", f"phantom-{phantom_key}"))
        self.gt_affine, self.maps, self.meta = load_gt_arrays(self.gt)
        check_affine(self.gt_affine)
        self.shape = self.maps["seg_label"].shape
        self.seg = self.maps["seg_label"].astype(int)

    def run_pair(self, label, series, override=None, gt_series=False, wrong=None, simasl=None):
        override = override or {}
        run_dir = os.path.join(WORK, f"run-{label}-{self.tag}")
        traj = None
        moving = any(any(v != 0 for v in series[k]) for k in ("rot_x", "rot_y", "rot_z", "transl_x", "transl_y", "transl_z"))
        if moving:
            # beside the run directory, which run_aslscan recreates
            traj = run_dir + "-trajectory.tsv"
            write_trajectory(traj, series, self.fov_centre(series), wrong=wrong)
        if simasl is None:
            simasl = run_simasl(self.gt, series, override, os.path.join(WORK, f"simasl-{label}-{self.tag}.zip"), gt_series)
        side, ctx, ov = translate(series, self.shape, self.gt_affine, self.meta, override, traj)
        a = run_aslscan(self.binary, self.phantom, side, ctx, ov, run_dir)
        check_affines(a, simasl)
        return a, simasl

    def fov_centre(self, series):
        """aslscan's rotation centre: the simulation grid's centre (= the acquisition grid's at
        oversample 1), voxel-centre origin: o + v (n - 1) / 2."""
        v = np.array([s / a for s, a in zip(self.shape, series["acq_matrix"])])
        return self.gt_affine[:3, 3] + v * (np.array(series["acq_matrix"]) - 1) / 2.0

    def identity_matrix(self):
        return list(self.shape)

    # ---- A and E: identity grid, criterion 1e-5 of peak (1e-4 for control - label) ----
    def exact(self, label, series):
        a, s = self.run_pair(label, series)
        ctx = series["asl_context"].split()
        rows = per_volume(a, s, ctx)
        ok = True
        for r in rows:
            tol = 1e-4 if r["volume"] == "control-label" else 1e-5
            r["criterion"] = tol
            r["pass"] = bool(r["all"]["max_rel"] <= tol)
            ok &= r["pass"]
        return {"rows": rows, "pass": ok}

    def bench_a(self):
        return self.exact("A", asl_series(self.identity_matrix()))

    def bench_e(self):
        n = 3
        return self.exact("E", asl_series(self.identity_matrix(), acq_contrast="ir", inversion_time=1.0,
                                          excitation_flip_angle=60.0, inversion_flip_angle=180.0,
                                          repetition_time=[10.0] + [5.0] * (n - 1)))

    # ---- B: [64, 64, 12], pure mask at 1e-3; all-voxel numbers and ground truth reported ----
    def bench_b(self):
        series = asl_series([64, 64, 12])
        a, s = self.run_pair("B", series, gt_series=True)
        voxel = [sh / m for sh, m in zip(self.shape, series["acq_matrix"])]
        mask = pure_mask(self.seg, a["affine"], series["acq_matrix"], self.gt_affine, voxel)
        spread = constant_per_label(self.maps)
        ctx = series["asl_context"].split()
        rows = per_volume(a, s, ctx, mask)
        ok = int(mask.sum()) >= MIN_PURE
        for r in rows:
            r["criterion"] = 1e-3
            r["pass"] = bool(r["pure"]["max_rel"] <= 1e-3)
            ok &= r["pass"]
        gt_rows = []
        pairs = [("perfusion", "perfusion_rate"), ("att", "transit_time"), ("T1map", "t1"), ("T2map", "t2"), ("M0map", "m0")]
        perf = a["ground_truth"]["perfusion"] > 0
        for desc, q in pairs:
            if q not in s["ground_truth"]:
                raise SystemExit(f"simasl's archive has no ground-truth {q}")
            m = mask & perf if desc == "att" else mask
            row = {"map": desc, "all": compare(a["ground_truth"][desc], s["ground_truth"][q]),
                   "pure": compare(a["ground_truth"][desc], s["ground_truth"][q], m), "criterion": 1e-3}
            row["pass"] = bool(row["pure"]["max_rel"] <= 1e-3)
            if desc == "att":
                # The CSF transit time is a 1000 s sentinel: a step ~1000x the tissue value, so
                # the spline tail still carries it past the 8-voxel reach (5.7 ms in WM on the
                # synthetic blocks). The bound assumes a step no larger than the map's peak;
                # simasl's ATT ground truth is reported, not gated.
                row["gated"] = False
            else:
                ok &= row["pass"]
            gt_rows.append(row)
        res = {"rows": rows, "ground_truth": gt_rows, "pure_voxels": int(mask.sum()),
               "pure_voxels_by_label": {int(l): int((mask & (a["ground_truth"]["dseg"] == l)).sum()) for l in (1, 2, 3)},
               "map_spread_within_label": spread, "gated": self.gated, "pass": bool(ok)}
        if not self.gated:
            res["criteria_met"], res["pass"] = bool(ok), True
        return res

    # ---- C: noise statistics over 8 seeds per side ----
    def bench_c(self):
        base = asl_series([64, 64, 12])
        ctx = base["asl_context"].split()
        a0, s0 = self.run_pair("C-clean", base)
        ref_a = a0["sidecar"]["AslscanSimulation"]["Compat"]["M0ReferenceMean"]
        nz = s0["m0_reference"][s0["m0_reference"] != 0]
        ref_s = float(np.mean(np.abs(nz)))
        noise_a, noise_s = [], []
        for seed in C_SEEDS:
            series = dict(base, desired_snr=50.0, random_seed=seed)
            a, s = self.run_pair(f"C-{seed}", series)
            noise_a.append(a["complex"] - a0["complex"])
            noise_s.append(np.stack([v - v0 for v, v0 in zip(s["volumes"], s0["volumes"])], axis=-1))
        res = {"reference_mean": {"aslscan": ref_a, "simasl": ref_s},
               "reference_voxels": {"aslscan": a0["sidecar"]["AslscanSimulation"]["Compat"]["M0ReferenceVoxels"],
                                    "simasl": int(nz.size)}}
        ok = True
        for side, fields, ref in (("aslscan", noise_a, ref_a), ("simasl", noise_s, ref_s)):
            st = noise_stats(fields, ctx, (ref / 50.0) ** 2)
            res[side] = st
            ok &= st["pass"]
        # distinct realizations
        for side, fields in (("aslscan", noise_a), ("simasl", noise_s)):
            same = [(i, j) for i in range(len(fields)) for j in range(i + 1, len(fields))
                    if np.array_equal(fields[i][..., 0], fields[j][..., 0])]
            res[side]["identical_pairs"] = same
            ok &= not same
        ratio = res["simasl"]["variance"] / res["aslscan"]["variance"]
        want = (ref_s / ref_a) ** 2
        res["cross"] = {"variance_ratio_simasl_over_aslscan": ratio, "reference_ratio_squared": want,
                        "pass": bool(abs(ratio / want - 1) <= 0.10)}
        ok &= res["cross"]["pass"]
        res["pass"] = bool(ok)
        return res

    # ---- D: motion conventions on the identity grid ----
    def bench_d(self):
        out = {"cases": [], "pass": True}
        cases = [("mixed", (2.0, -3.0, 4.0), (1.5, -2.0, 0.5), None, True),
                 ("single-axis", (0.0, 0.0, 5.0), (0.0, 0.0, 0.0), None, True),
                 ("wrong-order", (2.0, -3.0, 4.0), (1.5, -2.0, 0.5), "order", False),
                 ("wrong-centre", (2.0, -3.0, 4.0), (1.5, -2.0, 0.5), "centre", False)]
        simasl_runs = {}
        # On the synthetic blocks the right pose's floor is ~2e-5 of peak while the wrong
        # conversions misplace the M0 ramp by ~1e-3 at worst: 1e-4 separates them; 1e-3 would
        # not. On the anatomy (not gated) the addendum's 1e-3 is reported.
        tol = 1e-4 if self.gated else 1e-3
        for name, rot, tr, wrong, should_pass in cases:
            # simasl's BIDS writer needs a label row, and an m0scan row (or an M0 value, which as a
            # series key collides with the ground truth's): its default rows, all carrying the pose
            series = asl_series(self.identity_matrix(), rot_x=[rot[0]] * 3, rot_y=[rot[1]] * 3,
                                rot_z=[rot[2]] * 3, transl_x=[tr[0]] * 3, transl_y=[tr[1]] * 3, transl_z=[tr[2]] * 3)
            key = (rot, tr)
            a, s = self.run_pair(f"D-{name}", series, wrong=wrong, simasl=simasl_runs.get(key))
            simasl_runs[key] = s
            m = simasl_matrix(rot, tr)
            voxel = [1.0, 1.0, 1.0]
            mask = pure_mask(self.seg, a["affine"], self.shape, self.gt_affine, voxel, motion=m)
            rows = per_volume(a, s, ["m0scan", "control", "label"], mask)[:3]
            worst_pure = max(r["pure"]["max_rel"] for r in rows)
            # A case is evaluable only with a large enough mask and finite numbers on both
            # sides; a negative control must then show a finite error above the tolerance,
            # never a NaN or an empty mask standing in for a failure.
            valid = bool(int(mask.sum()) >= MIN_PURE and np.isfinite(worst_pure)
                         and np.all(np.isfinite(a["complex"])) and all(np.all(np.isfinite(v)) for v in s["volumes"]))
            passed = bool(valid and worst_pure <= tol)
            as_expected = bool(valid and (worst_pure <= tol if should_pass else worst_pure > tol))
            case = {"case": name, "rot_deg": rot, "transl_mm": tr, "wrong": wrong, "expected_pass": should_pass,
                    "pure_voxels": int(mask.sum()), "all": {"max_rel": max(r["all"]["max_rel"] for r in rows)},
                    "pure": {"max_rel": worst_pure}, "criterion": tol, "valid": valid,
                    "passed": passed, "as_expected": as_expected}
            out["cases"].append(case)
            out["pass"] &= case["as_expected"]
        out["gated"] = self.gated
        if not self.gated:
            out["criteria_met"], out["pass"] = bool(out["pass"]), True
        out["pass"] = bool(out["pass"])
        return out

    # ---- part C: the motion approximation on B's grid (reported) ----
    def bench_d_grid(self):
        rot, tr = (2.0, -3.0, 4.0), (1.5, -2.0, 0.5)
        series = asl_series([64, 64, 12], rot_x=[rot[0]] * 3, rot_y=[rot[1]] * 3,
                            rot_z=[rot[2]] * 3, transl_x=[tr[0]] * 3, transl_y=[tr[1]] * 3, transl_z=[tr[2]] * 3)
        a, s = self.run_pair("D-grid", series)
        voxel = [sh / m for sh, m in zip(self.shape, series["acq_matrix"])]
        mask = pure_mask(self.seg, a["affine"], series["acq_matrix"], self.gt_affine, voxel, motion=simasl_matrix(rot, tr))
        r = per_volume(a, s, ["m0scan", "control", "label"], mask)[1]
        return {"reported": True, "pure_voxels": int(mask.sum()), "all": r["all"], "pure": r["pure"], "pass": True}


def noise_stats(fields, ctx, predicted):
    """The addendum's per-side noise criteria over a list of (x, y, z, volume) complex fields."""
    allz = np.stack(fields, axis=-1)  # x, y, z, volume, realization
    re, im = np.real(allz).ravel(), np.imag(allz).ravel()
    n = re.size
    var_re, var_im = float(np.var(re)), float(np.var(im))
    sd = np.sqrt(0.5 * (var_re + var_im))
    st = {"n": int(n), "predicted_variance": predicted, "variance_re": var_re, "variance_im": var_im,
          "variance": 0.5 * (var_re + var_im), "mean_re": float(re.mean()), "mean_im": float(im.mean()),
          "rho_re_im": float(np.corrcoef(re, im)[0, 1])}
    # whiteness and the control - label variance, on both components
    adj, vcl = {}, {}
    c, l = ctx.index("control"), ctx.index("label")
    for part, comp in (("re", np.real(allz)), ("im", np.imag(allz))):
        for ax, nm in ((0, "x"), (1, "y"), (2, "z")):
            a = np.moveaxis(comp, ax, 0)
            adj[f"{part} {nm}"] = float(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1])
        dcl = (comp[..., c, :] - comp[..., l, :]).ravel()
        vcl[part] = float(np.var(dcl) / np.var(comp[..., c, :]))
    st["adjacent_rho"] = adj
    st["var_cl_over_var_c"] = vcl["re"]
    st["var_cl_over_var_c_by_part"] = vcl
    finite = bool(np.all(np.isfinite(allz)))
    checks = {
        "finite": finite,
        "mean": abs(st["mean_re"]) <= 3 * sd / np.sqrt(n) and abs(st["mean_im"]) <= 3 * sd / np.sqrt(n),
        "variance": abs(var_re / predicted - 1) <= 0.05 and abs(var_im / predicted - 1) <= 0.05,
        "re_im": abs(st["rho_re_im"]) < 0.02,
        "white": all(abs(v) < 0.02 for v in adj.values()),
        "control_minus_label": all(abs(v / 2 - 1) <= 0.10 for v in vcl.values()),
    }
    st["checks"] = {k: bool(v) for k, v in checks.items()}
    st["pass"] = bool(all(checks.values()))
    return st


# ------------------------------------------------------------------------------- reports

def fmt(x):
    return f"{x:.3e}" if isinstance(x, float) else str(x)


def markdown(bench, tag, res):
    out = [f"# Benchmark {bench} ({tag}): {'PASS' if res['pass'] else 'FAIL'}", ""]
    if "rows" in res:
        out += ["| volume | kind | max rel (all) | rms rel (all) | max rel (pure) | voxels (pure) | criterion | pass |",
                "|---|---|---|---|---|---|---|---|"]
        for r in res["rows"]:
            p = r.get("pure") or {}
            out.append(f"| {r['volume']} | {r['kind']} | {fmt(r['all']['max_rel'])} | {fmt(r['all']['rms_rel'])} | "
                       f"{fmt(p.get('max_rel', '-'))} | {p.get('voxels', '-')} | {r['criterion']} | {r['pass']} |")
    if "ground_truth" in res:
        out += ["", "| ground truth | max rel (all) | rms rel (all) | max rel (pure) | voxels | pass |", "|---|---|---|---|---|---|"]
        for r in res["ground_truth"]:
            out.append(f"| {r['map']} | {fmt(r['all']['max_rel'])} | {fmt(r['all']['rms_rel'])} | "
                       f"{fmt(r['pure']['max_rel'])} | {r['pure']['voxels']} | {r['pass']} |")
    if "cases" in res:
        out += ["| case | wrong | pure voxels | max rel (all) | max rel (pure) | passed | expected |", "|---|---|---|---|---|---|---|"]
        for c in res["cases"]:
            out.append(f"| {c['case']} | {c['wrong']} | {c['pure_voxels']} | {fmt(c['all']['max_rel'])} | "
                       f"{fmt(c['pure']['max_rel'])} | {c['passed']} | {c['expected_pass']} |")
    if "cross" in res:
        out += ["| side | ref mean | ref voxels | predicted var | var re | var im | rho re/im | adjacent rho | Var(C-L)/Var(C) | pass |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for side in ("aslscan", "simasl"):
            st = res[side]
            out.append(f"| {side} | {res['reference_mean'][side]:.4f} | {res['reference_voxels'][side]} | "
                       f"{fmt(st['predicted_variance'])} | {fmt(st['variance_re'])} | {fmt(st['variance_im'])} | "
                       f"{st['rho_re_im']:+.4f} | {', '.join(f'{k} {v:+.4f}' for k, v in st['adjacent_rho'].items())} | "
                       f"{st['var_cl_over_var_c']:.3f} | {st['pass']} |")
        cr = res["cross"]
        out += ["", f"Cross-side variance ratio simasl/aslscan {cr['variance_ratio_simasl_over_aslscan']:.4f}; "
                    f"squared reference ratio {cr['reference_ratio_squared']:.4f}; pass {cr['pass']}."]
    if res.get("reported"):
        out.append(f"Reported, not gated: control max rel (all) {fmt(res['all']['max_rel'])}, rms rel (all) "
                   f"{fmt(res['all']['rms_rel'])}; max rel (pure) {fmt(res['pure']['max_rel'])}.")
    for k in ("gated", "criteria_met", "pure_voxels", "pure_voxels_by_label", "map_spread_within_label"):
        if k in res:
            out.append(f"\n{k}: {res[k]}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", choices=["A", "B", "C", "D", "D-grid", "E", "all"])
    ap.add_argument("--phantom", choices=sorted(PHANTOMS) + ["synth"], default="3t",
                    help="synth: three tissue blocks on the 3 T grid, where the B and D pure masks exist")
    ap.add_argument("--crop", action="store_true", help="the checked-in crop's window of the 3 T ground truth")
    ap.add_argument("--aslscan", default=os.path.join(REPO, "target", "release", "aslscan"))
    a = ap.parse_args()
    if a.crop and a.phantom != "3t":
        raise SystemExit("--crop is a window of the 3 T ground truth")
    b = Bench(a, a.phantom, crop=a.crop)
    benches = ["A", "B", "C", "D", "D-grid", "E"] if a.bench == "all" else [a.bench]
    if a.crop and any(x not in ("A", "E") for x in benches):
        raise SystemExit("on the crop only A and E apply (the others need the full field of view)")
    runs = [(b, name) for name in benches]
    if a.bench == "all" and not b.gated:
        # B and D are report-only on the anatomy; their gates are the synthetic blocks', and
        # `all` must not succeed without them.
        synth = Bench(a, "synth")
        runs += [(synth, name) for name in ("B", "D", "D-grid")]
    ok = True
    for b, name in runs:
        res = getattr(b, "bench_" + name.lower().replace("-", "_"))()
        res["phantom_fingerprint"] = b.fingerprint
        stem = os.path.join(WORK, f"{name}-{b.tag}")
        json.dump(res, open(stem + ".json", "w"), indent=2, default=float)
        md = markdown(name, b.tag, res)
        open(stem + ".md", "w").write(md)
        print(md)
        ok &= res["pass"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
