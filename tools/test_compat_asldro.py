"""Tests of the ASLDRO compat driver's pure parts: the pose conversion and the parameter
translation. Run in the `simasl` environment from the repository root:

    micromamba run -n simasl python -m pytest tools/test_compat_asldro.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compat_asldro as ca  # noqa: E402

META = {"parameters": {"lambda_blood_brain": 0.9, "t1_arterial_blood": 1.65, "magnetic_field_strength": 3}}
SHAPE = (197, 233, 189)
AFFINE = np.array([[1.0, 0, 0, -98], [0, 1, 0, -134], [0, 0, 1, -72], [0, 0, 0, 1]])
CENTRE = np.array([-98.0 + 3.078125 * 31.5, -134 + 3.640625 * 31.5, -72 + 15.75 * 5.5])


def points(m, pts):
    return pts @ m[:3, :3].T + m[:3, 3]


def poses():
    rng = np.random.default_rng(20261001)
    out = [(rng.uniform(-30, 30, 3), rng.uniform(-10, 10, 3)) for _ in range(10)]
    # simasl's Rx(a) Ry(+-90) Rz(c) has R[2,0] = -cos(a + c) or cos(a - c): gimbal at c = -a, c = a
    out.append((np.array([12.0, 90.0, -12.0]), np.array([1.0, -2.0, 3.0])))
    out.append((np.array([-7.0, -90.0, -7.0]), np.array([0.5, 0.0, -1.0])))
    return out


def test_the_gimbal_poses_are_singular():
    for rot, _ in poses()[-2:]:
        r = ca.rot_x(rot[0]) @ ca.rot_y(rot[1]) @ ca.rot_z(rot[2])
        assert abs(abs(r[2, 0]) - 1) < 1e-12, (rot, r[2, 0])


def test_zyx_angles_recompose_the_matrix():
    for rot, _ in poses():
        r = ca.rot_x(rot[0]) @ ca.rot_y(rot[1]) @ ca.rot_z(rot[2])
        a = ca.zyx_angles(r)
        back = ca.rot_z(a[2]) @ ca.rot_y(a[1]) @ ca.rot_x(a[0])
        assert np.abs(back - r).max() < 1e-12, (rot, a)


def test_zyx_angles_take_the_gimbal_branch_for_both_signs():
    for a, b, c in [(25.0, 90.0, -40.0), (-13.0, -90.0, 70.0), (0.0, 90.0, 0.0)]:
        r = ca.rot_z(c) @ ca.rot_y(b) @ ca.rot_x(a)
        assert abs(abs(r[2, 0]) - 1) < 1e-15
        ang = ca.zyx_angles(r)
        assert ang[2] == 0.0 and abs(ang[1] - b) < 1e-9, ang  # the gimbal branch sets rz = 0
        back = ca.rot_z(ang[2]) @ ca.rot_y(ang[1]) @ ca.rot_x(ang[0])
        assert np.abs(back - r).max() < 1e-12, (a, b, c, ang)


def test_noise_stats_reject_correlated_imaginary_noise():
    rng = np.random.default_rng(3)
    ctx = ["m0scan", "control", "label"]
    shape = (24, 24, 8, 3)

    def field(correlated):
        re = rng.standard_normal(shape)
        if correlated:
            w = rng.standard_normal((shape[0] + 1,) + shape[1:])
            im = (w[:-1] + w[1:]) / np.sqrt(2)  # x-neighbours share a sample: rho 0.5
        else:
            im = rng.standard_normal(shape)
        return re + 1j * im

    good = ca.noise_stats([field(False) for _ in range(8)], ctx, 1.0)
    assert good["pass"], good["checks"]
    bad = ca.noise_stats([field(True) for _ in range(8)], ctx, 1.0)
    assert not bad["checks"]["white"] and bad["adjacent_rho"]["im x"] > 0.4, bad["adjacent_rho"]
    assert not bad["pass"]
    nan = [field(False) for _ in range(8)]
    nan[0][0, 0, 0, 0] = np.nan
    assert not ca.noise_stats(nan, ctx, 1.0)["pass"]


def test_converted_pose_moves_every_point_where_simasl_does():
    pts = np.random.default_rng(1).uniform(-120, 120, (50, 3))
    for origin in [(0.0, 0.0, 0.0), (3.0, -4.0, 10.0)]:
        for rot, tr in poses():
            s = ca.simasl_matrix(rot, tr, origin)
            ang, t = ca.pose_to_mrsim(rot, tr, CENTRE, origin)
            m = ca.mrsim_matrix(ang, t, CENTRE)
            assert np.abs(points(s, pts) - points(m, pts)).max() < 1e-9, (rot, tr, origin)


def test_each_wrong_conversion_moves_points_elsewhere():
    pts = np.random.default_rng(2).uniform(-120, 120, (50, 3))
    rot, tr = (2.0, -3.0, 4.0), (1.5, -2.0, 0.5)
    s = ca.simasl_matrix(rot, tr)
    for wrong in ("order", "centre"):
        ang, t = ca.pose_to_mrsim(rot, tr, CENTRE, wrong=wrong)
        m = ca.mrsim_matrix(ang, t, CENTRE)
        err = np.abs(points(s, pts) - points(m, pts)).max()
        assert err > 0.5, (wrong, err)
    # a single-axis rotation has no order to get wrong, but its centre still matters
    ang, t = ca.pose_to_mrsim((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), CENTRE, wrong="order")
    assert np.abs(points(ca.simasl_matrix((0, 0, 5), (0, 0, 0)), pts) - points(ca.mrsim_matrix(ang, t, CENTRE), pts)).max() < 1e-9


def test_defaults_translate_per_the_table():
    side, ctx, ov = ca.translate(ca.asl_series([64, 64, 12]), SHAPE, AFFINE, META)
    assert side["ArterialSpinLabelingType"] == "PCASL"
    assert side["PostLabelingDelay"] == pytest.approx(1.8) and side["LabelingDuration"] == 1.8
    assert side["RepetitionTimePreparation"] == [10.0, 5.0, 5.0]
    assert side["M0Type"] == "Included" and side["EchoTime"] == 0.01
    assert side["AcquisitionVoxelSize"] == [3.078125, 3.640625, 15.75]
    assert side["SliceTiming"] == [0.0] * 12 and side["MRAcquisitionType"] == "2D"
    assert side["LabelingEfficiency"] == 0.85 and side["BackgroundSuppression"] is False
    assert ctx == "volume_type\nm0scan\ncontrol\nlabel\n"
    assert "asldro = true" in ov and "lambda_blood_brain = 0.9" in ov and "t1_arterial_blood = 1.65" in ov
    assert "desired_snr = 0.0" in ov and "seed = 0" in ov and "[signal]" not in ov and "[motion]" not in ov


def test_pasl_maps_signal_time_to_the_pld_and_duration_to_the_cutoff():
    side, _, _ = ca.translate(ca.asl_series([64, 64, 12], label_type="pasl", label_duration=0.8, signal_time=1.8),
                              SHAPE, AFFINE, META)
    assert side["ArterialSpinLabelingType"] == "PASL"
    assert side["PostLabelingDelay"] == 1.8 and side["BolusCutOffDelayTime"] == 0.8 and side["BolusCutOffFlag"] is True
    assert "LabelingDuration" not in side


def test_parameter_override_reaches_the_kinetic_table():
    _, _, ov = ca.translate(ca.asl_series([64, 64, 12]), SHAPE, AFFINE, META, {"lambda_blood_brain": 0.98})
    assert "lambda_blood_brain = 0.98" in ov and "t1_arterial_blood = 1.65" in ov


def test_ir_and_motion_reach_the_overlay():
    s = ca.asl_series([64, 64, 12], acq_contrast="ir", inversion_time=1.0, excitation_flip_angle=60.0,
                      inversion_flip_angle=180.0)
    _, _, ov = ca.translate(s, SHAPE, AFFINE, META, trajectory="/tmp/t.tsv")
    assert 'acq_contrast = "ir"' in ov and "excitation_flip_angle = 60.0" in ov and "inversion_time = 1.0" in ov
    assert 'trajectory = "/tmp/t.tsv"' in ov


def test_a_flipped_or_oblique_source_affine_is_refused():
    flipped = AFFINE.copy()
    flipped[0, 0] = -1.0
    with pytest.raises(SystemExit):
        ca.translate(ca.asl_series([64, 64, 12]), SHAPE, flipped, META)
    two_mm = AFFINE.copy()
    two_mm[1, 1] = 2.0
    with pytest.raises(SystemExit):
        ca.translate(ca.asl_series([64, 64, 12]), SHAPE, two_mm, META)


def test_unequal_echo_times_and_gradient_echo_are_refused():
    with pytest.raises(SystemExit):
        ca.translate(ca.asl_series([64, 64, 12], echo_time=[0.01, 0.02, 0.01]), SHAPE, AFFINE, META)
    with pytest.raises(SystemExit):
        ca.translate(ca.asl_series([64, 64, 12], acq_contrast="ge"), SHAPE, AFFINE, META)


def test_pure_mask_on_a_two_label_slab():
    # label 1 for x < 30, label 2 beyond: a 1 mm identity grid's voxel is pure at least 8 voxels
    # from the boundary and from the edges, and nowhere else
    seg = np.ones((60, 30, 30), int)
    seg[30:] = 2
    aff = np.eye(4)
    m = ca.pure_mask(seg, aff, seg.shape, aff, [1.0, 1.0, 1.0])
    xs = np.where(m.any(axis=(1, 2)))[0]
    assert xs.min() == 8 and xs.max() == 51 and not m[22:38].any()
    assert m[8:22, 8:22, 8:22].all()
    # a translation by +3 mm in x moves the sample points back by 3 voxels
    shift = np.eye(4)
    shift[0, 3] = 3.0
    mm = ca.pure_mask(seg, aff, seg.shape, aff, [1.0, 1.0, 1.0], motion=shift)
    xs = np.where(mm.any(axis=(1, 2)))[0]
    assert xs.min() == 11 and xs.max() == 54
