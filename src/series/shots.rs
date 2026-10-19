//! Per-shot poses for 2D readouts (P9 addendum, part A).
//!
//! A shot is the set of slices excited together, from the normalized slice offsets
//! (`slice_schedule_from_times`). Shot `s` of volume `v` is excited at
//! `T(v, s) = start[v] + t_v + offset(s)`, and moves by `Q(v, s) = J(v, s) . P(v, s)`: `P` is the
//! volume's pose (P3) or, under `within_volume_sampling = "slice"`, the trajectory through the
//! volumes' poses placed at their first shots, sampled at `T(v, s)`; `J` is the pose whose six
//! parameters are the sum of the jumps of the volume's events at shots `<= s`. Both are rigid
//! transforms about the field-of-view centre, composed as matrices. The images are resampled once
//! per shot (`mrsim_acq::motion::apply_shot_transforms`).
//!
//! `start` and `t_v` come from the caller: `series` passes the protocol rows' clock, `p6` its raw
//! volumes' (`first_prep(v).start_s`), which under Hadamard are not the protocol's rows.

use mrsim_acq::grid::Grid;
use mrsim_acq::motion::{
    apply_shot_transforms, fov_center, sample_trajectory, slice_schedule_from_times, DropoutLaw, DroppedShot,
    MotionEvent, Pose, IDENTITY_MATRIX,
};

use crate::protocol::{MotionSpec, WithinVolumeSampling};

/// The resolved shots of a series.
#[derive(Debug, Clone)]
pub struct ShotPlan {
    /// Acquisition order of shots -> the slices excited in each.
    pub schedule: Vec<Vec<usize>>,
    /// `[volume][shot]` excitation times, seconds on the series clock.
    pub times: Vec<Vec<f64>>,
    /// `[volume][shot]` world-space transforms `Q = J . P`.
    pub transforms: Vec<Vec<[[f64; 4]; 4]>>,
    /// `[volume][shot]` dropout factors, one per event, in event order.
    pub attenuation: Vec<Vec<Vec<f32>>>,
    /// The events, drawn or read from the file.
    pub events: Vec<MotionEvent>,
    /// The rotation centre the transforms were built about.
    pub center: [f64; 3],
}

impl ShotPlan {
    /// `Q(v, s)` as a pose (for `desc-motionShots_gt.tsv`).
    pub fn pose(&self, v: usize, s: usize) -> Pose {
        Pose::from_matrix(self.transforms[v][s], self.center)
    }
}

/// `a . b` for 4x4 affine matrices.
fn matmul4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

fn matrix_of(p: Pose, c: [f64; 3]) -> [[f64; 4]; 4] {
    if p == Pose::IDENTITY { IDENTITY_MATRIX } else { p.to_matrix(c) }
}

/// The shot plan of a series of `n` volumes. `volume_poses` are P3's per-volume poses, `starts`
/// and `signal_t` each volume's start and signal time on the series clock, `offsets` the
/// normalized slice offsets. `draw` draws random events for a shot count (`series::draw_events`).
pub fn shot_plan(m: &MotionSpec, volume_poses: &[Pose], starts: &[f64], signal_t: &[f64], offsets: &[f64],
                 sim_grid: &Grid, draw: impl Fn(usize) -> Vec<MotionEvent>) -> ShotPlan {
    let n = volume_poses.len();
    assert!(starts.len() == n && signal_t.len() == n, "one start and one signal time per volume");
    let schedule = slice_schedule_from_times(offsets);
    let n_shots = schedule.len();
    let shot_offset: Vec<f64> = schedule.iter().map(|s| offsets[s[0]]).collect();
    let times: Vec<Vec<f64>> = (0..n).map(|v| shot_offset.iter().map(|o| starts[v] + signal_t[v] + o).collect()).collect();
    let events: Vec<MotionEvent> = match m.within.as_ref().and_then(|w| w.events_file.as_ref()) {
        Some(f) => f.events.iter().map(|e| MotionEvent {
            volume: e.volume, shot: e.shot, severity: e.severity, jump_mm: e.jump_mm, jump_deg: e.jump_deg,
        }).collect(),
        None => draw(n_shots),
    };
    let center = fov_center(sim_grid.dims, sim_grid.voxel_to_world);
    // the trajectory sampled per shot: a timed trace (P10, `Trace::pose_at`), else the volumes' poses
    // placed at their first shots (P9)
    let node_times: Vec<f64> = times.iter().map(|t| t[0]).collect();
    let mut transforms = vec![vec![IDENTITY_MATRIX; n_shots]; n];
    let mut attenuation = vec![vec![Vec::new(); n_shots]; n];
    for v in 0..n {
        let mut cum = Pose::IDENTITY;
        for s in 0..n_shots {
            for e in events.iter().filter(|e| e.volume == v && e.shot == s) {
                for i in 0..3 {
                    cum.trans_mm[i] += e.jump_mm[i];
                    cum.rot_deg[i] += e.jump_deg[i];
                }
                attenuation[v][s].push(DropoutLaw::Uniform.attenuation(v, e.severity));
            }
            let p = match m.sampling {
                WithinVolumeSampling::Volume => volume_poses[v],
                WithinVolumeSampling::Slice | WithinVolumeSampling::Shot => match &m.timed {
                    Some(t) => t.trace.pose_at(times[v][s]),
                    None => sample_trajectory(&node_times, volume_poses, times[v][s]),
                },
            };
            let (pm, jm) = (matrix_of(p, center), matrix_of(cum, center));
            transforms[v][s] = if jm == IDENTITY_MATRIX { pm } else if pm == IDENTITY_MATRIX { jm } else { matmul4(&jm, &pm) };
        }
    }
    ShotPlan { schedule, times, transforms, attenuation, events, center }
}

/// P10: the times a run queries a timed trace at, on the caller's clock (`starts` and `signal_t`
/// per volume, `offsets` per slice, normalized so the first shot is at 0): each volume's reference
/// time (its first shot) under one pose per volume, every shot's time under one pose per shot.
pub fn query_times(starts: &[f64], signal_t: &[f64], offsets: &[f64], per_shot: bool) -> Vec<f64> {
    let shot_offset: Vec<f64> = if per_shot {
        slice_schedule_from_times(offsets).iter().map(|s| offsets[s[0]]).collect()
    } else {
        vec![0.0]
    };
    starts.iter().zip(signal_t).flat_map(|(s, t)| shot_offset.iter().map(move |o| s + t + o)).collect()
}

/// P10 part C: a 3D readout's segment truth under per-shot sampling: `[unit][segment]` excitation
/// times on the series clock and absolute poses `J . P(t)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentShots {
    pub times: Vec<Vec<f64>>,
    pub poses: Vec<Vec<Pose>>,
}

/// P10 part C: the segment poses of one 3D motion unit. `cum[s]` is the summed jumps `J` of the
/// unit's events up to segment `s`, `times[s]` the segment's excitation time, `p_v` the pose the
/// unit's images were already moved by (its reference pose), `pose_at` the trajectory. Returns,
/// per segment, the pose to move those images by further (`J . P(t) . P_v^-1`) and the absolute
/// pose (`J . P(t)`). Where `P(t)` is `P_v` exactly the first is `J` itself, with no matrix
/// arithmetic, so a trajectory that holds still within a unit reproduces the per-volume path bit
/// for bit; the identity is the base images. A composed rotation the `Rz Ry Rx` decomposition
/// cannot represent (pitch at 90 degrees) is refused rather than resampled wrong.
pub fn segment_poses(cum: &[Pose], times: &[f64], p_v: Pose, pose_at: impl Fn(f64) -> Pose, center: [f64; 3])
    -> Result<(Vec<Pose>, Vec<Pose>), String>
{
    let pv_inv = {
        let m = matrix_of(p_v, center);
        rigid_inverse(&m)
    };
    // the decomposition reproduces the matrix, or the segment pose is refused
    let decompose = |m: [[f64; 4]; 4], s: usize| -> Result<Pose, String> {
        let p = Pose::from_matrix(m, center);
        let back = p.to_matrix(center);
        let err = back.iter().flatten().zip(m.iter().flatten()).fold(0.0f64, |e, (a, b)| e.max((a - b).abs()));
        if err > 1e-9 {
            return Err(format!(
                "segment {s} at {} s: its pose composes to a rotation at 90 degrees of pitch, which the Rz Ry Rx angles \
                 cannot represent (reconstruction error {err:.2e})", times[s]));
        }
        Ok(p)
    };
    let mut rel = Vec::with_capacity(cum.len());
    let mut abs = Vec::with_capacity(cum.len());
    for (s, &j) in cum.iter().enumerate() {
        let pt = pose_at(times[s]);
        let a = matmul4(&matrix_of(j, center), &matrix_of(pt, center));
        abs.push(if j == Pose::IDENTITY { pt } else if pt == Pose::IDENTITY { j } else { decompose(a, s)? });
        rel.push(if pt == p_v { j } else { decompose(matmul4(&a, &pv_inv), s)? });
    }
    Ok((rel, abs))
}

/// The number of distinct poses other than the identity among one unit's segments: the image sets
/// it holds beside its base images.
pub fn distinct_moved(poses: &[Pose]) -> usize {
    let mut distinct: Vec<Pose> = Vec::new();
    for &q in poses {
        if q != Pose::IDENTITY && !distinct.contains(&q) {
            distinct.push(q);
        }
    }
    distinct.len()
}

/// P10 part C: `sets` image sets of `n_images` compartment images of `nvox` voxels, all held at once
/// until the acquisition, within `limit_gib` (`[images] max_memory_gib`), checked before any is
/// made.
pub fn check_segment_memory(sets: usize, n_images: usize, nvox: usize, limit_gib: f64) -> Result<(), String> {
    let gib = sets as f64 * n_images as f64 * nvox as f64 * 4.0 / (1u64 << 30) as f64;
    if gib > limit_gib {
        return Err(format!(
            "within_volume_sampling = \"shot\" on this 3D readout makes {sets} segment image sets of {n_images} \
             compartment images ({gib:.2} GiB), over [images] max_memory_gib = {limit_gib}; raise it, sample fewer \
             distinct poses, or simulate fewer volumes"));
    }
    Ok(())
}

/// The inverse of a rigid world-space transform: `[R^T | -R^T t]`.
fn rigid_inverse(m: &[[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut out = IDENTITY_MATRIX;
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = m[j][i];
        }
        out[i][3] = -(0..3).map(|k| m[k][i] * m[k][3]).sum::<f64>();
    }
    out
}

/// P10, for the sidecar: the smallest interval between two shots of a volume (s): between a 2D
/// readout's distinct slice times, or a segmented 3D readout's repetitions; `None` with one shot.
pub fn min_shot_spacing(p: &crate::protocol::Protocol) -> Option<f64> {
    match &p.readout {
        Some(r) => (r.number_shots.0 > 1).then(|| p.rows.iter().map(|row| row.tr).fold(f64::INFINITY, f64::min)),
        None => {
            let shots: Vec<f64> = slice_schedule_from_times(&p.slice_offsets).iter().map(|s| p.slice_offsets[s[0]]).collect();
            shots.windows(2).map(|w| w[1] - w[0]).reduce(f64::min)
        }
    }
}

/// Each volume's reference time (its first shot): `starts[v] + signal_t[v]`.
pub fn reference_times(starts: &[f64], signal_t: &[f64]) -> Vec<f64> {
    starts.iter().zip(signal_t).map(|(s, t)| s + t).collect()
}

/// Move `images` (voxel-major interleaved, `n` volumes on the simulation grid) by the plan, and
/// return the dropped-shot truth (one per event, `1 - severity`, the shot's slices).
pub fn apply(plan: &ShotPlan, images: &mut [Vec<f32>], sim_grid: &Grid, n: usize) -> Vec<DroppedShot> {
    apply_shot_transforms(images, sim_grid.dims, n, sim_grid.voxel_to_world, &plan.schedule, &plan.transforms,
                          &plan.attenuation);
    dropped(plan)
}

/// The tissue's spin-history inputs (P9 part B): the phantom on its own grid and the series'
/// one preparation (`protocol` refuses spin history when the rows' preparations differ).
pub struct SpinInputs<'a> {
    pub grid: &'a Grid,
    pub t1: &'a [f32],
    pub m0: &'a [f32],
    /// The tissue class of each phantom voxel (`spin_history::NO_CLASS` outside the tissue).
    pub class: &'a [u8],
    pub n_classes: usize,
    pub suppression: Option<&'a crate::longitudinal::Suppression>,
    pub tr: f64,
    pub signal_t: f64,
    /// Each volume's start on the series clock.
    pub starts: &'a [f64],
}

/// P9 part B: the spin-history correction of the tissue classes on the simulation grid,
/// `[class][vox * n + v]`, before any factor the tissue carries, and the engine's merge count. The
/// reference is the closed form the tissue images were made with: `tissue_mz` at the shot's
/// readout under suppression, `tissue_se` without. The history uses the plan's shot transforms.
pub fn spin_history_correction(plan: &ShotPlan, inp: &SpinInputs, sim_grid: &Grid, offsets: &[f64]) -> (Vec<Vec<f32>>, u64) {
    use mrsim_acq::spin_history::{Pulse, Slabs, Source, SpinHistory};
    let n = plan.times.len();
    let [snx, sny, snz] = sim_grid.dims;
    let slab = snx * sny;
    let mut pulses = Vec::new();
    if let Some(s) = inp.suppression {
        for &start in inp.starts {
            pulses.extend(s.pulse_times.iter().map(|p| Pulse::Inversion { t: start + p, epsilon: s.epsilon }));
            if s.presaturation {
                pulses.push(Pulse::Saturation { t: start });
            }
        }
    }
    let mut h = SpinHistory::new(
        Source { dims: inp.grid.dims, v2w: inp.grid.voxel_to_world, t1: inp.t1, weight: inp.m0, class: inp.class, n_classes: inp.n_classes },
        Slabs { dims: sim_grid.dims, v2w: sim_grid.voxel_to_world },
        pulses,
    );
    // the fictitious volume: the first volume's shots one repetition early, at its first shot's pose
    let prime: Vec<(f64, &[usize])> = plan.schedule.iter().enumerate().map(|(s, sl)| (plan.times[0][s] - inp.tr, sl.as_slice())).collect();
    h.prime(&prime, plan.transforms[0][0]);
    let mut out = vec![vec![0.0f32; slab * snz * n]; inp.n_classes];
    for v in 0..n {
        for (s, slices) in plan.schedule.iter().enumerate() {
            let t_read = inp.signal_t + offsets[slices[0]];
            let (tr, sup) = (inp.tr, inp.suppression);
            let mz_ref = move |t1: f64| match sup {
                Some(s) => crate::longitudinal::tissue_mz(1.0, t1, tr, t_read, s),
                None => crate::mrsignal::tissue_se(1.0, t1, tr),
            };
            for term in h.excite(plan.times[v][s], plan.transforms[v][s], slices, &mz_ref, None) {
                for (c, img) in out.iter_mut().enumerate() {
                    for (cell, x) in term.num[c].iter().enumerate() {
                        img[(term.slice * slab + cell) * n + v] += *x;
                    }
                }
            }
        }
    }
    (out, h.merges())
}

/// The dropped-shot truth of the plan's events.
pub fn dropped(plan: &ShotPlan) -> Vec<DroppedShot> {
    plan.events.iter().map(|e| DroppedShot {
        volume: e.volume,
        shot: e.shot,
        slices: plan.schedule[e.shot].clone(),
        attenuation: DropoutLaw::Uniform.attenuation(e.volume, e.severity),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::WithinVolume;
    use mrsim_acq::motion::{apply_multiband_motion, slice_schedule, MotionMode};

    fn grid(dims: [usize; 3]) -> Grid {
        let mut v2w = IDENTITY_MATRIX;
        v2w[0][0] = 1.5;
        v2w[1][1] = 1.5;
        v2w[2][2] = 4.0;
        v2w[0][3] = -9.0;
        Grid { dims, voxel_to_world: v2w }
    }

    fn spec(sampling: WithinVolumeSampling) -> MotionSpec {
        MotionSpec {
            mode: MotionMode::Off,
            mode_name: "off".to_string(),
            within: Some(WithinVolume { dropout_rate: 0.0, severity: 0.0, jump_mm: [0.0; 3], jump_deg: [0.0; 3], events_file: None }),
            sampling,
            shot_path: true,
            spin_history: false,
            timed: None,
        }
    }

    fn smooth(dims: [usize; 3], ngrad: usize) -> Vec<f32> {
        let [nx, ny, nz] = dims;
        let mut v = vec![0.0f32; nx * ny * nz * ngrad];
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    for g in 0..ngrad {
                        let f = 1.0 + (0.5 * x as f64 + 0.2 * g as f64).sin() * (0.4 * y as f64).cos() + 0.3 * (0.7 * z as f64).sin();
                        v[(x + nx * (y + ny * z)) * ngrad + g] = f as f32;
                    }
                }
            }
        }
        v
    }

    #[test]
    fn slice_sampling_places_volume_poses_at_first_shots() {
        let g = grid([6, 5, 4]);
        let poses = [
            Pose::IDENTITY,
            Pose { trans_mm: [0.0, 0.0, 1.0], ..Pose::IDENTITY },
            Pose { trans_mm: [0.0, 0.0, 3.0], rot_deg: [0.0, 0.0, 2.0] },
        ];
        // descending single band: shots are slices 3, 2, 1, 0 at 0, 0.1, 0.2, 0.3
        let offsets = [0.3, 0.2, 0.1, 0.0];
        let starts = [0.0, 4.0, 8.0];
        let t = [1.0, 1.0, 1.0];
        let plan = shot_plan(&spec(WithinVolumeSampling::Slice), &poses, &starts, &t, &offsets, &g, |_| Vec::new());
        assert_eq!(plan.schedule, vec![vec![3], vec![2], vec![1], vec![0]]);
        assert!((plan.times[1][2] - 5.2).abs() < 1e-12);
        for (v, want) in poses.iter().enumerate() {
            let q = plan.pose(v, 0);
            for i in 0..3 {
                assert!((q.trans_mm[i] - want.trans_mm[i]).abs() < 1e-9 && (q.rot_deg[i] - want.rot_deg[i]).abs() < 1e-9, "{v}: {q:?}");
            }
        }
        // the last volume holds its pose; the middle of volume 1 is on the way to volume 2's
        assert!((plan.pose(2, 3).trans_mm[2] - 3.0).abs() < 1e-9);
        let mid = plan.pose(1, 2).trans_mm[2];
        assert!((mid - (1.0 + 2.0 * 0.2 / 4.0)).abs() < 1e-9, "{mid}");
        // volume sampling: every shot the volume's pose
        let plan = shot_plan(&spec(WithinVolumeSampling::Volume), &poses, &starts, &t, &offsets, &g, |_| Vec::new());
        assert!((plan.pose(1, 3).trans_mm[2] - 1.0).abs() < 1e-9);
    }

    /// P10: a timed trace linear between shot times gives, at the shots, the transform of the
    /// cumulative jumps that reach the same poses (the hand-written events of the P9 figures).
    #[test]
    fn a_linear_timed_trace_is_the_cumulative_jumps() {
        let g = grid([6, 5, 4]);
        let step = Pose { trans_mm: [0.0, 0.5, 0.75], rot_deg: [0.8, 0.0, -0.3] };
        let offsets = [0.0, 0.1, 0.2, 0.3];
        let mut m = spec(WithinVolumeSampling::Shot);
        let k = |s: f64| Pose { trans_mm: step.trans_mm.map(|x| x * s), rot_deg: step.rot_deg.map(|x| x * s) };
        m.timed = Some(crate::protocol::TimedTrajectory {
            path: "trace.tsv".into(),
            trace: mrsim_acq::motion::Trace { times: vec![0.5, 0.8, 2.0], poses: vec![Pose::IDENTITY, k(3.0), k(3.0)] },
            clock: "time column",
        });
        // shots at 0.5, 0.6, 0.7, 0.8: the trace is k(0), k(1), k(2), k(3) there
        let timed = shot_plan(&m, &[Pose::IDENTITY], &[0.0], &[0.5], &offsets, &g, |_| Vec::new());
        let jumps: Vec<MotionEvent> = (1..4).map(|s| MotionEvent {
            volume: 0, shot: s, severity: 0.0, jump_mm: step.trans_mm, jump_deg: step.rot_deg }).collect();
        let events = shot_plan(&spec(WithinVolumeSampling::Volume), &[Pose::IDENTITY], &[0.0], &[0.5], &offsets, &g,
                               move |_| jumps.clone());
        for s in 0..4 {
            for (a, b) in timed.transforms[0][s].iter().flatten().zip(events.transforms[0][s].iter().flatten()) {
                assert!((a - b).abs() < 1e-12, "shot {s}: {a} vs {b}");
            }
        }
        assert_ne!(timed.transforms[0][3], IDENTITY_MATRIX);
    }

    /// Implementation review: a relative rotation at 90 degrees of pitch (reference (10, -45, 5)
    /// degrees, shot about (17.05, 44.78, 14.98)) cannot be decomposed into the angles the
    /// resampler takes; it is refused, while an ordinary one round-trips.
    #[test]
    fn segment_poses_refuse_what_the_angles_cannot_represent() {
        let c = [1.0, -2.0, 3.0];
        let pv = Pose { rot_deg: [10.0, -45.0, 5.0], trans_mm: [0.0; 3] };
        let lock = Pose { rot_deg: [17.0532267, 44.7823850, 14.9810694], trans_mm: [0.0; 3] };
        let e = segment_poses(&[Pose::IDENTITY], &[2.0], pv, |_| lock, c).unwrap_err();
        assert!(e.contains("90 degrees") && e.contains("2 s"), "{e}");
        let ok = Pose { rot_deg: [12.0, -40.0, 7.0], trans_mm: [0.5, 0.0, 1.0] };
        let (rel, abs) = segment_poses(&[Pose::IDENTITY], &[2.0], pv, |_| ok, c).unwrap();
        assert_eq!(abs[0], ok);
        let back = matmul4(&rel[0].to_matrix(c), &pv.to_matrix(c));
        for (a, b) in back.iter().flatten().zip(ok.to_matrix(c).iter().flatten()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn query_times_are_the_reference_times_or_every_shot() {
        // interleaved single band: shots at 0, 0.04, 0.08 (slices 0, 2, 1)
        let offsets = [0.0, 0.08, 0.04];
        assert_eq!(query_times(&[0.0, 4.0], &[1.0, 1.5], &offsets, false), vec![1.0, 5.5]);
        let q = query_times(&[0.0, 4.0], &[1.0, 1.5], &offsets, true);
        let want = [1.0, 1.04, 1.08, 5.5, 5.54, 5.58];
        assert!(q.len() == 6 && q.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12), "{q:?}");
        assert_eq!(reference_times(&[0.0, 4.0], &[1.0, 1.5]), vec![1.0, 5.5]);
    }

    #[test]
    fn jumps_compose_with_the_volume_pose_as_matrices() {
        let g = grid([6, 5, 4]);
        let p = Pose { rot_deg: [0.0, 3.0, 5.0], trans_mm: [0.5, 0.0, 0.2] };
        let jump = MotionEvent { volume: 0, shot: 1, severity: 0.0, jump_mm: [0.0, 0.4, 0.0], jump_deg: [2.0, 0.0, -4.0] };
        let plan = shot_plan(&spec(WithinVolumeSampling::Volume), &[p], &[0.0], &[1.0], &[0.0, 0.1, 0.2, 0.3], &g, |_| vec![jump]);
        let c = plan.center;
        let j = Pose { rot_deg: jump.jump_deg, trans_mm: jump.jump_mm };
        let want = matmul4(&j.to_matrix(c), &p.to_matrix(c));
        assert_eq!(plan.transforms[0][1], want);
        assert_eq!(plan.transforms[0][0], p.to_matrix(c), "the jump starts at its shot");
        let summed = Pose {
            rot_deg: [p.rot_deg[0] + j.rot_deg[0], p.rot_deg[1] + j.rot_deg[1], p.rot_deg[2] + j.rot_deg[2]],
            trans_mm: [p.trans_mm[0] + j.trans_mm[0], p.trans_mm[1] + j.trans_mm[1], p.trans_mm[2] + j.trans_mm[2]],
        }.to_matrix(c);
        let diff = (0..3).flat_map(|r| (0..4).map(move |k| (r, k))).fold(0.0f64, |m, (r, k)| m.max((want[r][k] - summed[r][k]).abs()));
        assert!(diff > 1e-3, "rotating pose and jump: the product is not the summed pose");
    }

    #[test]
    fn identity_volume_poses_match_p3_multiband_motion() {
        let dims = [8usize, 6, 6];
        let g = grid(dims);
        let n = 2;
        let schedule = slice_schedule(6, 2, true);
        let mut offsets = vec![0.0; 6];
        for (k, s) in schedule.iter().enumerate() {
            for &z in s {
                offsets[z] = 0.05 * k as f64;
            }
        }
        let events = vec![
            MotionEvent { volume: 0, shot: 1, severity: 0.25, jump_mm: [0.6, 0.0, 0.8], jump_deg: [0.0, 0.0, 2.5] },
            MotionEvent { volume: 1, shot: 2, severity: 0.5, jump_mm: [0.0, -0.3, 0.0], jump_deg: [1.0, 0.0, 0.0] },
        ];
        let base = vec![smooth(dims, n)];
        let mut old = base.clone();
        let gt_old = apply_multiband_motion(&mut old, dims, n, g.voxel_to_world, 2, true, &DropoutLaw::Uniform, &events);
        let plan = shot_plan(&spec(WithinVolumeSampling::Volume), &[Pose::IDENTITY; 2], &[0.0, 4.0], &[1.8, 1.8], &offsets, &g,
                             |_| events.clone());
        assert_eq!(plan.schedule, schedule);
        let mut new = base.clone();
        let gt_new = apply(&plan, &mut new, &g, n);
        let peak = base[0].iter().fold(0.0f32, |m, x| m.max(x.abs()));
        for i in 0..base[0].len() {
            assert!((old[0][i] - new[0][i]).abs() <= 1e-6 * peak, "{i}: {} vs {}", old[0][i], new[0][i]);
        }
        assert_eq!(gt_old.len(), gt_new.len());
        for (a, b) in gt_old.iter().zip(&gt_new) {
            assert_eq!((a.volume, a.shot, &a.slices, a.attenuation), (b.volume, b.shot, &b.slices, b.attenuation));
        }
    }

    /// P9 part B on a uniform phantom (1 mm voxels, 5 mm slices aligned to them, sequential shots
    /// 40 ms apart, TR 4.6 s): a 0.5 mm through-plane jump at shot 2 of volume 1 corrects slice 2's
    /// tissue by `-0.1 (e_TR - e_dt)` per cell, and nothing else in that volume but the hair of slice
    /// 3; with no jump, nothing anywhere.
    #[test]
    fn spin_history_correction_meets_the_closed_form() {
        use crate::protocol::{EventsFile, FileEvent};
        let (t1, tr, sig, dt) = (1.33f64, 4.6f64, 3.6f64, 0.04f64);
        let phantom = Grid { dims: [4, 4, 20], voxel_to_world: IDENTITY_MATRIX };
        let mut s = IDENTITY_MATRIX;
        s[2][2] = 5.0;
        s[2][3] = 2.0;
        let sim = Grid { dims: [4, 4, 4], voxel_to_world: s };
        let nv = 4 * 4 * 20;
        let (t1v, m0v, class) = (vec![t1 as f32; nv], vec![1.0f32; nv], vec![0u8; nv]);
        let offsets = [0.0, dt, 2.0 * dt, 3.0 * dt];
        let starts = [0.0, tr, 2.0 * tr];
        let inputs = SpinInputs { grid: &phantom, t1: &t1v, m0: &m0v, class: &class, n_classes: 1, suppression: None, tr, signal_t: sig, starts: &starts };
        let mut m = spec(WithinVolumeSampling::Volume);
        let run = |m: &MotionSpec| {
            let plan = shot_plan(m, &[Pose::IDENTITY; 3], &starts, &[sig; 3], &offsets, &sim, |_| Vec::new());
            spin_history_correction(&plan, &inputs, &sim, &offsets)
        };
        let (still, merges) = run(&m);
        assert_eq!(merges, 0);
        assert!(still[0].iter().all(|x| x.abs() < 1e-7), "a still head needs no correction");
        let fe = FileEvent { volume: 1, shot: 2, severity: 0.0, jump_mm: [0.0, 0.0, 0.5], jump_deg: [0.0; 3] };
        m.within.as_mut().unwrap().events_file = Some(EventsFile { path: "x".into(), events: vec![fe] });
        let (corr, _) = run(&m);
        let (e_tr, e_dt) = (1.0 - (-tr / t1).exp(), 1.0 - (-dt / t1).exp());
        let at = |x: usize, y: usize, z: usize, v: usize| corr[0][(x + 4 * (y + 4 * z)) * 3 + v] as f64;
        let want = -0.1 * (e_tr - e_dt);
        for y in 0..4 {
            for x in 0..4 {
                assert!((at(x, y, 2, 1) - want).abs() < 1e-5, "({x},{y}): {} vs {want}", at(x, y, 2, 1));
                assert!(at(x, y, 0, 1).abs() < 1e-7 && at(x, y, 1, 1).abs() < 1e-7, "the slices before the jump");
                assert!(at(x, y, 3, 1).abs() < 1e-3, "slice 3: a tenth last excited TR + dt ago");
            }
        }
    }

    #[test]
    fn file_events_replace_the_draw() {
        use crate::protocol::{EventsFile, FileEvent};
        let g = grid([6, 5, 3]);
        let mut m = spec(WithinVolumeSampling::Volume);
        let fe = FileEvent { volume: 1, shot: 2, severity: 0.25, jump_mm: [0.0, 0.0, 0.5], jump_deg: [0.0; 3] };
        m.within.as_mut().unwrap().events_file = Some(EventsFile { path: "x".into(), events: vec![fe] });
        let plan = shot_plan(&m, &[Pose::IDENTITY; 2], &[0.0, 4.0], &[1.0, 1.0], &[0.0, 0.1, 0.2], &g, |_| panic!("no draw"));
        assert_eq!(plan.events.len(), 1);
        assert_eq!(plan.attenuation[1][2], vec![0.75]);
        assert_eq!(dropped(&plan)[0].slices, vec![2]);
    }
}
