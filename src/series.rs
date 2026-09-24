//! The orchestration: `aslcontext.tsv` rows -> compartment volumes on the simulation grid -> ONE
//! `simulate_acquisition_oversampled` call for the whole series (plus one for a separate M0
//! scan), and the ground truth on the acquisition grid.
//!
//! Row semantics (spec): tissue compartments `0..K` carry the post-excitation steady state at
//! the row's repetition time for `control`, `label` and `m0scan` rows and zero for `deltam`;
//! blood compartments `K..2K` carry `-delta_m` for `label`, `+delta_m` for `deltam`, zero
//! otherwise. `K` is the number of foreground labels in `class` mode and 1 in `voxel` mode.
//! Everything is evaluated per phantom voxel and only magnetization is averaged onto the
//! simulation grid.
//!
//! P3 (addendum): under background suppression the tissue of `control`/`label` rows is the
//! signed `longitudinal::tissue_mz` at each acquired slice's readout time and the blood carries
//! the row's label factor; under inversion recovery the tissue is `mrsignal::tissue_ir` and the
//! blood `blood_ir`; `m0scan` rows and the separate M0 scan always take the spin-echo steady
//! state at their own TR (a plain readout, recorded in the sidecar). Motion is applied to the
//! assembled simulation-grid compartments before the one call: per-volume poses through
//! `mrsim_acq::motion::apply_motion`, then multiband shot events through
//! `apply_multiband_motion`. The moved `delta_m` ground truth is the simulation-grid `delta_m`
//! moved by the same poses (events excluded) and block-averaged to the acquisition grid.
//!
//! One call per series is a hard rule (spec P0 change 5a): every random stream in the
//! acquisition stage is keyed on the volume index within a call, so calling once per volume
//! would give every volume the same noise and make control minus label noise-free.

use std::collections::HashMap;

use mrsim_acq::grid::Grid;
use mrsim_acq::io::hires_grid;
use mrsim_acq::kspace::{simulate_acquisition_oversampled, Acquisition, T2Volume};
use mrsim_acq::motion::{
    apply_motion, apply_multiband_motion, resolve_poses, slice_schedule, DropoutLaw, DroppedShot, MotionEvent, Pose,
};
use mrsim_acq::phase::PhaseModel;

use crate::kinetic::delta_m;
use crate::longitudinal::{label_factor, tissue_mz};
use crate::mrsignal::{blood_ir, blood_se, tissue_ir, tissue_se, Contrast};
use crate::phantom::{Phantom, Relaxation, T2Mode};
use crate::protocol::{M0Type, Protocol, Row, RowKind, WithinVolume};
use crate::resample::{acquisition_grid, axis_aligned_voxels, Resampler};
use crate::rng::SplitMix64;

/// The seed the separate M0 scan's call uses, derived from the series seed so the two calls
/// draw different noise (they would otherwise both be "volume 0").
pub const M0_SEED_SALT: u64 = 0x4D30_5343_414E;

/// The seed the motion draws use ("MOTION"), so that turning motion on leaves the acquisition
/// noise realization unchanged.
pub const MOTION_SEED_SALT: u64 = 0x4D4F_5449_4F4E;

/// Ground-truth maps on the acquisition grid (`resample` rules per map, see the spec).
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// `+delta_m` for `label` and `deltam` rows at their own timing, zero for other rows.
    /// Voxel-major interleaved like the data: `vox * n_volumes + v`. With motion on this is
    /// the MOVED truth (same per-volume poses as the data, no shot events, no suppression
    /// factor); `delta_m_static` then keeps the unmoved one.
    pub delta_m: Vec<f32>,
    pub delta_m_static: Option<Vec<f32>>,
    pub perfusion: Vec<f32>,
    /// Masked mean over perfused voxels only (the CSF sentinel must not leak).
    pub att: Vec<f32>,
    pub t1: Vec<f32>,
    pub t2: Vec<f32>,
    pub m0: Vec<f32>,
    pub dseg: Vec<i32>,
    /// `voxel` mode only: the rate-derived relaxation maps the acquisition actually used, on the
    /// SIMULATION grid (ms).
    pub acq_t2_ms: Option<Vec<f32>>,
    pub acq_t2p_ms: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct SeriesOutput {
    pub acq_grid: Grid,
    pub sim_grid: Grid,
    pub n_volumes: usize,
    /// Voxel-major interleaved on the acquisition grid, as the acquisition stage returns them.
    pub mag: Vec<f32>,
    pub phase: Vec<f32>,
    /// The separate M0 scan (one volume), when `M0Type` is `Separate`.
    pub m0: Option<(Vec<f32>, Vec<f32>)>,
    pub mode: T2Mode,
    pub labels: Vec<(i32, String)>,
    pub n_compartments: usize,
    pub fieldmap_present: bool,
    pub seeds: (u64, Option<u64>),
    pub acquisition: Acquisition,
    pub ground_truth: GroundTruth,
    /// Per row, the background-suppression blood factor (`None` without suppression).
    pub label_factors: Option<Vec<f64>>,
    /// The per-volume poses applied (identity everywhere without motion).
    pub poses: Vec<Pose>,
    pub motion_seed: Option<u64>,
    pub events: Vec<MotionEvent>,
    pub dropped: Vec<DroppedShot>,
}

/// Test hook: alter the row semantics to prove the linearity test is not slack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RowOverride {
    #[default]
    None,
    /// `label` rows carry `+delta_m` instead of `-delta_m`.
    FlipLabelSign,
    /// `control` and `label` rows swap their blood inputs.
    SwapControlLabel,
    /// The blood input of `label` rows lands in tissue compartment 0 instead of its own
    /// compartment. Label rows only, on purpose: a mis-wiring applied to every row alike is
    /// invisible to the linearity identity, because `I_L` and `I_B` then carry the same
    /// mis-wired term and still subtract to it. Only a row-dependent wiring error breaks it.
    BloodIntoTissue0,
}

/// Per-row blood sign under the row semantics (and the test overrides).
fn blood_sign(kind: RowKind, ov: RowOverride) -> f64 {
    let base = match kind {
        RowKind::Label => -1.0,
        RowKind::Deltam => 1.0,
        RowKind::Control | RowKind::M0scan => 0.0,
    };
    match ov {
        RowOverride::None | RowOverride::BloodIntoTissue0 => base,
        RowOverride::FlipLabelSign => if kind == RowKind::Label { 1.0 } else { base },
        RowOverride::SwapControlLabel => match kind {
            RowKind::Control => -1.0,
            RowKind::Label => 0.0,
            _ => base,
        },
    }
}

pub fn simulate(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel) -> Result<SeriesOutput, String> {
    simulate_impl(p, ph, mode, phase, RowOverride::None)
}

/// [`simulate`] with a [`RowOverride`]. Only for the linearity test's negative controls, so it
/// exists only under `cfg(test)` or the `test-hooks` feature (which `tests/end_to_end.rs` needs);
/// a production build has no way to produce a deliberately wrong dataset through it.
#[cfg(any(test, feature = "test-hooks"))]
pub fn simulate_with(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride)
    -> Result<SeriesOutput, String>
{
    simulate_impl(p, ph, mode, phase, ov)
}

/// Draw the within-volume shot events: each (volume, shot) has an event with probability
/// `dropout_rate`, its jumps uniform in `[-amp, amp]` per axis. Deterministic in the draw order.
fn draw_events(w: Option<&WithinVolume>, n_volumes: usize, n_shots: usize, seed: u64) -> Vec<MotionEvent> {
    let Some(w) = w else { return Vec::new() };
    let mut rng = SplitMix64(seed ^ 0x5348_4F54_5321);
    let mut events = Vec::new();
    for volume in 0..n_volumes {
        for shot in 0..n_shots {
            if rng.unit() < w.dropout_rate {
                let jump_mm = [rng.signed(w.jump_mm[0]), rng.signed(w.jump_mm[1]), rng.signed(w.jump_mm[2])];
                let jump_deg = [rng.signed(w.jump_deg[0]), rng.signed(w.jump_deg[1]), rng.signed(w.jump_deg[2])];
                events.push(MotionEvent { volume, shot, severity: w.severity as f32, jump_mm, jump_deg });
            }
        }
    }
    events
}

/// Block-average a voxel-major interleaved simulation-grid image (`o x o` in-plane oversampling,
/// same slices) onto the acquisition grid: the box mean of equal-volume cells.
fn block_mean_inplane(src: &[f32], sim_dims: [usize; 3], o: usize, n: usize) -> Vec<f32> {
    let [snx, sny, nz] = sim_dims;
    let (nx, ny) = (snx / o, sny / o);
    let mut out = vec![0.0f32; nx * ny * nz * n];
    let inv = 1.0 / (o * o) as f64;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let dst = x + nx * (y + ny * z);
                for v in 0..n {
                    let mut acc = 0.0f64;
                    for j in 0..o {
                        for i in 0..o {
                            let s = (x * o + i) + snx * ((y * o + j) + sny * z);
                            acc += src[s * n + v] as f64;
                        }
                    }
                    out[dst * n + v] = (acc * inv) as f32;
                }
            }
        }
    }
    out
}

fn simulate_impl(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride)
    -> Result<SeriesOutput, String>
{
    if let (Some(pf), fs) = (ph.params.and_then(|q| q.field_strength), p.field_strength) {
        if (pf - fs).abs() > 1e-9 {
            return Err(format!("phantom MagneticFieldStrength {pf} disagrees with the protocol's {fs}"));
        }
    }

    // ---- grids ----
    let pv = axis_aligned_voxels(&ph.grid)?;
    let acq_grid = acquisition_grid(&ph.grid, p.voxel_size_mm, p.acq.matrix)?;
    let o = p.acq.oversample;
    let sim_grid = hires_grid(&acq_grid, o);
    let [nx, ny, nz] = acq_grid.dims;
    let [snx, sny, _] = sim_grid.dims;
    if p.slice_offsets.len() != nz {
        return Err(format!(
            "SliceTiming has {} entries but the acquisition grid has {nz} slices (phantom extent {:.1} mm at \
             {} mm slices)", p.slice_offsets.len(), ph.grid.dims[2] as f64 * pv[2], p.voxel_size_mm[2]));
    }
    let dv = p.voxel_size_mm;
    let sim_vox = [dv[0] / o as f64, dv[1] / o as f64, dv[2]];
    let r_sim = Resampler::new(ph.grid.dims, pv, sim_grid.dims, sim_vox);
    let r_acq = Resampler::new(ph.grid.dims, pv, acq_grid.dims, dv);
    let nvox_sim = snx * sny * nz;
    let nvox_acq = nx * ny * nz;

    let mut acq = p.acquisition(nx, ny)?;
    acq.do_distortions = ph.fieldmap.is_some();
    let fmap_sim: Vec<f32> = match &ph.fieldmap {
        Some(f) => r_sim.mean(f),
        None => vec![0.0; nvox_sim],
    };

    // ---- relaxation and compartment layout ----
    let (relax, mode_used) = ph.relaxation(mode)?;
    let t2_blood_ms = p.t2_blood_ms();
    // Label masks: compartment i is label i (class) or everything foreground (voxel).
    let masks: Vec<Vec<bool>> = match &relax {
        Relaxation::Class { .. } => ph.labels.iter().map(|(l, _)| ph.dseg.iter().map(|d| d == l).collect()).collect(),
        Relaxation::Voxel { .. } => vec![ph.dseg.iter().map(|d| *d > 0).collect()],
    };
    let k = masks.len();
    let ncomp = 2 * k;
    // Owned map storage so the T2Volume slices below can borrow it.
    let (acq_t2_ms, acq_t2p_ms): (Option<Vec<f32>>, Option<Vec<f32>>) = match &relax {
        Relaxation::Voxel { t2_ms, t2p_ms } => (Some(r_sim.rate_mean(t2_ms, &ph.m0)), Some(r_sim.rate_mean(t2p_ms, &ph.m0))),
        Relaxation::Class { .. } => (None, None),
    };
    let (t2_vols, ti_vols): (Vec<T2Volume>, Vec<T2Volume>) = match &relax {
        Relaxation::Class { t2_ms, t2p_ms } => {
            let mut t2v = Vec::with_capacity(ncomp);
            let mut tiv = Vec::with_capacity(ncomp);
            for i in 0..k {
                t2v.push(T2Volume::Uniform(t2_ms[i]));
                tiv.push(T2Volume::Uniform(t2p_ms[i]));
            }
            for &tp in t2p_ms.iter().take(k) {
                t2v.push(T2Volume::Uniform(t2_blood_ms));
                tiv.push(T2Volume::Uniform(tp));
            }
            (t2v, tiv)
        }
        Relaxation::Voxel { .. } => {
            let t2m = acq_t2_ms.as_deref().unwrap();
            let tpm = acq_t2p_ms.as_deref().unwrap();
            (vec![T2Volume::Map(t2m), T2Volume::Uniform(t2_blood_ms)], vec![T2Volume::Map(tpm), T2Volume::Map(tpm)])
        }
    };

    // ---- the signal equations in use ----
    let ir = p.ir.as_ref().map(|s| s.params);
    let tissue_steady = |m0: f64, t1: f64, tr: f64, se: bool| -> f64 {
        match (p.contrast, ir, se) {
            (Contrast::InversionRecovery, Some(q), false) => tissue_ir(m0, t1, tr, &q),
            _ => tissue_se(m0, t1, tr),
        }
    };
    let blood_signal = |x: f64| -> f64 {
        match (p.contrast, ir) {
            (Contrast::InversionRecovery, Some(q)) => blood_ir(x, &q),
            _ => blood_se(x),
        }
    };
    // Per-row suppression: the pulse set (deduplicated so the tissue cache can key on it) and
    // the blood factor. Rows without events take the P1 steady-state path.
    let n = p.rows.len();
    let (suppression, pulse_set): (Vec<Option<crate::longitudinal::Suppression>>, Vec<usize>) = match &p.suppression {
        None => (vec![None; n], vec![0; n]),
        Some(spec) => {
            let mut sets: Vec<Vec<f64>> = Vec::new();
            let mut ids = Vec::with_capacity(n);
            let mut sup = Vec::with_capacity(n);
            for i in 0..n {
                let s = spec.for_row(i);
                let id = match sets.iter().position(|x| *x == s.pulse_times) {
                    Some(j) => j,
                    None => {
                        sets.push(s.pulse_times.clone());
                        sets.len() - 1
                    }
                };
                ids.push(id);
                sup.push(if s.has_events() && p.rows[i].kind != RowKind::M0scan { Some(s) } else { None });
            }
            (sup, ids)
        }
    };
    let label_factors = p.suppression.as_ref().map(|spec| (0..n).map(|i| label_factor(&spec.for_row(i))).collect::<Vec<f64>>());

    // ---- tissue signal per distinct (TR, equation), per compartment, on the sim grid ----
    let mut tissue_cache: HashMap<(u64, bool), Vec<Vec<f32>>> = HashMap::new();
    let mut tissue_for = |tr: f64, se: bool| -> Vec<Vec<f32>> {
        tissue_cache
            .entry((tr.to_bits(), se))
            .or_insert_with(|| {
                let sig: Vec<f64> = (0..ph.nvox()).map(|i| tissue_steady(ph.m0[i] as f64, ph.t1[i] as f64, tr, se)).collect();
                masks
                    .iter()
                    .map(|m| {
                        let masked: Vec<f32> = sig.iter().zip(m).map(|(s, &in_m)| if in_m { *s as f32 } else { 0.0 }).collect();
                        r_sim.mean(&masked)
                    })
                    .collect()
            })
            .clone()
    };
    // ---- suppressed tissue per slice, keyed on the complete resolved preparation AND the
    // slice: simultaneously excited (multiband) slices share a readout time but not anatomy ----
    let mut slice_cache: HashMap<(u64, u64, usize, usize), Vec<Vec<f32>>> = HashMap::new();
    let mut tissue_slice_for = |row: usize, z: usize| -> Vec<Vec<f32>> {
        let r = &p.rows[row];
        let s = suppression[row].as_ref().expect("suppressed rows only");
        let t_read = r.t + p.slice_offsets[z];
        slice_cache
            .entry((r.tr.to_bits(), t_read.to_bits(), pulse_set[row], z))
            .or_insert_with(|| {
                masks
                    .iter()
                    .map(|m| {
                        r_sim.mean_slice(z, |i| if m[i] { tissue_mz(ph.m0[i] as f64, ph.t1[i] as f64, r.tr, t_read, s) } else { 0.0 })
                    })
                    .collect()
            })
            .clone()
    };

    // ---- per-row blood images with per-slice timing ----
    let blood_for = |row: &Row, r: &Resampler, sign: f64, want_gt: bool| -> (Vec<Vec<f32>>, Vec<f32>) {
        let [dnx, dny, dnz] = r.dst_dims;
        let mut comps = vec![vec![0.0f32; dnx * dny * dnz]; k];
        let mut gt = if want_gt { vec![0.0f32; dnx * dny * dnz] } else { Vec::new() };
        if sign == 0.0 && !want_gt {
            return (comps, gt);
        }
        let kin = p.kinetic(row);
        for z in 0..dnz {
            let t = row.t + p.slice_offsets[z];
            // delta_m per phantom voxel of this slice's slab, computed on demand
            let dm = |i: usize| -> f64 {
                if ph.dseg[i] > 0 {
                    delta_m(&kin, ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64, t)
                } else {
                    0.0
                }
            };
            for (c, m) in masks.iter().enumerate() {
                if sign != 0.0 {
                    let sl = r.mean_slice(z, |i| if m[i] { blood_signal(sign * dm(i)) } else { 0.0 });
                    comps[c][z * dnx * dny..(z + 1) * dnx * dny].copy_from_slice(&sl);
                }
            }
            if want_gt {
                let sl = r.mean_slice(z, dm);
                gt[z * dnx * dny..(z + 1) * dnx * dny].copy_from_slice(&sl);
            }
        }
        (comps, gt)
    };

    // ---- assemble the 4D compartments, voxel-major interleaved ----
    let motion_on = p.motion.is_some();
    let mut images: Vec<Vec<f32>> = vec![vec![0.0f32; nvox_sim * n]; ncomp];
    let mut gt_static = vec![0.0f32; nvox_acq * n];
    let mut gt_sim = if motion_on { vec![0.0f32; nvox_sim * n] } else { Vec::new() };
    let slab = snx * sny;
    for (v, row) in p.rows.iter().enumerate() {
        let se = row.kind == RowKind::M0scan;
        let tissue = match (row.kind, &suppression[v]) {
            (RowKind::Deltam, _) => None,
            (_, Some(_)) => {
                let mut comps = vec![vec![0.0f32; nvox_sim]; k];
                for z in 0..nz {
                    let sl = tissue_slice_for(v, z);
                    for c in 0..k {
                        comps[c][z * slab..(z + 1) * slab].copy_from_slice(&sl[c]);
                    }
                }
                Some(comps)
            }
            (_, None) => Some(tissue_for(row.tr, se)),
        };
        let factor = label_factors.as_ref().map_or(1.0, |f| f[v]);
        let sign = blood_sign(row.kind, ov) * factor;
        let wants_gt = matches!(row.kind, RowKind::Label | RowKind::Deltam);
        let (blood, gt_s) = blood_for(row, &r_sim, sign, wants_gt && motion_on);
        if wants_gt {
            let (_, gt) = blood_for(row, &r_acq, 0.0, true);
            for vox in 0..nvox_acq {
                gt_static[vox * n + v] = gt[vox];
            }
            if motion_on {
                for vox in 0..nvox_sim {
                    gt_sim[vox * n + v] = gt_s[vox];
                }
            }
        }
        for c in 0..k {
            if let Some(t) = &tissue {
                for vox in 0..nvox_sim {
                    images[c][vox * n + v] = t[c][vox];
                }
            }
            let target = if ov == RowOverride::BloodIntoTissue0 && row.kind == RowKind::Label { 0 } else { k + c };
            for vox in 0..nvox_sim {
                images[target][vox * n + v] += blood[c][vox];
            }
        }
    }

    // ---- motion: per-volume poses, then multiband shot events ----
    let motion_seed = p.motion.as_ref().map(|_| p.seed ^ MOTION_SEED_SALT);
    let mut poses = vec![Pose::IDENTITY; n];
    let mut events = Vec::new();
    let mut dropped = Vec::new();
    let mut gt_moved: Option<Vec<f32>> = None;
    if let (Some(m), Some(seed)) = (&p.motion, motion_seed) {
        let v2w = sim_grid.voxel_to_world;
        poses = resolve_poses(&m.mode, n, seed);
        apply_motion(&mut images, sim_grid.dims, n, v2w, &poses);
        let mut gt_arr = [std::mem::take(&mut gt_sim)];
        apply_motion(&mut gt_arr, sim_grid.dims, n, v2w, &poses);
        let [moved] = gt_arr;
        gt_moved = Some(block_mean_inplane(&moved, sim_grid.dims, o, n));
        let n_shots = slice_schedule(nz, p.mb, p.mb_interleaved).len();
        events = draw_events(m.within.as_ref(), n, n_shots, seed);
        if !events.is_empty() {
            dropped = apply_multiband_motion(
                &mut images, sim_grid.dims, n, v2w, p.mb, p.mb_interleaved, &DropoutLaw::Uniform, &events,
            );
        }
    }

    // ---- the one call ----
    let eddy_drive = vec![None; n];
    let prep_drive = vec![None; n];
    let (mag, phase_out) = simulate_acquisition_oversampled(
        sim_grid.dims, acq_grid.dims, n, &images, &t2_vols, &fmap_sim, Some(&ti_vols), &acq,
        &eddy_drive, &prep_drive, phase, p.seed, None, None,
    );
    drop(images);

    // ---- the separate M0 scan: a plain spin-echo readout at its own TR ----
    let m0_seed = (p.m0_type == M0Type::Separate).then_some(p.seed ^ M0_SEED_SALT);
    let m0 = match m0_seed {
        Some(seed) => {
            let tr = p.m0_repetition_time_s.ok_or("M0Type Separate without an M0 repetition time")?;
            let tissue = tissue_for(tr, true);
            let mut imgs: Vec<Vec<f32>> = vec![vec![0.0f32; nvox_sim]; ncomp];
            for c in 0..k {
                imgs[c].copy_from_slice(&tissue[c]);
            }
            Some(simulate_acquisition_oversampled(
                sim_grid.dims, acq_grid.dims, 1, &imgs, &t2_vols, &fmap_sim, Some(&ti_vols), &acq,
                &[None], &[None], phase, seed, None, None,
            ))
        }
        None => None,
    };

    // ---- ground truth on the acquisition grid ----
    let perfused: Vec<bool> = ph.perfusion.iter().map(|f| *f > 0.0).collect();
    let (delta_m_gt, delta_m_static) = match gt_moved {
        Some(moved) => (moved, Some(gt_static)),
        None => (gt_static, None),
    };
    let ground_truth = GroundTruth {
        delta_m: delta_m_gt,
        delta_m_static,
        perfusion: r_acq.mean(&ph.perfusion),
        att: r_acq.masked_mean(&ph.att, &perfused),
        t1: r_acq.mean(&ph.t1),
        t2: r_acq.mean(&ph.t2),
        m0: r_acq.mean(&ph.m0),
        dseg: r_acq.majority(&ph.dseg),
        acq_t2_ms,
        acq_t2p_ms,
    };

    Ok(SeriesOutput {
        acq_grid, sim_grid, n_volumes: n, mag, phase: phase_out, m0, mode: mode_used,
        labels: ph.labels.clone(), n_compartments: ncomp, fieldmap_present: ph.fieldmap.is_some(),
        seeds: (p.seed, m0_seed), acquisition: acq, ground_truth, label_factors, poses, motion_seed,
        events, dropped,
    })
}

/// Reconstruct the complex image `(re, im)` from written magnitude and phase.
pub fn complex_from(mag: &[f32], phase: &[f32]) -> Vec<(f64, f64)> {
    mag.iter().zip(phase).map(|(&m, &p)| ((m as f64) * (p as f64).cos(), (m as f64) * (p as f64).sin())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phantom::load;
    use crate::protocol::{parse, Overlay};
    use serde_json::{json, Value};

    fn phantom() -> Phantom {
        load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap()
    }

    /// A PCASL sidecar on the crop: `voxel` mm, slice timing for the resulting slice count.
    fn sidecar(voxel: [f64; 3], nz: usize) -> Value {
        let timing: Vec<f64> = (0..nz).map(|z| 0.04 * z as f64).collect();
        json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": voxel,
            "MRAcquisitionType": "2D", "SliceTiming": timing, "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.012
        })
    }

    fn protocol_from(s: &Value, rows: &str, overlay: &str) -> Protocol {
        let ov: Overlay = toml::from_str(overlay).unwrap();
        let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
        parse(s, &ctx, Some(&ov), phantom().params.as_ref()).unwrap()
    }

    fn protocol(voxel: [f64; 3], nz: usize, rows: &str, overlay: &str) -> Protocol {
        protocol_from(&sidecar(voxel, nz), rows, overlay)
    }

    fn no_phase() -> PhaseModel {
        PhaseModel { global: 0.0, background: Default::default(), prep: None }
    }

    fn max_abs(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    #[test]
    fn series_rejects_a_field_mismatch() {
        let ph = phantom();
        let mut p = protocol([3.0, 3.0, 3.0], 2, "control,label", "");
        p.field_strength = 1.5;
        assert!(simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap_err().contains("MagneticFieldStrength"));
    }

    #[test]
    fn class_and_voxel_agree_on_a_homogeneous_grid() {
        // Acquisition grid == phantom grid, o = 1: every simulation voxel is one label, so the
        // weighted-rate collapse is exact and the two modes must agree.
        let ph = phantom();
        let p = protocol([1.0, 1.0, 1.0], 6, "control,label", "[acquisition]\noversample = 1\n");
        let a = simulate(&p, &ph, T2Mode::Class, &no_phase()).unwrap();
        let b = simulate(&p, &ph, T2Mode::Voxel, &no_phase()).unwrap();
        assert_eq!(a.mode, T2Mode::Class);
        assert_eq!(b.mode, T2Mode::Voxel);
        assert_eq!(a.n_compartments, 6);
        assert_eq!(b.n_compartments, 2);
        // Per-voxel relative agreement, with an absolute floor of 1e-6 of the peak for the
        // near-zero voxels (ringing outside the object) where a relative bound is meaningless.
        let scale = max_abs(&a.mag) as f64;
        assert!(scale > 0.0);
        let mut worst = 0.0f64;
        for (x, y) in a.mag.iter().zip(&b.mag) {
            let (x, y) = (*x as f64, *y as f64);
            worst = worst.max((x - y).abs() / (1e-5 * x.abs().max(y.abs()) + 1e-6 * scale));
        }
        assert!(worst <= 1.0, "class vs voxel on a homogeneous grid: worst residual {worst:.2}x tolerance");
        assert!(a.label_factors.is_none() && a.motion_seed.is_none() && a.ground_truth.delta_m_static.is_none());
        assert!(a.poses.iter().all(|q| *q == Pose::IDENTITY));
    }

    #[test]
    fn class_vs_voxel_boundary_discrepancy_is_tracked() {
        let ph = phantom();
        let p = protocol([3.0, 3.0, 3.0], 2, "control,label", "[acquisition]\noversample = 2\n");
        let a = simulate(&p, &ph, T2Mode::Class, &no_phase()).unwrap();
        let b = simulate(&p, &ph, T2Mode::Voxel, &no_phase()).unwrap();
        let scale = max_abs(&a.mag) as f64;
        let worst = a.mag.iter().zip(&b.mag).fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs() / scale));
        println!("class vs voxel boundary discrepancy (max, relative to peak): {worst:.3e}");
        assert!(worst < 0.05, "voxel-mode approximation drifted: {worst:e}");
        assert!(a.ground_truth.acq_t2_ms.is_none() && b.ground_truth.acq_t2_ms.is_some());
    }

    #[test]
    fn label_rows_carry_negative_delta_m() {
        let ph = phantom();
        let p = protocol([3.0, 3.0, 3.0], 2, "control,label", "[acquisition]\nsignal_scale = 1.0\n");
        let out = simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let nvox = out.acq_grid.dims.iter().product::<usize>();
        let mut diff = 0.0f64;
        let mut n_gm = 0;
        for vox in 0..nvox {
            if out.ground_truth.dseg[vox] == 1 {
                diff += out.mag[vox * 2] as f64 - out.mag[vox * 2 + 1] as f64;
                n_gm += 1;
            }
        }
        assert!(n_gm > 0 && diff > 0.0, "control minus label must be positive in GM: {diff} over {n_gm} voxels");
        // and the ground-truth delta_m is written for the label row only
        assert!(out.ground_truth.delta_m.iter().step_by(2).all(|v| *v == 0.0));
        assert!(out.ground_truth.delta_m.iter().skip(1).step_by(2).any(|v| *v > 0.0));
    }

    #[test]
    fn noise_is_uncorrelated_across_volumes_and_with_the_separate_m0() {
        let ph = phantom();
        let mk = |noise: f64| {
            protocol([3.0, 3.0, 3.0], 2, "control,control,control",
                     &format!("[acquisition]\nnoise_variance = {noise}\nsignal_scale = 1.0\n[m0]\nrepetition_time = 8.0\n"))
        };
        let mut clean = mk(0.0);
        clean.m0_type = M0Type::Separate;
        let mut noisy = mk(4.0);
        noisy.m0_type = M0Type::Separate;
        let a = simulate(&clean, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let b = simulate(&noisy, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let nvox = a.acq_grid.dims.iter().product::<usize>();
        let n = a.n_volumes;
        // noise realisations = noisy - clean, per volume and for the M0 scan (real parts)
        let re = |m: &[f32], p: &[f32]| complex_from(m, p).into_iter().map(|c| c.0).collect::<Vec<f64>>();
        let (ca, cb) = (re(&a.mag, &a.phase), re(&b.mag, &b.phase));
        let mut noise: Vec<Vec<f64>> = (0..n).map(|v| (0..nvox).map(|vox| cb[vox * n + v] - ca[vox * n + v]).collect()).collect();
        let (m0a, m0b) = (a.m0.as_ref().unwrap(), b.m0.as_ref().unwrap());
        let (ma, mb) = (re(&m0a.0, &m0a.1), re(&m0b.0, &m0b.1));
        noise.push((0..nvox).map(|vox| mb[vox] - ma[vox]).collect());
        let corr = |x: &[f64], y: &[f64]| {
            let (mx, my) = (x.iter().sum::<f64>() / x.len() as f64, y.iter().sum::<f64>() / y.len() as f64);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for (a, b) in x.iter().zip(y) {
                sxy += (a - mx) * (b - my);
                sxx += (a - mx).powi(2);
                syy += (b - my).powi(2);
            }
            sxy / (sxx * syy).sqrt()
        };
        for v in 0..noise.len() {
            let var = noise[v].iter().map(|x| x * x).sum::<f64>() / nvox as f64;
            assert!(var > 0.0, "volume {v} has no noise");
            for w in v + 1..noise.len() {
                let c = corr(&noise[v], &noise[w]);
                assert!(c.abs() < 0.25, "noise of volumes {v} and {w} correlate at {c:.3}");
            }
        }
        assert_eq!(b.seeds, (0, Some(M0_SEED_SALT)));
    }

    // ------------------------------------------------------------------ P3

    /// base timing: tau 1.8, PLD 1.8 (t = 3.6), TR 4.0.
    fn suppressed(pulses: &[f64]) -> Value {
        let mut s = sidecar([1.0, 1.0, 1.0], 6);
        s["BackgroundSuppression"] = json!(true);
        s["BackgroundSuppressionNumberPulses"] = json!(pulses.len());
        s["BackgroundSuppressionPulseTime"] = json!(pulses);
        s
    }

    /// Voxels of `label` whose in-plane 3x3 neighbourhood is all that label, on the
    /// acquisition grid, as (vox, z) pairs.
    fn interior(out: &SeriesOutput, label: i32) -> Vec<(usize, usize)> {
        let [nx, ny, nz] = out.acq_grid.dims;
        let d = &out.ground_truth.dseg;
        let mut v = Vec::new();
        for z in 0..nz {
            for y in 1..ny - 1 {
                for x in 1..nx - 1 {
                    let all = (-1i32..=1).all(|dy| (-1i32..=1).all(|dx| {
                        d[(x as i32 + dx) as usize + nx * ((y as i32 + dy) as usize + ny * z)] == label
                    }));
                    if all {
                        v.push((x + nx * (y + ny * z), z));
                    }
                }
            }
        }
        v
    }

    /// Homogeneous grid (acq == phantom, o = 1), no distortion, no noise: the acquired GM
    /// interior should follow tissue_mz / tissue_se at each slice's readout to a few percent
    /// (the readout's per-line relaxation blurs a little, and the crop's GM is thin).
    /// `multiband`: 3 shots of 2 slices, so slices z and z + 3 share a readout time but must
    /// keep their own anatomy (the per-slice cache is keyed on the slice as well).
    fn check_suppression_ratio(multiband: bool) {
        let ph = phantom();
        let ov = "[acquisition]\noversample = 1\n[background_suppression]\ninversion_efficiency = 1.0\n";
        let (mut s_on, mut s_off) = (suppressed(&[2.0, 3.2]), sidecar([1.0, 1.0, 1.0], 6));
        if multiband {
            for s in [&mut s_on, &mut s_off] {
                s["MultibandAccelerationFactor"] = json!(2);
                s["SliceTiming"] = json!([0.0, 0.04, 0.08, 0.0, 0.04, 0.08]);
            }
        }
        let on = protocol_from(&s_on, "control", ov);
        let off = protocol_from(&s_off, "control", "[acquisition]\noversample = 1\n");
        let a = simulate(&on, &ph, T2Mode::Class, &no_phase()).unwrap();
        let b = simulate(&off, &ph, T2Mode::Class, &no_phase()).unwrap();
        assert_eq!(a.label_factors.as_deref(), Some(&[1.0][..]));
        let s = on.suppression.as_ref().unwrap().for_row(0);
        let gm = interior(&a, 1);
        assert!(gm.len() >= 4, "too few interior GM voxels: {}", gm.len());
        let mut slices_seen = std::collections::HashSet::new();
        let mut worst = 0.0f64;
        for (vox, z) in gm {
            slices_seen.insert(z);
            let t_read = 3.6 + on.slice_offsets[z];
            let want = tissue_mz(1.0, 1.33, 4.0, t_read, &s) / tissue_se(1.0, 1.33, 4.0);
            let got = a.mag[vox] as f64 / b.mag[vox] as f64;
            worst = worst.max((got - want).abs() / want.abs());
        }
        println!("suppressed / unsuppressed GM ratio (multiband {multiband}): worst relative deviation {worst:.3e} over slices {slices_seen:?}");
        assert!(worst < 0.05, "{worst}");
        if multiband {
            assert!(slices_seen.iter().any(|z| *z >= 3), "the check must reach a slice that shares its readout time");
        }
    }

    #[test]
    fn suppression_scales_the_tissue_by_the_timeline() {
        check_suppression_ratio(false);
    }

    #[test]
    fn suppression_keeps_each_multiband_slices_own_anatomy() {
        check_suppression_ratio(true);
    }

    #[test]
    fn an_odd_pulse_count_flips_control_minus_label() {
        let ph = phantom();
        let mut s = suppressed(&[2.0]);
        s["AcquisitionVoxelSize"] = json!([3.0, 3.0, 3.0]);
        s["SliceTiming"] = json!([0.0, 0.04]);
        let p = protocol_from(&s, "control,label", "[acquisition]\nsignal_scale = 1.0\n[background_suppression]\ninversion_efficiency = 1.0\n");
        let out = simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap();
        assert_eq!(out.label_factors.as_deref(), Some(&[-1.0, -1.0][..]));
        let z = complex_from(&out.mag, &out.phase);
        let nvox = out.acq_grid.dims.iter().product::<usize>();
        let diff: f64 = (0..nvox).filter(|v| out.ground_truth.dseg[*v] == 1).map(|v| z[v * 2].0 - z[v * 2 + 1].0).sum();
        assert!(diff < 0.0, "control minus label must be negative in GM under one perfect pulse: {diff}");
        // the ground truth keeps the unsuppressed, positive delta_m
        assert!(out.ground_truth.delta_m.iter().skip(1).step_by(2).any(|v| *v > 0.0));
    }

    #[test]
    fn ir_at_90_degrees_without_inversion_is_the_spin_echo_run() {
        let ph = phantom();
        let se = protocol([3.0, 3.0, 3.0], 2, "control,label", "[acquisition]\nsignal_scale = 1.0\n");
        let ir = protocol([3.0, 3.0, 3.0], 2, "control,label",
                          "[acquisition]\nsignal_scale = 1.0\n[signal]\nacq_contrast = \"ir\"\ninversion_flip_angle = 0.0\n");
        let a = simulate(&se, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let b = simulate(&ir, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let scale = max_abs(&a.mag) as f64;
        let worst = a.mag.iter().zip(&b.mag).fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs() / scale));
        assert!(worst < 1e-6, "IR(fa 90, fa_inv 0) must equal spin echo: {worst:e}");
        // a 30-degree excitation scales the blood by sin(30) = 1/2 exactly: the deltam row has
        // no tissue, so its image is half the spin-echo deltam image
        let se_d = protocol([3.0, 3.0, 3.0], 2, "deltam", "[acquisition]\nsignal_scale = 1.0\n");
        let ir30_d = protocol([3.0, 3.0, 3.0], 2, "deltam",
                              "[acquisition]\nsignal_scale = 1.0\n[signal]\nacq_contrast = \"ir\"\nexcitation_flip_angle = 30.0\ninversion_flip_angle = 180.0\n");
        let d = simulate(&se_d, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let e = simulate(&ir30_d, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let dscale = max_abs(&d.mag) as f64;
        let worst = d.mag.iter().zip(&e.mag).fold(0.0f64, |m, (x, y)| m.max((0.5 * *x as f64 - *y as f64).abs() / dscale));
        assert!(worst < 1e-6, "deltam under fa 30 must be half the spin-echo deltam: {worst:e}");
    }

    #[test]
    fn ir_with_a_real_inversion_follows_the_closed_form_in_gm() {
        // TI 1 s, 180-degree inversion, 90-degree excitation on the homogeneous grid: the GM
        // interior control ratio IR / SE is tissue_ir / tissue_se (0.11 here, so a bypass to
        // sin(fa) * tissue_se, ratio 1, cannot pass).
        let ph = phantom();
        let se = protocol([1.0, 1.0, 1.0], 6, "control", "[acquisition]\noversample = 1\n");
        let ir = protocol([1.0, 1.0, 1.0], 6, "control",
                          "[acquisition]\noversample = 1\n[signal]\nacq_contrast = \"ir\"\ninversion_time = 1.0\n");
        let a = simulate(&se, &ph, T2Mode::Class, &no_phase()).unwrap();
        let b = simulate(&ir, &ph, T2Mode::Class, &no_phase()).unwrap();
        let q = ir.ir.as_ref().unwrap().params;
        let want = tissue_ir(1.0, 1.33, 4.0, &q) / tissue_se(1.0, 1.33, 4.0);
        assert!(want > 0.05 && want < 0.2, "closed form {want}");
        let gm = interior(&a, 1);
        assert!(gm.len() >= 4);
        let mut worst = 0.0f64;
        for (vox, _) in gm {
            let got = b.mag[vox] as f64 / a.mag[vox] as f64;
            worst = worst.max((got - want).abs() / want);
        }
        println!("IR / SE GM ratio: worst relative deviation {worst:.3e} from the closed form {want:.4}");
        assert!(worst < 0.05, "{worst}");
    }

    fn write_trajectory(rows: &[[f64; 6]]) -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir().join(format!("aslscan-series-{}-{}", std::process::id(), rows.len()));
        std::fs::create_dir_all(&dir).unwrap();
        let tsv = dir.join("motion.tsv");
        let mut text = String::from("trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n");
        for r in rows {
            text.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\n", r[0], r[1], r[2], r[3], r[4], r[5]));
        }
        std::fs::write(&tsv, text).unwrap();
        let path = tsv.to_string_lossy().replace('\\', "\\\\");
        (dir, path)
    }

    #[test]
    fn a_trajectory_translation_shifts_image_and_moved_ground_truth() {
        // acq == phantom grid at 1 mm, o = 1, no distortion or noise: +2 mm in x is +2 voxels.
        let ph = phantom();
        let (dir, path) = write_trajectory(&[[0.0; 6], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]);
        let p = protocol([1.0, 1.0, 1.0], 6, "label,label",
                         &format!("[acquisition]\noversample = 1\n[motion]\nmode = \"trajectory\"\ntrajectory = \"{path}\"\n"));
        let out = simulate(&p, &ph, T2Mode::Class, &no_phase()).unwrap();
        let [nx, ny, nz] = out.acq_grid.dims;
        assert_eq!(out.poses[1].trans_mm, [2.0, 0.0, 0.0]);
        assert_eq!(out.motion_seed, Some(MOTION_SEED_SALT));
        let (gt, gts) = (&out.ground_truth.delta_m, out.ground_truth.delta_m_static.as_ref().unwrap());
        // which way is +x in voxels? the grid's x axis may be flipped; find the sign from the GT
        let at = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
        let shift: i64 = if out.sim_grid.voxel_to_world[0][0] > 0.0 { 2 } else { -2 };
        let peak = max_abs(&out.mag) as f64;
        let (mut worst_img, mut n_checked) = (0.0f64, 0);
        for z in 0..nz {
            for y in 0..ny {
                for x in 3..nx - 3 {
                    let src = (x as i64 - shift) as usize;
                    // moved truth is exactly the static truth shifted (trilinear weights 0/1)
                    assert_eq!(gt[at(x, y, z) * 2 + 1], gts[at(src, y, z) * 2 + 1], "gt at {x},{y},{z}");
                    assert_eq!(gt[at(x, y, z) * 2], gts[at(x, y, z) * 2], "volume 0 is unmoved");
                    let d = (out.mag[at(x, y, z) * 2 + 1] as f64 - out.mag[at(src, y, z) * 2] as f64).abs() / peak;
                    worst_img = worst_img.max(d);
                    n_checked += 1;
                }
            }
        }
        println!("acquired magnitude under a +2 voxel shift: worst deviation {worst_img:.3e} of peak over {n_checked} voxels");
        assert!(worst_img < 1e-5, "{worst_img}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn random_motion_is_reproducible_and_leaves_the_noise_alone() {
        let ph = phantom();
        let mk = |motion: &str| {
            protocol([3.0, 3.0, 3.0], 2, "control,control",
                     &format!("seed = 3\n[acquisition]\nnoise_variance = 4.0\nsignal_scale = 1.0\n{motion}"))
        };
        let still = mk("");
        let moved = mk("[motion]\nmode = \"random\"\ntrans_mm = [2.0, 2.0, 0.0]\nrot_deg = [0.0, 0.0, 5.0]\nvolumes = [1]\n");
        let a = simulate(&still, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let b = simulate(&moved, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let c = simulate(&moved, &ph, T2Mode::Auto, &no_phase()).unwrap();
        assert_eq!(b.mag, c.mag, "same seed, same motion, same data");
        assert_eq!(b.poses, c.poses);
        assert_eq!(b.poses[0], Pose::IDENTITY);
        assert_ne!(b.poses[1], Pose::IDENTITY);
        assert_eq!(b.motion_seed, Some(3 ^ MOTION_SEED_SALT));
        // volume 0 did not move, and its noise stream is keyed on the volume index, so it is
        // bit-identical between the still and the moved run; volume 1 differs.
        let nvox = a.acq_grid.dims.iter().product::<usize>();
        for vox in 0..nvox {
            assert_eq!(a.mag[vox * 2], b.mag[vox * 2], "volume 0 changed at {vox}");
        }
        assert!((0..nvox).any(|vox| a.mag[vox * 2 + 1] != b.mag[vox * 2 + 1]));
        assert!(b.ground_truth.delta_m_static.is_some());
    }

    #[test]
    fn within_volume_events_attenuate_every_shot_when_certain() {
        // 6 slices at 1 mm, mb 2 (3 shots), sequential timing; dropout_rate 1 with severity 0.5
        // and no jumps halves every shot's signal: the whole image halves.
        let ph = phantom();
        let mut s = sidecar([1.0, 1.0, 1.0], 6);
        s["MultibandAccelerationFactor"] = json!(2);
        s["SliceTiming"] = json!([0.0, 0.04, 0.08, 0.0, 0.04, 0.08]);
        let base = protocol_from(&s, "control,label", "[acquisition]\noversample = 1\n");
        let ev = protocol_from(&s, "control,label",
                               "[acquisition]\noversample = 1\n[motion.within_volume]\ndropout_rate = 1.0\nseverity = 0.5\n");
        let a = simulate(&base, &ph, T2Mode::Class, &no_phase()).unwrap();
        let b = simulate(&ev, &ph, T2Mode::Class, &no_phase()).unwrap();
        assert_eq!(b.events.len(), 6);
        assert_eq!(b.dropped.len(), 6);
        assert!(b.dropped.iter().all(|d| (d.attenuation - 0.5).abs() < 1e-6 && d.slices.len() == 2));
        assert!(b.poses.iter().all(|q| *q == Pose::IDENTITY));
        let scale = max_abs(&a.mag) as f64;
        let worst = a.mag.iter().zip(&b.mag).fold(0.0f64, |m, (x, y)| m.max((0.5 * *x as f64 - *y as f64).abs() / scale));
        assert!(worst < 1e-6, "{worst}");
        // the moved ground truth excludes the events: identical to the static one
        assert_eq!(b.ground_truth.delta_m, *b.ground_truth.delta_m_static.as_ref().unwrap());
    }
}
