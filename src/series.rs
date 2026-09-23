//! The orchestration: `aslcontext.tsv` rows -> compartment volumes on the simulation grid -> ONE
//! `simulate_acquisition_oversampled` call for the whole series (plus one for a separate M0
//! scan), and the ground truth on the acquisition grid.
//!
//! Row semantics (spec): tissue compartments `0..K` carry the spin-echo steady state at the row's
//! repetition time for `control`, `label` and `m0scan` rows and zero for `deltam`; blood
//! compartments `K..2K` carry `-delta_m` for `label`, `+delta_m` for `deltam`, zero otherwise.
//! `K` is the number of foreground labels in `class` mode and 1 in `voxel` mode. Everything is
//! evaluated per phantom voxel and only magnetization is averaged onto the simulation grid.
//!
//! One call per series is a hard rule (spec P0 change 5a): every random stream in the
//! acquisition stage is keyed on the volume index within a call, so calling once per volume
//! would give every volume the same noise and make control minus label noise-free.

use std::collections::HashMap;

use mrsim_acq::grid::Grid;
use mrsim_acq::io::hires_grid;
use mrsim_acq::kspace::{simulate_acquisition_oversampled, Acquisition, T2Volume};
use mrsim_acq::phase::PhaseModel;

use crate::kinetic::delta_m;
use crate::mrsignal::{blood_se, tissue_se};
use crate::phantom::{Phantom, Relaxation, T2Mode};
use crate::protocol::{M0Type, Protocol, Row, RowKind};
use crate::resample::{acquisition_grid, axis_aligned_voxels, Resampler};

/// The seed the separate M0 scan's call uses, derived from the series seed so the two calls
/// draw different noise (they would otherwise both be "volume 0").
pub const M0_SEED_SALT: u64 = 0x4D30_5343_414E;

/// Ground-truth maps on the acquisition grid (`resample` rules per map, see the spec).
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// `+delta_m` for `label` and `deltam` rows at their own timing, zero for other rows.
    /// Voxel-major interleaved like the data: `vox * n_volumes + v`.
    pub delta_m: Vec<f32>,
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
    /// The blood input lands in tissue compartment 0 instead of its own compartment.
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
    simulate_with(p, ph, mode, phase, RowOverride::None)
}

/// [`simulate`] with a [`RowOverride`]; the override exists for the linearity test's negative
/// controls and must be `None` in every real run.
pub fn simulate_with(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride)
    -> Result<SeriesOutput, String>
{
    if p.background_suppression {
        return Err("asl.json: BackgroundSuppression is true; P1 does not model background suppression \
                    (it arrives with P3), and simulating without it would write a sidecar the data \
                    contradicts".to_string());
    }
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
    let t2_blood_ms = (p.t2_blood_s.0 * 1000.0) as f32;
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
            for i in 0..k {
                t2v.push(T2Volume::Uniform(t2_blood_ms));
                tiv.push(T2Volume::Uniform(t2p_ms[i]));
            }
            (t2v, tiv)
        }
        Relaxation::Voxel { .. } => {
            let t2m = acq_t2_ms.as_deref().unwrap();
            let tpm = acq_t2p_ms.as_deref().unwrap();
            (vec![T2Volume::Map(t2m), T2Volume::Uniform(t2_blood_ms)], vec![T2Volume::Map(tpm), T2Volume::Map(tpm)])
        }
    };

    // ---- tissue signal per distinct repetition time, per compartment, on the sim grid ----
    let mut tissue_cache: HashMap<u64, Vec<Vec<f32>>> = HashMap::new();
    let mut tissue_for = |tr: f64| -> Vec<Vec<f32>> {
        tissue_cache
            .entry(tr.to_bits())
            .or_insert_with(|| {
                let sig: Vec<f64> = (0..ph.nvox()).map(|i| tissue_se(ph.m0[i] as f64, ph.t1[i] as f64, tr)).collect();
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

    // ---- per-row blood images with per-slice timing ----
    let n = p.rows.len();
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
                    let sl = r.mean_slice(z, |i| if m[i] { blood_se(sign * dm(i)) } else { 0.0 });
                    comps[c][z * dnx * dny..(z + 1) * dnx * dny].copy_from_slice(&sl);
                }
            }
            if want_gt {
                let sl = r.mean_slice(z, |i| dm(i));
                gt[z * dnx * dny..(z + 1) * dnx * dny].copy_from_slice(&sl);
            }
        }
        (comps, gt)
    };

    // ---- assemble the 4D compartments, voxel-major interleaved ----
    let mut images: Vec<Vec<f32>> = vec![vec![0.0f32; nvox_sim * n]; ncomp];
    let mut gt_delta_m = vec![0.0f32; nvox_acq * n];
    for (v, row) in p.rows.iter().enumerate() {
        let tissue = if row.kind == RowKind::Deltam { None } else { Some(tissue_for(row.tr)) };
        let sign = blood_sign(row.kind, ov);
        let (blood, _) = blood_for(row, &r_sim, sign, false);
        let wants_gt = matches!(row.kind, RowKind::Label | RowKind::Deltam);
        if wants_gt {
            let (_, gt) = blood_for(row, &r_acq, 0.0, true);
            for vox in 0..nvox_acq {
                gt_delta_m[vox * n + v] = gt[vox];
            }
        }
        for c in 0..k {
            if let Some(t) = &tissue {
                for vox in 0..nvox_sim {
                    images[c][vox * n + v] = t[c][vox];
                }
            }
            let target = if ov == RowOverride::BloodIntoTissue0 { 0 } else { k + c };
            for vox in 0..nvox_sim {
                images[target][vox * n + v] += blood[c][vox];
            }
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

    // ---- the separate M0 scan ----
    let m0_seed = (p.m0_type == M0Type::Separate).then(|| p.seed ^ M0_SEED_SALT);
    let m0 = match m0_seed {
        Some(seed) => {
            let tr = p.m0_repetition_time_s.ok_or("M0Type Separate without an M0 repetition time")?;
            let tissue = tissue_for(tr);
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
    let ground_truth = GroundTruth {
        delta_m: gt_delta_m,
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
        seeds: (p.seed, m0_seed), acquisition: acq, ground_truth,
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
    use serde_json::json;

    fn phantom() -> Phantom {
        load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap()
    }

    /// A PCASL protocol on the crop: `voxel` mm, slice timing for the resulting slice count.
    fn protocol(voxel: [f64; 3], nz: usize, rows: &str, overlay: &str) -> Protocol {
        let timing: Vec<f64> = (0..nz).map(|z| 0.04 * z as f64).collect();
        let s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": voxel,
            "MRAcquisitionType": "2D", "SliceTiming": timing, "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.012
        });
        let ov: Overlay = toml::from_str(overlay).unwrap();
        let ctx = format!("volume_type\n{}\n", rows.split(',').collect::<Vec<_>>().join("\n"));
        parse(&s, &ctx, Some(&ov), phantom().params.as_ref()).unwrap()
    }

    fn no_phase() -> PhaseModel {
        PhaseModel { global: 0.0, background: Default::default(), prep: None }
    }

    fn max_abs(v: &[f32]) -> f32 {
        v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
    }

    #[test]
    fn series_rejects_background_suppression_and_field_mismatch() {
        let ph = phantom();
        let mut p = protocol([3.0, 3.0, 3.0], 2, "control,label", "");
        p.background_suppression = true;
        assert!(simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap_err().contains("P3"));
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
        let scale = max_abs(&a.mag) as f64;
        assert!(scale > 0.0);
        let mut worst = 0.0f64;
        for (x, y) in a.mag.iter().zip(&b.mag) {
            worst = worst.max((*x as f64 - *y as f64).abs() / scale);
        }
        assert!(worst < 1e-5, "class vs voxel on a homogeneous grid: max rel {worst:e}");
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
}
