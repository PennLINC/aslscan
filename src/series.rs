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

use crate::bolus::{arterial_factor, entry_offset, subbolus_factors, Region};
use crate::crushing::survival;
use crate::kinetic::{arterial_dm, delta_m, delta_m_iv, delta_m_iv_sub, delta_m_sub, LabelType};
use crate::physio::Physio;
use crate::longitudinal::{label_factor, tissue_mz};
use crate::mrsignal::{blood_ir, blood_se, tissue_ir, tissue_se, Contrast};
use crate::phantom::{Phantom, Relaxation, T2Mode};
use crate::protocol::{M0Type, Protocol, QuantitySource, Row, RowKind, SuppressionModel, WithinVolume};
use crate::resample::{acquisition_grid, axis_aligned_voxels, corner_offset, Resampler};
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
    /// P4, part A: the intravascular part of `delta_m` (kinetics and the exchange split), like
    /// `delta_m` per row and frame (moved under motion).
    pub delta_m_iv: Option<Vec<f32>>,
    /// P4, part D: `delta_m` after the bolus-position pulse factors (both parts).
    pub delta_m_suppressed: Option<Vec<f32>>,
    /// P4, part B: the arterial `delta_m` with `g = 1` (before factors and crushing).
    pub delta_m_arterial: Option<Vec<f32>>,
    /// P4, part B: `aBV` (mean) and `aATT` (mean over `aBV > 0`), static.
    pub abv: Option<Vec<f32>>,
    pub aatt: Option<Vec<f32>>,
}

/// The physiological factors applied (P4, part E): one line per volume and slice.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysioLine {
    pub volume: usize,
    pub slice: usize,
    /// s on the series clock
    pub time: f64,
    pub cardiac_phase: f64,
    pub respiratory_phase: f64,
    pub drift: f64,
    pub tissue_factor: f64,
    /// The volume's labeling window (s) and the window averages of `sin phi_c`, `sin phi_r` and
    /// the drift it was formed from.
    pub label_window: (f64, f64),
    pub label_means: [f64; 3],
    pub label_factor: f64,
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
    /// Under `[compat] asldro = true`: what the SNR resolved to.
    pub compat: Option<CompatFacts>,
    /// P4, part C: the arterial survival per row and label (in `labels` order).
    pub crush_survival: Option<Vec<Vec<f64>>>,
    /// P4, part E.
    pub physio: Option<Vec<PhysioLine>>,
}

/// The compat noise resolution (P2 addendum, part A).
#[derive(Debug, Clone, PartialEq)]
pub struct CompatFacts {
    pub desired_snr: Option<f64>,
    /// Per-component image variance handed to the acquisition (0 without an SNR).
    pub noise_variance: f64,
    /// Mean |M0| over the nonzero voxels of the acquisition-grid M0 ground truth.
    pub m0_reference_mean: f64,
    pub m0_reference_voxels: usize,
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
    /// P4: the `label` row's extravascular part lands in its blood compartment instead of the
    /// tissue compartment (the two have different T2, so the identity must break).
    ExtravascularIntoBlood,
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
        RowOverride::None | RowOverride::BloodIntoTissue0 | RowOverride::ExtravascularIntoBlood => base,
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

/// Everything P4 resolves once per series against the phantom (addendum parts B, C, D, E).
struct P4 {
    /// Any P4 part that changes the label is on, so the label takes the P4 path.
    label_path: bool,
    /// Index into `ph.labels` per phantom voxel (`usize::MAX` on background).
    label_of: Vec<usize>,
    /// Per phantom voxel, part B.
    abv: Option<Vec<f64>>,
    aatt: Option<Vec<f64>>,
    /// Per row, per label (part C); all ones without crushing.
    crush: Option<Vec<Vec<f64>>>,
    physio: Option<Physio>,
    /// Per row: the label factor and the window it was averaged over (part E).
    label_physio: Vec<(f64, (f64, f64), [f64; 3])>,
}

impl P4 {
    fn new(p: &Protocol, ph: &Phantom, bolus_region: Option<Region>) -> Result<P4, String> {
        let n = p.rows.len();
        let label_path = p.exchange_time.is_some() || bolus_region.is_some() || p.physio.is_some() || p.macrovascular.is_some();
        let label_of: Vec<usize> =
            ph.dseg.iter().map(|d| ph.labels.iter().position(|(l, _)| l == d).unwrap_or(usize::MAX)).collect();

        // Name-keyed tables: every foreground label needs a value, no unknown names, and the
        // names must be unique (phantom::load allows duplicates, phantom.rs:146).
        let tables: Vec<(&str, &std::collections::BTreeMap<String, f64>)> = {
            let mut t = Vec::new();
            if let Some(m) = &p.macrovascular {
                if let QuantitySource::Table(x) = &m.abv {
                    t.push(("macrovascular.arterial_blood_volume", x));
                }
                if let QuantitySource::Table(x) = &m.aatt {
                    t.push(("macrovascular.arterial_transit_time", x));
                }
            }
            if let Some(v) = p.crushing.as_ref().and_then(|c| c.arterial_velocity.as_ref()) {
                t.push(("vascular_crushing.arterial_velocity", v));
            }
            t
        };
        if !tables.is_empty() {
            for (i, (li, ni)) in ph.labels.iter().enumerate() {
                if let Some((lj, _)) = ph.labels[i + 1..].iter().find(|(_, nj)| nj == ni) {
                    return Err(format!(
                        "phantom labels {li} and {lj} share the name {ni:?}, so a name-keyed overlay table cannot \
                         tell them apart"));
                }
            }
            for (what, t) in &tables {
                for key in t.keys() {
                    if !ph.labels.iter().any(|(_, nm)| nm == key) {
                        return Err(format!("overlay: {what} names {key:?}, which is not a phantom label"));
                    }
                }
                for (l, nm) in &ph.labels {
                    if !t.contains_key(nm) {
                        return Err(format!("overlay: {what} has no value for label {l} ({nm:?})"));
                    }
                }
            }
        }
        let per_label = |t: &std::collections::BTreeMap<String, f64>| -> Vec<f64> {
            ph.labels.iter().map(|(_, nm)| t[nm]).collect()
        };
        let resolve = |src: &QuantitySource, map: &Option<Vec<f32>>, what: &str| -> Result<Vec<f64>, String> {
            match src {
                QuantitySource::Map => {
                    let m = map.as_ref().ok_or_else(|| format!("the phantom has no {what} map"))?;
                    Ok(m.iter().map(|v| *v as f64).collect())
                }
                QuantitySource::Table(t) => {
                    let v = per_label(t);
                    Ok(label_of.iter().map(|&l| if l == usize::MAX { 0.0 } else { v[l] }).collect())
                }
            }
        };
        let (abv, aatt) = match &p.macrovascular {
            Some(m) => (Some(resolve(&m.abv, &ph.abv, "abv")?), Some(resolve(&m.aatt, &ph.aatt, "aatt")?)),
            None => (None, None),
        };

        // Part D's slab entry may not come after a voxel's arrival.
        if let Some(Region::Slab(d)) = bolus_region {
            for i in 0..ph.nvox() {
                if ph.dseg[i] > 0 && ph.perfusion[i] > 0.0 && d > ph.att[i] as f64 {
                    return Err(format!(
                        "background_suppression.slab_entry_time {d} s exceeds the ATT {} s of perfused voxel {i}: the \
                         label would enter the slab after reaching the voxel", ph.att[i]));
                }
                if let (Some(b), Some(a)) = (&abv, &aatt) {
                    if b[i] > 0.0 && d > a[i] {
                        return Err(format!(
                            "background_suppression.slab_entry_time {d} s exceeds the arterial transit time {} s of \
                             voxel {i}", a[i]));
                    }
                }
            }
        }

        let crush = match (&p.crushing, p.macrovascular.is_some()) {
            (Some(c), true) => {
                let vel = per_label(c.arterial_velocity.as_ref().expect("protocol requires it with part B"));
                Some((0..n).map(|v| vel.iter().map(|&vm| survival(vm, c.venc[v])).collect()).collect())
            }
            _ => None,
        };

        let physio = p.physio.map(|params| {
            let horizon = p.row_start.last().copied().unwrap_or(0.0) + p.rows.last().map_or(0.0, |r| r.tr) + 1.0;
            Physio::new(params, p.seed, horizon)
        });
        let label_physio = (0..n)
            .map(|v| {
                let t0 = p.row_start[v];
                match &physio {
                    None => (1.0, (t0, t0), [0.0; 3]),
                    Some(phys) => {
                        let row = &p.rows[v];
                        let (f, m) = match p.label_type {
                            LabelType::Pasl => phys.label_factor_at(t0),
                            _ => phys.label_factor_window(t0, t0 + row.tau),
                        };
                        let w = match p.label_type {
                            LabelType::Pasl => (t0, t0),
                            _ => (t0, t0 + row.tau),
                        };
                        (f, w, m)
                    }
                }
            })
            .collect();
        Ok(P4 { label_path, label_of, abv, aatt, crush, physio, label_physio })
    }
}

fn simulate_impl(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride)
    -> Result<SeriesOutput, String>
{
    simulate_core(p, ph, mode, phase, ov, None)
}

/// [`simulate`] that also returns the compartment images handed to the acquisition (before
/// it), for the P4 tests that check the images themselves.
#[cfg(any(test, feature = "test-hooks"))]
pub fn simulate_compartments(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride)
    -> Result<(Vec<Vec<f32>>, SeriesOutput), String>
{
    let mut images = Vec::new();
    let out = simulate_core(p, ph, mode, phase, ov, Some(&mut images))?;
    Ok((images, out))
}

fn simulate_core(
    p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride, capture: Option<&mut Vec<Vec<f32>>>,
) -> Result<SeriesOutput, String> {
    if let (Some(pf), fs) = (ph.params.and_then(|q| q.field_strength), p.field_strength) {
        if (pf - fs).abs() > 1e-9 {
            return Err(format!("phantom MagneticFieldStrength {pf} disagrees with the protocol's {fs}"));
        }
    }

    // ---- grids ----
    let pv = axis_aligned_voxels(&ph.grid)?;
    let acq_grid = acquisition_grid(&ph.grid, p.voxel_size_mm, p.acq.matrix, p.grid_origin)?;
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
    let off = corner_offset(&ph.grid, &acq_grid)?;
    let r_sim = Resampler::with_offset(ph.grid.dims, pv, sim_grid.dims, sim_vox, off);
    let r_acq = Resampler::with_offset(ph.grid.dims, pv, acq_grid.dims, dv, off);
    let nvox_sim = snx * sny * nz;
    let nvox_acq = nx * ny * nz;

    if p.compat.is_some() && ph.fieldmap.is_some() {
        return Err("the phantom has a fieldmap under [compat] asldro = true: simasl cannot express it".to_string());
    }
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
    // P4, part B: K arterial compartments (class) or one (voxel) after the blood.
    let macro_on = p.macrovascular.is_some();
    let ncomp = if macro_on { 3 * k } else { 2 * k };
    let t2_arterial_ms = p.macrovascular.as_ref().map(|m| (m.t2_arterial.0 * 1000.0) as f32);
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
            if let Some(t2a) = t2_arterial_ms {
                for &tp in t2p_ms.iter().take(k) {
                    t2v.push(T2Volume::Uniform(t2a));
                    tiv.push(T2Volume::Uniform(tp));
                }
            }
            (t2v, tiv)
        }
        Relaxation::Voxel { .. } => {
            let t2m = acq_t2_ms.as_deref().unwrap();
            let tpm = acq_t2p_ms.as_deref().unwrap();
            let mut t2v = vec![T2Volume::Map(t2m), T2Volume::Uniform(t2_blood_ms)];
            let mut tiv = vec![T2Volume::Map(tpm), T2Volume::Map(tpm)];
            if let Some(t2a) = t2_arterial_ms {
                t2v.push(T2Volume::Uniform(t2a));
                tiv.push(T2Volume::Map(tpm));
            }
            (t2v, tiv)
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
    // P3's per-row global-bolus factor; absent under P4's bolus-position model, whose factors
    // are per parcel.
    let bolus_region = match p.suppression.as_ref().map(|s| s.model) {
        Some(SuppressionModel::BolusPosition(r)) => Some(r),
        _ => None,
    };
    let label_factors = match (&p.suppression, bolus_region) {
        (Some(spec), None) => Some((0..n).map(|i| label_factor(&spec.for_row(i))).collect::<Vec<f64>>()),
        _ => None,
    };

    // ---- P4: per-voxel arterial parameters, crushing, physiological noise ----
    let p4 = P4::new(p, ph, bolus_region)?;

    // ---- compat: simasl's one exp(-TE/T2) per phantom voxel, tissue and blood alike, with its
    // zero-T2 guard (`np.divide(.., where=t2 != 0)` leaves exp(0) = 1); the readout's own
    // relaxation is off (`Protocol::acquisition`) ----
    let te_factor: Option<Vec<f64>> = p.compat.as_ref().map(|_| {
        ph.t2.iter().map(|&t2| if t2 != 0.0 { (-p.echo_time_s / t2 as f64).exp() } else { 1.0 }).collect()
    });
    let te = |i: usize| -> f64 { te_factor.as_ref().map_or(1.0, |f| f[i]) };

    // ---- tissue signal per distinct (TR, equation), per compartment, on the sim grid ----
    let mut tissue_cache: HashMap<(u64, bool), Vec<Vec<f32>>> = HashMap::new();
    let mut tissue_for = |tr: f64, se: bool| -> Vec<Vec<f32>> {
        tissue_cache
            .entry((tr.to_bits(), se))
            .or_insert_with(|| {
                let sig: Vec<f64> = match &te_factor {
                    None => (0..ph.nvox()).map(|i| tissue_steady(ph.m0[i] as f64, ph.t1[i] as f64, tr, se)).collect(),
                    Some(f) => (0..ph.nvox()).map(|i| tissue_steady(ph.m0[i] as f64, ph.t1[i] as f64, tr, se) * f[i]).collect(),
                };
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
                    let sl = if te_factor.is_some() {
                        r.mean_slice(z, |i| if m[i] { blood_signal(sign * dm(i)) * te(i) } else { 0.0 })
                    } else {
                        r.mean_slice(z, |i| if m[i] { blood_signal(sign * dm(i)) } else { 0.0 })
                    };
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
    // P4's extra ground truth, per row and frame on the acquisition grid (static) or, under
    // motion, on the simulation grid (moved below like `gt_sim`).
    let (gnx, gny) = if motion_on { (snx, sny) } else { (nx, ny) };
    let gslab = gnx * gny;
    let r_gt = if motion_on { &r_sim } else { &r_acq };
    let alloc = |on: bool| if on { Some(vec![0.0f32; gslab * nz * n]) } else { None };
    let mut gt_iv = alloc(p.exchange_time.is_some());
    let mut gt_sup = alloc(bolus_region.is_some());
    let mut gt_art = alloc(macro_on);
    let mut physio_lines: Vec<PhysioLine> = Vec::new();
    let [pnx, pny, _] = ph.grid.dims;
    let pslab = pnx * pny;
    for (v, row) in p.rows.iter().enumerate() {
        // An m0scan row is a plain spin-echo readout (P3), except under compat, where it takes
        // the series' equation as simasl's does (P2 addendum, part A).
        let se = row.kind == RowKind::M0scan && p.compat.is_none();
        let mut tissue = match (row.kind, &suppression[v]) {
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
        // P4, part E: the tissue factor per slice at its readout (after the cache, so rows that
        // share a cache key but not a time get their own factor).
        if let Some(phys) = &p4.physio {
            let (lf, win, means) = p4.label_physio[v];
            for z in 0..nz {
                let time = p.row_start[v] + row.t + p.slice_offsets[z];
                let tf = phys.tissue_factor(time);
                if let Some(comps) = tissue.as_mut() {
                    for comp in comps.iter_mut() {
                        for x in &mut comp[z * slab..(z + 1) * slab] {
                            *x *= tf as f32;
                        }
                    }
                }
                physio_lines.push(PhysioLine {
                    volume: v, slice: z, time,
                    cardiac_phase: phys.cardiac.phase(time), respiratory_phase: phys.respiratory.phase(time),
                    drift: phys.drift.value(time), tissue_factor: tf,
                    label_window: win, label_means: means, label_factor: lf,
                });
            }
        }
        let factor = label_factors.as_ref().map_or(1.0, |f| f[v]);
        let sign = blood_sign(row.kind, ov) * factor;
        let wants_gt = matches!(row.kind, RowKind::Label | RowKind::Deltam);
        // The P1-P3 label path runs unchanged when no P4 part touches the label; the P4 path
        // computes only the ground-truth delta_m through it.
        let (blood, gt_s) = if p4.label_path {
            blood_for(row, &r_sim, 0.0, wants_gt && motion_on)
        } else {
            blood_for(row, &r_sim, sign, wants_gt && motion_on)
        };
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
            if !p4.label_path {
                let target = if ov == RowOverride::BloodIntoTissue0 && row.kind == RowKind::Label { 0 } else { k + c };
                for vox in 0..nvox_sim {
                    images[target][vox * n + v] += blood[c][vox];
                }
            }
        }

        // ---- P4: the label by parts (A), with parcel factors (D) and the physiological label
        // factor (E), and the arterial compartment (B) with crushing (C) ----
        if p4.label_path && wants_gt {
            let kin = p.kinetic(row);
            let sign0 = blood_sign(row.kind, ov) * factor * p4.label_physio[v].0;
            let bolus = bolus_region.map(|region| {
                let s = p.suppression.as_ref().unwrap().for_row(v);
                (region, s.pulse_times, s.epsilon)
            });
            let mut partitions: HashMap<u64, Vec<(f64, f64, f64)>> = HashMap::new();
            let blood_target = |c: usize| if ov == RowOverride::BloodIntoTissue0 && row.kind == RowKind::Label { 0 } else { k + c };
            let ev_target = |c: usize| if ov == RowOverride::ExtravascularIntoBlood && row.kind == RowKind::Label { k + c } else { c };
            for z in 0..nz {
                let zs = r_sim.z_slab(z);
                let (Some(zlo), Some(zhi)) = (zs.first().map(|p| p.0), zs.last().map(|p| p.0)) else { continue };
                let base = pslab * zlo;
                let len = pslab * (zhi - zlo + 1);
                let t = row.t + p.slice_offsets[z];
                let mut bl = vec![0.0f64; len];
                let mut ev = if p.exchange_time.is_some() { vec![0.0f64; len] } else { Vec::new() };
                let mut art = if macro_on { vec![0.0f64; len] } else { Vec::new() };
                let mut giv = if gt_iv.is_some() { vec![0.0f64; len] } else { Vec::new() };
                let mut gsup = if gt_sup.is_some() { vec![0.0f64; len] } else { Vec::new() };
                let mut gart = if gt_art.is_some() { vec![0.0f64; len] } else { Vec::new() };
                for j in 0..len {
                    let i = base + j;
                    if ph.dseg[i] <= 0 {
                        continue;
                    }
                    let (f_ml, att, t1t, m0) = (ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64);
                    // the tissue label: (scale, total, intravascular), the single-sub-bolus factor
                    // folded into the scale as P3 folds its factor into the sign
                    let (scale, dm, iv) = match &bolus {
                        Some((region, pulses, eps)) => {
                            let delta = entry_offset(p.label_type, *region, att);
                            let key = delta.map_or(u64::MAX, f64::to_bits);
                            let sb = partitions.entry(key).or_insert_with(|| subbolus_factors(pulses, *eps, kin.tau, delta));
                            if sb.len() == 1 {
                                let iv = p.exchange_time.map(|te| delta_m_iv(&kin, f_ml, att, t1t, m0, t, te));
                                (sb[0].2, delta_m(&kin, f_ml, att, t1t, m0, t), iv)
                            } else {
                                let dm: f64 = sb.iter().map(|&(a, b, f)| f * delta_m_sub(&kin, f_ml, att, t1t, m0, t, a, b)).sum();
                                let iv = p.exchange_time.map(|te| {
                                    sb.iter().map(|&(a, b, f)| f * delta_m_iv_sub(&kin, f_ml, att, t1t, m0, t, a, b, te)).sum::<f64>()
                                });
                                (1.0, dm, iv)
                            }
                        }
                        None => (1.0, delta_m(&kin, f_ml, att, t1t, m0, t), p.exchange_time.map(|te| delta_m_iv(&kin, f_ml, att, t1t, m0, t, te))),
                    };
                    let s = sign0 * scale;
                    match iv {
                        Some(iv) => {
                            bl[j] = blood_signal(s * iv);
                            ev[j] = blood_signal(s * (dm - iv));
                        }
                        None => bl[j] = blood_signal(s * dm),
                    }
                    if !giv.is_empty() {
                        // the unsuppressed intravascular part: kinetics and the split only
                        giv[j] = delta_m_iv(&kin, f_ml, att, t1t, m0, t, p.exchange_time.unwrap());
                    }
                    if !gsup.is_empty() {
                        gsup[j] = scale * dm;
                    }
                    if let (Some(abv), Some(aatt)) = (&p4.abv, &p4.aatt) {
                        let (va, a) = arterial_dm(&kin, abv[i], aatt[i], m0, t);
                        if let Some(a) = a {
                            let g = match &bolus {
                                Some((region, pulses, eps)) => {
                                    arterial_factor(pulses, *eps, a, entry_offset(p.label_type, *region, aatt[i]))
                                }
                                None => 1.0,
                            };
                            let c = p4.crush.as_ref().map_or(1.0, |cr| cr[v][p4.label_of[i]]);
                            art[j] = blood_signal(sign0 * g * c * va);
                        }
                        if !gart.is_empty() {
                            gart[j] = va;
                        }
                    }
                }
                // into the compartments, per mask
                for (c, m) in masks.iter().enumerate() {
                    let add = |images: &mut Vec<Vec<f32>>, target: usize, buf: &[f64]| {
                        let sl = r_sim.mean_slice(z, |i| if m[i] { buf[i - base] } else { 0.0 });
                        for (jj, x) in sl.iter().enumerate() {
                            images[target][(z * slab + jj) * n + v] += *x;
                        }
                    };
                    add(&mut images, blood_target(c), &bl);
                    if !ev.is_empty() {
                        add(&mut images, ev_target(c), &ev);
                    }
                    if !art.is_empty() {
                        add(&mut images, 2 * k + c, &art);
                    }
                }
                // the extra ground truth, over every foreground voxel
                for (gt, buf) in [(&mut gt_iv, &giv), (&mut gt_sup, &gsup), (&mut gt_art, &gart)] {
                    if let Some(gt) = gt.as_mut() {
                        let sl = r_gt.mean_slice(z, |i| buf[i - base]);
                        for (jj, x) in sl.iter().enumerate() {
                            gt[(z * gslab + jj) * n + v] = *x;
                        }
                    }
                }
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
        // P4's extra truths move with the same poses (no shot events), then block-average
        for gt in [&mut gt_iv, &mut gt_sup, &mut gt_art] {
            if let Some(g) = gt.take() {
                let mut arr = [g];
                apply_motion(&mut arr, sim_grid.dims, n, v2w, &poses);
                let [moved] = arr;
                *gt = Some(block_mean_inplane(&moved, sim_grid.dims, o, n));
            }
        }
        let n_shots = slice_schedule(nz, p.mb, p.mb_interleaved).len();
        events = draw_events(m.within.as_ref(), n, n_shots, seed);
        if !events.is_empty() {
            dropped = apply_multiband_motion(
                &mut images, sim_grid.dims, n, v2w, p.mb, p.mb_interleaved, &DropoutLaw::Uniform, &events,
            );
        }
    }

    // ---- compat noise: simasl's SNR against the mean |M0| over the nonzero voxels of the M0
    // ground truth on the acquisition grid, as a per-component image variance ----
    let m0_acq = r_acq.mean(&ph.m0);
    let compat_facts = p.compat.as_ref().map(|c| {
        let nz: Vec<f64> = m0_acq.iter().filter(|v| **v != 0.0).map(|v| v.abs() as f64).collect();
        let m0_reference_mean = if nz.is_empty() { 0.0 } else { nz.iter().sum::<f64>() / nz.len() as f64 };
        let noise_variance = match c.desired_snr {
            Some(snr) => (acq.signal_scale * m0_reference_mean / snr).powi(2),
            None => 0.0,
        };
        CompatFacts { desired_snr: c.desired_snr, noise_variance, m0_reference_mean, m0_reference_voxels: nz.len() }
    });
    if let Some(f) = &compat_facts {
        acq.noise_variance = f.noise_variance;
    }

    if let Some(c) = capture {
        *c = images.clone();
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
        m0: m0_acq,
        dseg: r_acq.majority(&ph.dseg),
        acq_t2_ms,
        acq_t2p_ms,
        delta_m_iv: gt_iv,
        delta_m_suppressed: gt_sup,
        delta_m_arterial: gt_art,
        abv: p4.abv.as_ref().map(|a| r_acq.mean(&a.iter().map(|x| *x as f32).collect::<Vec<f32>>())),
        aatt: match (&p4.abv, &p4.aatt) {
            (Some(b), Some(a)) => {
                let has: Vec<bool> = b.iter().map(|x| *x > 0.0).collect();
                Some(r_acq.masked_mean(&a.iter().map(|x| *x as f32).collect::<Vec<f32>>(), &has))
            }
            _ => None,
        },
    };

    Ok(SeriesOutput {
        acq_grid, sim_grid, n_volumes: n, mag, phase: phase_out, m0, mode: mode_used,
        labels: ph.labels.clone(), n_compartments: ncomp, fieldmap_present: ph.fieldmap.is_some(),
        seeds: (p.seed, m0_seed), acquisition: acq, ground_truth, label_factors, poses, motion_seed,
        events, dropped, compat: compat_facts,
        crush_survival: p4.crush.clone(), physio: p4.physio.as_ref().map(|_| physio_lines),
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
    use crate::kinetic::arterial_dm;
    use crate::crushing::survival;
    use crate::physio::Physio;
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

    // ------------------------------------------------------------------ P2

    /// A compat sidecar on the crop: `voxel` mm, all-zero slice timing for `nz` slices, simasl's
    /// default rows `m0scan control label` and TRs `[10, 5, 5]`, TE 10 ms.
    fn compat_sidecar(voxel: [f64; 3], nz: usize) -> Value {
        let mut s = sidecar(voxel, nz);
        s["SliceTiming"] = json!(vec![0.0; nz]);
        s["M0Type"] = json!("Included");
        s["RepetitionTimePreparation"] = json!([10.0, 5.0, 5.0]);
        s["EchoTime"] = json!(0.01);
        s["TotalReadoutTime"] = json!(0.001);
        s
    }

    fn re_of(out: &SeriesOutput) -> Vec<f64> {
        complex_from(&out.mag, &out.phase).into_iter().map(|z| z.0).collect()
    }

    /// The identity grid under compat: every acquired voxel is one phantom voxel, and the signed
    /// real image is simasl's closed form `(tissue + mag_enc) * exp(-TE/T2)` per voxel.
    fn check_compat_closed_form(contrast: &str) {
        let ph = phantom();
        let ov = format!("[compat]\nasldro = true\n{contrast}");
        let p = protocol_from(&compat_sidecar([1.0, 1.0, 1.0], 6), "m0scan,control,label", &ov);
        let out = simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap();
        assert_eq!(out.acq_grid.dims, ph.grid.dims);
        assert_eq!(out.acq_grid.voxel_to_world, ph.grid.voxel_to_world);
        assert!(!out.acquisition.do_relaxation && out.acquisition.noise_variance == 0.0);
        let ir = p.ir.as_ref().map(|s| s.params);
        let re = re_of(&out);
        let n = out.n_volumes;
        let (mut worst, mut peak) = (0.0f64, 0.0f64);
        for (v, row) in p.rows.iter().enumerate() {
            let kin = p.kinetic(row);
            for i in 0..ph.nvox() {
                let fg = ph.dseg[i] > 0;
                let (m0, t1) = (ph.m0[i] as f64, ph.t1[i] as f64);
                let tissue = match ir {
                    Some(q) => tissue_ir(m0, t1, row.tr, &q),
                    None => tissue_se(m0, t1, row.tr),
                };
                let enc = if row.kind == RowKind::Label {
                    -delta_m(&kin, ph.perfusion[i] as f64, ph.att[i] as f64, t1, m0, row.t)
                } else {
                    0.0
                };
                let blood = match ir {
                    Some(q) => blood_ir(enc, &q),
                    None => blood_se(enc),
                };
                let t2 = ph.t2[i] as f64;
                let te = if t2 != 0.0 { (-0.01 / t2).exp() } else { 1.0 };
                let want = if fg { (tissue + blood) * te } else { 0.0 };
                worst = worst.max((re[i * n + v] - want).abs());
                peak = peak.max(want.abs());
            }
        }
        println!("compat closed form ({contrast:?}): worst {:.3e} of peak {peak:.3}", worst / peak);
        assert!(worst <= 1e-5 * peak, "{worst} vs peak {peak}");
    }

    #[test]
    fn compat_is_the_closed_form_per_voxel_under_spin_echo() {
        check_compat_closed_form("");
    }

    #[test]
    fn compat_is_the_closed_form_per_voxel_under_ir_m0scan_row_included() {
        check_compat_closed_form("[signal]\nacq_contrast = \"ir\"\nexcitation_flip_angle = 60.0\n");
    }

    #[test]
    fn outside_compat_the_m0scan_row_of_an_ir_series_stays_spin_echo() {
        let ph = phantom();
        let mut s = compat_sidecar([3.0, 3.0, 3.0], 2);
        s["EchoTime"] = json!(0.012);
        s["TotalReadoutTime"] = json!(0.012);
        s["RepetitionTimePreparation"] = json!([10.0, 5.0]);
        let se = protocol_from(&s, "m0scan,control", "");
        let ir = protocol_from(&s, "m0scan,control", "[signal]\nacq_contrast = \"ir\"\n");
        let a = simulate(&se, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let b = simulate(&ir, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let nvox = a.acq_grid.dims.iter().product::<usize>();
        assert!((0..nvox).all(|v| a.mag[v * 2] == b.mag[v * 2]), "the m0scan row must not change");
        assert!((0..nvox).any(|v| a.mag[v * 2 + 1] != b.mag[v * 2 + 1]), "the control row must");
    }

    #[test]
    fn voxel_centre_grid_puts_a_point_source_where_the_affine_says() {
        // 2 mm in-plane on the 1 mm crop: acquisition voxel j is centred on phantom voxel 2j and
        // covers it fully (half the cell) and its two neighbours by a quarter each.
        let mut ph = phantom();
        let [nx, ny, _] = ph.grid.dims;
        let (px, py, pz) = (10usize, 12usize, 3usize);
        let at = |x: usize, y: usize, z: usize| x + nx * (y + ny * z);
        assert!(ph.dseg[at(px, py, pz)] > 0, "the source must sit in the foreground");
        ph.perfusion.iter_mut().for_each(|f| *f = 0.0);
        let keep = ph.m0[at(px, py, pz)];
        ph.m0.iter_mut().for_each(|m| *m = 0.0);
        ph.m0[at(px, py, pz)] = keep;
        let p = protocol_from(&compat_sidecar([2.0, 2.0, 1.0], 6), "m0scan,control,label", "[compat]\nasldro = true\n");
        let out = simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap();
        assert_eq!(out.acq_grid.dims, [12, 12, 6]);
        let [ax, ay, _] = out.acq_grid.dims;
        let re = re_of(&out);
        let n = out.n_volumes;
        let (best, _) = re.iter().enumerate().filter(|(i, _)| i % n == 1)
            .fold((0usize, 0.0f64), |b, (i, v)| if v.abs() > b.1 { (i / n, v.abs()) } else { b });
        assert_eq!(best, (px / 2) + ax * ((py / 2) + ay * pz), "the source landed in the wrong voxel");
        // the world position of that acquisition voxel is the phantom voxel's
        let (g, h) = (&out.acq_grid.voxel_to_world, &ph.grid.voxel_to_world);
        for (a, (q, r)) in [(px / 2, px), (py / 2, py), (pz, pz)].into_iter().enumerate() {
            assert!((g[a][a] * q as f64 + g[a][3] - (h[a][a] * r as f64 + h[a][3])).abs() < 1e-12);
        }
        // and it carries a quarter of the source (half in x, half in y), nothing elsewhere
        let te = (-0.01 / ph.t2[at(px, py, pz)] as f64).exp();
        let want = 0.25 * tissue_se(keep as f64, ph.t1[at(px, py, pz)] as f64, 5.0) * te;
        assert!((re[best * n + 1] - want).abs() < 1e-5 * want, "{} vs {want}", re[best * n + 1]);
        let rest = re.iter().enumerate().filter(|(i, _)| i % n == 1 && i / n != best).fold(0.0f64, |m, (_, v)| m.max(v.abs()));
        assert!(rest < 1e-5 * want, "signal leaked: {rest}");
    }

    #[test]
    fn compat_snr_resolves_to_the_reference_variance_and_is_realized() {
        let ph = phantom();
        let s = compat_sidecar([1.0, 1.0, 1.0], 6);
        let clean = protocol_from(&s, "m0scan,control,label", "[compat]\nasldro = true\n");
        let noisy = protocol_from(&s, "m0scan,control,label", "seed = 4\n[compat]\nasldro = true\ndesired_snr = 50.0\n");
        let a = simulate(&clean, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let b = simulate(&noisy, &ph, T2Mode::Auto, &no_phase()).unwrap();
        let fa = a.compat.as_ref().unwrap();
        assert_eq!((fa.desired_snr, fa.noise_variance), (None, 0.0));
        let f = b.compat.as_ref().unwrap();
        let nz: Vec<f64> = b.ground_truth.m0.iter().filter(|v| **v != 0.0).map(|v| v.abs() as f64).collect();
        let mean = nz.iter().sum::<f64>() / nz.len() as f64;
        assert_eq!(f.m0_reference_voxels, nz.len());
        assert!((f.m0_reference_mean - mean).abs() <= 1e-12 * mean);
        let want = (mean / 50.0).powi(2);
        assert!((f.noise_variance - want).abs() <= 1e-12 * want && b.acquisition.noise_variance == f.noise_variance);
        // the realized per-component variance (noisy - clean) over 3 volumes x 3456 voxels
        let (za, zb) = (complex_from(&a.mag, &a.phase), complex_from(&b.mag, &b.phase));
        let (mut sr, mut si) = (0.0f64, 0.0f64);
        for (x, y) in za.iter().zip(&zb) {
            sr += (y.0 - x.0).powi(2);
            si += (y.1 - x.1).powi(2);
        }
        let k = za.len() as f64;
        println!("compat SNR 50: predicted variance {want:.4e}, realized re {:.4e} im {:.4e}", sr / k, si / k);
        for got in [sr / k, si / k] {
            assert!((got / want - 1.0).abs() < 0.08, "{got} vs {want}");
        }
    }

    #[test]
    fn compat_refuses_a_fieldmap_phantom() {
        let mut ph = phantom();
        ph.fieldmap = Some(vec![0.0; ph.nvox()]);
        let p = protocol_from(&compat_sidecar([1.0, 1.0, 1.0], 6), "m0scan,control,label", "[compat]\nasldro = true\n");
        let e = simulate(&p, &ph, T2Mode::Auto, &no_phase()).unwrap_err();
        assert!(e.contains("fieldmap") && e.contains("simasl"), "{e}");
    }

    // ------------------------------------------------------------------ P4

    /// The bound for comparisons that regroup float32 compartment sums (plan, Task 5).
    fn within(a: f64, b: f64, peak: f64) -> bool {
        (a - b).abs() <= 1e-6 * peak + 1e-6 * b.abs()
    }

    const HOM: &str = "[acquisition]\noversample = 1\n";
    const TABLES: &str = "[macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
                          arterial_transit_time = { grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }\n";

    fn run(s: &Value, rows: &str, ov: &str, mode: T2Mode) -> SeriesOutput {
        simulate(&protocol_from(s, rows, ov), &phantom(), mode, &no_phase()).unwrap()
    }

    fn comps(s: &Value, rows: &str, ov: &str, mode: T2Mode) -> (Vec<Vec<f32>>, SeriesOutput) {
        simulate_compartments(&protocol_from(s, rows, ov), &phantom(), mode, &no_phase(), RowOverride::None).unwrap()
    }

    fn peak_of(v: &[f32]) -> f64 {
        max_abs(v) as f64
    }

    #[test]
    fn the_exchange_split_conserves_each_labels_total_and_its_limit_is_p1() {
        let s = sidecar([1.0, 1.0, 1.0], 6);
        let (off, _) = comps(&s, "label", HOM, T2Mode::Class);
        let (on, _) = comps(&s, "label", &format!("{HOM}[kinetic]\nexchange_time = 0.4\n"), T2Mode::Class);
        let k = 3;
        let peak = peak_of(&off[k]);
        assert!(peak > 0.0);
        let mut moved = 0.0f64;
        for c in 0..k {
            for vox in 0..off[0].len() {
                let a = on[c][vox] as f64 + on[k + c][vox] as f64;
                let b = off[c][vox] as f64 + off[k + c][vox] as f64;
                assert!(within(a, b, peak_of(&off[c]).max(peak)), "label {c} voxel {vox}: {a} vs {b}");
                moved = moved.max((off[k + c][vox] - on[k + c][vox]).abs() as f64);
            }
        }
        assert!(moved > 1e-3 * peak, "the split must move label out of the blood: {moved}");
        // tau_ex huge: the acquired images are P1's
        let a = run(&s, "label", HOM, T2Mode::Class);
        let b = run(&s, "label", &format!("{HOM}[kinetic]\nexchange_time = 1e9\n"), T2Mode::Class);
        let p = peak_of(&a.mag);
        for (x, y) in a.mag.iter().zip(&b.mag) {
            assert!(within(*y as f64, *x as f64, p) || (*x as f64 - *y as f64).abs() <= 1e-9 * p, "{x} vs {y}");
        }
        // the intravascular truth is below the total and positive in GM
        let gt = b.ground_truth.delta_m_iv.as_ref().unwrap();
        assert!(gt.iter().zip(&b.ground_truth.delta_m).all(|(iv, dm)| *iv <= *dm * (1.0 + 1e-6) + 1e-12));
    }

    #[allow(clippy::needless_range_loop)]
    fn check_arterial_image(s: &Value, rows: &str, aatt_gm: f64, expect_signal: bool) {
        let ph = phantom();
        let ov = format!("{HOM}[macrovascular]\narterial_blood_volume = {{ grey_matter = 0.03, white_matter = 0.0, csf = 0.0 }}\n\
                          arterial_transit_time = {{ grey_matter = {aatt_gm}, white_matter = 0.0, csf = 0.0 }}\n");
        let p = protocol_from(s, rows, &ov);
        let (img, out) = simulate_compartments(&p, &ph, T2Mode::Class, &no_phase(), RowOverride::None).unwrap();
        let k = out.labels.len();
        assert_eq!(out.n_compartments, 3 * k);
        let row = &p.rows[0];
        let kin = p.kinetic(row);
        let [nx, ny, _] = ph.grid.dims;
        let mut seen = 0.0f64;
        for i in 0..ph.nvox() {
            let z = i / (nx * ny);
            let t = row.t + p.slice_offsets[z];
            let want = if ph.dseg[i] == 1 { -arterial_dm(&kin, 0.03, aatt_gm, ph.m0[i] as f64, t).0 } else { 0.0 };
            let got = img[2 * k][i] as f64;
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1e-9), "voxel {i}: {got} vs {want}");
            seen = seen.max(got.abs());
        }
        assert_eq!(seen > 0.0, expect_signal, "arterial signal present: {seen}");
        // the arterial truth is the g = 1 term, positive where present
        assert!(out.ground_truth.delta_m_arterial.as_ref().unwrap().iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn the_arterial_compartment_is_the_closed_form_per_voxel() {
        // PCASL: t = 3.6 + offsets, window [2.5, 4.3)
        check_arterial_image(&sidecar([1.0, 1.0, 1.0], 6), "label", 2.5, true);
        check_arterial_image(&sidecar([1.0, 1.0, 1.0], 6), "label", 0.5, false);
        // PASL: t = PLD 1.8 + offsets, bolus 0.7, window [1.5, 2.2)
        let mut s = sidecar([1.0, 1.0, 1.0], 6);
        s["ArterialSpinLabelingType"] = json!("PASL");
        s["BolusCutOffFlag"] = json!(true);
        s["BolusCutOffTechnique"] = json!("Q2TIPS");
        s["BolusCutOffDelayTime"] = json!(0.7);
        s.as_object_mut().unwrap().remove("LabelingDuration");
        check_arterial_image(&s, "label", 1.5, true);
        check_arterial_image(&s, "label", 0.2, false);
    }

    fn crush_pair(mode: T2Mode) {
        let vel = "[vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n";
        let with = |venc: f64, abv: (f64, f64)| {
            let mut s = sidecar([1.0, 1.0, 1.0], 6);
            s["VascularCrushing"] = json!(true);
            s["VascularCrushingVENC"] = json!(venc);
            let ov = format!("{HOM}{vel}[macrovascular]\narterial_blood_volume = {{ grey_matter = {}, white_matter = {}, csf = 0.0 }}\n\
                              arterial_transit_time = {{ grey_matter = 2.5, white_matter = 2.7, csf = 0.0 }}\n", abv.0, abv.1);
            let o = run(&s, "label", &ov, mode);
            (complex_from(&o.mag, &o.phase), o)
        };
        let (z0, _) = with(0.0, (0.03, 0.015));
        let (z4, o4) = with(4.0, (0.03, 0.015));
        let (none, _) = with(0.0, (0.0, 0.0));
        let (gm, _) = with(0.0, (0.03, 0.0));
        let (wm, _) = with(0.0, (0.0, 0.015));
        let c = &o4.crush_survival.as_ref().unwrap()[0];
        assert!((c[0] - survival(10.0, 4.0)).abs() < 1e-15 && (c[1] - survival(6.0, 4.0)).abs() < 1e-15);
        assert!(c[0] != c[1]);
        let a_gm: Vec<(f64, f64)> = gm.iter().zip(&none).map(|(a, b)| (a.0 - b.0, a.1 - b.1)).collect();
        let a_wm: Vec<(f64, f64)> = wm.iter().zip(&none).map(|(a, b)| (a.0 - b.0, a.1 - b.1)).collect();
        let norm = |v: &[(f64, f64)]| v.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
        assert!(norm(&a_gm) > 0.0 && norm(&a_wm) > 0.0, "both labels need arterial signal");
        let peak = norm(&z0);
        let mut worst = 0.0f64;
        let mut pred_max = 0.0f64;
        for i in 0..z0.len() {
            let pred = ((c[0] - 1.0) * a_gm[i].0 + (c[1] - 1.0) * a_wm[i].0, (c[0] - 1.0) * a_gm[i].1 + (c[1] - 1.0) * a_wm[i].1);
            let got = (z4[i].0 - z0[i].0, z4[i].1 - z0[i].1);
            pred_max = pred_max.max(pred.0.hypot(pred.1));
            worst = worst.max((got.0 - pred.0).hypot(got.1 - pred.1) / peak);
        }
        assert!(pred_max > 1e-3 * peak, "the predicted difference must be measurable: {pred_max}");
        println!("matched VENC pair ({}): worst {worst:.2e} of peak", mode.as_str());
        assert!(worst < 2e-6, "{worst}");
    }

    #[test]
    fn a_matched_venc_pair_isolates_the_arterial_signal_per_label() {
        crush_pair(T2Mode::Class);
        crush_pair(T2Mode::Voxel);
    }

    fn bs(pulses: &[f64], eps: f64, model: &str) -> (Value, String) {
        let mut s = sidecar([1.0, 1.0, 1.0], 6);
        s["BackgroundSuppression"] = json!(true);
        s["BackgroundSuppressionNumberPulses"] = json!(pulses.len());
        s["BackgroundSuppressionPulseTime"] = json!(pulses);
        (s, format!("{HOM}[background_suppression]\ninversion_efficiency = {eps}\n{model}"))
    }

    #[test]
    fn global_bolus_position_is_the_global_bolus_model_bit_for_bit() {
        for eps in [0.95, 0.7] {
            let (s, a) = bs(&[2.0, 3.2], eps, "");
            let (_, b) = bs(&[2.0, 3.2], eps, "model = \"bolus-position\"\npulse_region = \"global\"\n");
            let x = run(&s, "control,label", &a, T2Mode::Class);
            let y = run(&s, "control,label", &b, T2Mode::Class);
            assert_eq!(x.mag, y.mag, "eps {eps}");
            assert_eq!(x.phase, y.phase);
            assert!(x.label_factors.is_some() && y.label_factors.is_none());
            // the suppressed truth carries the factor
            let f = x.label_factors.as_ref().unwrap()[1];
            let (sup, dm) = (y.ground_truth.delta_m_suppressed.as_ref().unwrap(), &y.ground_truth.delta_m);
            for (s1, d1) in sup.iter().zip(dm) {
                assert!((*s1 as f64 - f * *d1 as f64).abs() <= 1e-6 * d1.abs() as f64 + 1e-12);
            }
        }
    }

    #[test]
    fn a_zero_efficiency_slab_pulse_cuts_without_changing_either_part() {
        let model = "model = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.2\n[kinetic]\nexchange_time = 0.4\n";
        let (s1, a) = bs(&[1.0], 0.0, model);
        let (s0, b) = bs(&[], 0.0, model);
        let (x, _) = comps(&s1, "label", &a, T2Mode::Class);
        let (y, _) = comps(&s0, "label", &b, T2Mode::Class);
        for c in 0..x.len() {
            let peak = peak_of(&y[c]).max(peak_of(&y[3]));
            for (p, q) in x[c].iter().zip(&y[c]) {
                assert!(within(*p as f64, *q as f64, peak), "compartment {c}: {p} vs {q}");
            }
        }
    }

    #[test]
    fn multiband_and_the_split_keep_each_slices_anatomy() {
        // the same slice times with and without multiband give the same compartments: the
        // P4 path keys nothing on the shot
        let model = "model = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.2\n[kinetic]\nexchange_time = 0.4\n";
        let (mut s, ov) = bs(&[2.0, 3.2], 0.95, model);
        s["SliceTiming"] = json!([0.0, 0.04, 0.08, 0.0, 0.04, 0.08]);
        let (single, _) = comps(&s, "control,label", &ov, T2Mode::Class);
        s["MultibandAccelerationFactor"] = json!(2);
        let (multi, _) = comps(&s, "control,label", &ov, T2Mode::Class);
        assert_eq!(single, multi);
        // and slices sharing a readout time differ (their own anatomy)
        let [nx, ny, _] = phantom().grid.dims;
        let sl = |z: usize| &multi[0][z * nx * ny * 2..(z + 1) * nx * ny * 2];
        assert_ne!(sl(0), sl(3));
    }

    #[test]
    fn physio_scales_tissue_per_slice_and_reproduces_its_stream() {
        let s = sidecar([1.0, 1.0, 1.0], 6);
        let physio = "[physio]\ntissue_cardiac = 0.05\ntissue_respiratory = 0.03\ntissue_drift = 0.02\nlabel_cardiac = 0.04\n";
        let (base, _) = comps(&s, "control,control,label", &format!("seed = 77\n{HOM}"), T2Mode::Class);
        let (mod_, out) = comps(&s, "control,control,label", &format!("seed = 77\n{HOM}{physio}"), T2Mode::Class);
        let lines = out.physio.as_ref().unwrap();
        assert_eq!(lines.len(), 3 * 6);
        let n = 3;
        let [nx, ny, nz] = phantom().grid.dims;
        // two rows sharing the tissue cache key (same TR) get their own factors
        assert_ne!(lines[0].tissue_factor, lines[6].tissue_factor);
        for l in lines.iter().filter(|l| l.volume < 2) {
            let z = l.slice;
            for j in 0..nx * ny {
                let vox = z * nx * ny + j;
                let (a, b) = (mod_[0][vox * n + l.volume] as f64, base[0][vox * n + l.volume] as f64);
                assert!((a - b * l.tissue_factor).abs() <= 1e-6 * b.abs() + 1e-9, "v {} z {z}", l.volume);
            }
        }
        assert!(lines.iter().all(|l| l.slice < nz));
        // the reference stream through simulate: the salt has one owner
        let p = protocol_from(&s, "control,control,label", &format!("seed = 77\n{HOM}{physio}"));
        let ph = Physio::new(p.physio.unwrap(), 77, 100.0);
        assert_eq!(lines[0].cardiac_phase, ph.cardiac.phase(lines[0].time));
        assert_eq!(lines[0].drift, ph.drift.value(lines[0].time));
    }

    #[test]
    fn physio_leaves_the_acquisition_noise_alone() {
        let s = sidecar([1.0, 1.0, 1.0], 6);
        let physio = "[physio]\ntissue_cardiac = 0.05\nlabel_cardiac = 0.04\n";
        let ov = |noise: f64, ph: &str| format!("seed = 5\n[acquisition]\noversample = 1\nnoise_variance = {noise}\n{ph}");
        let a0 = run(&s, "control,label", &ov(0.0, ""), T2Mode::Class);
        let a1 = run(&s, "control,label", &ov(4.0, ""), T2Mode::Class);
        let b0 = run(&s, "control,label", &ov(0.0, physio), T2Mode::Class);
        let b1 = run(&s, "control,label", &ov(4.0, physio), T2Mode::Class);
        let res = |x: &SeriesOutput, y: &SeriesOutput| -> Vec<(f64, f64)> {
            complex_from(&y.mag, &y.phase).iter().zip(complex_from(&x.mag, &x.phase)).map(|(p, q)| (p.0 - q.0, p.1 - q.1)).collect()
        };
        let (ra, rb) = (res(&a0, &a1), res(&b0, &b1));
        let var: f64 = ra.iter().map(|z| z.0 * z.0).sum::<f64>() / ra.len() as f64;
        assert!(var > 0.1, "noise must be on: {var}");
        let peak = peak_of(&a1.mag);
        for (p, q) in ra.iter().zip(&rb) {
            assert!((p.0 - q.0).abs() <= 1e-5 * peak && (p.1 - q.1).abs() <= 1e-5 * peak, "{p:?} vs {q:?}");
        }
        assert_eq!(a1.seeds, b1.seeds);
        assert_eq!(a1.acquisition.noise_variance, b1.acquisition.noise_variance);
    }

    #[test]
    fn moved_p4_truths_are_the_static_ones_moved() {
        let ph = phantom();
        // its own directory: write_trajectory's is keyed on the row count, which the P3 test shares
        let dir = std::env::temp_dir().join(format!("aslscan-series-p4-moved-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tsv = dir.join("motion.tsv");
        std::fs::write(&tsv, "trans_x\ttrans_y\ttrans_z\trot_x\trot_y\trot_z\n0\t0\t0\t0\t0\t0\n2\t0\t0\t0\t0\t0\n").unwrap();
        let path = tsv.to_string_lossy().replace('\\', "\\\\");
        let (s, model) = bs(&[2.0, 3.2], 0.95, "model = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.2\n[kinetic]\nexchange_time = 0.4\n");
        let ov = format!("{model}{TABLES}");
        let moving = format!("{ov}[motion]\nmode = \"trajectory\"\ntrajectory = \"{path}\"\n");
        let still = run(&s, "label,label", &ov, T2Mode::Class);
        let moved = run(&s, "label,label", &moving, T2Mode::Class);
        let g = moved.acq_grid.clone();
        let n = 2;
        let nvox = g.dims.iter().product::<usize>();
        for (a, b) in [
            (&still.ground_truth.delta_m_iv, &moved.ground_truth.delta_m_iv),
            (&still.ground_truth.delta_m_suppressed, &moved.ground_truth.delta_m_suppressed),
            (&still.ground_truth.delta_m_arterial, &moved.ground_truth.delta_m_arterial),
        ] {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            let vol = |x: &[f32], v: usize| (0..nvox).map(|i| x[i * n + v]).collect::<Vec<f32>>();
            let want = mrsim_acq::motion::resample_by_pose(&vol(a, 1), g.dims, g.voxel_to_world, moved.poses[1]);
            let got = vol(b, 1);
            let peak = max_abs(&want) as f64;
            assert!(peak > 0.0);
            for (x, y) in got.iter().zip(&want) {
                assert!((*x as f64 - *y as f64).abs() <= 1e-6 * peak, "{x} vs {y}");
            }
            assert_eq!(vol(a, 0), vol(b, 0), "volume 0 does not move");
        }
        let _ = std::fs::remove_dir_all(&dir);
        let _ = ph;
    }

    #[test]
    fn class_and_voxel_agree_with_the_split_and_the_arterial_compartment() {
        let s = sidecar([1.0, 1.0, 1.0], 6);
        let ov = format!("{HOM}[kinetic]\nexchange_time = 0.4\n{TABLES}");
        let a = run(&s, "control,label", &ov, T2Mode::Class);
        let b = run(&s, "control,label", &ov, T2Mode::Voxel);
        assert_eq!((a.n_compartments, b.n_compartments), (9, 3));
        let scale = max_abs(&a.mag) as f64;
        let mut worst = 0.0f64;
        for (x, y) in a.mag.iter().zip(&b.mag) {
            let (x, y) = (*x as f64, *y as f64);
            worst = worst.max((x - y).abs() / (1e-5 * x.abs().max(y.abs()) + 1e-6 * scale));
        }
        assert!(worst <= 1.0, "class vs voxel with parts A and B: {worst:.2}x tolerance");
    }

    #[test]
    fn the_separate_m0_scan_is_untouched_by_every_part() {
        let s = sidecar([1.0, 1.0, 1.0], 6);
        let mk = |extra: &str| {
            let mut p = protocol_from(&s, "control,label", &format!("{HOM}[m0]\nrepetition_time = 8.0\n{extra}"));
            p.m0_type = M0Type::Separate;
            simulate(&p, &phantom(), T2Mode::Class, &no_phase()).unwrap()
        };
        let off = mk("");
        let on = mk(&format!("[kinetic]\nexchange_time = 0.4\n{TABLES}[physio]\ntissue_cardiac = 0.05\n"));
        assert_eq!(off.m0, on.m0);
        assert_ne!(off.mag, on.mag);
    }

    #[test]
    fn duplicate_or_unknown_label_names_are_refused_with_tables() {
        let mut ph = phantom();
        ph.labels[1].1 = "grey_matter".to_string();
        let p = protocol_from(&sidecar([1.0, 1.0, 1.0], 6), "label", &format!("{HOM}{TABLES}"));
        let e = simulate(&p, &ph, T2Mode::Class, &no_phase()).unwrap_err();
        assert!(e.contains("share the name"), "{e}");
        let p = protocol_from(&sidecar([1.0, 1.0, 1.0], 6), "label",
                              &format!("{HOM}[macrovascular]\narterial_blood_volume = {{ grey_matter = 0.03, white_matter = 0.0 }}\n\
                                        arterial_transit_time = {{ grey_matter = 1.0, white_matter = 1.0, csf = 0.0, other = 1.0 }}\n"));
        let e = simulate(&p, &phantom(), T2Mode::Class, &no_phase()).unwrap_err();
        assert!(e.contains("csf") || e.contains("other"), "{e}");
        // a slab entry after a voxel's arrival
        let (s, ov) = bs(&[2.0, 3.2], 0.95, "model = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 2.0\n");
        let e = simulate(&protocol_from(&s, "label", &ov), &phantom(), T2Mode::Class, &no_phase()).unwrap_err();
        assert!(e.contains("slab_entry_time") && e.contains("voxel"), "{e}");
    }
}
