//! The series of a protocol with a P6 feature (P6 addendum; the legacy dispatch is in the parent).
//!
//! The P1-P5 body of [`super::simulate_legacy`] copied, never shared, so that a P6 change cannot move a
//! byte of a legacy output: [`build`] is everything up to the acquisition (the compartment images,
//! the ground truth, the separate M0's tissue), run once per echo time where the echo time enters the
//! images (compat's `exp(-TE/T2)`), and [`simulate_p6`] the acquisition and the output. Part C
//! (multi-TE): every echo is read from the same excitation by `simulate_acquisition_echoes`.

use super::*;
use mrsim_acq::kspace::simulate_acquisition_echoes;
use crate::protocol::check_excitation_timing;

/// What [`build`] hands to the acquisition and the output.
struct Built {
    acq_grid: Grid,
    sim_grid: Grid,
    n: usize,
    nvox_sim: usize,
    images: Vec<Vec<f32>>,
    gt_static: Vec<f32>,
    gt_moved: Option<Vec<f32>>,
    gt_iv: Option<Vec<f32>>,
    gt_sup: Option<Vec<f32>>,
    gt_art: Option<Vec<f32>>,
    physio_lines: Vec<PhysioLine>,
    shot_physio: Vec<Vec<(f64, f64)>>,
    shot_gain: Vec<Vec<f64>>,
    shot_sets: Vec<Vec<ShotSet>>,
    n_shots: usize,
    events: Vec<MotionEvent>,
    dropped: Vec<DroppedShot>,
    poses: Vec<Pose>,
    motion_seed: Option<u64>,
    res3d: Option<ReadoutResolution>,
    acq: Acquisition,
    fmap_sim: Vec<f32>,
    relax: Relaxation,
    mode_used: T2Mode,
    k: usize,
    ncomp: usize,
    ev_group: bool,
    macro_on: bool,
    t2_arterial_ms: Option<f32>,
    t2_blood_ms: f32,
    acq_t2_ms: Option<Vec<f32>>,
    acq_t2p_ms: Option<Vec<f32>>,
    acq_t1_ms: Option<Vec<f32>>,
    needs_t1: bool,
    r_acq: Resampler,
    m0_acq: Vec<f32>,
    compat_facts: Option<CompatFacts>,
    p4: P4,
    label_factors: Option<Vec<f64>>,
    ge_flip: Option<f64>,
    ge_propagated_some: bool,
    m0_images: Option<Vec<Vec<f32>>>,
}

/// The P1-P5 series body up to the acquisition, with `echo_time_s` in place of the protocol's
/// `EchoTime` where the images depend on it (compat's per-voxel `exp(-TE/T2)`).
fn build(p: &Protocol, ph: &Phantom, mode: T2Mode, ov: RowOverride, echo_time_s: f64)
    -> Result<Built, String>
{
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
    // P5 part B: a 3D readout is completed with the grid here, before anything is simulated, and
    // has no slice timing (every partition shares the slab excitation)
    let res3d = resolve_readout(p, acq_grid.dims)?;
    if res3d.is_none() && p.slice_offsets.len() != nz {
        return Err(format!(
            "SliceTiming has {} entries but the acquisition grid has {nz} slices (phantom extent {:.1} mm at \
             {} mm slices)", p.slice_offsets.len(), ph.grid.dims[2] as f64 * pv[2], p.voxel_size_mm[2]));
    }
    let slice_offsets: Vec<f64> = if res3d.is_some() { vec![0.0; nz] } else { p.slice_offsets.clone() };
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
    // a 3D train refocused below 180 degrees has stimulated echoes, which depend on T1 (P5 part B)
    let needs_t1 = res3d.as_ref().is_some_and(|r| r.train.refocusing_deg != 180.0) && acq.do_relaxation;
    let (relax, mode_used) = ph.relaxation_for(mode, needs_t1)?;
    let t2_blood_ms = p.t2_blood_ms();
    // Label masks: compartment i is label i (class) or everything foreground (voxel).
    let masks: Vec<Vec<bool>> = match &relax {
        Relaxation::Class { .. } => ph.labels.iter().map(|(l, _)| ph.dseg.iter().map(|d| d == l).collect()).collect(),
        Relaxation::Voxel { .. } => vec![ph.dseg.iter().map(|d| *d > 0).collect()],
    };
    let k = masks.len();
    // P4, part B: K arterial compartments (class) or one (voxel) after the blood.
    let macro_on = p.macrovascular.is_some();
    // P5 part D: in 3D the physiological factors are per-shot line weights per compartment, so the
    // extravascular label (P4 part A), which P4 puts into the tissue compartment, gets its own group
    // after the others: the tissue's factor is not the label's
    let ev_group = res3d.is_some() && p.physio.is_some() && p.exchange_time.is_some();
    let ev_base = if macro_on { 3 * k } else { 2 * k };
    let ncomp = if ev_group { ev_base + k } else { ev_base };
    let t2_arterial_ms = p.macrovascular.as_ref().map(|m| (m.t2_arterial.0 * 1000.0) as f32);
    // Owned map storage so the T2Volume slices below can borrow it.
    let (acq_t2_ms, acq_t2p_ms): (Option<Vec<f32>>, Option<Vec<f32>>) = match &relax {
        Relaxation::Voxel { t2_ms, t2p_ms } => (Some(r_sim.rate_mean(t2_ms, &ph.m0)), Some(r_sim.rate_mean(t2p_ms, &ph.m0))),
        Relaxation::Class { .. } => (None, None),
    };
    // P5 part B: T1 for the echo amplitudes, only when the refocusing is below 180 degrees: per label
    // (class), or the M0-weighted rate mean on the simulation grid with an infinite background,
    // like the T2 maps (voxel); the blood and arterial compartments take the arterial blood T1
    let acq_t1_ms: Option<Vec<f32>> = (needs_t1 && matches!(relax, Relaxation::Voxel { .. })).then(|| {
        let t1_ms: Vec<f32> = ph.dseg.iter().zip(&ph.t1).map(|(&l, &t)| if l > 0 { t * 1000.0 } else { f32::INFINITY }).collect();
        r_sim.rate_mean(&t1_ms, &ph.m0)
    });

    // ---- the signal equations in use ----
    let ir = p.ir.as_ref().map(|s| s.params);
    // P5 part A: the gradient-echo excitation angle; the model's spoiled steady state, or
    // simasl's coherent form under compat
    let ge_flip = p.ge.as_ref().map(|g| g.flip_deg);
    let tissue_steady = |m0: f64, t1: f64, t2: f64, tr: f64, se: bool| -> f64 {
        match (p.contrast, ir, ge_flip, se) {
            (Contrast::InversionRecovery, Some(q), _, false) => tissue_ir(m0, t1, tr, &q),
            (Contrast::GradientEcho, _, Some(fa), false) if p.compat.is_some() => tissue_ge_simasl(m0, t1, t2, tr, fa),
            (Contrast::GradientEcho, _, Some(fa), false) => tissue_ge_spoiled(m0, t1, tr, fa),
            _ => tissue_se(m0, t1, tr),
        }
    };
    let blood_signal = |x: f64| -> f64 {
        match (p.contrast, ir, ge_flip) {
            (Contrast::InversionRecovery, Some(q), _) => blood_ir(x, &q),
            (Contrast::GradientEcho, _, Some(fa)) => blood_ge(x, fa),
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
        // under gradient echo simasl's transverse factor is exp(-TE/T2*), with the same guard
        // (`mri_signal_filter.py:194-198`)
        let tt = if p.contrast == Contrast::GradientEcho { &ph.t2star } else { &ph.t2 };
        tt.iter().map(|&t2| if t2 != 0.0 { (-echo_time_s / t2 as f64).exp() } else { 1.0 }).collect()
    });
    let te = |i: usize| -> f64 { te_factor.as_ref().map_or(1.0, |f| f[i]) };

    // ---- tissue signal per distinct (TR, equation), per compartment, on the sim grid ----
    let mut tissue_cache: HashMap<(u64, bool), Vec<Vec<f32>>> = HashMap::new();
    let mut tissue_for = |tr: f64, se: bool| -> Vec<Vec<f32>> {
        tissue_cache
            .entry((tr.to_bits(), se))
            .or_insert_with(|| {
                let sig: Vec<f64> = match &te_factor {
                    None => (0..ph.nvox()).map(|i| tissue_steady(ph.m0[i] as f64, ph.t1[i] as f64, ph.t2[i] as f64, tr, se)).collect(),
                    Some(f) => (0..ph.nvox()).map(|i| tissue_steady(ph.m0[i] as f64, ph.t1[i] as f64, ph.t2[i] as f64, tr, se) * f[i]).collect(),
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
        let t_read = r.t + slice_offsets[z];
        slice_cache
            .entry((r.tr.to_bits(), t_read.to_bits(), pulse_set[row], z))
            .or_insert_with(|| {
                masks
                    .iter()
                    .map(|m| {
                        r_sim.mean_slice(z, |i| {
                            if !m[i] {
                                0.0
                            } else if let Some(fa) = ge_flip {
                                // P5 part A: the gradient-echo steady state, sin(a) at the readout
                                fa.to_radians().sin() * tissue_mz_ge(ph.m0[i] as f64, ph.t1[i] as f64, r.tr, t_read, s, fa)
                            } else {
                                tissue_mz(ph.m0[i] as f64, ph.t1[i] as f64, r.tr, t_read, s)
                            }
                        })
                    })
                    .collect()
            })
            .clone()
    };

    // ---- P5 part A: a gradient echo below or above 90 degrees does not saturate the slab, so
    // when the rows' preparations differ (repetition time, readout time, pulse set,
    // presaturation) the longitudinal state carries from row to row and no single steady state
    // holds. Then every row's tissue comes from the propagation, per slice (each slice has its own
    // readout time). An m0scan row has no labeling; its readout is taken at the end of its
    // repetition, so its repetition time is recovery before the readout. Not under compat:
    // simasl treats each volume as its own steady state. ----
    let prep_key = |v: usize| {
        let r = &p.rows[v];
        let presat = suppression[v].as_ref().is_some_and(|s| s.presaturation);
        (r.tr.to_bits(), r.t.to_bits(), suppression[v].is_some(), pulse_set[v], presat, r.kind == RowKind::M0scan)
    };
    let uniform_prep = (0..n).all(|v| prep_key(v) == prep_key(0));
    let ge_propagated: Option<Vec<Vec<Vec<f32>>>> = match ge_flip {
        Some(fa) if fa != 90.0 && !uniform_prep && p.compat.is_none() => {
            let sin = fa.to_radians().sin();
            let [pnx, pny, _] = ph.grid.dims;
            let pslab = pnx * pny;
            let mut out = vec![vec![vec![0.0f32; nvox_sim]; k]; n];
            for z in 0..nz {
                let preps: Vec<Prep> = p.rows.iter().enumerate().map(|(v, r)| Prep {
                    tr: r.tr,
                    t_read: if r.kind == RowKind::M0scan { r.tr } else { r.t + slice_offsets[z] },
                    s: suppression[v].as_ref(),
                }).collect();
                // the sequence per phantom voxel of this slice's slab, computed once
                let cells: Vec<usize> = r_sim.z_slab(z).iter().map(|&(zs, _)| zs).collect();
                let local = |i: usize| -> usize {
                    let pos = cells.iter().position(|&zs| zs == i / pslab).expect("a voxel of this slab");
                    pos * pslab + i % pslab
                };
                let mut seq = vec![vec![0.0f64; cells.len() * pslab]; n];
                for (pos, &zs) in cells.iter().enumerate() {
                    for xy in 0..pslab {
                        let i = zs * pslab + xy;
                        if ph.dseg[i] <= 0 {
                            continue;
                        }
                        for (v, mz) in tissue_mz_ge_sequence(ph.m0[i] as f64, ph.t1[i] as f64, &preps, fa).into_iter().enumerate() {
                            seq[v][pos * pslab + xy] = sin * mz;
                        }
                    }
                }
                for v in 0..n {
                    for (c, m) in masks.iter().enumerate() {
                        let sl = r_sim.mean_slice(z, |i| if m[i] { seq[v][local(i)] } else { 0.0 });
                        out[v][c][z * snx * sny..(z + 1) * snx * sny].copy_from_slice(&sl);
                    }
                }
            }
            Some(out)
        }
        _ => None,
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
            let t = row.t + slice_offsets[z];
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
    // P5 part D: per (volume, shot) physiological factors (tissue, label) of a 3D series
    let n_shots = res3d.as_ref().map_or(1, |r| r.n_shots);
    let mut shot_physio: Vec<Vec<(f64, f64)>> = vec![vec![(1.0, 1.0); n_shots]; n];
    let [pnx, pny, _] = ph.grid.dims;
    let pslab = pnx * pny;
    for (v, row) in p.rows.iter().enumerate() {
        // An m0scan row is a plain spin-echo readout (P3), except under compat, where it takes
        // the series' equation as simasl's does (P2 addendum, part A), and under gradient echo,
        // whose M0 is the same excitation and readout without labeling (P5 addendum, part A).
        let se = row.kind == RowKind::M0scan && p.compat.is_none() && p.contrast != Contrast::GradientEcho;
        let mut tissue = match (row.kind, &suppression[v]) {
            (RowKind::Deltam, _) => None,
            _ if ge_propagated.is_some() => ge_propagated.as_ref().map(|g| g[v].clone()),
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
        if let (Some(phys), Some(_)) = (&p4.physio, &res3d) {
            // P5 part D: per shot, kept out of the images and applied as line weights; the shot
            // labels over its own window and excites at its own time
            for (s, sp) in shot_physio[v].iter_mut().enumerate() {
                let start = p.row_start[v] + s as f64 * row.tr;
                let time = start + row.t;
                let tf = phys.tissue_factor(time);
                let (lf, means) = match p.label_type {
                    LabelType::Pasl => phys.label_factor_at(start),
                    _ => phys.label_factor_window(start, start + row.tau),
                };
                let win = match p.label_type {
                    LabelType::Pasl => (start, start),
                    _ => (start, start + row.tau),
                };
                *sp = (tf, lf);
                physio_lines.push(PhysioLine {
                    volume: v, slice: s, time,
                    cardiac_phase: phys.cardiac.phase(time), respiratory_phase: phys.respiratory.phase(time),
                    drift: phys.drift.value(time), tissue_factor: tf,
                    label_window: win, label_means: means, label_factor: lf,
                });
            }
        } else if let Some(phys) = &p4.physio {
            let (lf, win, means) = p4.label_physio[v];
            for z in 0..nz {
                let time = p.row_start[v] + row.t + slice_offsets[z];
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
        // every row that carries label (the test controls can give a control row label), the
        // ground truth only for label and deltam rows
        if p4.label_path && (wants_gt || blood_sign(row.kind, ov) != 0.0) {
            let kin = p.kinetic(row);
            // in 3D the physiological label factor is a per-shot line weight instead (P5 part D)
            let lf_row = if res3d.is_some() { 1.0 } else { p4.label_physio[v].0 };
            let sign0 = blood_sign(row.kind, ov) * factor * lf_row;
            let bolus = bolus_region.map(|region| {
                let s = p.suppression.as_ref().unwrap().for_row(v);
                (region, s.pulse_times, s.epsilon)
            });
            let mut partitions: HashMap<u64, Vec<(f64, f64, f64)>> = HashMap::new();
            let blood_target = |c: usize| if ov == RowOverride::BloodIntoTissue0 && row.kind == RowKind::Label { 0 } else { k + c };
            let ev_target = |c: usize| {
                if ov == RowOverride::ExtravascularIntoBlood && row.kind == RowKind::Label {
                    k + c
                } else if ev_group && !(ov == RowOverride::ExtravascularIntoTissue && row.kind == RowKind::Label) {
                    ev_base + c
                } else {
                    c
                }
            };
            for z in 0..nz {
                let zs = r_sim.z_slab(z);
                let (Some(zlo), Some(zhi)) = (zs.first().map(|p| p.0), zs.last().map(|p| p.0)) else { continue };
                let base = pslab * zlo;
                let len = pslab * (zhi - zlo + 1);
                let t = row.t + slice_offsets[z];
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
                    if let (true, Some(gt)) = (wants_gt, gt.as_mut()) {
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
        if res3d.is_some() {
            // P5 part D: a shot is a readout segment; the events are drawn per shot as in 2D, but
            // act through per-shot images and shot gains below, not on slice groups
            events = draw_events(m.within.as_ref(), n, n_shots, seed);
        } else {
            let n_shots = slice_schedule(nz, p.mb, p.mb_interleaved).len();
            events = draw_events(m.within.as_ref(), n, n_shots, seed);
            if !events.is_empty() {
                dropped = apply_multiband_motion(
                    &mut images, sim_grid.dims, n, v2w, p.mb, p.mb_interleaved, &DropoutLaw::Uniform, &events,
                );
            }
        }
    }
    // P5 part D: a 3D volume's shots. Each event's jump persists for the later shots of its volume
    // (as apply_multiband_motion composes them); the shots sharing a pose other than the volume's
    // see the volume's images moved by it. Each event shot's lines are attenuated by
    // 1 - severity (DropoutLaw::Uniform), a shot gain.
    let mut shot_gain = vec![vec![1.0f64; n_shots]; n];
    let mut shot_sets: Vec<Vec<ShotSet>> = vec![Vec::new(); n];
    if res3d.is_some() && !events.is_empty() {
        let v2w = sim_grid.voxel_to_world;
        for (g, sets) in shot_sets.iter_mut().enumerate() {
            let evs: Vec<&MotionEvent> = events.iter().filter(|e| e.volume == g && e.shot < n_shots).collect();
            if evs.is_empty() {
                continue;
            }
            let mut cum = Pose::IDENTITY;
            let mut pose_of = vec![Pose::IDENTITY; n_shots];
            for (s, pose) in pose_of.iter_mut().enumerate() {
                for e in evs.iter().filter(|e| e.shot == s) {
                    for i in 0..3 {
                        cum.trans_mm[i] += e.jump_mm[i];
                        cum.rot_deg[i] += e.jump_deg[i];
                    }
                }
                *pose = cum;
            }
            let mut distinct: Vec<Pose> = Vec::new();
            for &q in &pose_of {
                if q != Pose::IDENTITY && !distinct.contains(&q) {
                    distinct.push(q);
                }
            }
            for q in distinct {
                let shots: Vec<usize> = (0..n_shots).filter(|&s| pose_of[s] == q).collect();
                let moved: Vec<Vec<f32>> = images.iter().map(|img| {
                    let vol: Vec<f32> = (0..nvox_sim).map(|vox| img[vox * n + g]).collect();
                    mrsim_acq::motion::resample_by_pose(&vol, sim_grid.dims, v2w, q)
                }).collect();
                sets.push(ShotSet { shots, images: moved });
            }
            for e in &evs {
                let atten = DropoutLaw::Uniform.attenuation(g, e.severity);
                shot_gain[g][e.shot] *= atten as f64;
                dropped.push(DroppedShot { volume: g, shot: e.shot, slices: Vec::new(), attenuation: atten });
            }
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

    // the separate M0's tissue: a plain readout at its own repetition time (P5 part A: or the
    // gradient-echo steady state)
    let m0_images = match p.m0_type {
        M0Type::Separate => {
            let tr = p.m0_repetition_time_s.ok_or("M0Type Separate without an M0 repetition time")?;
            let tissue = tissue_for(tr, p.contrast != Contrast::GradientEcho);
            let mut imgs: Vec<Vec<f32>> = vec![vec![0.0f32; nvox_sim]; ncomp];
            for c in 0..k {
                imgs[c].copy_from_slice(&tissue[c]);
            }
            Some(imgs)
        }
        _ => None,
    };
    let ge_propagated_some = ge_propagated.is_some();
    Ok(Built {
        acq_grid, sim_grid, n, nvox_sim, images, gt_static, gt_moved, gt_iv, gt_sup, gt_art, physio_lines, shot_physio, shot_gain, shot_sets, n_shots, events, dropped, poses, motion_seed, res3d, acq, fmap_sim, relax, mode_used, k, ncomp, ev_group, macro_on, t2_arterial_ms, t2_blood_ms, acq_t2_ms, acq_t2p_ms, acq_t1_ms, needs_t1, r_acq, m0_acq, compat_facts, p4, label_factors, ge_flip, ge_propagated_some, m0_images,
    })
}

/// The series of a protocol with a P6 feature: [`build`], then the acquisition and the output.
pub(super) fn simulate_p6(
    p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride, capture: Option<&mut Vec<Vec<f32>>>,
) -> Result<SeriesOutput, String> {
    let tes = p.echo_times_s.clone();
    let Built {
        acq_grid, sim_grid, n, nvox_sim, images, gt_static, gt_moved, gt_iv, gt_sup, gt_art, physio_lines, shot_physio, shot_gain, shot_sets, n_shots, events, dropped, poses, motion_seed, res3d, acq, fmap_sim, relax, mode_used, k, ncomp, ev_group, macro_on, t2_arterial_ms, t2_blood_ms, acq_t2_ms, acq_t2p_ms, acq_t1_ms, needs_t1, r_acq, m0_acq, compat_facts, p4, label_factors, ge_flip, ge_propagated_some, m0_images,
    } = build(p, ph, mode, ov, tes[0])?;
    // P6 part C: every echo's readout block on explicit intervals, on the acquired grid
    if tes.len() > 1 {
        check_excitation_timing(p, acq_grid.dims)?;
    }
    // P6 part C, compat: the echo-time decay is the signal stage's, so each echo has its own image
    // set, bounded before it is built
    let mut echo_images: Vec<Vec<Vec<f32>>> = Vec::new();
    if p.compat.is_some() && tes.len() > 1 {
        let bytes = 4.0 * tes.len() as f64 * ncomp as f64 * nvox_sim as f64 * n as f64;
        let limit = p.multi_te.as_ref().and_then(|m| m.max_image_memory_gib).map_or(4.0, |l| l.0);
        if bytes > limit * (1u64 << 30) as f64 {
            return Err(format!(
                "compat multi-TE needs {:.2} GiB of per-echo input images ({} echoes x {ncomp} compartments x {nvox_sim} \
                 voxels x {n} volumes x 4 bytes), over the limit of {limit} GiB (overlay multi_te.max_image_memory_gib)",
                bytes / (1u64 << 30) as f64, tes.len()));
        }
        for &te in &tes[1..] {
            echo_images.push(build(p, ph, mode, ov, te)?.images);
        }
    }

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
            if ev_group {
                // the extravascular label relaxes as the tissue it is in
                for i in 0..k {
                    t2v.push(T2Volume::Uniform(t2_ms[i]));
                    tiv.push(T2Volume::Uniform(t2p_ms[i]));
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
            if ev_group {
                t2v.push(T2Volume::Map(t2m));
                tiv.push(T2Volume::Map(tpm));
            }
            (t2v, tiv)
        }
    };
    let t1b_ms = (p.t1b.0 * 1000.0) as f32;
    let t1_vols: Option<Vec<T2Volume>> = needs_t1.then(|| {
        let tissue: Vec<T2Volume> = match &relax {
            Relaxation::Class { .. } => ph.labels.iter().map(|(l, _)| {
                let i = ph.dseg.iter().position(|d| d == l).expect("labels come from dseg");
                T2Volume::Uniform(ph.t1[i] * 1000.0)
            }).collect(),
            Relaxation::Voxel { .. } => vec![T2Volume::Map(acq_t1_ms.as_deref().unwrap())],
        };
        let mut v = tissue.clone();
        v.extend(std::iter::repeat_n(T2Volume::Uniform(t1b_ms), k));
        if macro_on {
            v.extend(std::iter::repeat_n(T2Volume::Uniform(t1b_ms), k));
        }
        if ev_group {
            v.extend(tissue);
        }
        v
    });


    if let Some(c) = capture {
        *c = images.clone();
    }

    // ---- the one call ----
    let eddy_drive = vec![None; n];
    let prep_drive = vec![None; n];
    // P5 part D: per (volume, shot, compartment) line weights, present when physiology or a dropout
    // event is: the tissue's factor on the tissue group, the label's on the blood, arterial and
    // extravascular-label groups, times the shot gain
    let line_weights = (res3d.is_some() && (p4.physio.is_some() || shot_gain.iter().flatten().any(|g| *g != 1.0)))
        .then(|| {
            let mut w = Vec::with_capacity(n * n_shots * ncomp);
            for g in 0..n {
                for s in 0..n_shots {
                    let (tf, lf) = shot_physio[g][s];
                    for c in 0..ncomp {
                        w.push(if c < k { tf } else { lf } * shot_gain[g][s]);
                    }
                }
            }
            LineWeights { n_shots, n_compartments: ncomp, w }
        });
    // P5 part C: a spiral's time segmentation, certified per slice before the acquisition runs it
    // (an uncertifiable rate rectangle is an error here, not a panic there)
    #[cfg(feature = "kspace")]
    let spiral_segmentation = match &res3d {
        Some(r3) if r3.spiral.is_some() => Some(mrsim_acq::kspace3d::spiral_segmentation(
            sim_grid.dims, acq_grid.dims, &t2_vols, t1_vols.as_deref(), &fmap_sim, Some(&ti_vols), &acq, &r3.train,
            &r3.readout,
        )?),
        _ => None,
    };
    #[cfg(not(feature = "kspace"))]
    let spiral_segmentation = None;
    let echo_ms: Vec<f64> = tes.iter().map(|te| te * 1000.0).collect();
    let mut more_mag: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let (mag, phase_out) = match &res3d {
        None if tes.len() > 1 => {
            let per: Vec<&[Vec<f32>]> = (0..tes.len())
                .map(|e| if e == 0 || echo_images.is_empty() { &images[..] } else { &echo_images[e - 1][..] })
                .collect();
            let mut all = simulate_acquisition_echoes(
                sim_grid.dims, acq_grid.dims, n, &per, &t2_vols, &fmap_sim, Some(&ti_vols), &acq, &echo_ms,
                &eddy_drive, &prep_drive, phase, p.seed, None, None,
            );
            more_mag = all.split_off(1);
            all.pop().expect("one echo at least")
        }
        None => simulate_acquisition_oversampled(
            sim_grid.dims, acq_grid.dims, n, &images, &t2_vols, &fmap_sim, Some(&ti_vols), &acq,
            &eddy_drive, &prep_drive, phase, p.seed, None, None,
        ),
        Some(r3) => simulate_acquisition_3d(
            sim_grid.dims, acq_grid.dims, n, &images, &t2_vols, t1_vols.as_deref(), &fmap_sim, Some(&ti_vols), &acq,
            &r3.train, &r3.readout, line_weights.as_ref(), shot_sets.iter().any(|s| !s.is_empty()).then_some(&shot_sets[..]),
            phase, p.seed,
        ),
    };
    drop(images);
    drop(echo_images);

    // ---- the separate M0 scan: a plain spin-echo readout at its own TR ----
    let m0_seed = (p.m0_type == M0Type::Separate).then_some(p.seed ^ M0_SEED_SALT);
    let mut more_m0: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
    let m0 = match (m0_seed, &m0_images) {
        (Some(seed), Some(imgs)) => Some(match &res3d {
            None if tes.len() > 1 => {
                let per: Vec<&[Vec<f32>]> = vec![&imgs[..]; tes.len()];
                let mut all = simulate_acquisition_echoes(
                    sim_grid.dims, acq_grid.dims, 1, &per, &t2_vols, &fmap_sim, Some(&ti_vols), &acq, &echo_ms,
                    &[None], &[None], phase, seed, None, None,
                );
                more_m0 = all.split_off(1);
                all.pop().expect("one echo at least")
            }
            None => simulate_acquisition_oversampled(
                sim_grid.dims, acq_grid.dims, 1, imgs, &t2_vols, &fmap_sim, Some(&ti_vols), &acq,
                &[None], &[None], phase, seed, None, None,
            ),
            // the same readout and train, no labeling, no physiology, no motion (P5 part B)
            Some(r3) => simulate_acquisition_3d(
                sim_grid.dims, acq_grid.dims, 1, imgs, &t2_vols, t1_vols.as_deref(), &fmap_sim, Some(&ti_vols),
                &acq, &r3.train, &r3.readout, None, None, phase, seed,
            ),
        }),
        (Some(_), None) => return Err("M0Type Separate without its images".to_string()),
        _ => None,
    };

    // ---- P5 part B: the echo amplitudes the sidecar records, per label and for the blood ----
    let echo_amplitudes = res3d.as_ref().map(|r3| {
        let tr = &r3.train;
        let amp = |t1_ms: f64, t2_ms: f64| -> Vec<f64> {
            mrsim_acq::epg::epg_cpmg(tr.etl, tr.esp_ms, tr.refocusing_deg, t1_ms, t2_ms).iter().map(|l| l.exp()).collect()
        };
        let mut v: Vec<(String, Vec<f64>)> = ph.labels.iter().map(|(l, name)| {
            let i = ph.dseg.iter().position(|d| d == l).expect("labels come from dseg");
            (name.clone(), amp(ph.t1[i] as f64 * 1000.0, ph.t2[i] as f64 * 1000.0))
        }).collect();
        v.push(("blood".to_string(), amp(p.t1b.0 * 1000.0, t2_blood_ms as f64)));
        v
    });

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
        acq_t1_ms: acq_t1_ms.clone(),
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
        readout: res3d.clone(),
        echo_amplitudes,
        spiral_segmentation,
        more_echoes: more_mag.into_iter().enumerate().map(|(e, (mag, phase))| EchoSeries {
            echo_time_s: tes[e + 1], mag, phase, m0: more_m0.get(e).cloned(),
        }).collect(),
        ge_rule: ge_flip.map(|fa| match (p.compat.is_some(), fa == 90.0, ge_propagated_some, p.suppression.is_some()) {
            (true, ..) => "simasl's coherent steady state per volume (compat)",
            (_, true, ..) => "90 degrees: the slab is saturated, each row independent (P3's timeline, sin(a) = 1)",
            (_, _, true, _) => "state propagated row to row (the rows' preparations differ)",
            (_, _, _, true) => "spoiled steady state of the repeated preparation (fixed point of the suppression timeline)",
            _ => "spoiled steady state (closed form)",
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phantom::load;
    use crate::protocol::{parse_echoes, Overlay};
    use serde_json::{json, Value};

    /// The crop in voxel mode with per-voxel T2 and T2* and a fieldmap gradient.
    fn heterogeneous_crop() -> Phantom {
        let mut ph = load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap();
        let [nx, _, _] = ph.grid.dims;
        for i in 0..ph.dseg.len() {
            let w = 1.0 + 0.2 * (i as f32 * 0.37).sin();
            ph.t2[i] *= w;
            ph.t2star[i] *= w * 0.9;
        }
        ph.fieldmap = Some((0..ph.dseg.len()).map(|i| 4.0 * ((i % nx) as f32 / nx as f32 - 0.5)).collect());
        ph
    }

    fn protocol(tes: &[f64], ge: bool) -> Protocol {
        let sidecars: Vec<Value> = tes.iter().map(|te| json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0,
            "EchoTime": te, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.07], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.012
        })).collect();
        let contrast = if ge { "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n" } else { "" };
        let ov: Overlay = toml::from_str(&format!("seed = 4\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n{contrast}")).unwrap();
        parse_echoes(&sidecars, "volume_type\ncontrol\nlabel\n", Some(&ov), None).unwrap()
    }

    /// Where relaxation varies within the object (voxel mode) the echoes are not scalar multiples:
    /// echo 2 is the acquisition at echo 1's timing of the object reweighted per voxel by
    /// exp(-dTE/T2) (spin echo, any fieldmap) or exp(-dTE (1/T2 + 1/T2')) (gradient echo, no
    /// fieldmap: its TE-dependent fieldmap phase is not a real image weight).
    #[test]
    fn a_later_echo_is_the_reweighted_object() {
        for ge in [false, true] {
            let mut ph = heterogeneous_crop();
            if ge {
                ph.fieldmap = None;
            }
            let tes = [0.015, 0.030];
            let p = protocol(&tes, ge);
            let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
            let out = simulate_p6(&p, &ph, T2Mode::Voxel, &phase, RowOverride::None, None).unwrap();
            let echo2 = complex_from(&out.more_echoes[0].mag, &out.more_echoes[0].phase);

            let b = build(&p, &ph, T2Mode::Voxel, RowOverride::None, tes[0]).unwrap();
            let (t2m, tpm) = (b.acq_t2_ms.as_deref().unwrap(), b.acq_t2p_ms.as_deref().unwrap());
            assert_eq!(b.images.len(), 2, "voxel mode: tissue and blood");
            let dte_ms = (tes[1] - tes[0]) * 1000.0;
            let mut images = b.images.clone();
            for (c, img) in images.iter_mut().enumerate() {
                for vox in 0..b.nvox_sim {
                    let t2 = if c == 0 { t2m[vox] as f64 } else { b.t2_blood_ms as f64 };
                    let rate = if ge { 1.0 / t2 + 1.0 / tpm[vox] as f64 } else { 1.0 / t2 };
                    let w = (-dte_ms * rate).exp() as f32;
                    for v in 0..b.n {
                        img[vox * b.n + v] *= w;
                    }
                }
            }
            let t2v = [T2Volume::Map(t2m), T2Volume::Uniform(b.t2_blood_ms)];
            let tiv = [T2Volume::Map(tpm), T2Volume::Map(tpm)];
            let (m, ph_out) = simulate_acquisition_oversampled(
                b.sim_grid.dims, b.acq_grid.dims, b.n, &images, &t2v, &b.fmap_sim, Some(&tiv), &b.acq,
                &vec![None; b.n], &vec![None; b.n], &phase, p.seed, None, None,
            );
            let want = complex_from(&m, &ph_out);
            let peak = want.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
            let worst = want.iter().zip(&echo2).map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1)).fold(0.0f64, f64::max) / peak;
            println!("ge {ge}: echo 2 vs the reweighted object at echo 1, worst {worst:e}");
            assert!(worst < 1e-5, "ge {ge}: {worst}");
            // and it is not a scalar multiple of echo 1 (the test would be vacuous)
            let echo1 = complex_from(&out.mag, &out.phase);
            let ratios: Vec<f64> = echo1.iter().zip(&echo2)
                .filter(|(a, _)| a.0.hypot(a.1) > 0.2 * peak)
                .map(|(a, b)| b.0.hypot(b.1) / a.0.hypot(a.1)).collect();
            let (lo, hi) = ratios.iter().fold((f64::MAX, f64::MIN), |(l, h), r| (l.min(*r), h.max(*r)));
            assert!(hi - lo > 0.01, "ge {ge}: ratios {lo}..{hi}");
        }
    }
}
