//! The series of a protocol with a P6 feature (P6 addendum; the legacy dispatch is in the parent).
//!
//! The P1-P5 body of [`super::simulate_legacy`] copied, never shared, so that a P6 change cannot move a
//! byte of a legacy output: [`build`] is everything up to the acquisition (the compartment images,
//! the ground truth, the separate M0's tissue), run once per echo time where the echo time enters the
//! images (compat's `exp(-TE/T2)`), and [`simulate_p6`] the acquisition and the output. Part C
//! (multi-TE): every echo is read from the same excitation by `simulate_acquisition_echoes`.

use super::*;
use mrsim_acq::kspace::simulate_acquisition_echoes;
use crate::kinetic::{delta_m_read, delta_m_read_parts, Kinetic, ReadParts};
use crate::longitudinal::{ll_legacy_dispatch, tissue_mz_ll_series, LlCycle};
use crate::protocol::{check_excitation_timing, HadamardSpec};
use crate::schedule::{Output, Schedule};

/// [`P4::new`] with its row-indexed parts on the schedule's raw volumes: the crushing survival
/// (each raw volume's VENC), the physiology horizon and the per-volume label factors and windows
/// (each from the raw volume's first preparation). On the identity schedule these are P4::new's
/// own expressions on the same values.
fn p4_for_schedule(p: &Protocol, ph: &Phantom, bolus_region: Option<Region>, sched: &Schedule) -> Result<P4, String> {
    let mut p4 = P4::new(p, ph, bolus_region)?;
    let n = sched.raw_rows.len();
    let start = |v: usize| sched.preps[sched.raws[v].prep].start_s;
    p4.crush = match (&p.crushing, p.macrovascular.is_some()) {
        (Some(c), true) => {
            let t = c.arterial_velocity.as_ref().expect("protocol requires it with part B");
            let vel: Vec<f64> = ph.labels.iter().map(|(_, nm)| t[nm]).collect();
            Some((0..n).map(|v| {
                let venc = sched.raws[v].venc_with(&sched.preps).expect("crushing is on");
                vel.iter().map(|&vm| survival(vm, venc)).collect()
            }).collect())
        }
        _ => None,
    };
    p4.physio = p.physio.map(|params| {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0) as f64;
        let horizon = (0..n).last().map(start).unwrap_or(0.0) + shots * sched.raw_rows.last().map_or(0.0, |r| r.tr) + 1.0;
        Physio::new(params, p.seed, horizon)
    });
    p4.label_physio = (0..n)
        .map(|v| {
            let t0 = start(v);
            match &p4.physio {
                None => (1.0, (t0, t0), [0.0; 3]),
                Some(phys) => {
                    let row = &sched.raw_rows[v];
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
    Ok(p4)
}

/// Raw volume `r` of an `n`-volume voxel-major series as complex `f64`, as `complex_from` forms it.
/// A Hadamard cycle's raw volumes by readout (P7 part B): readout `n`'s `H` raw volumes in
/// encoding-row order (the schedule lays a cycle out encoding-row-major, `M` readouts each).
/// Without Look-Locker one group, the cycle's raw volumes in order: P6's.
fn readout_groups(c: &crate::schedule::Cycle, order: usize) -> Vec<Vec<usize>> {
    let m = c.raws.len() / order;
    (0..m).map(|n| (0..order).map(|e| c.raws.start + e * m + n).collect()).collect()
}

/// The Look-Locker spec on the schedule's raw volumes (P7 part B). Without Hadamard the raw
/// volumes are the rows and this is the protocol's spec itself. Under Hadamard each preparation's
/// raw volumes are one cycle (an m0scan raw volume its own), each readout's flip that of sub-bolus
/// 1 of its row (the protocol checked they agree).
fn ll_on_raws(p: &Protocol, sched: &Schedule) -> Option<crate::protocol::LookLockerSpec> {
    let l = p.look_locker.as_ref()?;
    let Some(h) = &p.hadamard else { return Some(l.clone()) };
    let mut cycles: Vec<crate::protocol::LookLockerCycle> = Vec::new();
    let mut flip_deg = vec![0.0; sched.raws.len()];
    for (r, rv) in sched.raws.iter().enumerate() {
        if rv.readout == 0 {
            cycles.push(crate::protocol::LookLockerCycle { rows: Vec::new(), m0scan: rv.encoding_row.is_none() });
        }
        cycles.last_mut().expect("a cycle starts at readout 0").rows.push(r);
        flip_deg[r] = match (rv.cycle, rv.encoding_row) {
            (Some(c), Some(_)) => l.flip_deg[h.row(&h.cycles[c], 0, rv.readout)],
            // an m0scan raw volume: its source row is its preparation's
            _ => l.flip_deg[sched.preps[rv.prep].suppression],
        };
    }
    Some(crate::protocol::LookLockerSpec { cycles, flip_deg, ..l.clone() })
}

/// One complex volume, voxel-major.
type Image = Vec<(f64, f64)>;

fn complex_volume(mag: &[f32], phase: &[f32], n: usize, r: usize) -> Image {
    (0..mag.len() / n).map(|x| {
        let (m, p) = (mag[x * n + r] as f64, phase[x * n + r] as f64);
        (m * p.cos(), m * p.sin())
    }).collect()
}

/// Decode a raw series (`n_raw` volumes) to the outputs (P6 part A): each cycle's sub-boli by
/// `hadamard::decode` on the complex images in `f64`, `m0scan` raw volumes passed through as
/// acquired. Returns the outputs' magnitude and phase, voxel-major.
fn decode_series(sched: &Schedule, order: usize, mag: &[f32], phase: &[f32], n_raw: usize) -> (Vec<f32>, Vec<f32>) {
    let nvox = mag.len() / n_raw;
    let n_out = sched.outputs.len();
    // per cycle, per readout (P7 part B: one group without Look-Locker), the decoded sub-boli
    let decoded: Vec<Vec<Vec<Image>>> = sched.cycles.iter().map(|c| {
        readout_groups(c, order).iter().map(|g| {
            let imgs: Vec<Vec<(f64, f64)>> = g.iter().map(|&r| complex_volume(mag, phase, n_raw, r)).collect();
            let refs: Vec<&[(f64, f64)]> = imgs.iter().map(|v| v.as_slice()).collect();
            crate::hadamard::decode(&refs, order)
        }).collect()
    }).collect();
    let (mut om, mut op) = (vec![0.0f32; nvox * n_out], vec![0.0f32; nvox * n_out]);
    for (k, o) in sched.outputs.iter().enumerate() {
        match *o {
            Output::Raw(r) => {
                for x in 0..nvox {
                    om[x * n_out + k] = mag[x * n_raw + r];
                    op[x * n_out + k] = phase[x * n_raw + r];
                }
            }
            Output::Decoded { cycle, subbolus, readout } => {
                for (x, &(re, im)) in decoded[cycle][readout][subbolus].iter().enumerate() {
                    om[x * n_out + k] = re.hypot(im) as f32;
                    op[x * n_out + k] = im.atan2(re) as f32;
                }
            }
        }
    }
    (om, op)
}

/// The ideal decoded truth (P6 part A, "Ground truth"), static on the acquisition grid, one volume
/// per output: sub-bolus `j`'s `delta_m_sub` from the kinetics alone at the raw readout time
/// (with its intravascular part, its bolus-position pulse factors and its arterial parcels, where
/// those parts are on); zero for `m0scan` outputs.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn decoded_truth(
    p: &Protocol, ph: &Phantom, sched: &Schedule, h: &HadamardSpec, r_acq: &Resampler, slice_offsets: &[f64], p4: &P4,
    bolus_region: Option<Region>,
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let [nx, ny, nz] = r_acq.dst_dims;
    let (slab, n_out) = (nx * ny, sched.outputs.len());
    let mut dm = vec![0.0f32; slab * nz * n_out];
    let mut iv = p.exchange_time.map(|_| vec![0.0f32; slab * nz * n_out]);
    let mut sup = bolus_region.map(|_| vec![0.0f32; slab * nz * n_out]);
    let mut art = p4.abv.as_ref().map(|_| vec![0.0f32; slab * nz * n_out]);
    for (k, o) in sched.outputs.iter().enumerate() {
        let Output::Decoded { cycle, subbolus, readout } = *o else { continue };
        let r0 = readout_groups(&sched.cycles[cycle], h.order)[readout][0];
        let row = &sched.raw_rows[r0];
        let kin = p.kinetic(row);
        let (aj, bj) = h.spans[subbolus];
        let pulses = p.suppression.as_ref().map(|s| s.for_row(sched.preps[sched.raws[r0].prep].suppression));
        for z in 0..nz {
            let t = row.t + slice_offsets[z];
            let put = |out: &mut Vec<f32>, f: &dyn Fn(usize) -> f64| {
                let sl = r_acq.mean_slice(z, |i| if ph.dseg[i] > 0 { f(i) } else { 0.0 });
                for (jj, x) in sl.iter().enumerate() {
                    out[(z * slab + jj) * n_out + k] = *x;
                }
            };
            let vox = |i: usize| (ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64);
            put(&mut dm, &|i| {
                let (f, att, t1, m0) = vox(i);
                delta_m_sub(&kin, f, att, t1, m0, t, aj, bj)
            });
            if let (Some(out), Some(te)) = (iv.as_mut(), p.exchange_time) {
                put(out, &|i| {
                    let (f, att, t1, m0) = vox(i);
                    delta_m_iv_sub(&kin, f, att, t1, m0, t, aj, bj, te)
                });
            }
            if let (Some(out), Some(region), Some(s)) = (sup.as_mut(), bolus_region, pulses.as_ref()) {
                put(out, &|i| {
                    let (f, att, t1, m0) = vox(i);
                    subbolus_factors(&s.pulse_times, s.epsilon, kin.tau, entry_offset(p.label_type, region, att))
                        .iter()
                        .map(|&(a, b, fac)| {
                            let (lo, hi) = (a.max(aj), b.min(bj));
                            if hi > lo { fac * delta_m_sub(&kin, f, att, t1, m0, t, lo, hi) } else { 0.0 }
                        })
                        .sum()
                });
            }
            if let (Some(out), Some(abv), Some(aatt)) = (art.as_mut(), &p4.abv, &p4.aatt) {
                put(out, &|i| {
                    let (va, a) = arterial_dm(&kin, abv[i], aatt[i], ph.m0[i] as f64, t);
                    if a.is_some_and(|a| aj <= a && a < bj) { va } else { 0.0 }
                });
            }
        }
    }
    (dm, iv, sup, art)
}


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
    slice_offsets: Vec<f64>,
    /// P6 part A: the tissue alone (moved, events applied) and its shot sets, for the leakage.
    tissue_only: Option<Vec<Vec<f32>>>,
    tissue_shot_sets: Vec<Vec<ShotSet>>,
    /// P6 part A: per cycle, the unsuppressed tissue steady state at its repetition time.
    m0_ref_images: Vec<Vec<Vec<f32>>>,
    /// P6 part B: the depleted-read truth and the per-readout table.
    gt_read: Option<Vec<f32>>,
    ll_lines: Vec<LlLine>,
    /// P7 part A: the read's intravascular and extravascular parts (exchange) and the arterial
    /// read (the arterial compartment), each times `sin(a_n)` as `gt_read` is.
    gt_read_iv: Option<Vec<f32>>,
    gt_read_ev: Option<Vec<f32>>,
    gt_read_art: Option<Vec<f32>>,
}

/// The P1-P5 series body up to the acquisition, with `echo_time_s` in place of the protocol's
/// `EchoTime` where the images depend on it (compat's per-voxel `exp(-TE/T2)`).
fn build(p: &Protocol, ph: &Phantom, mode: T2Mode, ov: RowOverride, echo_time_s: f64, sched: &Schedule)
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
    // P6 part B: Look-Locker readouts. One readout per cycle at one flip is P5's series, which this
    // body already is (the legacy dispatch); every other Look-Locker series takes the readout
    // timeline for the tissue and the depleted label for the blood, each readout its own flip.
    // P7 part B: on the raw volumes (under Hadamard they are not the rows)
    let ll_raw = ll_on_raws(p, sched);
    let ll = ll_raw.as_ref().filter(|l| !ll_legacy_dispatch(l.cycles.iter().map(|c| c.rows.len()), l.flip_array));
    let tissue_steady = |m0: f64, t1: f64, t2: f64, tr: f64, se: bool| -> f64 {
        match (p.contrast, ir, ge_flip, se) {
            (Contrast::InversionRecovery, Some(q), _, false) => tissue_ir(m0, t1, tr, &q),
            (Contrast::GradientEcho, _, Some(fa), false) if p.compat.is_some() => tissue_ge_simasl(m0, t1, t2, tr, fa),
            (Contrast::GradientEcho, _, Some(fa), false) => tissue_ge_spoiled(m0, t1, tr, fa),
            _ => tissue_se(m0, t1, tr),
        }
    };
    let blood_signal = |v: usize, x: f64| -> f64 {
        match (p.contrast, ir, ge_flip) {
            (Contrast::InversionRecovery, Some(q), _) => blood_ir(x, &q),
            (Contrast::GradientEcho, _, Some(fa)) => blood_ge(x, ll.map_or(fa, |l| l.flip_deg[v])),
            _ => blood_se(x),
        }
    };
    // Per-row suppression: the pulse set (deduplicated so the tissue cache can key on it) and
    // the blood factor. Rows without events take the P1 steady-state path.
    // the rows simulated are the schedule's raw volumes (the input rows on the identity schedule)
    let rows = &sched.raw_rows;
    let n = rows.len();
    let first_prep = |v: usize| &sched.preps[sched.raws[v].prep];
    // P6 part A: an encoded raw volume's labeling weights per sub-bolus, with their spans
    let enc_w: Vec<Option<Vec<u8>>> = match &p.hadamard {
        Some(h) => {
            let e = crate::hadamard::encoding(h.order);
            sched.raws.iter().map(|r| r.encoding_row.map(|i| crate::hadamard::weights(&e[i]))).collect()
        }
        None => vec![None; n],
    };
    let spans: &[(f64, f64)] = p.hadamard.as_ref().map_or(&[], |h| &h.spans[..]);
    // the encoded kinetic sum, or the row's own delta_m
    let dm_of = |v: usize, kin: &Kinetic, f: f64, att: f64, t1t: f64, m0: f64, t: f64| -> f64 {
        match &enc_w[v] {
            None => delta_m(kin, f, att, t1t, m0, t),
            Some(w) => spans.iter().zip(w).filter(|(_, &w)| w == 1)
                .map(|(&(a, b), _)| delta_m_sub(kin, f, att, t1t, m0, t, a, b)).sum(),
        }
    };
    // P6 part B: a Look-Locker readout's excitation times in its cycle up to it (this slice's) and
    // the flips of the earlier ones, for kinetic::delta_m_read; None outside Look-Locker
    let ll_read = |v: usize, off: f64| -> Option<(Vec<f64>, Vec<f64>)> {
        let l = ll?;
        if rows[v].kind == RowKind::M0scan {
            return None;
        }
        // the cycle's first readout: the first raw volume of v's preparation
        let r0 = sched.preps[sched.raws[v].prep].raw;
        Some(((r0..=v).map(|r| rows[r].t + off).collect(), (r0..v).map(|r| l.flip_deg[r]).collect()))
    };
    // the encoded weight of the parcel labeled at `a` (the arterial compartment's single parcel)
    let weight_at = |v: usize, a: f64| -> f64 {
        match &enc_w[v] {
            None => 1.0,
            Some(w) => spans.iter().zip(w).find(|(&(lo, hi), _)| lo <= a && a < hi).map_or(0.0, |(_, &w)| w as f64),
        }
    };
    let (suppression, pulse_set): (Vec<Option<crate::longitudinal::Suppression>>, Vec<usize>) = match &p.suppression {
        None => (vec![None; n], vec![0; n]),
        Some(spec) => {
            let mut sets: Vec<Vec<f64>> = Vec::new();
            let mut ids = Vec::with_capacity(n);
            let mut sup = Vec::with_capacity(n);
            for (i, row) in rows.iter().enumerate() {
                let s = spec.for_row(first_prep(i).suppression);
                let id = match sets.iter().position(|x| *x == s.pulse_times) {
                    Some(j) => j,
                    None => {
                        sets.push(s.pulse_times.clone());
                        sets.len() - 1
                    }
                };
                ids.push(id);
                sup.push(if s.has_events() && row.kind != RowKind::M0scan { Some(s) } else { None });
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
        (Some(spec), None) => Some((0..n).map(|i| label_factor(&spec.for_row(first_prep(i).suppression))).collect::<Vec<f64>>()),
        _ => None,
    };

    // ---- P4: per-voxel arterial parameters, crushing, physiological noise ----
    let p4 = p4_for_schedule(p, ph, bolus_region, sched)?;

    // P7 part A: under Look-Locker with exchange or bolus-position suppression the read label is
    // split and cut as P4 splits and cuts it, each part depleted by the later readouts. Without
    // either, the read is P6's `delta_m_read` itself (the arterial compartment, crushing and the
    // physiological factor do not change the tissue label's read).
    let ll_parts_on = ll.is_some() && (p.exchange_time.is_some() || bolus_region.is_some());
    // the pulses of raw volume v's preparation, under the bolus-position model
    let bolus_of = |v: usize| {
        bolus_region.map(|region| {
            let s = p.suppression.as_ref().unwrap().for_row(first_prep(v).suppression);
            (region, s.pulse_times, s.epsilon)
        })
    };
    // the read of phantom voxel i over the labeled spans `spans` of the bolus: every sub-bolus
    // with its parcel factor, the exchange split
    let ll_parts_spans = |v: usize, kin: &Kinetic, i: usize, e: &[f64], fl: &[f64], spans: &[(f64, f64)]| -> ReadParts {
        let att = ph.att[i] as f64;
        let parcels = match bolus_of(v) {
            Some((region, pulses, eps)) => subbolus_factors(&pulses, eps, kin.tau, entry_offset(p.label_type, region, att)),
            None => vec![(0.0, kin.tau, 1.0)],
        };
        let mut subs = Vec::with_capacity(parcels.len() * spans.len());
        for &(aj, bj) in spans {
            for &(a, b, f) in &parcels {
                let (lo, hi) = (a.max(aj), b.min(bj));
                if hi > lo {
                    subs.push((lo, hi, f));
                }
            }
        }
        delta_m_read_parts(kin, ph.perfusion[i] as f64, att, ph.t1[i] as f64, ph.m0[i] as f64, e, fl, &subs, p.exchange_time, 0.0)
    };
    // P7 part B: an encoded raw volume reads its labeled sub-boli; any other the whole bolus
    let labeled_spans = |v: usize, tau: f64| -> Vec<(f64, f64)> {
        match &enc_w[v] {
            Some(w) => spans.iter().zip(w).filter(|(_, &w)| w == 1).map(|(&s, _)| s).collect(),
            None => vec![(0.0, tau)],
        }
    };
    let ll_parts = |v: usize, kin: &Kinetic, i: usize, e: &[f64], fl: &[f64]| -> ReadParts {
        ll_parts_spans(v, kin, i, e, fl, &labeled_spans(v, kin.tau))
    };

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
        let r = &rows[row];
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
        let r = &rows[v];
        let presat = suppression[v].as_ref().is_some_and(|s| s.presaturation);
        (r.tr.to_bits(), r.t.to_bits(), suppression[v].is_some(), pulse_set[v], presat, r.kind == RowKind::M0scan)
    };
    let uniform_prep = (0..n).all(|v| prep_key(v) == prep_key(0));
    let ge_propagated: Option<Vec<Vec<Vec<f32>>>> = match ge_flip {
        Some(fa) if fa != 90.0 && !uniform_prep && p.compat.is_none() && ll.is_none() => {
            let sin = fa.to_radians().sin();
            let [pnx, pny, _] = ph.grid.dims;
            let pslab = pnx * pny;
            let mut out = vec![vec![vec![0.0f32; nvox_sim]; k]; n];
            for z in 0..nz {
                let preps: Vec<Prep> = rows.iter().enumerate().map(|(v, r)| Prep {
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

    // ---- P6 part B: the Look-Locker tissue, per slice from the readout timeline (each readout
    // sin(a_n) Mz before its pulse), and the per-readout table of the mean Mz per label ----
    let mut ll_lines: Vec<LlLine> = Vec::new();
    let ll_tissue: Option<Vec<Vec<Vec<f32>>>> = ll.map(|l| {
        let [pnx, pny, _] = ph.grid.dims;
        let pslab = pnx * pny;
        let mut out = vec![vec![vec![0.0f32; nvox_sim]; k]; n];
        let m0scan: Vec<bool> = l.cycles.iter().map(|c| c.m0scan).collect();
        let n_labels = ph.labels.len();
        // per readout, slice and phantom label: the sum of Mz and the voxel count (pooled per group below)
        let mut mz_sum = vec![vec![vec![0.0f64; n_labels]; nz]; n];
        let mut mz_count = vec![vec![0usize; n_labels]; nz];
        for z in 0..nz {
            let cycles: Vec<LlCycle> = l.cycles.iter().map(|cy| LlCycle {
                tr: rows[cy.rows[0]].tr,
                s: if cy.m0scan { None } else { suppression[cy.rows[0]].as_ref() },
                t_read: if cy.m0scan { vec![slice_offsets[z]] } else { cy.rows.iter().map(|&r| rows[r].t + slice_offsets[z]).collect() },
                flip_deg: cy.rows.iter().map(|&r| l.flip_deg[r]).collect(),
            }).collect();
            let cells: Vec<usize> = r_sim.z_slab(z).iter().map(|&(zs, _)| zs).collect();
            let local = |i: usize| -> usize {
                let pos = cells.iter().position(|&zs| zs == i / pslab).expect("a voxel of this slab");
                pos * pslab + i % pslab
            };
            let mut seq = vec![vec![0.0f64; cells.len() * pslab]; n];
            let mut counts = vec![0usize; n_labels];
            for (pos, &zs) in cells.iter().enumerate() {
                for xy in 0..pslab {
                    let i = zs * pslab + xy;
                    if ph.dseg[i] <= 0 {
                        continue;
                    }
                    // the report is per phantom label (voxel mode has one compartment for all of them)
                    let c = ph.labels.iter().position(|(l, _)| *l == ph.dseg[i]).expect("labels come from dseg");
                    counts[c] += 1;
                    let mz = tissue_mz_ll_series(ph.m0[i] as f64, ph.t1[i] as f64, &cycles, &m0scan, None, false);
                    for (v, x) in mz.into_iter().flatten().enumerate() {
                        seq[v][pos * pslab + xy] = l.flip_deg[v].to_radians().sin() * x;
                        mz_sum[v][z][c] += x;
                    }
                }
            }
            mz_count[z] = counts;
            for v in 0..n {
                for (c, m) in masks.iter().enumerate() {
                    let sl = r_sim.mean_slice(z, |i| if m[i] { seq[v][local(i)] } else { 0.0 });
                    out[v][c][z * snx * sny..(z + 1) * snx * sny].copy_from_slice(&sl);
                }
            }
        }
        // one line per readout and excitation group (slices sharing an offset)
        let mut groups: Vec<f64> = slice_offsets.clone();
        groups.sort_by(f64::total_cmp);
        groups.dedup();
        for (c, cy) in l.cycles.iter().enumerate() {
            for (rd, &v) in cy.rows.iter().enumerate() {
                for (g, &off) in groups.iter().enumerate() {
                    let zs: Vec<usize> = (0..nz).filter(|&z| slice_offsets[z] == off).collect();
                    let t_read = if cy.m0scan { off } else { rows[v].t + off };
                    ll_lines.push(LlLine {
                        cycle: c, readout: rd, group: g, time: first_prep(v).start_s + t_read, flip_deg: l.flip_deg[v],
                        // the group's mean: its slices' sums over their counts, pooled (a label absent
                        // from a slice adds nothing to either)
                        tissue_mz: (0..n_labels).map(|lab| {
                            let count: usize = zs.iter().map(|&z| mz_count[z][lab]).sum();
                            if count == 0 { 0.0 } else { zs.iter().map(|&z| mz_sum[v][z][lab]).sum::<f64>() / count as f64 }
                        }).collect(),
                    });
                }
            }
        }
        out
    });

    // ---- per-row blood images with per-slice timing ----
    let blood_for = |v: usize, row: &Row, r: &Resampler, sign: f64, want_gt: bool| -> (Vec<Vec<f32>>, Vec<f32>) {
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
                    dm_of(v, &kin, ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64, t)
                } else {
                    0.0
                }
            };
            // what the readout reads: under Look-Locker the label depleted by the earlier readouts
            let read = ll_read(v, slice_offsets[z]);
            let dm_read = |i: usize| -> f64 {
                match &read {
                    // P7 part B: an encoded raw volume's labeled sub-boli, depleted
                    Some((e, fl)) if ph.dseg[i] > 0 && enc_w[v].is_some() => ll_parts(v, &kin, i, e, fl).total(),
                    Some((e, fl)) if ph.dseg[i] > 0 => delta_m_read(
                        &kin, ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64, e, fl),
                    Some(_) => 0.0,
                    None => dm(i),
                }
            };
            for (c, m) in masks.iter().enumerate() {
                if sign != 0.0 {
                    let sl = if te_factor.is_some() {
                        r.mean_slice(z, |i| if m[i] { blood_signal(v, sign * dm_read(i)) * te(i) } else { 0.0 })
                    } else {
                        r.mean_slice(z, |i| if m[i] { blood_signal(v, sign * dm_read(i)) } else { 0.0 })
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
    // P6 part A: the tissue alone, for the decoded leakage; the extravascular label later joins
    // the tissue compartments of `images`, so this is kept apart from them
    let leakage_on = p.hadamard.as_ref().is_some_and(|h| h.report_leakage.0);
    let mut tissue_only: Option<Vec<Vec<f32>>> = leakage_on.then(|| vec![vec![0.0f32; nvox_sim * n]; ncomp]);
    let mut gt_static = vec![0.0f32; nvox_acq * n];
    let mut gt_read: Option<Vec<f32>> = ll.map(|_| vec![0.0f32; nvox_acq * n]);
    let read_alloc = |on: bool| if ll.is_some() && on { Some(vec![0.0f32; nvox_acq * n]) } else { None };
    let mut gt_read_iv = read_alloc(ll_parts_on && p.exchange_time.is_some());
    let mut gt_read_ev = read_alloc(ll_parts_on && p.exchange_time.is_some());
    let mut gt_read_art = read_alloc(macro_on);
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
    for (v, row) in rows.iter().enumerate() {
        // An m0scan row is a plain spin-echo readout (P3), except under compat, where it takes
        // the series' equation as simasl's does (P2 addendum, part A), and under gradient echo,
        // whose M0 is the same excitation and readout without labeling (P5 addendum, part A).
        let se = row.kind == RowKind::M0scan && p.compat.is_none() && p.contrast != Contrast::GradientEcho;
        let mut tissue = match (row.kind, &suppression[v]) {
            (RowKind::Deltam, _) => None,
            _ if ll_tissue.is_some() => ll_tissue.as_ref().map(|g| g[v].clone()),
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
                let start = sched.preps_of(v)[s].start_s;
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
                let time = first_prep(v).start_s + row.t + slice_offsets[z];
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
            blood_for(v, row, &r_sim, 0.0, wants_gt && motion_on)
        } else {
            blood_for(v, row, &r_sim, sign, wants_gt && motion_on)
        };
        if let (true, Some(gr), true) = (wants_gt, gt_read.as_mut(), p.hadamard.is_none()) {
            // P6 part B: what the readout read, sin(a_n) times the depleted label, static
            let kin = p.kinetic(row);
            let fa = ll.map_or(90.0, |l| l.flip_deg[v]).to_radians().sin();
            for z in 0..nz {
                let read = ll_read(v, slice_offsets[z]);
                if ll_parts_on {
                    // P7 part A: the parts, each with its parcel factor, over this slice's slab
                    let zs = r_acq.z_slab(z);
                    let (zlo, zhi) = match (zs.first(), zs.last()) {
                        (Some(a), Some(b)) => (a.0, b.0),
                        _ => (0, 0),
                    };
                    let base = pslab * zlo;
                    let parts: Vec<ReadParts> = (base..pslab * (zhi + 1))
                        .map(|i| match &read {
                            Some((e, fl)) if ph.dseg[i] > 0 => ll_parts(v, &kin, i, e, fl),
                            _ => ReadParts::default(),
                        })
                        .collect();
                    let put = |dst: &mut Vec<f32>, f: &dyn Fn(&ReadParts) -> f64| {
                        let sl = r_acq.mean_slice(z, |i| fa * f(&parts[i - base]));
                        for (jj, x) in sl.iter().enumerate() {
                            dst[(z * nx * ny + jj) * n + v] = *x;
                        }
                    };
                    put(gr, &|q| q.total());
                    if let Some(g) = gt_read_iv.as_mut() {
                        put(g, &|q| q.iv);
                    }
                    if let Some(g) = gt_read_ev.as_mut() {
                        put(g, &|q| q.ev);
                    }
                } else {
                    let sl = r_acq.mean_slice(z, |i| match &read {
                        Some((e, fl)) if ph.dseg[i] > 0 => fa * delta_m_read(
                            &kin, ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64, e, fl),
                        _ => 0.0,
                    });
                    for (jj, x) in sl.iter().enumerate() {
                        gr[(z * nx * ny + jj) * n + v] = *x;
                    }
                }
                // P7 part A: the arterial read, fresh (2D), with its crushing survival and parcel
                // factor, sin(a_n) as the image takes it
                if let (Some(ga), Some(abv), Some(aatt)) = (gt_read_art.as_mut(), &p4.abv, &p4.aatt) {
                    let t = row.t + slice_offsets[z];
                    let bolus = bolus_of(v);
                    let sl = r_acq.mean_slice(z, |i| {
                        if ph.dseg[i] <= 0 {
                            return 0.0;
                        }
                        let (va, a) = arterial_dm(&kin, abv[i], aatt[i], ph.m0[i] as f64, t);
                        let Some(a) = a else { return 0.0 };
                        let g = match &bolus {
                            Some((region, pulses, eps)) => {
                                arterial_factor(pulses, *eps, a, entry_offset(p.label_type, *region, aatt[i]))
                            }
                            None => 1.0,
                        };
                        let c = p4.crush.as_ref().map_or(1.0, |cr| cr[v][p4.label_of[i]]);
                        fa * g * c * va
                    });
                    for (jj, x) in sl.iter().enumerate() {
                        ga[(z * nx * ny + jj) * n + v] = *x;
                    }
                }
            }
        }
        if wants_gt {
            let (_, gt) = blood_for(v, row, &r_acq, 0.0, true);
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
                if let Some(to) = tissue_only.as_mut() {
                    for vox in 0..nvox_sim {
                        to[c][vox * n + v] = t[c][vox];
                    }
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
                let s = p.suppression.as_ref().unwrap().for_row(first_prep(v).suppression);
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
                let read = ll_read(v, slice_offsets[z]);
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
                    let (scale, dm, iv) = match (&enc_w[v], &bolus) {
                        // P6 part A: the labeled sub-boli, each cut by the parcels' pulse factors
                        (Some(w), _) => {
                            let parcels: Vec<(f64, f64, f64)> = match &bolus {
                                Some((region, pulses, eps)) => {
                                    let delta = entry_offset(p.label_type, *region, att);
                                    let key = delta.map_or(u64::MAX, f64::to_bits);
                                    partitions.entry(key).or_insert_with(|| subbolus_factors(pulses, *eps, kin.tau, delta)).clone()
                                }
                                None => vec![(0.0, kin.tau, 1.0)],
                            };
                            let (mut dm, mut iv) = (0.0f64, 0.0f64);
                            for (&(aj, bj), _) in spans.iter().zip(w).filter(|(_, &w)| w == 1) {
                                for &(a, b, f) in &parcels {
                                    let (lo, hi) = (a.max(aj), b.min(bj));
                                    if hi > lo {
                                        dm += f * delta_m_sub(&kin, f_ml, att, t1t, m0, t, lo, hi);
                                        if let Some(te) = p.exchange_time {
                                            iv += f * delta_m_iv_sub(&kin, f_ml, att, t1t, m0, t, lo, hi, te);
                                        }
                                    }
                                }
                            }
                            (1.0, dm, p.exchange_time.map(|_| iv))
                        }
                        (None, Some((region, pulses, eps))) => {
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
                        (None, None) => (1.0, match &read {
                            // P6 part B: the depleted read (exchange is refused with Look-Locker)
                            Some((e, fl)) => delta_m_read(&kin, f_ml, att, t1t, m0, e, fl),
                            None => delta_m(&kin, f_ml, att, t1t, m0, t),
                        }, p.exchange_time.map(|te| delta_m_iv(&kin, f_ml, att, t1t, m0, t, te))),
                    };
                    // P7 part A: under Look-Locker the images read the depleted parts, each with its
                    // parcel factor; the P4 truths below keep their P4 meaning (at t, undepleted)
                    let (s, dm_img, iv_img) = match (&read, ll_parts_on || enc_w[v].is_some()) {
                        (Some((e, fl)), true) => {
                            let pr = ll_parts(v, &kin, i, e, fl);
                            (sign0, pr.total(), p.exchange_time.map(|_| pr.iv))
                        }
                        _ => (sign0 * scale, dm, iv),
                    };
                    match iv_img {
                        Some(iv) => {
                            bl[j] = blood_signal(v, s * iv);
                            ev[j] = blood_signal(v, s * (dm_img - iv));
                        }
                        None => bl[j] = blood_signal(v, s * dm_img),
                    }
                    if !giv.is_empty() {
                        // the unsuppressed intravascular part: kinetics and the split only
                        let te = p.exchange_time.unwrap();
                        giv[j] = match &enc_w[v] {
                            None => delta_m_iv(&kin, f_ml, att, t1t, m0, t, te),
                            Some(w) => spans.iter().zip(w).filter(|(_, &w)| w == 1)
                                .map(|(&(a, b), _)| delta_m_iv_sub(&kin, f_ml, att, t1t, m0, t, a, b, te)).sum(),
                        };
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
                            // P6 part A: the arterial parcel counts only where its sub-bolus is labeled
                            if enc_w[v].is_some() {
                                art[j] = blood_signal(v, sign0 * g * c * va * weight_at(v, a));
                            } else {
                                art[j] = blood_signal(v, sign0 * g * c * va);
                            }
                        }
                        if !gart.is_empty() {
                            gart[j] = if enc_w[v].is_some() { va * a.map_or(0.0, |a| weight_at(v, a)) } else { va };
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

    // ---- P7 part B: under Hadamard the read truths are per decoded output: sub-bolus j as readout
    // n read it, sin(a_n) times its depleted label (each part, with its parcel factors) and its
    // fresh arterial parcel; zero for m0scan outputs ----
    if let (Some(h), Some(_)) = (&p.hadamard, ll) {
        let n_out = sched.outputs.len();
        let decoded_alloc = |on: bool| on.then(|| vec![0.0f32; nvox_acq * n_out]);
        let mut d_read = vec![0.0f32; nvox_acq * n_out];
        let (mut d_iv, mut d_ev) = (decoded_alloc(gt_read_iv.is_some()), decoded_alloc(gt_read_ev.is_some()));
        let mut d_art = decoded_alloc(gt_read_art.is_some());
        let [pnx, pny, _] = ph.grid.dims;
        let pslab = pnx * pny;
        for (k_out, o) in sched.outputs.iter().enumerate() {
            let Output::Decoded { cycle, subbolus, readout } = *o else { continue };
            let r = readout_groups(&sched.cycles[cycle], h.order)[readout][0];
            let row = &rows[r];
            let kin = p.kinetic(row);
            let fa = ll.map_or(90.0, |l| l.flip_deg[r]).to_radians().sin();
            let span = [h.spans[subbolus]];
            for z in 0..nz {
                let read = ll_read(r, slice_offsets[z]);
                let zs = r_acq.z_slab(z);
                let (zlo, zhi) = match (zs.first(), zs.last()) {
                    (Some(a), Some(b)) => (a.0, b.0),
                    _ => (0, 0),
                };
                let base = pslab * zlo;
                let parts: Vec<ReadParts> = (base..pslab * (zhi + 1))
                    .map(|i| match &read {
                        Some((e, fl)) if ph.dseg[i] > 0 => ll_parts_spans(r, &kin, i, e, fl, &span),
                        _ => ReadParts::default(),
                    })
                    .collect();
                let put = |dst: &mut Vec<f32>, f: &dyn Fn(usize) -> f64| {
                    let sl = r_acq.mean_slice(z, f);
                    for (jj, x) in sl.iter().enumerate() {
                        dst[(z * nx * ny + jj) * n_out + k_out] = *x;
                    }
                };
                put(&mut d_read, &|i| fa * parts[i - base].total());
                if let Some(g) = d_iv.as_mut() {
                    put(g, &|i| fa * parts[i - base].iv);
                }
                if let Some(g) = d_ev.as_mut() {
                    put(g, &|i| fa * parts[i - base].ev);
                }
                if let (Some(g), Some(abv), Some(aatt)) = (d_art.as_mut(), &p4.abv, &p4.aatt) {
                    let t = row.t + slice_offsets[z];
                    let bolus = bolus_of(r);
                    let (aj, bj) = h.spans[subbolus];
                    put(g, &|i| {
                        if ph.dseg[i] <= 0 {
                            return 0.0;
                        }
                        let (va, a) = arterial_dm(&kin, abv[i], aatt[i], ph.m0[i] as f64, t);
                        match a {
                            Some(a) if aj <= a && a < bj => {
                                let g = match &bolus {
                                    Some((region, pulses, eps)) => {
                                        arterial_factor(pulses, *eps, a, entry_offset(p.label_type, *region, aatt[i]))
                                    }
                                    None => 1.0,
                                };
                                let c = p4.crush.as_ref().map_or(1.0, |cr| cr[r][p4.label_of[i]]);
                                fa * g * c * va
                            }
                            _ => 0.0,
                        }
                    });
                }
            }
        }
        gt_read = Some(d_read);
        gt_read_iv = d_iv;
        gt_read_ev = d_ev;
        gt_read_art = d_art;
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
        if let Some(to) = tissue_only.as_mut() {
            apply_motion(to, sim_grid.dims, n, v2w, &poses);
        }
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
                if let Some(to) = tissue_only.as_mut() {
                    apply_multiband_motion(to, sim_grid.dims, n, v2w, p.mb, p.mb_interleaved, &DropoutLaw::Uniform, &events);
                }
            }
        }
    }
    // P5 part D: a 3D volume's shots. Each event's jump persists for the later shots of its volume
    // (as apply_multiband_motion composes them); the shots sharing a pose other than the volume's
    // see the volume's images moved by it. Each event shot's lines are attenuated by
    // 1 - severity (DropoutLaw::Uniform), a shot gain.
    let mut shot_gain = vec![vec![1.0f64; n_shots]; n];
    let mut shot_sets: Vec<Vec<ShotSet>> = vec![Vec::new(); n];
    // the tissue-only images' own shot sets (the motion shot sets are copies of the images)
    let mut tissue_shot_sets: Vec<Vec<ShotSet>> = vec![Vec::new(); n];
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
                if let Some(to) = &tissue_only {
                    let moved_t: Vec<Vec<f32>> = to.iter().map(|img| {
                        let vol: Vec<f32> = (0..nvox_sim).map(|vox| img[vox * n + g]).collect();
                        mrsim_acq::motion::resample_by_pose(&vol, sim_grid.dims, v2w, q)
                    }).collect();
                    tissue_shot_sets[g].push(ShotSet { shots: shots.clone(), images: moved_t });
                }
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
            let tissue = match p.look_locker.as_ref().and_then(|l| l.m0_flip_deg) {
                // P6 part B: the M0 excitation is its own ([m0] flip_angle)
                Some((fa, _)) if Some(fa) != ge_flip => masks.iter().map(|m| {
                    let sig: Vec<f32> = (0..ph.nvox()).map(|i| {
                        if m[i] { tissue_ge_spoiled(ph.m0[i] as f64, ph.t1[i] as f64, tr, fa) as f32 } else { 0.0 }
                    }).collect();
                    r_sim.mean(&sig)
                }).collect(),
                _ => tissue_for(tr, p.contrast != Contrast::GradientEcho),
            };
            let mut imgs: Vec<Vec<f32>> = vec![vec![0.0f32; nvox_sim]; ncomp];
            for c in 0..k {
                imgs[c].copy_from_slice(&tissue[c]);
            }
            Some(imgs)
        }
        _ => None,
    };
    let m0_ref_images: Vec<Vec<Vec<f32>>> = match (&p.hadamard, leakage_on) {
        (Some(_), true) => sched.cycles.iter().map(|c| tissue_for(rows[c.raws.start].tr, false)).collect(),
        _ => Vec::new(),
    };
    let ge_propagated_some = ge_propagated.is_some();
    Ok(Built {
        acq_grid, sim_grid, n, nvox_sim, images, gt_static, gt_moved, gt_iv, gt_sup, gt_art, physio_lines, shot_physio, shot_gain, shot_sets, n_shots, events, dropped, poses, motion_seed, res3d, acq, fmap_sim, relax, mode_used, k, ncomp, ev_group, macro_on, t2_arterial_ms, t2_blood_ms, acq_t2_ms, acq_t2p_ms, acq_t1_ms, needs_t1, r_acq, m0_acq, compat_facts, p4, label_factors, ge_flip, ge_propagated_some, m0_images, slice_offsets, tissue_only, tissue_shot_sets, m0_ref_images, gt_read, ll_lines, gt_read_iv, gt_read_ev, gt_read_art,
    })
}

/// The series of a protocol with a P6 feature: [`build`], then the acquisition and the output.
pub(super) fn simulate_p6(
    p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride, capture: Option<&mut Vec<Vec<f32>>>,
) -> Result<SeriesOutput, String> {
    let tes = p.echo_times_s.clone();
    // P7 part C: the 3D gradient-echo train is its own series
    if p.ge3d().is_some() {
        return super::ge3d::simulate_ge3d(p, ph, mode, phase, ov);
    }
    let sched = Schedule::new(p);
    // P6 part C, compat: the echo-time decay is the signal stage's, so each echo has its own image
    // set, bounded before any is built: the grids and the compartment count resolved alone
    if p.compat.is_some() && tes.len() > 1 {
        let ag = acquisition_grid(&ph.grid, p.voxel_size_mm, p.acq.matrix, p.grid_origin)?;
        let sg = hires_grid(&ag, p.acq.oversample);
        let k = match ph.relaxation_for(mode, false)?.0 {
            Relaxation::Class { .. } => ph.labels.len(),
            Relaxation::Voxel { .. } => 1,
        };
        // compat refuses the arterial compartment and 3D, so tissue and blood per label
        let (ncomp, nvox_sim, n) = (2 * k, sg.dims.iter().product::<usize>(), sched.raw_rows.len());
        let bytes = 4.0 * tes.len() as f64 * ncomp as f64 * nvox_sim as f64 * n as f64;
        let limit = p.multi_te.as_ref().and_then(|m| m.max_image_memory_gib).map_or(4.0, |l| l.0);
        if bytes > limit * (1u64 << 30) as f64 {
            return Err(format!(
                "compat multi-TE needs {:.2} GiB of per-echo input images ({} echoes x {ncomp} compartments x {nvox_sim} \
                 voxels x {n} volumes x 4 bytes), over the limit of {limit} GiB (overlay multi_te.max_image_memory_gib)",
                bytes / (1u64 << 30) as f64, tes.len()));
        }
    }
    let Built {
        acq_grid, sim_grid, n, nvox_sim, images, gt_static, gt_moved, gt_iv, gt_sup, gt_art, physio_lines, shot_physio, shot_gain, shot_sets, n_shots, events, dropped, poses, motion_seed, res3d, acq, fmap_sim, relax, mode_used, k, ncomp, ev_group, macro_on, t2_arterial_ms, t2_blood_ms, acq_t2_ms, acq_t2p_ms, acq_t1_ms, needs_t1, r_acq, m0_acq, compat_facts, p4, label_factors, ge_flip, ge_propagated_some, m0_images, slice_offsets, tissue_only, tissue_shot_sets, m0_ref_images, gt_read, ll_lines, gt_read_iv, gt_read_ev, gt_read_art,
    } = build(p, ph, mode, ov, tes[0], &sched)?;
    // P6 part C: every echo's readout block on explicit intervals, on the acquired grid
    if tes.len() > 1 || p.look_locker.is_some() {
        check_excitation_timing(p, acq_grid.dims)?;
    }
    let mut echo_images: Vec<Vec<Vec<f32>>> = Vec::new();
    if p.compat.is_some() && tes.len() > 1 {
        for &te in &tes[1..] {
            echo_images.push(build(p, ph, mode, ov, te, &sched)?.images);
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

    // ---- P6 part A: the tissue leakage. The tissue-only raw volumes and one reference volume
    // per cycle (its unsuppressed steady state) in one call with noise, spikes and GRAPPA off,
    // decoded per cycle; the norms over the brain mask ----
    let leakage: Option<Vec<Vec<HadamardLeakage>>> = match (&p.hadamard, tissue_only) {
        (Some(h), Some(to)) => {
            let nc = sched.cycles.len();
            let nt = n + nc;
            let mut imgs = vec![vec![0.0f32; nvox_sim * nt]; ncomp];
            for (c, img) in imgs.iter_mut().enumerate() {
                for vox in 0..nvox_sim {
                    img[vox * nt..vox * nt + n].copy_from_slice(&to[c][vox * n..(vox + 1) * n]);
                    if c < k {
                        for (cy, r) in m0_ref_images.iter().enumerate() {
                            img[vox * nt + n + cy] = r[c][vox];
                        }
                    }
                }
            }
            drop(to);
            let quiet = Acquisition { noise_variance: 0.0, n_spikes: 0, accel: 1, ..acq.clone() };
            // every echo's (P6 part C: Hadamard x multi-TE), each acquired at its own TE
            let per_echo: Vec<(Vec<f32>, Vec<f32>)> = match &res3d {
                None if tes.len() > 1 => {
                    let per: Vec<&[Vec<f32>]> = vec![&imgs[..]; tes.len()];
                    simulate_acquisition_echoes(
                        sim_grid.dims, acq_grid.dims, nt, &per, &t2_vols, &fmap_sim, Some(&ti_vols), &quiet, &echo_ms,
                        &vec![None; nt], &vec![None; nt], phase, p.seed, None, None,
                    )
                }
                None => vec![simulate_acquisition_oversampled(
                    sim_grid.dims, acq_grid.dims, nt, &imgs, &t2_vols, &fmap_sim, Some(&ti_vols), &quiet,
                    &vec![None; nt], &vec![None; nt], phase, p.seed, None, None,
                )],
                Some(r3) => {
                    let lw = line_weights.as_ref().map(|l| {
                        let mut w = l.w.clone();
                        w.extend(std::iter::repeat_n(1.0, nc * n_shots * ncomp));
                        LineWeights { n_shots, n_compartments: ncomp, w }
                    });
                    let mut sets = tissue_shot_sets;
                    sets.extend(std::iter::repeat_n(Vec::new(), nc));
                    vec![simulate_acquisition_3d(
                        sim_grid.dims, acq_grid.dims, nt, &imgs, &t2_vols, t1_vols.as_deref(), &fmap_sim, Some(&ti_vols),
                        &quiet, &r3.train, &r3.readout, lw.as_ref(), sets.iter().any(|s| !s.is_empty()).then_some(&sets[..]),
                        phase, p.seed,
                    )]
                }
            };
            let brain: Vec<bool> = r_acq.majority(&ph.dseg).iter().map(|l| *l > 0).collect();
            let norm = |img: &[(f64, f64)]| -> f64 {
                img.iter().zip(&brain).filter(|(_, b)| **b).map(|(z, _)| z.0 * z.0 + z.1 * z.1).sum::<f64>().sqrt()
            };
            let eps = 1e-12 * acq.signal_scale;
            // per cycle, per readout (P7 part B: one group without Look-Locker)
            Some(per_echo.iter().map(|(tm, tp)| sched.cycles.iter().enumerate().flat_map(|(cy, c)| {
                let reference = norm(&complex_volume(tm, tp, nt, n + cy));
                readout_groups(c, h.order).into_iter().enumerate().map(move |(readout, g)| {
                    let raw: Vec<Vec<(f64, f64)>> = g.iter().map(|&r| complex_volume(tm, tp, nt, r)).collect();
                    let refs: Vec<&[(f64, f64)]> = raw.iter().map(|v| v.as_slice()).collect();
                    let per_subbolus = crate::hadamard::decode(&refs, h.order).iter().map(|l| {
                        let a = norm(l);
                        (a, a / reference.max(eps))
                    }).collect();
                    HadamardLeakage { cycle: cy, readout, reference_norm: reference, per_subbolus }
                }).collect::<Vec<_>>()
            }).collect()).collect())
        }
        _ => None,
    };

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
    // P6 part A: the main dataset is the decoded series with the ideal sub-bolus truth; the raw
    // series, its truth and its per-preparation records go to `HadamardSeries` (sourcedata)
    let (mag, phase_out, more_mag, n_out, delta_m_gt, delta_m_static, gt_iv, gt_sup, gt_art, hadamard) = match &p.hadamard {
        None => (mag, phase_out, more_mag, n, delta_m_gt, delta_m_static, gt_iv, gt_sup, gt_art, None),
        Some(h) => {
            let (dm, iv, sup, art) = decoded_truth(p, ph, &sched, h, &r_acq, &slice_offsets, &p4,
                                                    match p.suppression.as_ref().map(|s| s.model) {
                                                        Some(SuppressionModel::BolusPosition(r)) => Some(r),
                                                        _ => None,
                                                    });
            let (dmag, dphase) = decode_series(&sched, h.order, &mag, &phase_out, n);
            let dmore: Vec<(Vec<f32>, Vec<f32>)> = more_mag.iter().map(|(m, ph_)| decode_series(&sched, h.order, m, ph_, n)).collect();
            let prep_factors: Vec<PrepFactors> = sched.preps.iter().map(|pr| {
                let v = pr.raw;
                let (tissue, label) = match (&res3d, &p4.physio) {
                    (Some(_), Some(_)) => shot_physio[v][pr.shot],
                    (None, Some(_)) => (
                        f64::NAN, // per slice: the physiology table's lines of this volume
                        p4.label_physio[v].0,
                    ),
                    _ => (1.0, 1.0),
                };
                PrepFactors {
                    raw: v, shot: pr.shot, encoding_row: sched.raws[v].encoding_row, start_s: pr.start_s,
                    labeling_window: pr.labeling_window, label, tissue,
                    suppression: label_factors.as_ref().map(|f| f[v]),
                    shot_gain: shot_gain[v][pr.shot],
                }
            }).collect();
            let series = HadamardSeries {
                n_raw: n, raw_mag: mag, raw_phase: phase_out, raw_more_echoes: more_mag,
                raw_delta_m: if res3d.is_some() { delta_m_static.clone().unwrap_or(delta_m_gt) } else { delta_m_gt },
                raw_delta_m_static: if res3d.is_some() { None } else { delta_m_static },
                raw_delta_m_iv: gt_iv, raw_delta_m_suppressed: gt_sup, raw_delta_m_arterial: gt_art,
                schedule: sched.clone(),
                leakage: leakage.as_ref().map(|l| l[0].clone()),
                leakage_more_echoes: leakage.map(|l| l[1..].to_vec()).unwrap_or_default(),
                prep_factors,
                flags: HadamardFlags {
                    grappa: acq.accel > 1, spikes: acq.n_spikes > 0, motion: p.motion.is_some(),
                    shot_factors: line_weights.is_some(), transients: ge_propagated_some, physiology: p.physio.is_some(),
                },
            };
            (dmag, dphase, dmore, sched.outputs.len(), dm, None, iv, sup, art, Some(series))
        }
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
        acq_grid, sim_grid, n_volumes: n_out, mag, phase: phase_out, m0, mode: mode_used,
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
        hadamard,
        ge3d: None,
        look_locker: ll_on_raws(p, &sched).map(|l| LookLockerSeries {
            delta_m_read: gt_read,
            lines: ll_lines,
            legacy_dispatch: ll_legacy_dispatch(l.cycles.iter().map(|c| c.rows.len()), l.flip_array),
            read_iv: gt_read_iv,
            read_ev: gt_read_ev,
            read_arterial: gt_read_art,
            p4_parts: [
                ("exchange (P4 part A)", p.exchange_time.is_some()),
                ("the arterial compartment (P4 part B)", p.macrovascular.is_some()),
                ("crushing (P4 part C)", p.crushing.is_some()),
                ("bolus-position suppression (P4 part D)",
                 p.suppression.as_ref().is_some_and(|s| s.model != SuppressionModel::GlobalBolus)),
            ]
            .into_iter()
            .filter_map(|(name, on)| on.then_some(name))
            .collect(),
        }),
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

            let b = build(&p, &ph, T2Mode::Voxel, RowOverride::None, tes[0], &Schedule::new(&p)).unwrap();
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

    // ---- P6 part A: Hadamard

    fn crop() -> Phantom {
        load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap()
    }

    const TAU: f64 = 0.25;
    const PLD: f64 = 1.5;

    fn sidecar(ld: Vec<f64>, pld: Vec<f64>, tr: Vec<f64>, m0: &str, ge: bool) -> Value {
        let mut s = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": m0, "RepetitionTimePreparation": tr, "EchoTime": 0.012,
            "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "2D",
            "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.012
        });
        if ge {
            s["FlipAngle"] = json!(60);
        }
        s
    }

    fn overlay(extra: &str, ge: bool) -> Overlay {
        let contrast = if ge { "[signal]\nacq_contrast = \"ge\"\nexcitation_flip_angle = 60\n" } else { "" };
        toml::from_str(&format!("seed = 11\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n{contrast}{extra}")).unwrap()
    }

    /// One Hadamard cycle of `order` on the crop (equal sub-boli), an m0scan row first at 6 s
    /// with `m0_first`.
    fn hadamard_p(order: usize, m0_first: bool, extra: &str, ge: bool) -> Protocol {
        let n = order - 1;
        let (mut ld, mut pld, mut tr, mut ctx) = (Vec::new(), Vec::new(), Vec::new(), String::from("volume_type\n"));
        if m0_first {
            ld.push(0.0);
            pld.push(0.0);
            tr.push(6.0);
            ctx.push_str("m0scan\n");
        }
        for j in 0..n {
            ld.push(TAU);
            pld.push(PLD + TAU * (n - 1 - j) as f64);
            tr.push(4.0);
            ctx.push_str("deltam\n");
        }
        let s = sidecar(ld, pld, tr, if m0_first { "Included" } else { "Absent" }, ge);
        parse_echoes(&[s], &ctx, Some(&overlay(&format!("[hadamard]\norder = {order}\n{extra}"), ge)), None).unwrap()
    }

    /// One row of `kind` with its own duration and delay.
    fn single_p(kind: &str, tau: f64, pld: f64, extra: &str) -> Protocol {
        let s = sidecar(vec![tau], vec![pld], vec![4.0], "Absent", false);
        parse_echoes(&[s], &format!("volume_type\n{kind}\n"), Some(&overlay(extra, false)), None).unwrap()
    }

    fn class_volumes(b: &Built) -> (Vec<T2Volume<'static>>, Vec<T2Volume<'static>>) {
        let Relaxation::Class { t2_ms, t2p_ms } = &b.relax else { panic!("class mode") };
        let k = b.k;
        let mut t2v: Vec<T2Volume> = (0..k).map(|i| T2Volume::Uniform(t2_ms[i])).collect();
        t2v.extend((0..k).map(|_| T2Volume::Uniform(b.t2_blood_ms)));
        let mut tiv: Vec<T2Volume> = (0..k).map(|i| T2Volume::Uniform(t2p_ms[i])).collect();
        tiv.extend((0..k).map(|i| T2Volume::Uniform(t2p_ms[i])));
        (t2v, tiv)
    }

    fn cvol(mag: &[f32], phase: &[f32], n: usize, v: usize) -> Vec<(f64, f64)> {
        complex_volume(mag, phase, n, v)
    }

    fn zero_phase() -> PhaseModel {
        PhaseModel { global: 0.0, background: Default::default(), prep: None }
    }

    /// Over every sub-bolus the encoded sum is the whole bolus (the synthetic all-ones row).
    #[test]
    fn the_sub_boli_partition_the_bolus() {
        let p = hadamard_p(8, false, "", false);
        let h = p.hadamard.as_ref().unwrap();
        let sched = Schedule::new(&p);
        let kin = p.kinetic(&sched.raw_rows[0]);
        for (f, att, t1, m0) in [(60.0, 1.2, 1.33, 74.6), (20.0, 1.6, 0.83, 60.0), (60.0, 0.5, 1.33, 74.6)] {
            for t in [0.6, 1.4, 2.0, 3.25, 3.9] {
                let whole = delta_m(&kin, f, att, t1, m0, t);
                let sum: f64 = h.spans.iter().map(|&(a, b)| delta_m_sub(&kin, f, att, t1, m0, t, a, b)).sum();
                assert!((sum - whole).abs() <= 1e-12 * whole.abs().max(1e-300), "t {t}: {sum} vs {whole}");
            }
        }
    }

    /// Physiology (with exchange) in stationary 2D: the series against the per-component reference
    /// built from single rows without physiology (the tissue of a control row, and each sub-bolus's
    /// intravascular and extravascular label from its own deltam row), each scaled by the factor
    /// the series recorded, assembled per raw volume, acquired and decoded. The negative control
    /// gives the extravascular label the tissue factor and must fail.
    #[test]
    fn physiology_decodes_per_component() {
        let ph = crop();
        let order = 4;
        let kin = "[kinetic]\nexchange_time = 0.5\n";
        let phys = "[physio]\ntissue_cardiac = 0.06\nlabel_cardiac = 0.08\nlabel_drift = 0.03\n";
        let p = hadamard_p(order, false, &format!("{kin}{phys}"), false);
        let out = simulate_p6(&p, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let h = out.hadamard.as_ref().unwrap();
        let lines = out.physio.as_ref().unwrap();
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        let ctl = single_p("control", TAU * 3.0, PLD, kin);
        let tissue = build(&ctl, &ph, T2Mode::Class, RowOverride::None, 0.012, &Schedule::new(&ctl)).unwrap().images;
        let comps: Vec<Vec<Vec<f32>>> = (0..order - 1).map(|j| {
            let q = single_p("deltam", TAU, PLD + TAU * (order - 2 - j) as f64, kin);
            build(&q, &ph, T2Mode::Class, RowOverride::None, 0.012, &Schedule::new(&q)).unwrap().images
        }).collect();
        let (n, nsim, k, ncomp) = (order, b.nvox_sim, b.k, b.ncomp);
        assert_eq!(ncomp, 2 * k);
        let [snx, sny, _] = b.sim_grid.dims;
        let enc = crate::hadamard::encoding(order);
        let make = |ev_takes_tissue_factor: bool| -> Vec<Vec<f32>> {
            let mut img = vec![vec![0.0f32; nsim * n]; ncomp];
            for i in 0..n {
                let w = crate::hadamard::weights(&enc[i]);
                let lf = h.prep_factors[i].label;
                for vox in 0..nsim {
                    let z = vox / (snx * sny);
                    let tf = lines.iter().find(|l| l.volume == i && l.slice == z).unwrap().tissue_factor;
                    for c in 0..ncomp {
                        let mut x = if c < k { tf * tissue[c][vox] as f64 } else { 0.0 };
                        for j in (0..order - 1).filter(|&j| w[j] == 1) {
                            let f = if c < k && ev_takes_tissue_factor { tf } else { lf };
                            x -= f * comps[j][c][vox] as f64;
                        }
                        img[c][vox * n + i] = x as f32;
                    }
                }
            }
            img
        };
        let (t2v, tiv) = class_volumes(&b);
        let acquire = |img: &[Vec<f32>]| simulate_acquisition_oversampled(
            b.sim_grid.dims, b.acq_grid.dims, n, img, &t2v, &b.fmap_sim, Some(&tiv), &b.acq, &vec![None; n], &vec![None; n],
            &zero_phase(), p.seed, None, None);
        let decoded_of = |m: &[f32], ph_: &[f32]| -> Vec<Vec<(f64, f64)>> {
            let raw: Vec<Vec<(f64, f64)>> = (0..n).map(|i| cvol(m, ph_, n, i)).collect();
            let refs: Vec<&[(f64, f64)]> = raw.iter().map(|v| v.as_slice()).collect();
            crate::hadamard::decode(&refs, order)
        };
        let series: Vec<Vec<(f64, f64)>> = (0..order - 1).map(|j| cvol(&out.mag, &out.phase, out.n_volumes, j)).collect();
        let smax = (0..n).flat_map(|i| cvol(&h.raw_mag, &h.raw_phase, n, i)).map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
        let worst = |d: &[Vec<(f64, f64)>]| -> f64 {
            series.iter().zip(d).map(|(a, r)| {
                let max_ref = r.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
                a.iter().zip(r).map(|(x, y)| (x.0 - y.0).hypot(x.1 - y.1)).fold(0.0f64, f64::max) / (1e-6 * smax + 1e-4 * max_ref)
            }).fold(0.0f64, f64::max)
        };
        let (m, ph_) = acquire(&make(false));
        let good = worst(&decoded_of(&m, &ph_));
        let (m, ph_) = acquire(&make(true));
        let bad = worst(&decoded_of(&m, &ph_));
        println!("physiology per component: worst {good:.3e} of the bound; extravascular with the tissue factor {bad:.3e}");
        assert!(good <= 1.0, "{good}");
        assert!(bad > 10.0, "{bad}");
        // the factors did vary between the raw volumes (the test is not of a common factor)
        let lfs: Vec<f64> = h.prep_factors.iter().map(|f| f.label).collect();
        assert!(lfs.iter().any(|f| (f - lfs[0]).abs() > 1e-3), "{lfs:?}");
    }

    /// Gradient echo after an included M0 at another repetition time: each raw volume's tissue is
    /// the state carried through the raw volumes' actual history, an independent
    /// `tissue_mz_ge_sequence` over the raw rows per phantom voxel, resampled per label.
    #[test]
    fn gradient_echo_tissue_follows_the_raw_history() {
        use crate::longitudinal::{tissue_mz_ge_sequence, Prep};
        use crate::resample::{axis_aligned_voxels, corner_offset};
        let ph = crop();
        let p = hadamard_p(4, true, "", true);
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        assert!(b.ge_propagated_some);
        let to = b.tissue_only.as_ref().unwrap();
        let o = p.acq.oversample;
        let dv = p.voxel_size_mm;
        let r_sim = Resampler::with_offset(ph.grid.dims, axis_aligned_voxels(&ph.grid).unwrap(), b.sim_grid.dims,
                                           [dv[0] / o as f64, dv[1] / o as f64, dv[2]], corner_offset(&ph.grid, &b.acq_grid).unwrap());
        let [snx, sny, nz] = b.sim_grid.dims;
        let n = sched.raw_rows.len();
        assert_eq!(n, 5);
        let sin = 60f64.to_radians().sin();
        let mut worst = 0.0f64;
        for z in 0..nz {
            let preps: Vec<Prep> = sched.raw_rows.iter().map(|r| Prep {
                tr: r.tr, t_read: if r.kind == RowKind::M0scan { r.tr } else { r.t + b.slice_offsets[z] }, s: None,
            }).collect();
            for v in 0..n {
                for (c, (label, _)) in ph.labels.iter().enumerate() {
                    let want = r_sim.mean_slice(z, |i| {
                        if ph.dseg[i] == *label {
                            sin * tissue_mz_ge_sequence(ph.m0[i] as f64, ph.t1[i] as f64, &preps, 60.0)[v]
                        } else {
                            0.0
                        }
                    });
                    for (jj, w) in want.iter().enumerate() {
                        let got = to[c][(z * snx * sny + jj) * n + v];
                        worst = worst.max((got - w).abs() as f64 / w.abs().max(1e-3) as f64);
                    }
                }
            }
        }
        println!("gradient-echo history: worst relative difference {worst:e}");
        assert!(worst < 1e-5, "{worst}");
        // and the encoded raw volumes' tissue differs (the transient)
        let first: Vec<f32> = (0..snx * sny * nz).map(|vox| to[0][vox * n + 1]).collect();
        let last: Vec<f32> = (0..snx * sny * nz).map(|vox| to[0][vox * n + 4]).collect();
        assert!(first.iter().zip(&last).any(|(a, b)| (a - b).abs() > 1e-3 * a.abs().max(1.0)));
    }

    /// The decoded noise SD is the raw SD times 2/sqrt(H) (measured, recorded).
    #[test]
    fn decoded_noise_scales_by_two_over_root_h() {
        let ph = crop();
        let order = 8;
        let clean = simulate_p6(&hadamard_p(order, false, "", false), &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let mut p = hadamard_p(order, false, "", false);
        p.acq.noise_variance = 1.0;
        let noisy = simulate_p6(&p, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let sd = |a: &[(f64, f64)], b: &[(f64, f64)]| -> f64 {
            let v: Vec<f64> = a.iter().zip(b).flat_map(|(x, y)| [x.0 - y.0, x.1 - y.1]).collect();
            (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
        };
        let (hn, hc) = (noisy.hadamard.as_ref().unwrap(), clean.hadamard.as_ref().unwrap());
        let raw: Vec<f64> = (0..order).map(|i| sd(&cvol(&hn.raw_mag, &hn.raw_phase, order, i), &cvol(&hc.raw_mag, &hc.raw_phase, order, i))).collect();
        let dec: Vec<f64> = (0..order - 1).map(|j| sd(&cvol(&noisy.mag, &noisy.phase, order - 1, j), &cvol(&clean.mag, &clean.phase, order - 1, j))).collect();
        let (r, d) = (raw.iter().sum::<f64>() / raw.len() as f64, dec.iter().sum::<f64>() / dec.len() as f64);
        let want = 2.0 / (order as f64).sqrt();
        println!("decoded noise SD / raw: {:.4} (2/sqrt(H) = {want:.4}); raw SD {r:.4}", d / r);
        assert!((d / r / want - 1.0).abs() < 0.1, "{} vs {want}", d / r);
    }

    fn sidecar3d(ld: Vec<f64>, pld: Vec<f64>, tr: Vec<f64>) -> Value {
        json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": tr, "EchoTime": 0.012,
            "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "3D",
            "PulseSequenceType": "3Dgrase", "PhaseEncodingDirection": "j-", "EffectiveEchoSpacing": 0.0005,
            "NumberShots": 2, "FlipAngle": 150
        })
    }

    /// 3D GRASE (two shots per raw volume) with exchange, physiology and shot events: every shot of
    /// a raw volume repeats its encoding row and takes its own physiological factors, gain and pose.
    /// The reference is built per preparation from single rows without them (the control row's
    /// tissue, each sub-bolus's intravascular and extravascular label), the recorded factors given
    /// to the 3D call as line weights and the recorded poses as shot sets, then decoded. The
    /// negative control gives the extravascular label the tissue factor.
    #[test]
    fn three_d_shots_carry_their_own_factors_and_poses() {
        use mrsim_acq::motion::resample_by_pose;
        let ph = crop();
        let order = 4;
        let kin = "[kinetic]\nexchange_time = 0.5\n";
        let extra = format!("{kin}[physio]\ntissue_cardiac = 0.2\nlabel_cardiac = 0.08\nlabel_drift = 0.03\n\
                             [motion]\nwithin_volume = {{ dropout_rate = 0.5, severity = 0.3, jump_mm = [0.6, 0.0, 0.0], jump_deg = [0.0, 0.0, 1.5] }}\n");
        let n_sub = order - 1;
        let (ld, pld): (Vec<f64>, Vec<f64>) = (0..n_sub).map(|j| (TAU, PLD + TAU * (n_sub - 1 - j) as f64)).unzip();
        let ctx = format!("volume_type\n{}", "deltam\n".repeat(n_sub));
        let p = parse_echoes(&[sidecar3d(ld, pld, vec![4.0; n_sub])], &ctx,
                             Some(&overlay(&format!("[hadamard]\norder = {order}\n{extra}"), false)), None).unwrap();
        let out = simulate_p6(&p, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let h = out.hadamard.as_ref().unwrap();
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        let r3 = b.res3d.clone().unwrap();
        let (n, shots, k, ncomp, nsim) = (order, r3.n_shots, b.k, b.ncomp, b.nvox_sim);
        assert_eq!((shots, ncomp), (2, 3 * k), "two shots; tissue, blood and the extravascular group");
        assert!(!out.events.is_empty() && h.flags.shot_factors);
        let single = |row: &str, tau: f64, pld: f64| -> Vec<Vec<f32>> {
            let q = parse_echoes(&[sidecar3d(vec![tau], vec![pld], vec![4.0])], &format!("volume_type\n{row}\n"),
                                 Some(&overlay(kin, false)), None).unwrap();
            build(&q, &ph, T2Mode::Class, RowOverride::None, 0.012, &Schedule::new(&q)).unwrap().images
        };
        let tissue = single("control", TAU * n_sub as f64, PLD);
        let comps: Vec<Vec<Vec<f32>>> = (0..n_sub).map(|j| single("deltam", TAU, PLD + TAU * (n_sub - 1 - j) as f64)).collect();
        let enc = crate::hadamard::encoding(order);
        let mut img = vec![vec![0.0f32; nsim * n]; ncomp];
        for i in 0..n {
            let w = crate::hadamard::weights(&enc[i]);
            for vox in 0..nsim {
                for c in 0..k {
                    img[c][vox * n + i] = tissue[c][vox];
                    let (mut iv, mut ev) = (0.0f64, 0.0f64);
                    for j in (0..n_sub).filter(|&j| w[j] == 1) {
                        iv -= comps[j][k + c][vox] as f64;
                        ev -= comps[j][c][vox] as f64;
                    }
                    img[k + c][vox * n + i] = iv as f32;
                    img[2 * k + c][vox * n + i] = ev as f32;
                }
            }
        }
        // the poses of each raw volume's shots, from the recorded events, as shot sets
        let v2w = b.sim_grid.voxel_to_world;
        let sets: Vec<Vec<ShotSet>> = (0..n).map(|g| {
            let mut cum = Pose::IDENTITY;
            let mut pose_of = Vec::new();
            for s in 0..shots {
                for e in out.events.iter().filter(|e| e.volume == g && e.shot == s) {
                    for a in 0..3 {
                        cum.trans_mm[a] += e.jump_mm[a];
                        cum.rot_deg[a] += e.jump_deg[a];
                    }
                }
                pose_of.push(cum);
            }
            let mut distinct: Vec<Pose> = Vec::new();
            for &q in &pose_of {
                if q != Pose::IDENTITY && !distinct.contains(&q) {
                    distinct.push(q);
                }
            }
            distinct.into_iter().map(|q| ShotSet {
                shots: (0..shots).filter(|&s| pose_of[s] == q).collect(),
                images: img.iter().map(|im| {
                    let vol: Vec<f32> = (0..nsim).map(|vox| im[vox * n + g]).collect();
                    resample_by_pose(&vol, b.sim_grid.dims, v2w, q)
                }).collect(),
            }).collect()
        }).collect();
        let Relaxation::Class { t2_ms, t2p_ms } = &b.relax else { panic!("class mode") };
        let t1_of: Vec<f32> = ph.labels.iter().map(|(l, _)| ph.t1[ph.dseg.iter().position(|d| d == l).unwrap()] * 1000.0).collect();
        let group = |t: &dyn Fn(usize) -> f32, blood: f32| -> Vec<T2Volume<'static>> {
            (0..k).map(|i| T2Volume::Uniform(t(i))).chain((0..k).map(|_| T2Volume::Uniform(blood)))
                .chain((0..k).map(|i| T2Volume::Uniform(t(i)))).collect()
        };
        let t2v = group(&|i| t2_ms[i], b.t2_blood_ms);
        let tiv: Vec<T2Volume> = (0..3).flat_map(|_| (0..k).map(|i| T2Volume::Uniform(t2p_ms[i]))).collect();
        let t1v = group(&|i| t1_of[i], (p.t1b.0 * 1000.0) as f32);
        let acquire = |ev_takes_tissue_factor: bool| {
            let mut w = Vec::with_capacity(n * shots * ncomp);
            for g in 0..n {
                for s in 0..shots {
                    let f = &h.prep_factors[g * shots + s];
                    assert_eq!((f.raw, f.shot), (g, s));
                    for c in 0..ncomp {
                        let x = if c < k || (c >= 2 * k && ev_takes_tissue_factor) { f.tissue } else { f.label };
                        w.push(x * f.shot_gain);
                    }
                }
            }
            let lw = LineWeights { n_shots: shots, n_compartments: ncomp, w };
            simulate_acquisition_3d(b.sim_grid.dims, b.acq_grid.dims, n, &img, &t2v, Some(&t1v), &b.fmap_sim, Some(&tiv), &b.acq,
                                    &r3.train, &r3.readout, Some(&lw), Some(&sets), &zero_phase(), p.seed)
        };
        let decoded_of = |(m, ph_): (Vec<f32>, Vec<f32>)| -> Vec<Vec<(f64, f64)>> {
            let raw: Vec<Vec<(f64, f64)>> = (0..n).map(|i| cvol(&m, &ph_, n, i)).collect();
            let refs: Vec<&[(f64, f64)]> = raw.iter().map(|v| v.as_slice()).collect();
            crate::hadamard::decode(&refs, order)
        };
        let series: Vec<Vec<(f64, f64)>> = (0..n_sub).map(|j| cvol(&out.mag, &out.phase, out.n_volumes, j)).collect();
        let smax = (0..n).flat_map(|i| cvol(&h.raw_mag, &h.raw_phase, n, i)).map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
        let worst = |d: &[Vec<(f64, f64)>]| -> f64 {
            series.iter().zip(d).map(|(a, r)| {
                let max_ref = r.iter().map(|z| z.0.hypot(z.1)).fold(0.0f64, f64::max);
                a.iter().zip(r).map(|(x, y)| (x.0 - y.0).hypot(x.1 - y.1)).fold(0.0f64, f64::max) / (1e-6 * smax + 1e-4 * max_ref)
            }).fold(0.0f64, f64::max)
        };
        let good = worst(&decoded_of(acquire(false)));
        let bad = worst(&decoded_of(acquire(true)));
        println!("3D per preparation: worst {good:.3e} of the bound; extravascular with the tissue factor {bad:.3e}");
        assert!(good <= 1.0, "{good}");
        assert!(bad > 10.0, "{bad}");
        // the preparation table: one row per shot with the factors actually applied, which differ
        assert_eq!(h.prep_factors.len(), n * shots);
        assert!(h.prep_factors.iter().all(|f| f.tissue.is_finite() && f.tissue != 1.0));
        assert!(h.prep_factors.chunks(2).any(|c| c[0].tissue != c[1].tissue && c[0].label != c[1].label));
        assert!(h.prep_factors.iter().any(|f| f.shot_gain != 1.0));
    }

    /// Hadamard x multi-TE: each echo's raw series is decoded separately, and echo e is the
    /// one-echo Hadamard series at TE_e bit for bit (noise off).
    #[test]
    fn each_echo_decodes_as_its_own_series() {
        let ph = crop();
        let order = 4;
        let n_sub = order - 1;
        let (ld, pld): (Vec<f64>, Vec<f64>) = (0..n_sub).map(|j| (TAU, PLD + TAU * (n_sub - 1 - j) as f64)).unzip();
        let ctx = format!("volume_type\n{}", "deltam\n".repeat(n_sub));
        let side = |te: f64| {
            let mut s = sidecar(ld.clone(), pld.clone(), vec![4.0; n_sub], "Absent", true);
            s["EchoTime"] = json!(te);
            s["SliceTiming"] = json!([0.0, 0.07]);
            s
        };
        let ov = overlay(&format!("[hadamard]\norder = {order}\n"), true);
        let tes = [0.013, 0.032];
        let p = parse_echoes(&[side(tes[0]), side(tes[1])], &ctx, Some(&ov), None).unwrap();
        let out = simulate_p6(&p, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        assert_eq!(out.more_echoes.len(), 1);
        for (e, &te) in tes.iter().enumerate() {
            let q = parse_echoes(&[side(te)], &ctx, Some(&ov), None).unwrap();
            let one = simulate_p6(&q, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
            let (m, phs) = if e == 0 { (&out.mag, &out.phase) } else { (&out.more_echoes[0].mag, &out.more_echoes[0].phase) };
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(m), bits(&one.mag), "echo {e}");
            assert_eq!(bits(phs), bits(&one.phase), "echo {e}");
        }
        assert_eq!(out.hadamard.as_ref().unwrap().raw_more_echoes.len(), 1);
    }

    /// The transient with shot motion (2D, multiband shot events, gradient echo after an included
    /// M0): the leakage the series reports is the decoded tissue part of its own data, i.e. the
    /// decoded full images minus the decoded blood alone (the acquisition is linear in its
    /// compartments), so the tissue-only path carries the same poses and events as the images.
    #[test]
    fn the_leakage_is_the_decoded_tissue_part_under_shot_motion() {
        let ph = crop();
        let order = 4;
        let n_sub = order - 1;
        let (mut ld, mut pld, mut tr) = (vec![0.0], vec![0.0], vec![6.0]);
        for j in 0..n_sub {
            ld.push(TAU);
            pld.push(PLD + TAU * (n_sub - 1 - j) as f64);
            tr.push(4.0);
        }
        let mut s = sidecar(ld, pld, tr, "Included", true);
        s["SliceTiming"] = json!([0.0, 0.0]);
        s["MultibandAccelerationFactor"] = json!(2);
        let ctx = format!("volume_type\nm0scan\n{}", "deltam\n".repeat(n_sub));
        let ov = overlay(&format!("[hadamard]\norder = {order}\n[motion]\nwithin_volume = {{ dropout_rate = 0.6, severity = 0.4, \
                                   jump_mm = [0.8, 0.0, 0.0], jump_deg = [0.0, 0.0, 2.0] }}\n"), true);
        let p = parse_echoes(&[s], &ctx, Some(&ov), None).unwrap();
        let out = simulate_p6(&p, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let h = out.hadamard.as_ref().unwrap();
        assert!(h.flags.transients && h.flags.motion && !out.events.is_empty());
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        let n = sched.raw_rows.len();
        let to = b.tissue_only.as_ref().unwrap();
        let blood: Vec<Vec<f32>> = b.images.iter().zip(to).map(|(a, t)| a.iter().zip(t).map(|(x, y)| x - y).collect()).collect();
        let (t2v, tiv) = class_volumes(&b);
        let acquire = |img: &[Vec<f32>]| simulate_acquisition_oversampled(
            b.sim_grid.dims, b.acq_grid.dims, n, img, &t2v, &b.fmap_sim, Some(&tiv), &b.acq, &vec![None; n], &vec![None; n],
            &zero_phase(), p.seed, None, None);
        let decoded = |(m, ph_): (Vec<f32>, Vec<f32>)| -> Vec<Vec<(f64, f64)>> {
            let raw: Vec<Vec<(f64, f64)>> = sched.cycles[0].raws.clone().map(|i| cvol(&m, &ph_, n, i)).collect();
            let refs: Vec<&[(f64, f64)]> = raw.iter().map(|v| v.as_slice()).collect();
            crate::hadamard::decode(&refs, order)
        };
        let (full, bl) = (decoded(acquire(&b.images)), decoded(acquire(&blood)));
        let brain: Vec<bool> = b.r_acq.majority(&ph.dseg).iter().map(|l| *l > 0).collect();
        let l = &h.leakage.as_ref().unwrap()[0];
        for j in 0..n_sub {
            let norm = full[j].iter().zip(&bl[j]).zip(&brain).filter(|(_, m)| **m)
                .map(|((a, c), _)| (a.0 - c.0).powi(2) + (a.1 - c.1).powi(2)).sum::<f64>().sqrt();
            let (abs, _) = l.per_subbolus[j];
            println!("shot motion: sub-bolus {j} leakage {abs:.4}, decoded full - blood {norm:.4}");
            assert!(abs > 1e-4 * l.reference_norm, "sub-bolus {j}: no leakage");
            assert!((norm - abs).abs() <= 1e-4 * abs, "sub-bolus {j}: {norm} vs {abs}");
        }
    }

    /// Review fixes: a 3D Hadamard raw volume's kinetic truth is static under volume motion (its
    /// shots have their own poses), so it equals the motionless series' raw truth; and each echo's
    /// leakage is acquired at its own TE, equal to the one-echo series' at that TE.
    #[test]
    fn three_d_raw_truth_is_static_and_leakage_is_per_echo() {
        let ph = crop();
        let order = 4;
        let n_sub = order - 1;
        let (ld, pld): (Vec<f64>, Vec<f64>) = (0..n_sub).map(|j| (TAU, PLD + TAU * (n_sub - 1 - j) as f64)).unzip();
        let ctx = format!("volume_type\n{}", "deltam\n".repeat(n_sub));
        let run3d = |motion: &str| {
            let p = parse_echoes(&[sidecar3d(ld.clone(), pld.clone(), vec![4.0; n_sub])], &ctx,
                                 Some(&overlay(&format!("[hadamard]\norder = {order}\n{motion}"), false)), None).unwrap();
            simulate_p6(&p, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap()
        };
        let still = run3d("");
        let moved = run3d("[motion]\nmode = \"random\"\ntrans_mm = [1.0, 1.0, 0.0]\nrot_deg = [0.0, 0.0, 2.0]\n");
        let (hs, hm) = (still.hadamard.as_ref().unwrap(), moved.hadamard.as_ref().unwrap());
        assert!(hm.raw_delta_m_static.is_none());
        assert_eq!(hm.raw_delta_m, hs.raw_delta_m);
        assert!(moved.poses.iter().all(|q| *q != Pose::IDENTITY));

        // two echoes after an included M0 (a gradient-echo transient, so leakage is nonzero)
        let (mut ld2, mut pld2, mut tr2) = (vec![0.0], vec![0.0], vec![6.0]);
        ld2.extend(&ld);
        pld2.extend(&pld);
        tr2.extend(vec![4.0; n_sub]);
        let side = |te: f64| {
            let mut s = sidecar(ld2.clone(), pld2.clone(), tr2.clone(), "Included", true);
            s["EchoTime"] = json!(te);
            s["SliceTiming"] = json!([0.0, 0.07]);
            s
        };
        let ctx2 = format!("volume_type\nm0scan\n{}", "deltam\n".repeat(n_sub));
        let ov = overlay(&format!("[hadamard]\norder = {order}\n"), true);
        let two = parse_echoes(&[side(0.013), side(0.032)], &ctx2, Some(&ov), None).unwrap();
        let out = simulate_p6(&two, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let h = out.hadamard.as_ref().unwrap();
        let one = parse_echoes(&[side(0.032)], &ctx2, Some(&ov), None).unwrap();
        let o1 = simulate_p6(&one, &ph, T2Mode::Class, &zero_phase(), RowOverride::None, None).unwrap();
        let (e2, single) = (&h.leakage_more_echoes[0], o1.hadamard.as_ref().unwrap().leakage.as_ref().unwrap());
        assert_eq!(e2, single);
        assert!(e2[0].per_subbolus[0].0 > 0.0);
        assert_ne!(h.leakage.as_ref().unwrap()[0].per_subbolus[0].0, e2[0].per_subbolus[0].0);
    }

    /// The Look-Locker table's group means pool the group's slices (sums over counts): two slices
    /// excited together, each label's mean over every one of its phantom voxels in both slabs,
    /// from an independent call of the timeline per voxel.
    #[test]
    fn look_locker_group_means_pool_the_slices() {
        use crate::longitudinal::{tissue_mz_ll_sequence, LlCycle};
        use crate::resample::{axis_aligned_voxels, corner_offset};
        let ph = crop();
        let pld: Vec<f64> = (0..4).map(|n| 0.6 + 0.3 * n as f64).collect();
        let mut s = sidecar(vec![1.0; 4], pld, vec![4.5; 4], "Absent", true);
        s["LookLocker"] = json!(true);
        s["FlipAngle"] = json!(35);
        s["SliceTiming"] = json!([0.0, 0.0]);
        let p = parse_echoes(&[s], "volume_type\ncontrol\ncontrol\ncontrol\ncontrol\n", Some(&overlay("", true)), None).unwrap();
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        assert_eq!(b.ll_lines.len(), 4, "one group of two slices, four readouts");
        let o = p.acq.oversample;
        let dv = p.voxel_size_mm;
        let r_sim = Resampler::with_offset(ph.grid.dims, axis_aligned_voxels(&ph.grid).unwrap(), b.sim_grid.dims,
                                           [dv[0] / o as f64, dv[1] / o as f64, dv[2]], corner_offset(&ph.grid, &b.acq_grid).unwrap());
        let [pnx, pny, _] = ph.grid.dims;
        // the resolved flips (the test overlay's excitation_flip_angle, 60, over the sidecar's)
        let flips = p.look_locker.as_ref().unwrap().flip_deg.clone();
        assert_eq!(flips, vec![60.0; 4]);
        let cyc = LlCycle { tr: 4.5, s: None, t_read: p.rows.iter().map(|r| r.t).collect(), flip_deg: flips };
        for (lab, (l, _)) in ph.labels.iter().enumerate() {
            let (mut sum, mut count) = (vec![0.0f64; 4], 0usize);
            // the slabs of the two slices, each voxel counted once per slab it belongs to (as the series does)
            for z in 0..b.sim_grid.dims[2] {
                for &(zs, _) in r_sim.z_slab(z) {
                    for xy in 0..pnx * pny {
                        let i = zs * pnx * pny + xy;
                        if ph.dseg[i] == *l {
                            let mz = tissue_mz_ll_sequence(ph.m0[i] as f64, ph.t1[i] as f64, std::slice::from_ref(&cyc)).remove(0);
                            for (n, x) in mz.iter().enumerate() {
                                sum[n] += x;
                            }
                            count += 1;
                        }
                    }
                }
            }
            for (n, line) in b.ll_lines.iter().enumerate() {
                let want = sum[n] / count as f64;
                assert!((line.tissue_mz[lab] - want).abs() <= 1e-9 * want.abs(), "label {l} readout {n}: {} vs {want}", line.tissue_mz[lab]);
            }
        }
    }

    // ---- P7 part A: the P4 parts under Look-Locker

    /// A label cycle of six readouts (PCASL, 1 s labeling, readouts 0.3 s apart from PLD 0.6 s, 60
    /// degrees) with exchange, the arterial compartment, crushing alternating per readout and a
    /// bolus-position slab pulse after the labeling whose entry cut splits the bolus.
    fn quasar_like(parts: &str) -> Protocol {
        let m = 6;
        let pld: Vec<f64> = (0..m).map(|n| 0.6 + 0.3 * n as f64).collect();
        let mut s = sidecar(vec![1.0; m], pld, vec![4.5; m], "Absent", true);
        s["LookLocker"] = json!(true);
        s["BackgroundSuppression"] = json!(true);
        s["BackgroundSuppressionNumberPulses"] = json!(1);
        s["BackgroundSuppressionPulseTime"] = json!([1.2]);
        s["VascularCrushing"] = json!(true);
        s["VascularCrushingVENC"] = json!([0.0, 4.0, 0.0, 4.0, 0.0, 4.0]);
        let ctx = format!("volume_type\n{}", "label\n".repeat(m));
        parse_echoes(&[s], &ctx, Some(&overlay(parts, true)), None).unwrap()
    }

    const QUASAR_PARTS: &str = "[kinetic]\nexchange_time = 0.6\n\
        [macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
        arterial_transit_time = { grey_matter = 1.2, white_matter = 1.8, csf = 0.0 }\n\
        [vascular_crushing]\narterial_velocity = { grey_matter = 10.0, white_matter = 6.0, csf = 3.0 }\n\
        [background_suppression]\nmodel = \"bolus-position\"\npulse_region = \"slab\"\nslab_entry_time = 0.3\n";

    /// Every Part A truth of the series against the parcel reference, voxel by voxel through the
    /// series' own resampler: the read's intravascular and extravascular parts, their sum, and the
    /// fresh arterial read with its per-readout survival; and the blood and arterial images
    /// (noiseless, static) against the same references on the simulation grid.
    #[test]
    fn look_locker_reads_every_p4_part_as_the_parcel_reference() {
        use crate::crushing::survival;
        use crate::kinetic::parcel_ref::{Case, Region as PRegion};
        let ph = crop();
        let p = quasar_like(QUASAR_PARTS);
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        let (iv_gt, ev_gt, art_gt, tot_gt) = (b.gt_read_iv.as_ref().unwrap(), b.gt_read_ev.as_ref().unwrap(),
                                              b.gt_read_art.as_ref().unwrap(), b.gt_read.as_ref().unwrap());
        let n = b.n;
        let eps = p.suppression.as_ref().unwrap().for_row(0).epsilon;
        let vel = [0.0, 10.0, 6.0, 3.0];
        let abv = [0.0, 0.03, 0.015, 0.0];
        let aatt = [0.0, 1.2, 1.8, 0.0];
        let fl = p.look_locker.as_ref().unwrap().flip_deg.clone();
        let [nx, ny, nz] = b.acq_grid.dims;
        let r_sim = Resampler::with_offset(ph.grid.dims, crate::resample::axis_aligned_voxels(&ph.grid).unwrap(), b.sim_grid.dims,
            [p.voxel_size_mm[0] / p.acq.oversample as f64, p.voxel_size_mm[1] / p.acq.oversample as f64, p.voxel_size_mm[2]],
            crate::resample::corner_offset(&ph.grid, &b.acq_grid).unwrap());
        let [snx, sny, _] = b.sim_grid.dims;
        let k = b.k;
        let mut checked = 0;
        for v in 0..n {
            let row = &p.rows[v];
            let kin = p.kinetic(row);
            let fa = fl[v].to_radians().sin();
            let venc = [0.0, 4.0][v % 2];
            for z in 0..nz {
                let t = row.t + b.slice_offsets[z];
                let excitations: Vec<(f64, f64)> = (0..v).map(|r| (p.rows[r].t + b.slice_offsets[z], fl[r])).collect();
                let reference = |i: usize| -> (f64, f64, f64) {
                    let l = ph.dseg[i];
                    if l <= 0 {
                        return (0.0, 0.0, 0.0);
                    }
                    let c = Case {
                        k: kin, f: ph.perfusion[i] as f64, att: ph.att[i] as f64, t1t: ph.t1[i] as f64, m0: ph.m0[i] as f64, t,
                        excitations: excitations.clone(), entry_lead: 0.0, pulses: vec![1.2], epsilon: eps,
                        region: PRegion::Slab(0.3), tau_ex: Some(0.6), span: (0.0, kin.tau),
                    };
                    let (iv, total) = c.read(4);
                    let l = l as usize;
                    let art = c.arterial(abv[l], aatt[l], 0.0) * survival(vel[l], venc);
                    (iv, total - iv, art)
                };
                // the truths, on the acquired grid
                for (which, gt) in [(0, iv_gt), (1, ev_gt), (2, art_gt), (3, tot_gt)] {
                    let want = b.r_acq.mean_slice(z, |i| {
                        let (iv, ev, art) = reference(i);
                        fa * [iv, ev, art, iv + ev][which]
                    });
                    let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
                    for (jj, w) in want.iter().enumerate() {
                        let got = gt[(z * nx * ny + jj) * n + v];
                        assert!((got - w).abs() <= 1e-5 * peak, "readout {v} slice {z} part {which} voxel {jj}: {got} vs {w}");
                        checked += usize::from(w.abs() > 0.01 * peak);
                    }
                }
                // the images: a label row's blood compartments read -sin(a) iv, its arterial ones
                // -sin(a) times the arterial read
                for (first, which) in [(k, 0), (2 * k, 2)] {
                    let want = r_sim.mean_slice(z, |i| { let r = reference(i); -fa * [r.0, r.1, r.2][which] });
                    let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-6);
                    for (jj, w) in want.iter().enumerate() {
                        let got: f32 = (first..first + k).map(|c| b.images[c][(z * snx * sny + jj) * n + v]).sum();
                        assert!((got - w).abs() <= 1e-5 * peak, "image readout {v} slice {z} from {first} voxel {jj}: {got} vs {w}");
                    }
                }
            }
        }
        assert!(checked > 100, "too few nonzero truth voxels: {checked}");
        // the arterial truth is not vacuous: readouts 0 and 1 (1.6 and 1.9 s) are inside grey
        // matter's arterial window [1.2, 2.2), and readout 1 is crushed (the slab pulse inverts
        // the arterial parcel, so both are negative)
        let sum = |gt: &[f32], v: usize| (0..nx * ny * nz).map(|j| gt[j * n + v] as f64).sum::<f64>().abs();
        assert!(sum(art_gt, 0) > 0.0 && sum(art_gt, 1) > 0.0, "{} {}", sum(art_gt, 0), sum(art_gt, 1));
        assert!(sum(art_gt, 1) < sum(art_gt, 0), "crushing: {} vs {}", sum(art_gt, 1), sum(art_gt, 0));
    }

    /// Without a Part A input the Look-Locker series writes no part truths, and a series with only
    /// the arterial compartment reads the label as P6 does (`delta_m_read`).
    #[test]
    fn look_locker_part_truths_follow_their_parts() {
        let ph = crop();
        let mut s = sidecar(vec![1.0; 4], (0..4).map(|n| 0.6 + 0.3 * n as f64).collect(), vec![4.5; 4], "Absent", true);
        s["LookLocker"] = json!(true);
        let ctx = format!("volume_type\n{}", "label\n".repeat(4));
        let p = parse_echoes(&[s.clone()], &ctx, Some(&overlay("", true)), None).unwrap();
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &Schedule::new(&p)).unwrap();
        assert!(b.gt_read.is_some() && b.gt_read_iv.is_none() && b.gt_read_ev.is_none() && b.gt_read_art.is_none());
        let macro_only = "[macrovascular]\narterial_blood_volume = { grey_matter = 0.03, white_matter = 0.015, csf = 0.0 }\n\
            arterial_transit_time = { grey_matter = 0.5, white_matter = 0.7, csf = 0.0 }\n";
        let pm = parse_echoes(&[s], &ctx, Some(&overlay(macro_only, true)), None).unwrap();
        let bm = build(&pm, &ph, T2Mode::Class, RowOverride::None, pm.echo_times_s[0], &Schedule::new(&pm)).unwrap();
        assert!(bm.gt_read_art.is_some() && bm.gt_read_iv.is_none());
        // the label read is P6's, bit for bit
        assert_eq!(b.gt_read.as_ref().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                   bm.gt_read.as_ref().unwrap().iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    }

    // ---- P7 part B: Look-Locker x Hadamard

    const LLH_PLDS: [f64; 3] = [1.0, 1.3, 1.6];
    const LLH_TAU: f64 = 0.3;

    /// Hadamard-4 PCASL on the crop (three 0.3 s sub-boli), each encoded preparation read at
    /// PLD_n = 1.0, 1.3, 1.6 s (60 degrees), `cycles` encoding cycles, an m0scan row first with
    /// `m0`; the context the decoded volumes, readout-major.
    fn ll_hadamard_p(cycles: usize, m0: bool, extra: &str) -> Protocol {
        let (mut ld, mut pld, mut tr, mut ctx) = (Vec::new(), Vec::new(), Vec::new(), String::from("volume_type\n"));
        if m0 {
            ld.push(0.0);
            pld.push(0.0);
            tr.push(6.0);
            ctx.push_str("m0scan\n");
        }
        for _ in 0..cycles {
            for p_n in LLH_PLDS {
                for j in 0..3 {
                    ld.push(LLH_TAU);
                    pld.push(p_n + LLH_TAU * (2 - j) as f64);
                    tr.push(4.0);
                    ctx.push_str("deltam\n");
                }
            }
        }
        let mut s = sidecar(ld, pld, tr, if m0 { "Included" } else { "Absent" }, true);
        s["LookLocker"] = json!(true);
        parse_echoes(&[s], &ctx, Some(&overlay(&format!("[hadamard]\norder = 4\n[look_locker]\nreadouts_per_cycle = 3\n{extra}"), true)), None)
            .unwrap()
    }

    /// The raw images summed over compartments, as magnitude and phase (0 or pi), voxel-major.
    fn summed(b: &Built) -> (Vec<f32>, Vec<f32>) {
        let n = b.n;
        let tot: Vec<f32> = (0..b.nvox_sim * n).map(|x| b.images.iter().map(|c| c[x]).sum::<f32>()).collect();
        (tot.iter().map(|x| x.abs()).collect(), tot.iter().map(|x| if *x < 0.0 { std::f32::consts::PI } else { 0.0 }).collect())
    }

    /// Decoded output `k` at simulation voxel `x`, as a real number.
    fn real_at(m: &[f32], ph: &[f32], n_out: usize, x: usize, k: usize) -> f64 {
        m[x * n_out + k] as f64 * (ph[x * n_out + k] as f64).cos()
    }

    /// Decoded (j, n) of a noiseless steady-state series is sin(a_n) times sub-bolus j as readout n
    /// read it (depleted by the readouts before it in its preparation; with exchange), against the
    /// parcel reference through the series' resampler: the raw images decoded per readout by the
    /// production decode. Two encoding rows swapped, or one raw volume's sign flipped, fail it.
    #[test]
    fn look_locker_hadamard_decodes_per_readout() {
        use crate::kinetic::parcel_ref::{Case, Region as PRegion};
        let ph = crop();
        let p = ll_hadamard_p(2, false, "[kinetic]\nexchange_time = 0.6\n");
        let sched = Schedule::new(&p);
        let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
        let n = b.n;
        assert_eq!(n, 2 * 4 * 3);
        let r_sim = Resampler::with_offset(ph.grid.dims, crate::resample::axis_aligned_voxels(&ph.grid).unwrap(), b.sim_grid.dims,
            [p.voxel_size_mm[0] / p.acq.oversample as f64, p.voxel_size_mm[1] / p.acq.oversample as f64, p.voxel_size_mm[2]],
            crate::resample::corner_offset(&ph.grid, &b.acq_grid).unwrap());
        let [snx, sny, nz] = b.sim_grid.dims;
        let n_out = sched.outputs.len();
        let fa = 60f64.to_radians();
        let h = p.hadamard.as_ref().unwrap();
        let kin = p.kinetic(&sched.raw_rows[0]);
        // the reference of every output, per slice
        let want: Vec<Vec<Vec<f32>>> = sched.outputs.iter().map(|o| {
            let Output::Decoded { subbolus, readout, .. } = *o else { unreachable!("no m0scan") };
            (0..nz).map(|z| {
                let off = b.slice_offsets[z];
                let e = |m: usize| 0.9 + LLH_PLDS[m] + off;
                r_sim.mean_slice(z, |i| {
                    if ph.dseg[i] <= 0 {
                        return 0.0;
                    }
                    let c = Case {
                        k: kin, f: ph.perfusion[i] as f64, att: ph.att[i] as f64, t1t: ph.t1[i] as f64, m0: ph.m0[i] as f64,
                        t: e(readout), excitations: (0..readout).map(|m| (e(m), 60.0)).collect(), entry_lead: 0.0, pulses: vec![],
                        epsilon: 0.0, region: PRegion::Global, tau_ex: Some(0.6), span: h.spans[subbolus],
                    };
                    fa.sin() * c.read(4).1
                })
            }).collect()
        }).collect();
        let peak = want.iter().flatten().flatten().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
        assert!(peak > 0.0);
        let worst = |m: &[f32], phs: &[f32]| -> f64 {
            let mut w = 0.0f64;
            for (k, per) in want.iter().enumerate() {
                for (z, sl) in per.iter().enumerate() {
                    for (jj, &x) in sl.iter().enumerate() {
                        w = w.max((real_at(m, phs, n_out, z * snx * sny + jj, k) - x as f64).abs());
                    }
                }
            }
            w
        };
        let (m, phs) = summed(&b);
        let (dm, dp) = decode_series(&sched, 4, &m, &phs, n);
        let err = worst(&dm, &dp);
        assert!(err <= 1e-4 * peak, "decoded vs reference: {err:.3e} of peak {peak:.3e}");
        // negative controls: encoding rows 1 and 2 swapped at readout 0 of cycle 0; raw volume 3's sign
        let mut swapped = (m.clone(), phs.clone());
        for x in 0..b.nvox_sim {
            swapped.0.swap(x * n + 3, x * n + 6);
            swapped.1.swap(x * n + 3, x * n + 6);
        }
        let (sm, sp) = decode_series(&sched, 4, &swapped.0, &swapped.1, n);
        assert!(worst(&sm, &sp) > 1e-2 * peak, "a swapped column still decodes");
        let mut flipped = phs.clone();
        for x in 0..b.nvox_sim {
            flipped[x * n + 3] = std::f32::consts::PI - flipped[x * n + 3];
        }
        let (fm, fp) = decode_series(&sched, 4, &m, &flipped, n);
        assert!(worst(&fm, &fp) > 1e-2 * peak, "a sign flip still decodes");
        // the decoded read truth is the same reference, on the acquired grid, its parts summing to it
        let gt = b.gt_read.as_ref().unwrap();
        let (giv, gev) = (b.gt_read_iv.as_ref().unwrap(), b.gt_read_ev.as_ref().unwrap());
        let [anx, any, _] = b.acq_grid.dims;
        assert_eq!(gt.len(), anx * any * nz * n_out);
        for (k, o) in sched.outputs.iter().enumerate() {
            let Output::Decoded { subbolus, readout, .. } = *o else { unreachable!() };
            for z in 0..nz {
                let off = b.slice_offsets[z];
                let e = |m: usize| 0.9 + LLH_PLDS[m] + off;
                let w = b.r_acq.mean_slice(z, |i| {
                    if ph.dseg[i] <= 0 {
                        return 0.0;
                    }
                    let c = Case {
                        k: kin, f: ph.perfusion[i] as f64, att: ph.att[i] as f64, t1t: ph.t1[i] as f64, m0: ph.m0[i] as f64,
                        t: e(readout), excitations: (0..readout).map(|m| (e(m), 60.0)).collect(), entry_lead: 0.0, pulses: vec![],
                        epsilon: 0.0, region: PRegion::Global, tau_ex: Some(0.6), span: h.spans[subbolus],
                    };
                    fa.sin() * c.read(4).1
                });
                for (jj, x) in w.iter().enumerate() {
                    let at = (z * anx * any + jj) * n_out + k;
                    assert!((gt[at] - x).abs() <= 1e-5 * peak as f32, "truth output {k} slice {z}: {} vs {x}", gt[at]);
                    assert!((giv[at] + gev[at] - gt[at]).abs() <= 1e-5 * peak as f32);
                }
            }
        }
    }

    /// In the steady state the encoding rows' tissue is the same and decodes away at every readout;
    /// an included M0 before the first cycle leaves the first rows' tissue in a transient, and the
    /// decoded tissue part L_{j,n} is nonzero.
    #[test]
    fn look_locker_hadamard_tissue_residual() {
        let ph = crop();
        for (m0, transient) in [(false, false), (true, true)] {
            let p = ll_hadamard_p(1, m0, "");
            let sched = Schedule::new(&p);
            let b = build(&p, &ph, T2Mode::Class, RowOverride::None, p.echo_times_s[0], &sched).unwrap();
            let n = b.n;
            let to = b.tissue_only.as_ref().unwrap();
            let tot: Vec<f32> = (0..b.nvox_sim * n).map(|x| to.iter().map(|c| c[x]).sum::<f32>()).collect();
            let peak = tot.iter().fold(0.0f32, |a, x| a.max(x.abs())) as f64;
            let (mm, pp): (Vec<f32>, Vec<f32>) = (tot.iter().map(|x| x.abs()).collect(),
                                                  tot.iter().map(|x| if *x < 0.0 { std::f32::consts::PI } else { 0.0 }).collect());
            let (dm, dp) = decode_series(&sched, 4, &mm, &pp, n);
            let n_out = sched.outputs.len();
            let mut resid = 0.0f64;
            for (k, o) in sched.outputs.iter().enumerate() {
                if matches!(o, Output::Decoded { .. }) {
                    for x in 0..b.nvox_sim {
                        resid = resid.max(real_at(&dm, &dp, n_out, x, k).abs());
                    }
                }
            }
            if transient {
                assert!(resid > 1e-3 * peak, "the transient leaves no residual: {resid:.3e} of {peak:.3e}");
            } else {
                assert!(resid < 1e-5 * peak, "the steady state leaves a residual: {resid:.3e} of {peak:.3e}");
            }
        }
    }
}
