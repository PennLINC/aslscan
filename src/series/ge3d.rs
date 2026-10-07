//! The series of a 3D gradient-echo train (P7 addendum, part C; `[readout] type = "epi3d"`).
//!
//! Each kz partition is read by its own small-flip excitation, so the train's longitudinal state
//! enters the acquisition as one weight per excitation (`mrsim_acq::kspace3d::GeVolume`). This
//! module computes those weights: the tissue's from the excitation-train timeline, exactly, by
//! grouping each tissue compartment by `T1` (tissue `Mz` is `M0(r) m_j(T1(r))`); the label's (Task
//! 14) by depletion from slab entry and a piecewise-linear expansion over the train.

use super::*;
use crate::longitudinal::{tissue_mz_ll, tissue_mz_ll_sequence, LlCycle, Suppression};
use crate::protocol::SlabEntryTime;
use crate::schedule::Schedule;

/// One tissue group (P7 addendum, part C, "Tissue is separable"): the phantom voxels of compartment
/// `compartment` (a label in class mode; every foreground voxel in voxel mode) whose `T1` is `t1_s`.
/// Their tissue signal before excitation `j` is `M0(r) m_j(t1_s)`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TissueGroup {
    pub compartment: usize,
    pub t1_s: f64,
    /// Phantom-grid membership.
    pub mask: Vec<bool>,
}

/// The tissue groups of a phantom: each compartment mask (`masks`, phantom grid: one per label in
/// class mode, one in voxel mode) split by distinct `T1` value at the phantom voxels, before any
/// resampling. More than `max_groups` is refused, naming the count: a smooth `T1` map is not
/// approximated.
pub(super) fn tissue_groups(ph: &Phantom, masks: &[Vec<bool>], max_groups: usize) -> Result<Vec<TissueGroup>, String> {
    let mut groups: Vec<TissueGroup> = Vec::new();
    for (c, m) in masks.iter().enumerate() {
        let mut t1s: Vec<u32> = ph.t1.iter().zip(m).filter(|(_, &on)| on).map(|(t, _)| t.to_bits()).collect();
        t1s.sort_unstable();
        t1s.dedup();
        if groups.len() + t1s.len() > max_groups {
            let total = groups.len() + t1s.len() + masks[c + 1..].len();
            return Err(format!(
                "the phantom's tissue has at least {total} distinct (compartment, T1) groups, more than [readout] \
                 max_t1_groups = {max_groups}: a 3D gradient-echo train's tissue weights are exact per T1 group, and a \
                 smooth T1 map is refused rather than approximated (raise max_t1_groups, or use label-wise T1)"));
        }
        for bits in t1s {
            let mask: Vec<bool> = ph.t1.iter().zip(m).map(|(t, &on)| on && t.to_bits() == bits).collect();
            groups.push(TissueGroup { compartment: c, t1_s: f32::from_bits(bits) as f64, mask });
        }
    }
    Ok(groups)
}

/// The excitations of one train (P7 addendum, part C, "Excitation times"): `n_exc` at
/// `start + j spacing` (s from the start of the repetition), each at `flip_deg`.
pub(super) fn train_times(start_s: f64, n_exc: usize, spacing_s: f64) -> Vec<f64> {
    (0..n_exc).map(|j| start_s + j as f64 * spacing_s).collect()
}

/// One preparation's cycle: its suppression pulses before the train, then the train's excitations
/// as events on `Mz`, recovery to `tr`. An m0scan preparation's train starts at the start of its
/// repetition, with no pulses.
pub(super) fn train_cycle<'a>(tr: f64, s: Option<&'a Suppression>, excitations: Vec<f64>, flip_deg: f64) -> LlCycle<'a> {
    let n = excitations.len();
    LlCycle { tr, s, t_read: excitations, flip_deg: vec![flip_deg; n] }
}

/// `m_j(T1)` for every excitation of every preparation, in acquisition order, at unit `M0`: the
/// first preparation from its own steady state (the dummy repetitions are taken to have run it),
/// each later one from the end of the one before (P5's two rules: a series of identical
/// preparations stays at the fixed point). It is the Look-Locker timeline itself
/// ([`tissue_mz_ll_sequence`]): every excitation is an event.
pub(super) fn train_mz(t1_s: f64, cycles: &[LlCycle]) -> Vec<Vec<f64>> {
    tissue_mz_ll_sequence(1.0, t1_s, cycles)
}

/// The cycle raw volume `v` is read in (P7 addendum, part C, "3D Look-Locker"): its preparation's raw
/// volumes, one per readout, each a sub-train of `n_exc` excitations from its row's excitation time
/// (an m0scan row's from the start of its repetition) at that raw volume's flip. Returns every
/// excitation of the cycle, their flips, and where `v`'s sub-train starts among them. Without
/// Look-Locker the cycle is `v`'s own train.
pub(super) fn cycle_of(sched: &Schedule, raw_flip: &[f64], n_exc: usize, spacing_s: f64, v: usize)
    -> (Vec<f64>, Vec<f64>, usize)
{
    let prep = sched.raws[v].prep;
    let r0 = sched.preps[prep].raw;
    let m = sched.raws[r0..].iter().take_while(|r| r.prep == prep).count();
    let (mut times, mut flips) = (Vec::with_capacity(m * n_exc), Vec::with_capacity(m * n_exc));
    for (row, &fa) in sched.raw_rows[r0..r0 + m].iter().zip(&raw_flip[r0..r0 + m]) {
        let start = if row.kind == RowKind::M0scan { 0.0 } else { row.t };
        times.extend(train_times(start, n_exc, spacing_s));
        flips.extend(std::iter::repeat_n(fa, n_exc));
    }
    (times, flips, (v - r0) * n_exc)
}

/// The separate M0 scan's train (P7 addendum, part C, "M0"): the same excitations without
/// labeling or suppression, from the start of its repetition, at its steady state; `m_j(T1)` at
/// unit `M0`.
pub(super) fn m0_train_mz(t1_s: f64, m0_tr_s: f64, n_exc: usize, spacing_s: f64, flip_deg: f64) -> Vec<f64> {
    tissue_mz_ll(1.0, t1_s, &train_cycle(m0_tr_s, None, train_times(0.0, n_exc, spacing_s), flip_deg))
}

// ---- P7 Task 14: depletion from slab entry, the blood interpolation ----

/// How long before its arrival a voxel's label entered the slab (P7 addendum, part C, "Depletion
/// from slab entry"): `arrival - d`, zero for `"arrival"`. A slab entry after the arrival is
/// refused, naming the voxel (`what` names the arrival: "ATT" or "aATT").
pub(super) fn entry_lead(entry: SlabEntryTime, arrival_s: f64, voxel: usize, what: &str) -> Result<f64, String> {
    match entry {
        SlabEntryTime::Arrival => Ok(0.0),
        SlabEntryTime::Seconds(d) if d <= arrival_s => Ok(arrival_s - d),
        SlabEntryTime::Seconds(d) => Err(format!(
            "slab_entry_time {d} s is after phantom voxel {voxel}'s {what} {arrival_s} s: its label would enter the slab \
             after reaching the voxel")),
    }
}

/// The node selection of the blood interpolation (P7 addendum, part C, "Blood is not separable"):
/// `values[r][j]` is a family's read at voxel `r` and excitation `j` of one train. Nodes start at
/// the first, the last and `centre`, and the interval with the worst error is bisected until every
/// voxel's piecewise-linear interpolant is within `tol` of the largest value over the whole train
/// (computed exactly, at every excitation). Nodes at every excitation are exact, so it terminates. A
/// train with no label read needs no nodes.
pub(super) fn select_nodes(values: &[Vec<f64>], centre: usize, tol: f64) -> Vec<usize> {
    let n = values.first().map_or(0, |v| v.len());
    let peak = values.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
    if n == 0 || peak == 0.0 {
        return Vec::new();
    }
    let mut nodes = vec![0, centre.min(n - 1), n - 1];
    nodes.sort_unstable();
    nodes.dedup();
    let bound = tol * peak;
    loop {
        // the worst interval between consecutive nodes
        let mut worst = (0.0f64, 0usize);
        for (k, w) in nodes.windows(2).enumerate() {
            let (a, b) = (w[0], w[1]);
            for v in values {
                for j in a + 1..b {
                    let x = (j - a) as f64 / (b - a) as f64;
                    let err = (v[j] - ((1.0 - x) * v[a] + x * v[b])).abs();
                    if err > worst.0 {
                        worst = (err, k);
                    }
                }
            }
        }
        if worst.0 <= bound {
            return nodes;
        }
        let (a, b) = (nodes[worst.1], nodes[worst.1 + 1]);
        nodes.insert(worst.1 + 1, (a + b) / 2);
    }
}

/// The hat weights of excitation `j` on `nodes` (sorted): `(node index, weight)`, two neighbours
/// (or one at a node), summing to one inside the span; nothing outside it.
pub(super) fn hat(nodes: &[usize], j: usize) -> Vec<(usize, f64)> {
    match nodes.binary_search(&j) {
        Ok(k) => vec![(k, 1.0)],
        Err(0) => Vec::new(),
        Err(k) if k == nodes.len() => Vec::new(),
        Err(k) => {
            let (a, b) = (nodes[k - 1], nodes[k]);
            let x = (j - a) as f64 / (b - a) as f64;
            vec![(k - 1, 1.0 - x), (k, x)]
        }
    }
}

/// The interpolant of `values` (one voxel) at every excitation on `nodes`.
pub(super) fn interpolate(values: &[f64], nodes: &[usize]) -> Vec<f64> {
    (0..values.len()).map(|j| hat(nodes, j).iter().map(|&(k, w)| w * values[nodes[k]]).sum()).collect()
}

/// The images one volume holds while it is simulated (P7 addendum, part C, "Memory"), in bytes:
/// `compartments` images of `nvox_sim` `f32` voxels, times the volumes in flight (the worker
/// threads). Over `limit_gib` it is refused, naming the estimate and the remedies.
pub(super) fn check_memory(nvox_sim: usize, compartments: usize, in_flight: usize, limit_gib: f64, what: &str) -> Result<f64, String> {
    let bytes = 4.0 * nvox_sim as f64 * compartments as f64 * in_flight as f64;
    let gib = bytes / (1u64 << 30) as f64;
    if gib > limit_gib {
        return Err(format!(
            "the 3D gradient-echo series would hold {gib:.2} GiB of images at once ({compartments} compartment images of \
             {nvox_sim} voxels x {in_flight} volumes in flight; {what}), over the limit of {limit_gib} GiB ([images] \
             max_memory_gib): more kz segments (shorter trains need fewer interpolation nodes), a looser \
             [readout] node_tolerance, fewer worker threads, or a higher limit"));
    }
    Ok(gib)
}


// ---- P7 Task 15: the series ----

/// The label families of a 3D gradient-echo series, each its own compartment family: the label
/// not yet exchanged (all of it without exchange; the blood compartment, `T2_blood`), the
/// exchanged label (the tissue compartment's `T2`), the arterial label (`T2_arterial`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Family {
    Blood,
    Extravascular,
    Arterial,
}

impl Family {
    pub fn as_str(&self) -> &'static str {
        match self {
            Family::Blood => "blood (the label not yet exchanged; all of it without exchange)",
            Family::Extravascular => "extravascular (the exchanged label)",
            Family::Arterial => "arterial",
        }
    }
}

/// Everything a 3D gradient-echo series resolves before its acquisition (P7 Task 15): the
/// per-volume builder [`volume_inputs`] reads it, from the worker that simulates the volume.
pub(super) struct Ge3dBuild<'a> {
    pub p: &'a Protocol,
    pub ph: &'a Phantom,
    pub res: crate::protocol::Ge3dResolution,
    pub table: mrsim_acq::readout::Ge3dTable,
    pub sched: Schedule,
    pub r_sim: Resampler,
    pub masks: Vec<Vec<bool>>,
    pub groups: Vec<TissueGroup>,
    /// Per tissue group, its `M0` on the simulation grid (static).
    pub group_images: Vec<Vec<f32>>,
    /// `m_j(T1)` per group, per preparation, per excitation (unit `M0`).
    pub group_mz: Vec<Vec<Vec<f64>>>,
    pub families: Vec<Family>,
    /// Node slots per family and compartment mask.
    pub k_nodes: usize,
    /// Per raw volume, per family: the node excitations (empty: no label read).
    pub nodes: Vec<Vec<Vec<usize>>>,
    /// Per preparation: the tissue and label physiological factors.
    pub prep_factors: Vec<(f64, f64)>,
    /// Per raw volume: the label's sign times P3's global suppression factor.
    pub label_scale: Vec<f64>,
    pub p4: P4,
    pub bolus_region: Option<Region>,
    /// Per phantom voxel: the slab-entry lead of its label and of its arterial label (s).
    pub lead: Vec<f64>,
    pub lead_a: Vec<f64>,
    /// The separate M0's excitation (degrees), and each raw volume's (a Look-Locker FlipAngle array
    /// gives each readout its own).
    pub flip_deg: f64,
    pub raw_flip: Vec<f64>,
    pub spacing_s: f64,
    /// Motion (P5 part D, as on the 3D spin-echo path): each raw volume's pose, each shot's pose
    /// within its volume (the cumulative jumps of the events before it), each shot's gain (an
    /// event shot's `1 - severity`), and the simulation grid's voxel-to-world map.
    pub poses: Vec<Pose>,
    pub shot_pose: Vec<Vec<Pose>>,
    pub shot_gain: Vec<Vec<f64>>,
    pub v2w: [[f64; 4]; 4],
    pub sim_dims: [usize; 3],
}

impl Ge3dBuild<'_> {
    pub fn n_compartments(&self) -> usize {
        self.groups.len() + self.families.len() * self.masks.len() * self.k_nodes
    }

    /// The compartment of node slot `k` of family `f` (index into `families`) and mask `c`.
    pub fn comp(&self, f: usize, c: usize, k: usize) -> usize {
        self.groups.len() + (f * self.masks.len() + c) * self.k_nodes + k
    }

    /// The cycle of raw volume `v` ([`cycle_of`]).
    pub fn cycle(&self, v: usize) -> (Vec<f64>, Vec<f64>, usize) {
        cycle_of(&self.sched, &self.raw_flip, self.res.n_exc, self.spacing_s, v)
    }

    /// The label left of what arrived before raw volume `v`'s sub-train, by the excitations of its
    /// cycle before it: the product of their `cos(a)` (1 without Look-Locker).
    pub fn cumulative_depletion(&self, v: usize) -> f64 {
        let (_, flips, offset) = self.cycle(v);
        flips[..offset].iter().map(|a| a.to_radians().cos()).product()
    }

    /// The reads of phantom voxel `i` in raw volume `v` at every excitation of its sub-train, per
    /// family (before `sin(a)`, sign and factors): the label depleted from slab entry by every
    /// excitation of its cycle before it (earlier sub-trains included), with every P4 part
    /// (sub-boli with their parcel factors, the exchange split), and the arterial read with its
    /// crushing survival and parcel factor.
    pub fn reads(&self, v: usize, i: usize) -> Vec<Vec<f64>> {
        let p = self.p;
        let ph = self.ph;
        let row = &self.sched.raw_rows[v];
        let kin = p.kinetic(row);
        let (e, flips, offset) = self.cycle(v);
        let n = self.res.n_exc;
        let (f_ml, att, t1t, m0) = (ph.perfusion[i] as f64, ph.att[i] as f64, ph.t1[i] as f64, ph.m0[i] as f64);
        let bolus = self.bolus_region.map(|region| {
            let s = p.suppression.as_ref().unwrap().for_row(self.sched.preps[self.sched.raws[v].prep].suppression);
            (region, s.pulse_times, s.epsilon)
        });
        let subs = match &bolus {
            Some((region, pulses, eps)) => subbolus_factors(pulses, *eps, kin.tau, entry_offset(p.label_type, *region, att)),
            None => vec![(0.0, kin.tau, 1.0)],
        };
        let all = crate::kinetic::delta_m_read_all(&kin, f_ml, att, t1t, m0, &e, &flips, &subs, p.exchange_time, self.lead[i]);
        let parts = &all[offset..offset + n];
        self.families.iter().map(|fam| match fam {
            Family::Blood => parts.iter().map(|q| q.iv).collect(),
            Family::Extravascular => parts.iter().map(|q| q.ev).collect(),
            Family::Arterial => {
                let (Some(abv), Some(aatt)) = (&self.p4.abv, &self.p4.aatt) else { return vec![0.0; n] };
                let crush = self.p4.crush.as_ref().map_or(1.0, |cr| cr[v][self.p4.label_of[i]]);
                (offset..offset + n).map(|x| {
                    let g = match &bolus {
                        Some((region, pulses, eps)) => arterial_factor(pulses, *eps, e[x] - aatt[i],
                                                                       entry_offset(p.label_type, *region, aatt[i])),
                        None => 1.0,
                    };
                    let earlier: Vec<(f64, f64)> = e[..x].iter().copied().zip(flips[..x].iter().copied()).collect();
                    crate::kinetic::arterial_read(&kin, abv[i], aatt[i], m0, e[x], &earlier, self.lead_a[i], g * crush)
                }).collect()
            }
        }).collect()
    }

    /// Whether raw volume `v` carries label.
    pub fn labeled(&self, v: usize) -> bool {
        self.label_scale[v] != 0.0
    }
}

/// Raw volume `g`'s input to the acquisition (P7 addendum, part C): the tissue groups' `M0` and,
/// per family and mask, the label's reads at its node excitations (masked, resampled), with the
/// weights of every excitation: the tissue's `sin(a) m_j(T1) x` its physiological factor, the
/// label's `sin(a) x` its sign and suppression factor `x` its physiological factor `x` the hat
/// weight of the excitation on the node.
pub(super) fn volume_inputs(b: &Ge3dBuild, g: usize) -> mrsim_acq::kspace3d::GeVolume {
    let ph = b.ph;
    let ncomp = b.n_compartments();
    let nk = b.masks.len();
    let mut images = vec![Vec::new(); ncomp];
    let row = &b.sched.raw_rows[g];
    let tissue_on = row.kind != RowKind::Deltam;
    if tissue_on {
        for (gi, im) in b.group_images.iter().enumerate() {
            images[gi] = im.clone();
        }
    }
    let labeled = b.labeled(g);
    if labeled {
        // per family and node, the reads at the node excitation over the phantom
        let nvox = ph.nvox();
        let mut node_vals: Vec<Vec<Vec<f32>>> = b.families.iter().enumerate()
            .map(|(f, _)| vec![vec![0.0f32; nvox]; b.nodes[g][f].len()]).collect();
        for i in (0..nvox).filter(|&i| ph.dseg[i] > 0) {
            let r = b.reads(g, i);
            for (f, fam) in r.iter().enumerate() {
                for (k, &j) in b.nodes[g][f].iter().enumerate() {
                    node_vals[f][k][i] = fam[j] as f32;
                }
            }
        }
        for (f, per_node) in node_vals.iter().enumerate() {
            for (k, vals) in per_node.iter().enumerate() {
                for (c, m) in b.masks.iter().enumerate() {
                    let masked: Vec<f32> = vals.iter().zip(m).map(|(v, &on)| if on { *v } else { 0.0 }).collect();
                    images[b.comp(f, c, k)] = b.r_sim.mean(&masked);
                }
            }
        }
    }
    // motion (P5 part D): the volume's pose moves its images; the shots after a within-volume event
    // see them moved again by the event's jump (each distinct pose one shot set)
    let pose = b.poses[g];
    if pose != Pose::IDENTITY {
        for im in images.iter_mut().filter(|im| !im.is_empty()) {
            *im = mrsim_acq::motion::resample_by_pose(im, b.sim_dims, b.v2w, pose);
        }
    }
    let mut distinct: Vec<Pose> = Vec::new();
    for &q in &b.shot_pose[g] {
        if q != Pose::IDENTITY && !distinct.contains(&q) {
            distinct.push(q);
        }
    }
    let shot_images: Option<Vec<mrsim_acq::kspace3d::ShotSet>> = (!distinct.is_empty()).then(|| {
        distinct.iter().map(|&q| mrsim_acq::kspace3d::ShotSet {
            shots: (0..b.shot_pose[g].len()).filter(|&s| b.shot_pose[g][s] == q).collect(),
            images: images.iter().map(|im| {
                if im.is_empty() { Vec::new() } else { mrsim_acq::motion::resample_by_pose(im, b.sim_dims, b.v2w, q) }
            }).collect(),
        }).collect()
    });
    let ky_segments = b.res.readout.ky_segments;
    let nz = b.table.nz;
    let sin_a = b.raw_flip[g].to_radians().sin();
    let preps = b.sched.preps_of(g);
    let first_prep = b.sched.raws[g].prep;
    let (_, _, offset) = b.cycle(g);
    let mut weights = vec![0.0; nz * ky_segments * ncomp];
    for p in 0..nz {
        for sy in 0..ky_segments {
            let line = b.table.line(p, sy);
            let (shot, j) = (line.shot, line.excitation);
            let prep = first_prep + shot;
            debug_assert_eq!(preps[shot].shot, shot);
            let (tf, lf) = b.prep_factors[prep];
            let gain = b.shot_gain[g][shot];
            let (tf, lf) = (tf * gain, lf * gain);
            let at = |c: usize| (p * ky_segments + sy) * ncomp + c;
            if tissue_on {
                for (gi, mz) in b.group_mz.iter().enumerate() {
                    weights[at(gi)] = sin_a * mz[prep][offset + j] * tf;
                }
            }
            if labeled {
                for f in 0..b.families.len() {
                    for (k, w) in hat(&b.nodes[g][f], j) {
                        for c in 0..nk {
                            weights[at(b.comp(f, c, k))] = sin_a * b.label_scale[g] * lf * w;
                        }
                    }
                }
            }
        }
    }
    mrsim_acq::kspace3d::GeVolume { images, weights, shot_images }
}

/// What [`prepare_ge3d`] resolves for [`simulate_ge3d`]: the build and the series' grids and
/// records.
pub(super) struct Ge3dPrepared<'a> {
    pub b: Ge3dBuild<'a>,
    pub acq_grid: Grid,
    pub sim_grid: Grid,
    pub r_acq: Resampler,
    pub relax: Relaxation,
    pub mode_used: T2Mode,
    pub acq: Acquisition,
    pub fmap_sim: Vec<f32>,
    pub physio_lines: Vec<PhysioLine>,
    pub label_factors: Option<Vec<f64>>,
    pub achieved: f64,
    /// The phantom's foreground voxels.
    pub fg: Vec<usize>,
    pub motion_seed: Option<u64>,
    pub events: Vec<MotionEvent>,
    pub dropped: Vec<DroppedShot>,
}

/// Everything before the acquisition (P7 Task 15): the grids, the train, the tissue groups and
/// their timeline, the P4 parts and the factors, the slab-entry leads and the interpolation nodes.
pub(super) fn prepare_ge3d<'a>(p: &'a Protocol, ph: &'a Phantom, mode: T2Mode, ov: RowOverride)
    -> Result<Ge3dPrepared<'a>, String>
{
    let g3 = p.ge3d().expect("a 3D gradient-echo protocol");
    if ov != RowOverride::None {
        return Err("the test-hook row overrides are not available on the 3D gradient-echo series".to_string());
    }
    for (what, on) in [
        ("[hadamard] (P7 plan, Task 18)", p.hadamard.is_some()),
    ] {
        if on {
            return Err(format!("[readout] type \"epi3d\" with {what}: not available"));
        }
    }
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
    let res = crate::protocol::resolve_ge3d(p, acq_grid.dims)?.expect("an epi3d readout resolves");
    let table = mrsim_acq::readout::ge3d_lines(&res.train, &res.readout, ny, nz)?;
    let dv = p.voxel_size_mm;
    let off = corner_offset(&ph.grid, &acq_grid)?;
    let r_sim = Resampler::with_offset(ph.grid.dims, pv, sim_grid.dims, [dv[0] / o as f64, dv[1] / o as f64, dv[2]], off);
    let r_acq = Resampler::with_offset(ph.grid.dims, pv, acq_grid.dims, dv, off);
    let nvox_sim = snx * sny * nz;
    let mut acq = p.acquisition(nx, ny)?;
    acq.do_distortions = ph.fieldmap.is_some();
    let fmap_sim: Vec<f32> = match &ph.fieldmap {
        Some(f) => r_sim.mean(f),
        None => vec![0.0; nvox_sim],
    };

    // ---- compartments: tissue groups, then each family's node slots per mask ----
    let (relax, mode_used) = ph.relaxation_for(mode, false)?;
    let masks: Vec<Vec<bool>> = match &relax {
        Relaxation::Class { .. } => ph.labels.iter().map(|(l, _)| ph.dseg.iter().map(|d| d == l).collect()).collect(),
        Relaxation::Voxel { .. } => vec![ph.dseg.iter().map(|d| *d > 0).collect()],
    };
    let groups = tissue_groups(ph, &masks, g3.max_t1_groups.0)?;
    let group_images: Vec<Vec<f32>> = groups.iter()
        .map(|gr| r_sim.mean(&ph.m0.iter().zip(&gr.mask).map(|(m, &on)| if on { *m } else { 0.0 }).collect::<Vec<f32>>()))
        .collect();
    let mut families = vec![Family::Blood];
    if p.exchange_time.is_some() {
        families.push(Family::Extravascular);
    }
    if p.macrovascular.is_some() {
        families.push(Family::Arterial);
    }

    // ---- the schedule, its suppression, its trains and the tissue timeline ----
    let sched = Schedule::new(p);
    let n = sched.raw_rows.len();
    let ge = p.ge.as_ref().expect("epi3d is gradient echo");
    // the separate M0's excitation: Look-Locker's [m0] flip_angle, else the series'
    let flip_deg = p.look_locker.as_ref().and_then(|l| l.m0_flip_deg).map_or(ge.flip_deg, |f| f.0);
    // each raw volume's: a Look-Locker FlipAngle array gives each readout (raw volume) its own
    let raw_flip: Vec<f64> = (0..n).map(|v| p.look_locker.as_ref().map_or(ge.flip_deg, |l| l.flip_deg[v])).collect();
    let spacing_s = res.train.exc_spacing_ms / 1000.0;
    let sups: Vec<Option<crate::longitudinal::Suppression>> = sched.preps.iter().map(|pr| {
        let row = &sched.raw_rows[pr.raw];
        p.suppression.as_ref().map(|s| s.for_row(pr.suppression)).filter(|s| s.has_events() && row.kind != RowKind::M0scan)
    }).collect();
    // each preparation's cycle: every sub-train of its raw volumes (one without Look-Locker)
    let cycles: Vec<LlCycle> = sched.preps.iter().zip(&sups).map(|(pr, s)| {
        let row = &sched.raw_rows[pr.raw];
        let (times, flips, _) = cycle_of(&sched, &raw_flip, res.n_exc, spacing_s, pr.raw);
        LlCycle { tr: row.tr, s: s.as_ref(), t_read: times, flip_deg: flips }
    }).collect();
    let group_mz: Vec<Vec<Vec<f64>>> = groups.iter().map(|gr| train_mz(gr.t1_s, &cycles)).collect();

    // ---- P4 parts, physiology per preparation, the label's scale per raw volume ----
    let bolus_region = match p.suppression.as_ref().map(|s| s.model) {
        Some(SuppressionModel::BolusPosition(r)) => Some(r),
        _ => None,
    };
    let p4 = P4::new(p, ph, bolus_region)?;
    let mut physio_lines: Vec<PhysioLine> = Vec::new();
    let prep_factors: Vec<(f64, f64)> = sched.preps.iter().map(|pr| {
        let row = &sched.raw_rows[pr.raw];
        match &p4.physio {
            None => (1.0, 1.0),
            Some(phys) => {
                let time = pr.start_s + row.t;
                let tf = phys.tissue_factor(time);
                let (lf, means) = match p.label_type {
                    LabelType::Pasl => phys.label_factor_at(pr.start_s),
                    _ => phys.label_factor_window(pr.start_s, pr.start_s + row.tau),
                };
                physio_lines.push(PhysioLine {
                    volume: pr.raw, slice: pr.shot, time,
                    cardiac_phase: phys.cardiac.phase(time), respiratory_phase: phys.respiratory.phase(time),
                    drift: phys.drift.value(time), tissue_factor: tf, label_window: (pr.labeling_window[0], pr.labeling_window[1]),
                    label_means: means, label_factor: lf,
                });
                (tf, lf)
            }
        }
    }).collect();
    let label_factors: Option<Vec<f64>> = match (&p.suppression, bolus_region) {
        (Some(spec), None) => Some((0..n).map(|v| label_factor(&spec.for_row(sched.preps[sched.raws[v].prep].suppression))).collect()),
        _ => None,
    };
    let label_scale: Vec<f64> = (0..n).map(|v| {
        blood_sign(sched.raw_rows[v].kind, ov) * label_factors.as_ref().map_or(1.0, |f| f[v])
    }).collect();

    // ---- depletion from slab entry: each voxel's leads, refusing an entry after its arrival ----
    let mut lead = vec![0.0; ph.nvox()];
    let mut lead_a = vec![0.0; ph.nvox()];
    for i in 0..ph.nvox() {
        if ph.dseg[i] > 0 && ph.perfusion[i] > 0.0 {
            lead[i] = entry_lead(g3.slab_entry.0, ph.att[i] as f64, i, "ATT")?;
        }
        if let (Some(abv), Some(aatt)) = (&p4.abv, &p4.aatt) {
            if ph.dseg[i] > 0 && abv[i] > 0.0 {
                lead_a[i] = entry_lead(g3.slab_entry.0, aatt[i], i, "aATT")?;
            }
        }
    }

    // ---- motion (P5 part D): per-volume poses; within-volume events per shot, each jump persisting
    // for the later shots of its volume, each event shot attenuated by 1 - severity ----
    let n_shots = res.n_shots;
    let motion_seed = p.motion.as_ref().map(|_| p.seed ^ MOTION_SEED_SALT);
    let mut poses = vec![Pose::IDENTITY; n];
    let mut events = Vec::new();
    if let (Some(m), Some(seed)) = (&p.motion, motion_seed) {
        poses = resolve_poses(&m.mode, n, seed);
        events = draw_events(m.within.as_ref(), n, n_shots, seed);
    }
    let mut shot_pose = vec![vec![Pose::IDENTITY; n_shots]; n];
    let mut shot_gain = vec![vec![1.0f64; n_shots]; n];
    let mut dropped = Vec::new();
    for (g, sp) in shot_pose.iter_mut().enumerate() {
        let evs: Vec<&MotionEvent> = events.iter().filter(|e| e.volume == g && e.shot < n_shots).collect();
        let mut cum = Pose::IDENTITY;
        for (sh, pose) in sp.iter_mut().enumerate() {
            for e in evs.iter().filter(|e| e.shot == sh) {
                for k in 0..3 {
                    cum.trans_mm[k] += e.jump_mm[k];
                    cum.rot_deg[k] += e.jump_deg[k];
                }
            }
            *pose = cum;
        }
        for e in &evs {
            let atten = DropoutLaw::Uniform.attenuation(g, e.severity);
            shot_gain[g][e.shot] *= atten as f64;
            dropped.push(DroppedShot { volume: g, shot: e.shot, slices: Vec::new(), attenuation: atten });
        }
    }

    let mut b = Ge3dBuild {
        p, ph, res, table, sched, r_sim, masks, groups, group_images, group_mz, families, k_nodes: 0, nodes: Vec::new(),
        prep_factors, label_scale, p4, bolus_region, lead, lead_a, flip_deg, raw_flip, spacing_s, poses, shot_pose, shot_gain,
        v2w: sim_grid.voxel_to_world, sim_dims: sim_grid.dims,
    };

    // ---- the node selection, per raw volume and family; one count for the series ----
    let fg: Vec<usize> = (0..ph.nvox()).filter(|&i| ph.dseg[i] > 0).collect();
    let mut achieved = 0.0f64;
    let mut nodes = Vec::with_capacity(n);
    for v in 0..n {
        if !b.labeled(v) {
            nodes.push(vec![Vec::new(); b.families.len()]);
            continue;
        }
        let per_voxel: Vec<Vec<Vec<f64>>> = fg.iter().map(|&i| b.reads(v, i)).collect();
        let mut per_family = Vec::with_capacity(b.families.len());
        for f in 0..b.families.len() {
            let vals: Vec<Vec<f64>> = per_voxel.iter().map(|r| r[f].clone()).collect();
            let nd = select_nodes(&vals, b.res.e_c, g3.node_tolerance.0);
            let peak = vals.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            if peak > 0.0 {
                for vv in &vals {
                    for (a, c) in vv.iter().zip(interpolate(vv, &nd)) {
                        achieved = achieved.max((a - c).abs() / peak);
                    }
                }
            }
            per_family.push(nd);
        }
        nodes.push(per_family);
    }
    b.k_nodes = nodes.iter().flatten().map(|nd| nd.len()).max().unwrap_or(0);
    b.nodes = nodes;
    Ok(Ge3dPrepared {
        b, acq_grid, sim_grid, r_acq, relax, mode_used, acq, fmap_sim, physio_lines, label_factors, achieved, fg,
        motion_seed, events, dropped,
    })
}

/// The 3D gradient-echo series (P7 addendum, part C): the train's tissue from its timeline per T1
/// group, the label depleted from slab entry on interpolation nodes, one acquisition call with
/// volumes built on demand, the separate M0 as the same train at its own repetition time.
pub(super) fn simulate_ge3d(p: &Protocol, ph: &Phantom, mode: T2Mode, phase: &PhaseModel, ov: RowOverride)
    -> Result<SeriesOutput, String>
{
    let g3 = p.ge3d().expect("a 3D gradient-echo protocol");
    let Ge3dPrepared {
        b, acq_grid, sim_grid, r_acq, relax, mode_used, acq, fmap_sim, physio_lines, label_factors, achieved, fg, motion_seed,
        events, dropped,
    } = prepare_ge3d(p, ph, mode, ov)?;
    let n = b.sched.raw_rows.len();
    let [_, _, nz] = acq_grid.dims;
    let nvox_sim = sim_grid.dims.iter().product::<usize>();
    let nvox_acq = acq_grid.dims.iter().product::<usize>();
    let flip_deg = b.flip_deg;
    let spacing_s = b.spacing_s;
    let ncomp = b.n_compartments();
    #[cfg(feature = "par")]
    let in_flight = std::thread::available_parallelism().map_or(1, |x| x.get()).min(n.max(1));
    #[cfg(not(feature = "par"))]
    let in_flight = 1;
    let what = format!("{} tissue group(s), {} label famil(ies) x {} mask(s) x {} node(s)", b.groups.len(), b.families.len(),
                       b.masks.len(), b.k_nodes);
    let memory_gib = check_memory(nvox_sim, ncomp, in_flight, g3.max_memory_gib.0, &what)?;

    // ---- relaxation per compartment ----
    let t2_blood_ms = p.t2_blood_ms();
    let t2_arterial_ms = p.macrovascular.as_ref().map(|m| (m.t2_arterial.0 * 1000.0) as f32);
    let (acq_t2_ms, acq_t2p_ms): (Option<Vec<f32>>, Option<Vec<f32>>) = match &relax {
        Relaxation::Voxel { t2_ms, t2p_ms } => (Some(b.r_sim.rate_mean(t2_ms, &ph.m0)), Some(b.r_sim.rate_mean(t2p_ms, &ph.m0))),
        Relaxation::Class { .. } => (None, None),
    };
    let tissue_t2 = |c: usize| -> T2Volume {
        match (&relax, &acq_t2_ms) {
            (Relaxation::Class { t2_ms, .. }, _) => T2Volume::Uniform(t2_ms[c]),
            (_, Some(m)) => T2Volume::Map(m),
            _ => unreachable!("voxel mode has its maps"),
        }
    };
    let tissue_t2p = |c: usize| -> T2Volume {
        match (&relax, &acq_t2p_ms) {
            (Relaxation::Class { t2p_ms, .. }, _) => T2Volume::Uniform(t2p_ms[c]),
            (_, Some(m)) => T2Volume::Map(m),
            _ => unreachable!("voxel mode has its maps"),
        }
    };
    let mut t2_vols: Vec<T2Volume> = b.groups.iter().map(|gr| tissue_t2(gr.compartment)).collect();
    let mut ti_vols: Vec<T2Volume> = b.groups.iter().map(|gr| tissue_t2p(gr.compartment)).collect();
    for fam in &b.families {
        for c in 0..b.masks.len() {
            for _ in 0..b.k_nodes {
                t2_vols.push(match fam {
                    Family::Blood => T2Volume::Uniform(t2_blood_ms),
                    Family::Extravascular => tissue_t2(c),
                    Family::Arterial => T2Volume::Uniform(t2_arterial_ms.expect("the arterial family has its T2")),
                });
                // the label inherits its compartment's T2' (P6)
                ti_vols.push(tissue_t2p(c));
            }
        }
    }

    // ---- the acquisition: every echo, volumes on demand ----
    let echoes = mrsim_acq::kspace3d::simulate_acquisition_3d_ge(
        sim_grid.dims, acq_grid.dims, n, &t2_vols, &fmap_sim, Some(&ti_vols), &acq, &b.res.train, &b.res.readout,
        &|g| volume_inputs(&b, g), phase, p.seed,
    );

    // ---- the separate M0: the same train without labeling or suppression at its own TR ----
    let m0_seed = (p.m0_type == M0Type::Separate).then_some(p.seed ^ M0_SEED_SALT);
    let m0_echoes: Option<Vec<(Vec<f32>, Vec<f32>)>> = match (m0_seed, p.m0_repetition_time_s) {
        (Some(seed), Some(tr)) => {
            let mz: Vec<Vec<f64>> = b.groups.iter().map(|gr| m0_train_mz(gr.t1_s, tr, b.res.n_exc, spacing_s, flip_deg)).collect();
            let sin_a = flip_deg.to_radians().sin();
            let ky_segments = b.res.readout.ky_segments;
            let m0_volume = |_g: usize| {
                let mut images = vec![Vec::new(); ncomp];
                for (gi, im) in b.group_images.iter().enumerate() {
                    images[gi] = im.clone();
                }
                let mut weights = vec![0.0; nz * ky_segments * ncomp];
                for pp in 0..nz {
                    for sy in 0..ky_segments {
                        let j = b.table.line(pp, sy).excitation;
                        for (gi, m) in mz.iter().enumerate() {
                            weights[(pp * ky_segments + sy) * ncomp + gi] = sin_a * m[j];
                        }
                    }
                }
                mrsim_acq::kspace3d::GeVolume { images, weights, shot_images: None }
            };
            Some(mrsim_acq::kspace3d::simulate_acquisition_3d_ge(
                sim_grid.dims, acq_grid.dims, 1, &t2_vols, &fmap_sim, Some(&ti_vols), &acq, &b.res.train, &b.res.readout,
                &m0_volume, phase, seed,
            ))
        }
        (Some(_), None) => return Err("M0Type Separate without an M0 repetition time".to_string()),
        _ => None,
    };

    // ---- ground truth: the label at the kz-centre excitation, static ----
    let e_c = b.res.e_c;
    let mut delta_m = vec![0.0f32; nvox_acq * n];
    let mut gt_iv = p.exchange_time.map(|_| vec![0.0f32; nvox_acq * n]);
    let mut gt_art = p.macrovascular.as_ref().map(|_| vec![0.0f32; nvox_acq * n]);
    for v in 0..n {
        if !matches!(sched_kind(&b, v), RowKind::Label | RowKind::Deltam) {
            continue;
        }
        let mut per: Vec<Vec<f32>> = vec![vec![0.0f32; ph.nvox()]; b.families.len()];
        for &i in &fg {
            for (f, r) in b.reads(v, i).iter().enumerate() {
                per[f][i] = r[e_c] as f32;
            }
        }
        let fam = |x: Family| b.families.iter().position(|y| *y == x);
        let total: Vec<f32> = (0..ph.nvox()).map(|i| {
            per[0][i] + fam(Family::Extravascular).map_or(0.0, |f| per[f][i])
        }).collect();
        let put = |dst: &mut Vec<f32>, src: &[f32]| {
            for (vox, x) in r_acq.mean(src).iter().enumerate() {
                dst[vox * n + v] = *x;
            }
        };
        put(&mut delta_m, &total);
        if let Some(g) = gt_iv.as_mut() {
            put(g, &per[0]);
        }
        if let (Some(g), Some(f)) = (gt_art.as_mut(), fam(Family::Arterial)) {
            put(g, &per[f]);
        }
    }
    // under motion the truths are moved by each volume's pose on the simulation grid (no shot events)
    // and block-averaged, the unmoved kept as delta_m_static, as on the 3D spin-echo path
    let (delta_m, delta_m_static, gt_iv, gt_art) = if p.motion.is_some() {
        let o = p.acq.oversample;
        let mv = |sim: Vec<f32>| -> Vec<f32> {
            let mut arr = [sim];
            apply_motion(&mut arr, sim_grid.dims, n, sim_grid.voxel_to_world, &b.poses);
            let [moved] = arr;
            block_mean_inplane(&moved, sim_grid.dims, o, n)
        };
        let moved = |which: usize| -> Vec<f32> {
            let mut sim = vec![0.0f32; nvox_sim * n];
            for v in 0..n {
                if !matches!(sched_kind(&b, v), RowKind::Label | RowKind::Deltam) {
                    continue;
                }
                let mut per: Vec<Vec<f32>> = vec![vec![0.0f32; ph.nvox()]; b.families.len()];
                for &i in &fg {
                    for (f, r) in b.reads(v, i).iter().enumerate() {
                        per[f][i] = r[e_c] as f32;
                    }
                }
                let fam = |x: Family| b.families.iter().position(|y| *y == x);
                let src: Vec<f32> = match which {
                    0 => (0..ph.nvox()).map(|i| per[0][i] + fam(Family::Extravascular).map_or(0.0, |f| per[f][i])).collect(),
                    1 => per[0].clone(),
                    _ => fam(Family::Arterial).map_or(vec![0.0; ph.nvox()], |f| per[f].clone()),
                };
                for (vox, x) in b.r_sim.mean(&src).iter().enumerate() {
                    sim[vox * n + v] = *x;
                }
            }
            mv(sim)
        };
        (moved(0), Some(delta_m), gt_iv.map(|_| moved(1)), gt_art.map(|_| moved(2)))
    } else {
        (delta_m, None, gt_iv, gt_art)
    };
    let perfused: Vec<bool> = ph.perfusion.iter().map(|f| *f > 0.0).collect();
    let ground_truth = GroundTruth {
        delta_m,
        delta_m_static,
        perfusion: r_acq.mean(&ph.perfusion),
        att: r_acq.masked_mean(&ph.att, &perfused),
        t1: r_acq.mean(&ph.t1),
        t2: r_acq.mean(&ph.t2),
        m0: r_acq.mean(&ph.m0),
        dseg: r_acq.majority(&ph.dseg),
        acq_t2_ms: acq_t2_ms.clone(),
        acq_t2p_ms: acq_t2p_ms.clone(),
        acq_t1_ms: None,
        delta_m_iv: gt_iv,
        delta_m_suppressed: None,
        delta_m_arterial: gt_art,
        abv: b.p4.abv.as_ref().map(|a| r_acq.mean(&a.iter().map(|x| *x as f32).collect::<Vec<f32>>())),
        aatt: match (&b.p4.abv, &b.p4.aatt) {
            (Some(bv), Some(a)) => {
                let has: Vec<bool> = bv.iter().map(|x| *x > 0.0).collect();
                Some(r_acq.masked_mean(&a.iter().map(|x| *x as f32).collect::<Vec<f32>>(), &has))
            }
            _ => None,
        },
    };

    let ge3d = Ge3dSeries {
        resolution: b.res.clone(),
        tissue_groups: b.groups.iter().map(|gr| {
            let name = match &relax {
                Relaxation::Class { .. } => ph.labels[gr.compartment].1.clone(),
                Relaxation::Voxel { .. } => "tissue".to_string(),
            };
            (name, gr.t1_s)
        }).collect(),
        families: b.families.iter().map(|f| f.as_str()).collect(),
        k_nodes: b.k_nodes,
        nodes: b.nodes.clone(),
        achieved_error: achieved,
        memory_gib,
        slab_entry: g3.slab_entry,
        cumulative_depletion: (0..n).map(|v| b.cumulative_depletion(v)).collect(),
        readouts_per_cycle: p.look_locker.as_ref().map(|_| {
            let prep = b.sched.raws[0].prep;
            b.sched.raws.iter().take_while(|r| r.prep == prep).count()
        }),
        mean_lead_s: ph.labels.iter().enumerate().map(|(li, (_, nm))| {
            let (s, c) = (0..ph.nvox()).filter(|&i| b.p4.label_of[i] == li && ph.perfusion[i] > 0.0)
                .fold((0.0, 0usize), |(s, c), i| (s + b.lead[i], c + 1));
            (nm.clone(), if c > 0 { s / c as f64 } else { 0.0 })
        }).collect(),
    };
    let mut echoes = echoes.into_iter();
    let (mag, phase_out) = echoes.next().expect("one echo at least");
    let mut m0_iter = m0_echoes.map(|v| v.into_iter());
    let m0 = m0_iter.as_mut().and_then(|it| it.next());
    let tes = &p.echo_times_s;
    let more_echoes: Vec<EchoSeries> = echoes.enumerate().map(|(e, (mag, phase))| EchoSeries {
        echo_time_s: tes[e + 1], mag, phase, m0: m0_iter.as_mut().and_then(|it| it.next()),
    }).collect();
    Ok(SeriesOutput {
        acq_grid, sim_grid, n_volumes: n, mag, phase: phase_out, m0, mode: mode_used, labels: ph.labels.clone(),
        n_compartments: ncomp, fieldmap_present: ph.fieldmap.is_some(), seeds: (p.seed, m0_seed), acquisition: acq,
        ground_truth, label_factors, poses: b.poses.clone(), motion_seed, events, dropped, compat: None, crush_survival: b.p4.crush.clone(),
        physio: b.p4.physio.as_ref().map(|_| physio_lines), ge_rule: Some(
            "every excitation of the train an event on the tissue's timeline (spoiled): the first preparation at its \
             steady state, each later one from the end of the one before"),
        readout: None, echo_amplitudes: None, spiral_segmentation: None, more_echoes, hadamard: None, look_locker: None,
        ge3d: Some(ge3d),
    })
}

fn sched_kind(b: &Ge3dBuild, v: usize) -> RowKind {
    b.sched.raw_rows[v].kind
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longitudinal::tissue_mz_ge;

    /// An independent reference: the train's events run one by one from `Mz = M0`, the first cycle
    /// repeated until its start state settles, then the sequence, recording `Mz` before each
    /// excitation. Nothing of the production timeline (no affine fixed point).
    fn brute(t1: f64, cycles: &[LlCycle]) -> Vec<Vec<f64>> {
        let run = |mz0: f64, c: &LlCycle| -> (Vec<f64>, f64) {
            let rec = |m: f64, dt: f64| 1.0 + (m - 1.0) * (-dt / t1).exp();
            let mut events: Vec<(f64, f64, bool)> = Vec::new();
            let first = c.t_read.first().copied().unwrap_or(c.tr);
            let mut m = mz0;
            if let Some(s) = c.s {
                if s.presaturation {
                    m = 0.0;
                }
                for &p in s.pulse_times.iter().filter(|&&p| p < first) {
                    events.push((p, 1.0 - 2.0 * s.epsilon, false));
                }
            }
            for (&t, &a) in c.t_read.iter().zip(&c.flip_deg) {
                events.push((t, a.to_radians().cos(), true));
            }
            let mut t = 0.0;
            let mut before = Vec::new();
            for (te, f, read) in events {
                m = rec(m, te - t);
                if read {
                    before.push(m);
                }
                m *= f;
                t = te;
            }
            (before, rec(m, c.tr - t))
        };
        let mut m = 1.0;
        for _ in 0..2000 {
            m = run(m, &cycles[0]).1;
        }
        let mut out = Vec::new();
        for c in cycles {
            let (b, end) = run(m, c);
            out.push(b);
            m = end;
        }
        out
    }

    /// The train timeline against the brute-force run: identical preparations (the fixed point),
    /// preparations that differ (suppression on some, a longer TR), and an m0scan-like train; and a
    /// one-excitation train is P5's single gradient-echo excitation.
    #[test]
    fn the_train_timeline_matches_a_brute_force_run() {
        let s = Suppression::new(vec![1.2, 2.4], 0.92, false);
        let (t1, spacing, n_exc, fa) = (1.33, 0.045, 12, 10.0);
        let same: Vec<LlCycle> = (0..4).map(|_| train_cycle(4.5, Some(&s), train_times(2.8, n_exc, spacing), fa)).collect();
        let mixed = vec![
            train_cycle(4.5, Some(&s), train_times(2.8, n_exc, spacing), fa),
            train_cycle(4.5, None, train_times(2.8, n_exc, spacing), fa),
            train_cycle(6.0, None, train_times(0.0, n_exc, spacing), fa),
            train_cycle(4.5, Some(&s), train_times(2.8, n_exc, spacing), 15.0),
        ];
        for cycles in [same, mixed] {
            let got = train_mz(t1, &cycles);
            let want = brute(t1, &cycles);
            for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
                assert!((g - w).abs() <= 1e-12, "{g} vs {w}");
            }
        }
        // without suppression (the tissue near recovery) the train depletes it: the last excitation
        // reads less than the first; under suppression it may be recovering from the null instead
        let plain = train_mz(t1, &[train_cycle(4.5, None, train_times(2.8, n_exc, spacing), fa)]);
        assert!(plain[0][n_exc - 1] < 0.99 * plain[0][0], "{:?}", plain[0]);
        // the separate M0's train at its steady state
        let m0 = m0_train_mz(t1, 6.0, n_exc, spacing, fa);
        let want = brute(t1, &[train_cycle(6.0, None, train_times(0.0, n_exc, spacing), fa)]);
        for (g, w) in m0.iter().zip(&want[0]) {
            assert!((g - w).abs() <= 1e-12);
        }
        // one excitation per train (kz_segments = nz) is P5's single gradient-echo excitation
        for sup in [None, Some(&s)] {
            let one = train_mz(t1, &[train_cycle(4.5, sup, vec![2.8], fa)]);
            let p5 = tissue_mz_ge(1.0, t1, 4.5, 2.8, sup.unwrap_or(&Suppression::new(vec![], 0.0, false)), fa);
            assert!((one[0][0] - p5).abs() <= 1e-12, "{} vs {p5}", one[0][0]);
        }
    }

    /// Groups are formed per compartment at the phantom voxels, by distinct T1; their masks
    /// partition each compartment; too many is refused with the count.
    #[test]
    fn tissue_groups_split_by_t1() {
        let mut ph = crate::phantom::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop")))
            .unwrap();
        let masks: Vec<Vec<bool>> = ph.labels.iter().map(|(l, _)| ph.dseg.iter().map(|d| d == l).collect()).collect();
        let g0 = tissue_groups(&ph, &masks, 16).unwrap();
        // the crop is label-wise in T1: one group per label that has voxels
        let used = masks.iter().filter(|m| m.iter().any(|&b| b)).count();
        assert_eq!(g0.len(), used);
        // two T1 values inside the first label: one more group, and the groups still partition it
        let first: Vec<usize> = (0..ph.dseg.len()).filter(|&i| masks[0][i]).collect();
        for &i in first.iter().step_by(2) {
            ph.t1[i] += 0.1;
        }
        let g1 = tissue_groups(&ph, &masks, 16).unwrap();
        assert_eq!(g1.len(), used + 1);
        for (c, mask) in masks.iter().enumerate() {
            for (i, &on) in mask.iter().enumerate() {
                let n = g1.iter().filter(|g| g.compartment == c && g.mask[i]).count();
                assert_eq!(n, usize::from(on), "compartment {c} voxel {i}");
            }
        }
        // a smooth T1 map: refused, with the count
        for (n, &i) in first.iter().enumerate() {
            ph.t1[i] = 1.0 + 0.001 * n as f32;
        }
        let e = tissue_groups(&ph, &masks, 16).unwrap_err();
        assert!(e.contains("max_t1_groups = 16") && e.contains("distinct"), "{e}");
    }

    // ---- Task 14

    use crate::kinetic::{delta_m_read_all, Kinetic, LabelType};

    const K: Kinetic = Kinetic { label_type: LabelType::Pcasl, tau: 1.8, alpha: 0.85, lambda: 0.9, t1b: 1.65 };

    /// One train's blood reads (the whole label, with exchange's split summed) at 32 excitations 40
    /// ms apart from 2.0 s, for voxels whose ATT puts their arrival before, inside and after it.
    fn train_reads(atts: &[f64], lead_from: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        let e = train_times(2.0, 32, 0.04);
        let flips = vec![10.0; e.len()];
        let vals = atts.iter().map(|&att| {
            let lead = entry_lead(SlabEntryTime::Seconds(lead_from), att, 0, "ATT").unwrap();
            delta_m_read_all(&K, 60.0, att, 1.33, 74.6, &e, &flips, &[(0.0, K.tau, 1.0)], Some(0.6), lead)
                .iter().map(|p| p.total()).collect()
        }).collect();
        (e, vals)
    }

    /// The interpolation meets its tolerance at every voxel and excitation; an arrival inside the
    /// train needs more than the three starting nodes; nodes at every excitation are exact; a
    /// centre before the arrival (centric order) still meets the tolerance, the reference being the
    /// largest value over the train; no label, no nodes.
    #[test]
    fn the_blood_interpolation_meets_its_tolerance() {
        let tol = 1e-4;
        for (atts, centre) in [(vec![0.5, 0.8, 1.2], 16), (vec![2.3, 2.6], 16), (vec![2.9], 0)] {
            let (_, vals) = train_reads(&atts, 0.2);
            let nodes = select_nodes(&vals, centre, tol);
            let peak = vals.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            assert!(peak > 0.0);
            for v in &vals {
                for (a, b) in v.iter().zip(interpolate(v, &nodes)) {
                    assert!((a - b).abs() <= tol * peak, "atts {atts:?}: {a} vs {b}");
                }
            }
            // arrival (ATT 2.3, 2.6, 2.9 s) inside the 2.0-3.24 s train: more than the three starting nodes
            if atts[0] > 2.0 {
                assert!(nodes.len() > 3, "atts {atts:?}: nodes {nodes:?}");
            }
        }
        let (_, vals) = train_reads(&[1.0], 0.2);
        let all: Vec<usize> = (0..32).collect();
        assert_eq!(interpolate(&vals[0], &all), vals[0]);
        assert!(select_nodes(&[vec![0.0; 32]], 16, tol).is_empty());
        // the hat weights sum to one inside the span
        let nodes = [0, 5, 31];
        for j in 0..32 {
            let s: f64 = hat(&nodes, j).iter().map(|w| w.1).sum();
            assert!((s - 1.0).abs() < 1e-15);
        }
    }

    /// The slab-entry lead per voxel; an entry after the arrival is refused, naming the voxel; the
    /// memory estimate and its refusal.
    #[test]
    fn slab_entry_leads_and_the_memory_bound() {
        assert_eq!(entry_lead(SlabEntryTime::Arrival, 1.2, 0, "ATT").unwrap(), 0.0);
        assert!((entry_lead(SlabEntryTime::Seconds(0.5), 1.2, 0, "ATT").unwrap() - 0.7).abs() < 1e-15);
        let e = entry_lead(SlabEntryTime::Seconds(0.9), 0.8, 417, "aATT").unwrap_err();
        assert!(e.contains("voxel 417") && e.contains("aATT"), "{e}");
        // with depletion before arrival the read is smaller than with none
        let (_, early) = train_reads(&[1.0], 0.2);
        let e = train_times(2.0, 32, 0.04);
        let none: Vec<f64> = delta_m_read_all(&K, 60.0, 1.0, 1.33, 74.6, &e, &vec![10.0; 32], &[(0.0, K.tau, 1.0)], Some(0.6), 0.0)
            .iter().map(|p| p.total()).collect();
        assert!(early[0][31] < none[31] && early[0][0] <= none[0]);
        // memory: 2^20 voxels x 64 compartments x 4 bytes x 8 in flight = 2 GiB
        assert!((check_memory(1 << 20, 64, 8, 4.0, "test").unwrap() - 2.0).abs() < 1e-12);
        let err = check_memory(1 << 20, 64, 8, 1.5, "18 tissue groups, K = 6").unwrap_err();
        assert!(err.contains("2.00 GiB") && err.contains("max_memory_gib") && err.contains("K = 6"), "{err}");
    }

    // ---- Task 15

    use crate::kinetic::parcel_ref::{Case, Region as PRegion};
    use crate::protocol::{parse, Overlay};
    use serde_json::{json, Value};

    fn crop() -> Phantom {
        crate::phantom::load(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/phantom-crop"))).unwrap()
    }

    /// A 3D EPI PCASL protocol on the crop: 0.75 mm partitions (8 of them), two kz and two ky
    /// segments (four shots, four excitations each, 40 ms apart, 12 degrees), PLD 0.5 s so the
    /// label arrives during the train, one suppression pulse before it, exchange, the label
    /// entering the slab 0.5 s after labeling.
    fn ge3d_protocol(extra: &str) -> Protocol {
        let s: Value = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 0.5,
            "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 1, "BackgroundSuppressionPulseTime": [2.0],
            "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3,
            "AcquisitionVoxelSize": [2.0, 2.0, 0.75], "MRAcquisitionType": "3D", "PulseSequenceType": "3D EPI",
            "PhaseEncodingDirection": "j-", "EffectiveEchoSpacing": 0.0005, "NumberShots": 4, "FlipAngle": 12
        });
        let ov: Overlay = toml::from_str(&format!(
            "seed = 5\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n[signal]\nacq_contrast = \"ge\"\n\
             [readout]\nexcitation_spacing = 40.0\nslab_entry_time = 0.5\nkz_segments = 2\n[kinetic]\nexchange_time = 0.4\n{extra}"))
            .unwrap();
        parse(&s, "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap()
    }

    /// Every raw volume's input, at every partition's excitation, against independent references:
    /// the weighted sum of its images is sin(a) times the tissue `M0 m_j` (the brute-force event
    /// run) plus the label's sign times its read (the parcel reference, depleted from slab entry,
    /// exchange split, suppression by its pulse), resampled. The same reference without the
    /// slab-entry lead misses it: the depletion before arrival is in the images.
    #[test]
    fn each_excitation_sees_the_reference_object() {
        let ph = crop();
        let p = ge3d_protocol("");
        let pr = prepare_ge3d(&p, &ph, T2Mode::Class, RowOverride::None).unwrap();
        let b = &pr.b;
        assert_eq!((b.table.nz, b.res.n_exc, b.res.n_shots), (8, 4, 4));
        assert!(b.k_nodes >= 2);
        let spec = p.suppression.as_ref().unwrap();
        let eps = spec.epsilon.0;
        let sin_a = 12f64.to_radians().sin();
        let spacing = 0.040;
        // the brute-force tissue timeline over every preparation
        let sups: Vec<Option<Suppression>> = b.sched.preps.iter().map(|_| Some(Suppression::new(vec![2.0], eps, false))).collect();
        let cycles: Vec<LlCycle> = b.sched.preps.iter().zip(&sups).map(|(pp, s)| {
            let row = &b.sched.raw_rows[pp.raw];
            train_cycle(row.tr, s.as_ref(), train_times(row.t, 4, spacing), 12.0)
        }).collect();
        let ncomp = b.n_compartments();
        let ky = b.res.readout.ky_segments;
        let [snx, sny, _] = pr.sim_grid.dims;
        for g in 0..2 {
            let vol = volume_inputs(b, g);
            let row = &b.sched.raw_rows[g];
            let kin = p.kinetic(row);
            let e = train_times(row.t, 4, spacing);
            let sign = if row.kind == RowKind::Label { -1.0 } else { 0.0 };
            for pp in 0..8 {
                for sy in 0..ky {
                    let line = b.table.line(pp, sy);
                    let (j, prep) = (line.excitation, b.sched.raws[g].prep + line.shot);
                    let got: Vec<f64> = (0..snx * sny * 8).map(|x| {
                        (0..ncomp).filter(|&c| !vol.images[c].is_empty())
                            .map(|c| vol.weights[(pp * ky + sy) * ncomp + c] * vol.images[c][x] as f64).sum()
                    }).collect();
                    let reference = |lead_on: bool| -> Vec<f32> {
                        let per: Vec<f32> = (0..ph.nvox()).map(|i| {
                            if ph.dseg[i] <= 0 {
                                return 0.0;
                            }
                            let m = brute(ph.t1[i] as f64, &cycles)[prep][j];
                            let mut v = ph.m0[i] as f64 * m;
                            if sign != 0.0 && ph.perfusion[i] > 0.0 {
                                let c = Case {
                                    k: kin, f: ph.perfusion[i] as f64, att: ph.att[i] as f64, t1t: ph.t1[i] as f64,
                                    m0: ph.m0[i] as f64, t: e[j], excitations: e[..j].iter().map(|&t| (t, 12.0)).collect(),
                                    entry_lead: if lead_on { ph.att[i] as f64 - 0.5 } else { 0.0 }, pulses: vec![2.0],
                                    epsilon: eps, region: PRegion::Global, tau_ex: Some(0.4), span: (0.0, kin.tau),
                                };
                                v += sign * c.read(4).1;
                            }
                            (sin_a * v) as f32
                        }).collect();
                        pr.b.r_sim.mean(&per)
                    };
                    let want = reference(true);
                    let tissue_peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
                    let worst = got.iter().zip(&want).map(|(a, w)| (a - *w as f64).abs()).fold(0.0, f64::max);
                    // the label is about a hundredth of the tissue: the tolerance covers the
                    // interpolation (1e-4 of the label's peak) and f32
                    assert!(worst <= 3e-6 * tissue_peak, "volume {g} partition {pp} segment {sy}: {worst:e} of {tissue_peak:e}");
                    if g == 1 && j == 3 {
                        let no_lead = reference(false);
                        let miss = got.iter().zip(&no_lead).map(|(a, w)| (a - *w as f64).abs()).fold(0.0, f64::max);
                        assert!(miss > 10.0 * worst.max(1e-9 * tissue_peak), "the slab-entry depletion is not in the images: {miss:e} vs {worst:e}");
                    }
                }
            }
        }
    }

    /// The series refuses what it does not model yet, and runs a protocol end to end.
    #[test]
    fn the_series_runs_and_refuses_what_it_does_not_model() {
        let ph = crop();
        let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
        let out = simulate_ge3d(&ge3d_protocol(""), &ph, T2Mode::Auto, &phase, RowOverride::None).unwrap();
        assert_eq!(out.n_volumes, 2);
        assert!(out.ge3d.is_some() && out.readout.is_none());
        assert!(out.mag.iter().all(|x| x.is_finite()) && out.mag.iter().any(|x| *x > 0.0));
        assert!(simulate_ge3d(&ge3d_protocol(""), &ph, T2Mode::Auto, &phase, RowOverride::BloodIntoTissue0).is_err());
    }

    const MOTION: &str = "[motion]\nmode = \"random\"\ntrans_mm = [1.0, 1.0, 0.0]\nrot_deg = [0.0, 0.0, 2.0]\nvolumes = [1]\n\
                          [motion.within_volume]\ndropout_rate = 1.0\nseverity = 0.3\njump_mm = [0.5, 0.0, 0.0]\njump_deg = [0.0, 0.0, 1.0]\n";

    /// Motion on the 3D gradient-echo train (P5 part D, as on the spin-echo path): each volume's
    /// images are the static ones moved by its pose; the shots after a within-volume event see them
    /// moved again by the cumulative jump (one shot set per distinct pose); an event shot's weights
    /// are scaled by 1 - severity; the truth is moved by the volume poses, the static one kept.
    #[test]
    fn motion_moves_the_volumes_and_their_shots() {
        let ph = crop();
        let (ps, pm) = (ge3d_protocol(""), ge3d_protocol(MOTION));
        let still = prepare_ge3d(&ps, &ph, T2Mode::Class, RowOverride::None).unwrap();
        let moving = prepare_ge3d(&pm, &ph, T2Mode::Class, RowOverride::None).unwrap();
        let (bs, bm) = (&still.b, &moving.b);
        assert_eq!(bm.poses[0], Pose::IDENTITY);
        assert_ne!(bm.poses[1], Pose::IDENTITY);
        assert!(!moving.events.is_empty() && moving.dropped.len() == moving.events.len());
        let ky = bm.res.readout.ky_segments;
        let ncomp = bm.n_compartments();
        assert_eq!(ncomp, bs.n_compartments());
        let mut sets_seen = 0;
        for g in 0..2 {
            let (vs, vm) = (volume_inputs(bs, g), volume_inputs(bm, g));
            for (a, b) in vs.images.iter().zip(&vm.images) {
                if a.is_empty() {
                    assert!(b.is_empty());
                    continue;
                }
                let want = if bm.poses[g] == Pose::IDENTITY { a.clone() }
                           else { mrsim_acq::motion::resample_by_pose(a, bm.sim_dims, bm.v2w, bm.poses[g]) };
                assert_eq!(b, &want, "volume {g}: the volume's pose");
            }
            for set in vm.shot_images.iter().flatten() {
                sets_seen += 1;
                let q = bm.shot_pose[g][set.shots[0]];
                assert!(q != Pose::IDENTITY && set.shots.iter().all(|&s| bm.shot_pose[g][s] == q));
                for (im, base) in set.images.iter().zip(&vm.images) {
                    if !base.is_empty() {
                        assert_eq!(im, &mrsim_acq::motion::resample_by_pose(base, bm.sim_dims, bm.v2w, q));
                    }
                }
            }
            for pp in 0..8 {
                for sy in 0..ky {
                    let gain = bm.shot_gain[g][bm.table.line(pp, sy).shot];
                    for c in 0..ncomp {
                        let at = (pp * ky + sy) * ncomp + c;
                        assert!((vm.weights[at] - gain * vs.weights[at]).abs() <= 1e-12 * vs.weights[at].abs().max(1e-300));
                    }
                }
            }
            assert!(bm.shot_gain[g].iter().any(|&x| (x - 0.7).abs() < 1e-6), "an event shot in every volume (rate 1)");
        }
        assert!(sets_seen > 0, "an event before the last shot moves the later ones");
        // the series: the truth moved by the volume poses, the static kept; poses and events recorded
        let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
        let out_s = simulate_ge3d(&ps, &ph, T2Mode::Class, &phase, RowOverride::None).unwrap();
        let out_m = simulate_ge3d(&pm, &ph, T2Mode::Class, &phase, RowOverride::None).unwrap();
        let gt = &out_m.ground_truth;
        assert_eq!(gt.delta_m_static.as_ref().unwrap(), &out_s.ground_truth.delta_m);
        assert_ne!(gt.delta_m, out_s.ground_truth.delta_m);
        assert_eq!((out_m.poses.len(), out_m.events.len(), out_m.motion_seed.is_some()), (2, moving.events.len(), true));
        assert_ne!(out_m.mag, out_s.mag);
    }

    // ---- Task 16: 3D Look-Locker

    /// `ge3d_protocol`'s train read by Look-Locker: control then label cycles of `plds.len()`
    /// readouts (sub-trains), `flips` the scalar or per-volume FlipAngle.
    fn ll3d_protocol(plds: &[f64], flips: Value, extra: &str) -> Protocol {
        let m = plds.len();
        let pld: Vec<f64> = (0..2).flat_map(|_| plds.iter().copied()).collect();
        let s: Value = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": pld,
            "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 1, "BackgroundSuppressionPulseTime": [2.0],
            "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012, "MagneticFieldStrength": 3,
            "AcquisitionVoxelSize": [2.0, 2.0, 0.75], "MRAcquisitionType": "3D", "PulseSequenceType": "3D EPI",
            "PhaseEncodingDirection": "j-", "EffectiveEchoSpacing": 0.0005, "NumberShots": 4, "FlipAngle": flips,
            "LookLocker": true
        });
        let ctx = format!("volume_type\n{}{}", "control\n".repeat(m), "label\n".repeat(m));
        let ov: Overlay = toml::from_str(&format!(
            "seed = 5\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n[signal]\nacq_contrast = \"ge\"\n\
             [readout]\nexcitation_spacing = 40.0\nslab_entry_time = 0.5\nkz_segments = 2\n[kinetic]\nexchange_time = 0.4\n\
             [look_locker]\nreadouts_per_cycle = {m}\n{extra}")).unwrap();
        parse(&s, &ctx, Some(&ov), crop().params.as_ref()).unwrap()
    }

    /// 3D Look-Locker: the schedule (a group of four shots' cycles per cycle, every readout reading
    /// each), and every raw volume's input at every partition's excitation against the brute-force
    /// timeline (every sub-train of the cycle an event) and the parcel reference (depleted by every
    /// earlier excitation of the cycle, earlier sub-trains included, from slab entry); the recorded
    /// cumulative depletion.
    #[test]
    fn look_locker_sub_trains_see_the_reference_object() {
        let ph = crop();
        let plds = [0.5, 0.75, 1.0];
        let p = ll3d_protocol(&plds, json!([12, 15, 18, 12, 15, 18]), "");
        let pr = prepare_ge3d(&p, &ph, T2Mode::Class, RowOverride::None).unwrap();
        let b = &pr.b;
        // the schedule: per cycle four preparations (shots) and three raw volumes reading them
        assert_eq!((b.sched.preps.len(), b.sched.raws.len()), (2 * 4, 6));
        for (v, r) in b.sched.raws.iter().enumerate() {
            assert_eq!((r.prep, r.n_preps, r.readout), ((v / 3) * 4, 4, v % 3));
        }
        assert!((b.sched.preps[4].start_s - 4.0 * 4.0).abs() < 1e-12 && (b.sched.preps[1].start_s - 4.0).abs() < 1e-12);
        let spec = p.suppression.as_ref().unwrap();
        let eps = spec.epsilon.0;
        let spacing = 0.040;
        let flips = [12.0, 15.0, 18.0];
        // the brute-force tissue timeline: each preparation's cycle every sub-train
        let sups: Vec<Option<Suppression>> = b.sched.preps.iter().map(|_| Some(Suppression::new(vec![2.0], eps, false))).collect();
        let cyc_times = |r0: usize| -> (Vec<f64>, Vec<f64>) {
            let mut t = Vec::new();
            let mut f = Vec::new();
            for (n, &fa) in flips.iter().enumerate() {
                t.extend(train_times(b.sched.raw_rows[r0 + n].t, 4, spacing));
                f.extend(std::iter::repeat_n(fa, 4));
            }
            (t, f)
        };
        let cycles: Vec<LlCycle> = b.sched.preps.iter().zip(&sups).map(|(pp, s)| {
            let (t, f) = cyc_times(pp.raw);
            LlCycle { tr: 4.0, s: s.as_ref(), t_read: t, flip_deg: f }
        }).collect();
        let ncomp = b.n_compartments();
        let ky = b.res.readout.ky_segments;
        let [snx, sny, _] = pr.sim_grid.dims;
        for g in 0..6 {
            let vol = volume_inputs(b, g);
            let (r0, n) = ((g / 3) * 3, g % 3);
            let row = &b.sched.raw_rows[g];
            let kin = p.kinetic(row);
            let (e_all, f_all) = cyc_times(r0);
            let sign = if row.kind == RowKind::Label { -1.0 } else { 0.0 };
            let sin_a = flips[n].to_radians().sin();
            // the recorded cumulative depletion: every excitation of the earlier sub-trains
            let want_dep: f64 = f_all[..4 * n].iter().map(|a: &f64| a.to_radians().cos()).product();
            assert!((b.cumulative_depletion(g) - want_dep).abs() < 1e-15);
            for pp in [0usize, 3, 4, 7] {
                for sy in 0..ky {
                    let line = b.table.line(pp, sy);
                    let (j, prep) = (line.excitation, b.sched.raws[g].prep + line.shot);
                    let x = 4 * n + j;
                    let got: Vec<f64> = (0..snx * sny * 8).map(|v| {
                        (0..ncomp).filter(|&c| !vol.images[c].is_empty())
                            .map(|c| vol.weights[(pp * ky + sy) * ncomp + c] * vol.images[c][v] as f64).sum()
                    }).collect();
                    let per: Vec<f32> = (0..ph.nvox()).map(|i| {
                        if ph.dseg[i] <= 0 {
                            return 0.0;
                        }
                        let mut v = ph.m0[i] as f64 * brute(ph.t1[i] as f64, &cycles)[prep][x];
                        if sign != 0.0 && ph.perfusion[i] > 0.0 {
                            let c = Case {
                                k: kin, f: ph.perfusion[i] as f64, att: ph.att[i] as f64, t1t: ph.t1[i] as f64,
                                m0: ph.m0[i] as f64, t: e_all[x],
                                excitations: e_all[..x].iter().copied().zip(f_all[..x].iter().copied()).collect(),
                                entry_lead: ph.att[i] as f64 - 0.5, pulses: vec![2.0], epsilon: eps, region: PRegion::Global,
                                tau_ex: Some(0.4), span: (0.0, kin.tau),
                            };
                            v += sign * c.read(4).1;
                        }
                        (sin_a * v) as f32
                    }).collect();
                    let want = b.r_sim.mean(&per);
                    let peak = want.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;
                    let worst = got.iter().zip(&want).map(|(a, w)| (a - *w as f64).abs()).fold(0.0, f64::max);
                    assert!(worst <= 3e-6 * peak, "volume {g} partition {pp} segment {sy}: {worst:e} of {peak:e}");
                }
            }
        }
    }

    /// One readout per cycle at a scalar flip is the plain 3D EPI series: every volume's input the
    /// same, bit for bit. A GRASE protocol still refuses Look-Locker, naming the 3D EPI readout;
    /// overlapping sub-trains are refused, naming both rows.
    #[test]
    fn look_locker_reductions_and_refusals() {
        let ph = crop();
        let one = ll3d_protocol(&[0.5], json!(12), "");
        let plain = ge3d_protocol("");
        let (a, b) = (prepare_ge3d(&one, &ph, T2Mode::Class, RowOverride::None).unwrap(),
                      prepare_ge3d(&plain, &ph, T2Mode::Class, RowOverride::None).unwrap());
        for g in 0..2 {
            let (va, vb) = (volume_inputs(&a.b, g), volume_inputs(&b.b, g));
            assert_eq!(va.images, vb.images, "volume {g}");
            assert_eq!(va.weights.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                       vb.weights.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "volume {g}");
        }
        // GRASE with Look-Locker
        let mut g: Value = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": [0.5, 0.8],
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012,
            "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "3D",
            "PulseSequenceType": "3Dgrase", "PhaseEncodingDirection": "j-", "EffectiveEchoSpacing": 0.0005,
            "NumberShots": 2, "FlipAngle": 150
        });
        g["LookLocker"] = json!(true);
        let ov: Overlay = toml::from_str("[signal]\nacq_contrast = \"ge\"\n").unwrap();
        let e = parse(&g, "volume_type\nlabel\nlabel\n", Some(&ov), None).unwrap_err();
        assert!(e.contains("epi3d") || e.contains("GRASE"), "{e}");
        // readouts 0.1 s apart cannot hold a 4-excitation sub-train of 40 ms spacing
        let tight = ll3d_protocol(&[0.5, 0.6], json!(12), "");
        let e = crate::protocol::resolve_ge3d(&tight, [12, 12, 8]).unwrap_err();
        assert!(e.contains("sub-train") && e.contains("row 0") && e.contains("row 1"), "{e}");
    }

    // ---- Task 17: multi-TE 3D

    /// `ge3d_protocol` read at several echo times, one sidecar per echo.
    fn ge3d_echoes(tes: &[f64], extra: &str) -> Protocol {
        let sides: Vec<Value> = tes.iter().map(|te| json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 0.5,
            "BackgroundSuppression": true, "BackgroundSuppressionNumberPulses": 1, "BackgroundSuppressionPulseTime": [2.0],
            "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": te, "MagneticFieldStrength": 3,
            "AcquisitionVoxelSize": [2.0, 2.0, 0.75], "MRAcquisitionType": "3D", "PulseSequenceType": "3D EPI",
            "PhaseEncodingDirection": "j-", "EffectiveEchoSpacing": 0.0005, "NumberShots": 4, "FlipAngle": 12
        })).collect();
        let ov: Overlay = toml::from_str(&format!(
            "seed = 5\n[acquisition]\noversample = 2\nsignal_scale = 100.0\n[signal]\nacq_contrast = \"ge\"\n\
             [readout]\nexcitation_spacing = 40.0\nslab_entry_time = 0.5\nkz_segments = 2\n[kinetic]\nexchange_time = 0.4\n{extra}"))
            .unwrap();
        crate::protocol::parse_echoes(&sides, "volume_type\ncontrol\nlabel\n", Some(&ov), crop().params.as_ref()).unwrap()
    }

    /// Several echoes per excitation: noise off, each echo is the single-echo series at its echo time
    /// bit for bit; noise on, echo 1 is still the single-echo series bit for bit and the echoes' noise
    /// differs; the excitation spacing bounds the echo count (the last block must end before the
    /// next excitation pulse); GRASE still refuses several echoes.
    #[test]
    fn several_echoes_per_excitation() {
        let ph = crop();
        let phase = PhaseModel { global: 0.0, background: Default::default(), prep: None };
        let run = |p: &Protocol| simulate_ge3d(p, &ph, T2Mode::Auto, &phase, RowOverride::None).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let tes = [0.012, 0.024];
        let multi = run(&ge3d_echoes(&tes, ""));
        assert_eq!(multi.more_echoes.len(), 1);
        for (e, te) in tes.iter().enumerate() {
            let one = run(&ge3d_echoes(&[*te], ""));
            let (m, ph_) = if e == 0 { (&multi.mag, &multi.phase) } else { (&multi.more_echoes[0].mag, &multi.more_echoes[0].phase) };
            assert_eq!((bits(m), bits(ph_)), (bits(&one.mag), bits(&one.phase)), "echo {}", e + 1);
        }
        assert_ne!(bits(&multi.mag), bits(&multi.more_echoes[0].mag), "the echoes decay differently");
        let with_noise = |tes: &[f64]| {
            let mut p = ge3d_echoes(tes, "");
            p.acq.noise_variance = 0.05;
            run(&p)
        };
        let (mn, on) = (with_noise(&tes), with_noise(&tes[..1]));
        assert_eq!((bits(&mn.mag), bits(&mn.phase)), (bits(&on.mag), bits(&on.phase)), "echo 1 under noise");
        // echo 2's noise is its own: its residual against the noise-off echo 2 differs from echo 1's
        let r = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) as f64).collect::<Vec<f64>>();
        let (r1, r2) = (r(&mn.mag, &multi.mag), r(&mn.more_echoes[0].mag, &multi.more_echoes[0].mag));
        assert!(r1.iter().any(|x| *x != 0.0) && r1 != r2);
        // the grid's blocks: 6 lines of 1 ms, half-width 3 ms, centred at EchoTime + 0.5 ms; with 40 ms
        // between excitations and a 1 ms pulse the last echo's block must end by 39 ms: EchoTime <= 35.5 ms
        let dims = [12, 12, 8];
        crate::protocol::resolve_ge3d(&ge3d_echoes(&[0.012, 0.020, 0.028, 0.035], ""), dims).unwrap();
        let e = crate::protocol::resolve_ge3d(&ge3d_echoes(&[0.012, 0.020, 0.028, 0.036], ""), dims).unwrap_err();
        assert!(e.contains("next excitation pulse"), "{e}");
        // GRASE refuses several echoes
        let mut g = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": 1.8, "PostLabelingDelay": 1.8,
            "BackgroundSuppression": false, "M0Type": "Absent", "RepetitionTimePreparation": 4.0, "EchoTime": 0.012,
            "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [2.0, 2.0, 3.0], "MRAcquisitionType": "3D",
            "PulseSequenceType": "3Dgrase", "PhaseEncodingDirection": "j-", "EffectiveEchoSpacing": 0.0005,
            "NumberShots": 2, "FlipAngle": 150
        });
        let g2 = { let mut x = g.clone(); x["EchoTime"] = json!(0.03); x };
        g["EchoTime"] = json!(0.012);
        let e = crate::protocol::parse_echoes(&[g, g2], "volume_type\nlabel\n", None, None).unwrap_err();
        assert!(e.contains("multi-echo spin-echo"), "{e}");
    }
}
