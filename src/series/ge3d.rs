//! The series of a 3D gradient-echo train (P7 addendum, part C; `[readout] type = "epi3d"`).
//!
//! Each kz partition is read by its own small-flip excitation, so the train's longitudinal state
//! enters the acquisition as one weight per excitation (`mrsim_acq::kspace3d::GeVolume`). This
//! module computes those weights: the tissue's from the excitation-train timeline, exactly, by
//! grouping each tissue compartment by `T1` (tissue `Mz` is `M0(r) m_j(T1(r))`); the label's (Task
//! 14) by depletion from slab entry and a piecewise-linear expansion over the train.

// the series that calls these is P7 Task 15; until then they are exercised by their tests
#![cfg_attr(not(test), allow(dead_code))]

use super::*;
use crate::longitudinal::{tissue_mz_ll, tissue_mz_ll_sequence, LlCycle, Suppression};
use crate::protocol::SlabEntryTime;

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
}
