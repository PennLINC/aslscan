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
}
