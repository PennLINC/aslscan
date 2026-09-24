//! The longitudinal-magnetization timeline under background suppression (P3 addendum, part A).
//! Pure std, seconds throughout.
//!
//! Per row, `t = 0` at labeling start. The static tissue recovers toward `m0` with its T1 from
//! what the previous excitation left, each suppression pulse at `p_k` scales `Mz` by
//! `1 - 2 epsilon`, and the slice is read at `t_read`. The labeled blood's difference signal
//! sees the same pulses: each multiplies it by `1 - 2 epsilon`, the recovery toward `m0`
//! cancelling between control and label. That blood factor treats every pulse as inverting the
//! whole bolus wherever it is (the addendum's "global-bolus" approximation, named in the
//! sidecar); `protocol` guarantees every pulse precedes the first slice's readout, so the pulse
//! set is the same for every slice of a row and only `t_read` differs.
//!
//! simasl v2.2.0 has no background-suppression model, so the tests here are closed forms and
//! the addendum's hand-worked `asl002` example, not fixtures.

/// The pulse train and its efficiency for one row. `pulse_times` are seconds from labeling
/// start, kept sorted.
#[derive(Debug, Clone, PartialEq)]
pub struct Suppression {
    pub pulse_times: Vec<f64>,
    /// The fraction of longitudinal magnetization each pulse inverts, in `[0, 1]`.
    pub epsilon: f64,
    /// A saturation pulse at labeling start: `Mz(0) = 0`.
    pub presaturation: bool,
}

impl Suppression {
    pub fn new(mut pulse_times: Vec<f64>, epsilon: f64, presaturation: bool) -> Suppression {
        pulse_times.sort_by(|a, b| a.partial_cmp(b).expect("pulse times are finite"));
        Suppression { pulse_times, epsilon, presaturation }
    }

    /// Whether the timeline differs from the P1 steady state at all.
    pub fn has_events(&self) -> bool {
        self.presaturation || !self.pulse_times.is_empty()
    }
}

/// Recovery from `mz` toward `m0` over `dt` seconds with time constant `t1` (positive).
fn recover(mz: f64, m0: f64, t1: f64, dt: f64) -> f64 {
    m0 - (m0 - mz) * (-dt / t1).exp()
}

/// Signed tissue `Mz` at `t_read` (s) for a row with repetition time `tr` (s): recovery for
/// `tr - t_read` after the previous 90-degree excitation (or zero under presaturation), the
/// pulses before `t_read` applied in order with recovery between them, then recovery to
/// `t_read`. A zero T1 gives 0, the guard [`crate::mrsignal::tissue_se`] applies (an infinite
/// rate is "no signal" in P1; run through `exp(-x/0)` it would instead give `m0`). Pulses at or
/// after `t_read` are ignored; `protocol` rejects them for the first slice.
///
/// With no events this is `m0 (1 - exp(-(tr - t_read)/t1) exp(-t_read/t1))`, the P1 steady
/// state in exact arithmetic but not bit for bit, which is why `series` keeps `tissue_se` for
/// that case.
pub fn tissue_mz(m0: f64, t1: f64, tr: f64, t_read: f64, s: &Suppression) -> f64 {
    if t1 == 0.0 {
        return 0.0;
    }
    let mut mz = if s.presaturation { 0.0 } else { recover(0.0, m0, t1, tr - t_read) };
    let mut t = 0.0;
    for &p in s.pulse_times.iter().filter(|&&p| p < t_read) {
        mz = recover(mz, m0, t1, p - t);
        mz *= 1.0 - 2.0 * s.epsilon;
        t = p;
    }
    recover(mz, m0, t1, t_read - t)
}

/// The blood difference signal's factor: `prod (1 - 2 epsilon)` over every pulse. `(-1)^N` for
/// perfect pulses; an odd count flips `control - label`.
pub fn label_factor(s: &Suppression) -> f64 {
    s.pulse_times.iter().map(|_| 1.0 - 2.0 * s.epsilon).product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mrsignal::tissue_se;

    fn none() -> Suppression {
        Suppression::new(vec![], 1.0, false)
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()).max(1e-300)
    }

    #[test]
    fn no_events_is_the_p1_steady_state() {
        for (m0, t1, tr, t_read) in [(74.6, 1.33, 4.0, 3.6), (88.0, 0.83, 4.57, 3.8), (1.0, 3.0, 8.0, 2.0)] {
            let got = tissue_mz(m0, t1, tr, t_read, &none());
            let want = tissue_se(m0, t1, tr);
            assert!(close(got, want, 1e-12), "{got} vs {want}");
        }
        assert!(!none().has_events());
        assert_eq!(tissue_mz(74.6, 0.0, 4.0, 3.6, &none()), 0.0);
    }

    #[test]
    fn zero_efficiency_is_the_identity() {
        let s = Suppression::new(vec![2.05, 3.276], 0.0, false);
        assert!(s.has_events());
        let got = tissue_mz(1.0, 1.33, 4.57, 3.8, &s);
        assert!(close(got, tissue_mz(1.0, 1.33, 4.57, 3.8, &none()), 1e-12));
        assert_eq!(label_factor(&s), 1.0);
    }

    #[test]
    fn one_perfect_pulse_just_before_readout_negates() {
        let s = Suppression::new(vec![3.8 - 1e-12], 1.0, false);
        let got = tissue_mz(1.0, 1.33, 4.57, 3.8, &s);
        let unsupp = tissue_mz(1.0, 1.33, 4.57, 3.8, &none());
        assert!(close(got, -unsupp, 1e-9), "{got} vs {}", -unsupp);
        assert_eq!(label_factor(&s), -1.0);
    }

    #[test]
    fn two_coincident_perfect_pulses_cancel() {
        let s = Suppression::new(vec![2.5, 2.5], 1.0, false);
        let got = tissue_mz(1.0, 1.33, 4.57, 3.8, &s);
        assert!(close(got, tissue_mz(1.0, 1.33, 4.57, 3.8, &none()), 1e-12));
        assert_eq!(label_factor(&s), 1.0);
    }

    #[test]
    fn presaturation_starts_from_zero() {
        let s = Suppression::new(vec![], 1.0, true);
        assert!(s.has_events());
        let got = tissue_mz(2.0, 1.33, 4.57, 3.8, &s);
        let want = 2.0 * (1.0 - (-3.8f64 / 1.33).exp());
        assert!(close(got, want, 1e-12));
    }

    #[test]
    fn pulses_at_or_after_readout_are_ignored() {
        let s = Suppression::new(vec![3.8, 4.0], 1.0, false);
        assert!(close(tissue_mz(1.0, 1.33, 4.57, 3.8, &s), tissue_mz(1.0, 1.33, 4.57, 3.8, &none()), 1e-12));
    }

    /// The addendum's asl002 example: PCASL tau 1.8, PLD 2.0, TR 4.5717, pulses 2.05 and 3.276,
    /// perfect pulses. First slice reads at 3.8, the last at 3.8 + 0.7315.
    #[test]
    fn asl002_worked_example() {
        let s = Suppression::new(vec![2.05, 3.276], 1.0, false);
        let tr = 4.57168115234375;
        for (t1, t_read, want, unsupp) in [
            (1.33, 3.8, 0.156, 0.968),
            (0.83, 3.8, 0.175, 0.996),
            (3.0, 3.8, 0.219, 0.782),
            (1.33, 3.8 + 0.7315, 0.499, 0.968),
            (0.83, 3.8 + 0.7315, 0.656, 0.996),
        ] {
            let got = tissue_mz(1.0, t1, tr, t_read, &s);
            assert!((got - want).abs() < 1e-3, "T1 {t1} at {t_read}: {got} vs {want}");
            let u = tissue_mz(1.0, t1, tr, t_read, &none());
            assert!((u - unsupp).abs() < 1e-3, "unsuppressed T1 {t1}: {u} vs {unsupp}");
        }
        assert_eq!(label_factor(&s), 1.0);
        let s95 = Suppression::new(vec![2.05, 3.276], 0.95, false);
        assert!(close(label_factor(&s95), 0.81, 1e-12));
        let s3 = Suppression::new(vec![2.0, 2.5, 3.0], 1.0, false);
        assert_eq!(label_factor(&s3), -1.0);
    }

    #[test]
    fn pulse_times_are_sorted() {
        let s = Suppression::new(vec![3.276, 2.05], 1.0, false);
        assert_eq!(s.pulse_times, vec![2.05, 3.276]);
        assert!(close(tissue_mz(1.0, 1.33, 4.57168115234375, 3.8, &s), 0.156, 1e-2));
    }
}
