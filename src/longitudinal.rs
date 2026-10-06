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
    let mz = if s.presaturation { 0.0 } else { recover(0.0, m0, t1, tr - t_read) };
    timeline(mz, m0, t1, t_read, s)
}

/// From `Mz = mz0` at labeling start, the pulses before `t_read` with recovery between them, then
/// recovery to `t_read`. Affine in `mz0` (each recovery is `m0 (1 - E) + E Mz`, each pulse a
/// scaling), which the gradient-echo steady state below uses.
fn timeline(mz0: f64, m0: f64, t1: f64, t_read: f64, s: &Suppression) -> f64 {
    let mut mz = mz0;
    let mut t = 0.0;
    for &p in s.pulse_times.iter().filter(|&&p| p < t_read) {
        mz = recover(mz, m0, t1, p - t);
        mz *= 1.0 - 2.0 * s.epsilon;
        t = p;
    }
    recover(mz, m0, t1, t_read - t)
}

/// Signed tissue `Mz` at `t_read` under a gradient-echo excitation of `flip_deg`, in the steady
/// state of the same preparation repeated (P5 addendum, part A). The excitation leaves
/// `cos(a) Mz(t_read)`, so the start of the next repetition is
/// `Mz(0) = m0 (1 - Er) + cos(a) Er (A Mz(0) + B)`, `Er = exp(-(tr - t_read)/t1)`, with the
/// timeline `Mz(t_read) = A Mz(0) + B`; solved for `Mz(0)`. At exactly 90 degrees this is P3's
/// [`tissue_mz`], called as such so its bits do not move; presaturation (`Mz(0) = 0`) needs no
/// fixed point. The readout compartment is `sin(a)` times this.
pub fn tissue_mz_ge(m0: f64, t1: f64, tr: f64, t_read: f64, s: &Suppression, flip_deg: f64) -> f64 {
    if flip_deg == 90.0 {
        return tissue_mz(m0, t1, tr, t_read, s);
    }
    if t1 == 0.0 {
        return 0.0;
    }
    if s.presaturation {
        return timeline(0.0, m0, t1, t_read, s);
    }
    let b = timeline(0.0, m0, t1, t_read, s);
    let a = timeline(1.0, m0, t1, t_read, s) - b;
    let er = (-(tr - t_read) / t1).exp();
    let ca = flip_deg.to_radians().cos();
    // |a| <= 1 and er < 1, so the denominator is positive
    let mz0 = (m0 * (1.0 - er) + ca * er * b) / (1.0 - ca * er * a);
    a * mz0 + b
}

/// One row of a series in acquisition order, for [`tissue_mz_ge_sequence`]: its repetition time,
/// this slice's readout time (from labeling start), and its suppression (`None`: no events).
#[derive(Debug, Clone, Copy)]
pub struct Prep<'a> {
    pub tr: f64,
    pub t_read: f64,
    pub s: Option<&'a Suppression>,
}

/// Signed tissue `Mz` at each row's readout under a gradient-echo excitation of `flip_deg` when
/// the rows' preparations differ (P5 addendum, part A, "a series whose rows differ"): the state is
/// carried row to row, `Mz(0)` of a row being `m0 (1 - Er) + cos(a) Er Mz_read` of the row before
/// (its `Er`); the first row starts from its own isolated steady state (the dummy repetitions are
/// taken to have used it), and a presaturated row starts from zero. At exactly 90 degrees nothing
/// carries over and each row is [`tissue_mz`] (or the plain recovery when it has no events).
pub fn tissue_mz_ge_sequence(m0: f64, t1: f64, preps: &[Prep], flip_deg: f64) -> Vec<f64> {
    let empty = Suppression::new(vec![], 0.0, false);

    if t1 == 0.0 {
        return vec![0.0; preps.len()];
    }
    if flip_deg == 90.0 {
        return preps.iter().map(|p| tissue_mz(m0, t1, p.tr, p.t_read, p.s.unwrap_or(&empty))).collect();
    }
    let ca = flip_deg.to_radians().cos();
    let mut out = Vec::with_capacity(preps.len());
    let mut prev: Option<(f64, f64)> = None; // (Er, Mz_read) of the previous row
    for p in preps {
        let s = p.s.unwrap_or(&empty);
        let mz_read = match prev {
            None => tissue_mz_ge(m0, t1, p.tr, p.t_read, s, flip_deg),
            Some(_) if s.presaturation => timeline(0.0, m0, t1, p.t_read, s),
            Some((er, mz)) => timeline(m0 * (1.0 - er) + ca * er * mz, m0, t1, p.t_read, s),
        };
        out.push(mz_read);
        prev = Some(((-(p.tr - p.t_read) / t1).exp(), mz_read));
    }
    out
}

// ---- P6 part B: Look-Locker readouts ----

/// One Look-Locker cycle of one slice (P6 addendum, part B): a preparation (its suppression,
/// pulses before the first readout) followed by readouts at `t_read[n]` (s from the start of the
/// labeling, this slice's, increasing) with flips `flip_deg[n]`, every one an event on `Mz`: the
/// readout's signal is `sin(a) Mz` just before it and it leaves `cos(a) Mz`. The cycle lasts `tr`.
/// An `m0scan` cycle is one readout at the start of its own repetition, no labeling, no pulses.
#[derive(Debug, Clone)]
pub struct LlCycle<'a> {
    pub tr: f64,
    pub s: Option<&'a Suppression>,
    pub t_read: Vec<f64>,
    pub flip_deg: Vec<f64>,
}

/// One cycle from `Mz(0) = mz0` at its start: the pulses before the first readout with recovery
/// between them, then each readout (recovery to it, the state recorded, the `cos(a)` applied),
/// then recovery to `tr`. Returns `Mz` before each readout and `Mz` at the end. Affine in `mz0`.
fn ll_cycle(mz0: f64, m0: f64, t1: f64, c: &LlCycle) -> (Vec<f64>, f64) {
    let empty = Suppression::new(vec![], 0.0, false);
    let s = c.s.unwrap_or(&empty);
    let first = c.t_read.first().copied().unwrap_or(c.tr);
    let mut mz = if s.presaturation { 0.0 } else { mz0 };
    let mut t = 0.0;
    for &p in s.pulse_times.iter().filter(|&&p| p < first) {
        mz = recover(mz, m0, t1, p - t);
        mz *= 1.0 - 2.0 * s.epsilon;
        t = p;
    }
    let mut before = Vec::with_capacity(c.t_read.len());
    for (&tr_n, &fa) in c.t_read.iter().zip(&c.flip_deg) {
        mz = recover(mz, m0, t1, tr_n - t);
        before.push(mz);
        mz *= fa.to_radians().cos();
        t = tr_n;
    }
    (before, recover(mz, m0, t1, c.tr - t))
}

/// Signed tissue `Mz` before each readout of a Look-Locker cycle repeated to its steady state:
/// the cycle maps its starting `Mz` affinely, `end = A Mz(0) + B`, so the steady state is
/// `Mz(0) = B / (1 - A)` (a presaturated cycle starts from zero and needs none). A zero T1 gives
/// zeros, as [`tissue_mz`] does.
pub fn tissue_mz_ll(m0: f64, t1: f64, c: &LlCycle) -> Vec<f64> {
    if t1 == 0.0 {
        return vec![0.0; c.t_read.len()];
    }
    let (_, b) = ll_cycle(0.0, m0, t1, c);
    let (_, ab) = ll_cycle(1.0, m0, t1, c);
    let a = ab - b;
    // |a| < 1: every readout and pulse scales by at most 1 and the recovery by exp(-tr/t1) < 1
    let mz0 = b / (1.0 - a);
    ll_cycle(mz0, m0, t1, c).0
}

/// Signed tissue `Mz` before each readout of each cycle of a series whose cycles differ: the first
/// cycle starts from its own steady state ([`tissue_mz_ll`]'s, the dummy cycles taken to have run
/// it) and each later one from the end state of the cycle before (a presaturated one from zero).
pub fn tissue_mz_ll_sequence(m0: f64, t1: f64, cycles: &[LlCycle]) -> Vec<Vec<f64>> {
    if t1 == 0.0 {
        return cycles.iter().map(|c| vec![0.0; c.t_read.len()]).collect();
    }
    let mut out = Vec::with_capacity(cycles.len());
    let mut mz = None;
    for c in cycles {
        let start = match mz {
            None => {
                let (_, b) = ll_cycle(0.0, m0, t1, c);
                let (_, ab) = ll_cycle(1.0, m0, t1, c);
                b / (1.0 - (ab - b))
            }
            Some(m) => m,
        };
        let (before, end) = ll_cycle(start, m0, t1, c);
        out.push(before);
        mz = Some(end);
    }
    out
}

/// Whether a Look-Locker series takes the legacy dispatch (P6 addendum, part B): every cycle one
/// readout and one scalar `FlipAngle` for the whole series (P5's sequence takes one flip; an array,
/// even of equal values, gives every volume its own excitation and takes the generalized timeline).
pub fn ll_legacy_dispatch(readouts_per_cycle: impl IntoIterator<Item = usize>, flip_array: bool) -> bool {
    !flip_array && readouts_per_cycle.into_iter().all(|m| m == 1)
}

/// The Look-Locker tissue of a series. With `legacy_flip` (the series' scalar flip, given when
/// [`ll_legacy_dispatch`] holds) it is P5's [`tissue_mz_ge`] (one cycle repeated, `uniform`) or
/// [`tissue_mz_ge_sequence`] themselves, so the result is P5's bit for bit, `m0scan` cycles taking
/// P5's convention (read at the end of their repetition, `t_read = tr`). Otherwise the generalized
/// timeline, where an `m0scan` cycle is read at the start of its repetition (its own `t_read`).
pub fn tissue_mz_ll_series(m0: f64, t1: f64, cycles: &[LlCycle], m0scan: &[bool], legacy_flip: Option<f64>, uniform: bool)
    -> Vec<Vec<f64>>
{
    match legacy_flip {
        Some(fa) => {
            assert!(cycles.iter().all(|c| c.t_read.len() == 1), "the legacy dispatch has one readout per cycle");
            let empty = Suppression::new(vec![], 0.0, false);
            if uniform {
                cycles.iter().map(|c| vec![tissue_mz_ge(m0, t1, c.tr, c.t_read[0], c.s.unwrap_or(&empty), fa)]).collect()
            } else {
                let preps: Vec<Prep> = cycles.iter().zip(m0scan).map(|(c, &m)| Prep {
                    tr: c.tr, t_read: if m { c.tr } else { c.t_read[0] }, s: c.s,
                }).collect();
                tissue_mz_ge_sequence(m0, t1, &preps, fa).into_iter().map(|x| vec![x]).collect()
            }
        }
        None => tissue_mz_ll_sequence(m0, t1, cycles),
    }
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

    /// An independent step-by-step simulation of repetitions: recovery between events, each pulse
    /// scaling `Mz` by `1 - 2 eps`, the excitation leaving `cos(a) Mz_read`. Returns `Mz_read` of
    /// every repetition from `Mz(0) = mz_start`.
    fn brute(m0: f64, t1: f64, flip_deg: f64, mz_start: f64, reps: &[(f64, f64, Vec<f64>, f64)]) -> Vec<f64> {
        let rec = |mz: f64, dt: f64| m0 - (m0 - mz) * (-dt / t1).exp();
        let ca = flip_deg.to_radians().cos();
        let mut mz = mz_start;
        let mut out = Vec::new();
        for (tr, t_read, pulses, eps) in reps {
            let mut t = 0.0;
            for &p in pulses.iter().filter(|&&p| p < *t_read) {
                mz = rec(mz, p - t) * (1.0 - 2.0 * eps);
                t = p;
            }
            let read = rec(mz, t_read - t);
            out.push(read);
            mz = rec(ca * read, tr - t_read);
        }
        out
    }

    #[test]
    fn gradient_echo_at_ninety_degrees_is_p3_bit_for_bit() {
        for s in [none(), Suppression::new(vec![2.05, 3.276], 0.95, false), Suppression::new(vec![1.0], 1.0, true)] {
            for (m0, t1, tr, t_read) in [(74.6, 1.33, 4.57, 3.8), (88.0, 0.83, 4.57, 4.53), (1.0, 3.0, 8.0, 2.0)] {
                assert_eq!(tissue_mz_ge(m0, t1, tr, t_read, &s, 90.0).to_bits(), tissue_mz(m0, t1, tr, t_read, &s).to_bits());
            }
        }
    }

    #[test]
    fn gradient_echo_fixed_point_is_the_converged_repetition() {
        for s in [none(), Suppression::new(vec![2.05, 3.276], 0.95, false), Suppression::new(vec![1.5, 2.7], 1.0, false)] {
            for (m0, t1, tr, t_read) in [(74.6, 1.33, 4.57, 3.8), (88.0, 0.83, 4.57, 4.53), (1.0, 3.0, 4.0, 3.6)] {
                for fa in [30.0f64, 60.0, -30.0, 120.0] {
                    let got = tissue_mz_ge(m0, t1, tr, t_read, &s, fa);
                    // iterate until successive starting values agree to 1e-15 m0; the multiplier
                    // cos(a) Er A has magnitude below 1, so this terminates
                    let mut start = m0;
                    let mut n = 0;
                    let read = loop {
                        let r = brute(m0, t1, fa, start, &[(tr, t_read, s.pulse_times.clone(), s.epsilon)])[0];
                        let next = m0 - (m0 - fa.to_radians().cos() * r) * (-(tr - t_read) / t1).exp();
                        n += 1;
                        if (next - start).abs() <= 1e-15 * m0 || n > 100_000 {
                            break r;
                        }
                        start = next;
                    };
                    assert!(n < 100_000, "did not converge");
                    assert!(close(got, read, 1e-12) || (got - read).abs() <= 1e-12 * m0, "fa {fa}: {got} vs {read} after {n}");
                }
            }
        }
        // no events: sin(a) times this is the spoiled closed form
        for fa in [30.0f64, 60.0, 90.0, -30.0] {
            let got = fa.to_radians().sin() * tissue_mz_ge(74.6, 1.33, 4.0, 3.6, &none(), fa);
            let want = crate::mrsignal::tissue_ge_spoiled(74.6, 1.33, 4.0, fa);
            assert!(close(got, want, 1e-12), "fa {fa}: {got} vs {want}");
        }
    }

    #[test]
    fn a_series_whose_rows_differ_carries_its_state() {
        // the second review's example: TR 4 s, T1 3 s, 30 degrees; row A reads at 2 s after a
        // perfect pulse at 1.9 s, row B at 3 s after perfect pulses at 1 s and 2.9 s
        let (m0, t1, fa) = (1.0, 3.0, 30.0f64);
        let a = Suppression::new(vec![1.9], 1.0, false);
        let b = Suppression::new(vec![1.0, 2.9], 1.0, false);
        let sin = fa.to_radians().sin();
        // isolated fixed points (wrong for the alternation)
        assert!((sin * tissue_mz_ge(m0, t1, 4.0, 2.0, &a, fa) + 0.273079).abs() < 1e-6);
        assert!((sin * tissue_mz_ge(m0, t1, 4.0, 3.0, &b, fa) + 0.110918).abs() < 1e-6);
        let n = 40;
        let preps: Vec<Prep> = (0..n).map(|i| if i % 2 == 0 { Prep { tr: 4.0, t_read: 2.0, s: Some(&a) } }
                                                else { Prep { tr: 4.0, t_read: 3.0, s: Some(&b) } }).collect();
        let got = tissue_mz_ge_sequence(m0, t1, &preps, fa);
        // the brute force from the same start: row A's isolated steady state at labeling start
        let er = (-(4.0 - 2.0) / t1).exp();
        let start = m0 * (1.0 - er) + fa.to_radians().cos() * er * tissue_mz_ge(m0, t1, 4.0, 2.0, &a, fa);
        let reps: Vec<(f64, f64, Vec<f64>, f64)> = (0..n).map(|i| if i % 2 == 0 { (4.0, 2.0, vec![1.9], 1.0) }
                                                              else { (4.0, 3.0, vec![1.0, 2.9], 1.0) }).collect();
        let want = brute(m0, t1, fa, start, &reps);
        for i in 0..n {
            assert!((got[i] - want[i]).abs() <= 1e-12, "row {i}: {} vs {}", got[i], want[i]);
        }
        // and it settles on the alternation's own steady state
        assert!((sin * got[n - 2] + 0.254639).abs() < 1e-6, "{}", sin * got[n - 2]);
        assert!((sin * got[n - 1] + 0.089888).abs() < 1e-6, "{}", sin * got[n - 1]);
        // a uniform series is the fixed point throughout
        let same: Vec<Prep> = (0..6).map(|_| Prep { tr: 4.0, t_read: 2.0, s: Some(&a) }).collect();
        let fp = tissue_mz_ge(m0, t1, 4.0, 2.0, &a, fa);
        assert!(tissue_mz_ge_sequence(m0, t1, &same, fa).iter().all(|v| (v - fp).abs() <= 1e-12));
        // a presaturated row starts from zero whatever came before
        let ps = Suppression::new(vec![1.0], 1.0, true);
        let mixed = [Prep { tr: 4.0, t_read: 2.0, s: Some(&a) }, Prep { tr: 4.0, t_read: 3.0, s: Some(&ps) }];
        let v = tissue_mz_ge_sequence(m0, t1, &mixed, fa);
        assert!((v[1] - brute(m0, t1, fa, 0.0, &[(4.0, 3.0, vec![1.0], 1.0)])[0]).abs() <= 1e-12);
        // 90 degrees: independent rows, P3's values
        let v90 = tissue_mz_ge_sequence(m0, t1, &preps[..4], 90.0);
        assert_eq!(v90[1].to_bits(), tissue_mz(m0, t1, 4.0, 3.0, &b).to_bits());
    }

    // ---- P6 part B: the Look-Locker timeline

    /// An independent reference, not sharing `ll_cycle`: each cycle written as an explicit
    /// time-sorted event list (its suppression pulses, then its readouts), `Mz` advanced between
    /// events by exact exponential recovery; repeated `reps` times from `Mz = 0`, the first cycle
    /// alone until the last repetition. Returns the states before the readouts of the last
    /// repetition. `eps_override` replaces the pulses' efficiency (the negative control).
    fn brute_ll_with(m0: f64, t1: f64, cycles: &[LlCycle], reps: usize, eps_override: Option<f64>) -> Vec<Vec<f64>> {
        let step = |mz: f64, dt: f64| m0 + (mz - m0) * (-dt / t1).exp();
        let mut mz = 0.0;
        let mut last = Vec::new();
        for r in 0..reps {
            let seq: &[LlCycle] = if r + 1 < reps { &cycles[..1] } else { cycles };
            last.clear();
            for c in seq {
                let mut events: Vec<(f64, f64, bool)> = Vec::new();
                if let Some(s) = c.s {
                    if s.presaturation {
                        mz = 0.0;
                    }
                    for &p in &s.pulse_times {
                        events.push((p, 1.0 - 2.0 * eps_override.unwrap_or(s.epsilon), false));
                    }
                }
                for (&t, &a) in c.t_read.iter().zip(&c.flip_deg) {
                    events.push((t, a.to_radians().cos(), true));
                }
                events.sort_by(|a, b| a.0.total_cmp(&b.0));
                let (mut t, mut before) = (0.0, Vec::new());
                for (te, f, readout) in events {
                    mz = step(mz, te - t);
                    t = te;
                    if readout {
                        before.push(mz);
                    }
                    mz *= f;
                }
                mz = step(mz, c.tr - t);
                last.push(before);
            }
        }
        last
    }

    fn brute_ll(m0: f64, t1: f64, cycles: &[LlCycle], reps: usize) -> Vec<Vec<f64>> {
        brute_ll_with(m0, t1, cycles, reps, None)
    }

    /// The reference detects a changed event: a different pulse efficiency moves the states.
    #[test]
    fn the_look_locker_reference_is_sensitive() {
        let s = Suppression::new(vec![0.5, 0.8], 0.95, false);
        let t: Vec<f64> = (0..6).map(|n| 1.0 + 0.3 * n as f64).collect();
        let c = cycle(5.0, Some(&s), &t, &[35.0; 6]);
        let got = tissue_mz_ll(74.6, 1.33, &c);
        let other = brute_ll_with(74.6, 1.33, std::slice::from_ref(&c), 400, Some(0.9)).remove(0);
        assert!(got.iter().zip(&other).any(|(g, w)| (g - w).abs() > 1e-3), "{got:?} vs {other:?}");
    }

    fn cycle<'a>(tr: f64, s: Option<&'a Suppression>, t: &[f64], fa: &[f64]) -> LlCycle<'a> {
        LlCycle { tr, s, t_read: t.to_vec(), flip_deg: fa.to_vec() }
    }

    #[test]
    fn the_look_locker_fixed_point_is_the_repeated_cycle() {
        let s = Suppression::new(vec![0.5, 0.8], 0.95, false);
        let t: Vec<f64> = (0..12).map(|n| 1.0 + 0.3 * n as f64).collect();
        for (sup, fa) in [(None, 35.0), (Some(&s), 35.0), (Some(&s), 12.0), (None, 89.0)] {
            let c = cycle(5.0, sup, &t, &[fa; 12]);
            let got = tissue_mz_ll(74.6, 1.33, &c);
            let want = brute_ll(74.6, 1.33, std::slice::from_ref(&c), 400).remove(0);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-15 * 74.6 * 10.0, "{sup:?} {fa}: {g} vs {w}");
            }
        }
        // varying flips within the cycle (QUASAR-like ramps)
        let fa: Vec<f64> = (0..12).map(|n| 20.0 + 5.0 * n as f64).collect();
        let c = cycle(5.0, Some(&s), &t, &fa);
        let got = tissue_mz_ll(74.6, 1.33, &c);
        let want = brute_ll(74.6, 1.33, std::slice::from_ref(&c), 400).remove(0);
        assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() <= 1e-12), "{got:?} vs {want:?}");
        // each readout leaves cos(a) of the state it read, and recovery runs from there
        let next = recover(got[0] * 20f64.to_radians().cos(), 74.6, 1.33, 0.3);
        assert!((got[1] - next).abs() <= 1e-12 && (next - recover(got[0], 74.6, 1.33, 0.3)).abs() > 1e-3);
        // presaturation starts from zero
        let ps = Suppression::new(vec![0.5], 0.95, true);
        let c = cycle(5.0, Some(&ps), &t, &[35.0; 12]);
        assert_eq!(tissue_mz_ll(74.6, 1.33, &c), ll_cycle(123.0, 74.6, 1.33, &c).0);
    }

    /// One readout per cycle and one flip: P5's functions themselves, bit for bit, through the
    /// series dispatch; with varying flips the generalized timeline, against brute force.
    #[test]
    fn one_readout_dispatches_to_p5() {
        let s = Suppression::new(vec![2.0, 2.7], 0.9, false);
        let fa = 35.0;
        let cycles = [cycle(4.0, Some(&s), &[3.6], &[fa]), cycle(4.0, Some(&s), &[2.8], &[fa]), cycle(6.0, None, &[6.0], &[fa])];
        let m0scan = [false, false, true];
        let got = tissue_mz_ll_series(74.6, 1.33, &cycles, &m0scan, Some(fa), false);
        let preps = [Prep { tr: 4.0, t_read: 3.6, s: Some(&s) }, Prep { tr: 4.0, t_read: 2.8, s: Some(&s) },
                     Prep { tr: 6.0, t_read: 6.0, s: None }];
        let want = tissue_mz_ge_sequence(74.6, 1.33, &preps, fa);
        assert_eq!(got.iter().map(|v| v[0].to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        let uni = tissue_mz_ll_series(74.6, 1.33, &cycles[..1], &[false], Some(fa), true);
        assert_eq!(uni[0][0].to_bits(), tissue_mz_ge(74.6, 1.33, 4.0, 3.6, &s, fa).to_bits());
        // M = 1 with varying flips: the generalized timeline
        let vary = [cycle(4.0, Some(&s), &[3.6], &[35.0]), cycle(4.0, Some(&s), &[3.6], &[50.0])];
        let got = tissue_mz_ll_series(74.6, 1.33, &vary, &[false, false], None, false);
        let want = brute_ll(74.6, 1.33, &vary, 400);
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!((g - w).abs() <= 1e-12, "{g} vs {w}");
        }
    }

    /// An m0scan cycle between two Look-Locker cycles: its own readout at the start of its
    /// repetition, the state carried across it, against brute-force propagation.
    #[test]
    fn an_m0scan_between_cycles_carries_the_state() {
        let s = Suppression::new(vec![0.8], 0.95, false);
        let t: Vec<f64> = (0..6).map(|n| 1.2 + 0.4 * n as f64).collect();
        let ll = cycle(4.5, Some(&s), &t, &[30.0; 6]);
        let m0c = cycle(4.5, None, &[0.02], &[15.0]);
        let cycles = [ll.clone(), m0c, ll];
        let got = tissue_mz_ll_series(74.6, 1.33, &cycles, &[false, true, false], None, false);
        let want = brute_ll(74.6, 1.33, &cycles, 400);
        for (g, w) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!((g - w).abs() <= 1e-12, "{g} vs {w}");
        }
        // the cycle after the m0scan starts from a different state than the first
        assert!((got[2][0] - got[0][0]).abs() > 1e-6);
        assert_eq!((got[0].len(), got[1].len(), got[2].len()), (6, 1, 6));
    }
}
