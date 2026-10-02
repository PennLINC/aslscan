//! The Buxton general kinetic model, per voxel, as simasl's `GkmFilter` computes it
//! (`src/asldro/filters/gkm_filter.py`), so that the two can be diffed. Seconds throughout.
//!
//! Faithful means faithful to the guards too: the delivery-state masks (`gkm_filter.py:157-161`),
//! every `np.divide(.., out=zeros, where=..)` site, and the two *different* tests on a zero
//! arterial T1 (`> 0` in the PASL branch at `:205`, `!= 0` in the CASL/PCASL branch at `:252`).
//! The fixture diff at 1e-9 relative (`tests/fixtures/gkm.txt`) is the referee, and its exact
//! zeros come from masks, so the Rust reproduces them as exact zeros rather than as tiny values.

/// Which labeling scheme the series uses. Selects the kinetic branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelType {
    Pasl,
    Casl,
    Pcasl,
}

/// Per-row kinetic constants, in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kinetic {
    pub label_type: LabelType,
    /// Bolus duration `tau` (s): `LabelingDuration` for (P)CASL, `BolusCutOffDelayTime[0]` for
    /// PASL.
    pub tau: f64,
    /// Label efficiency `alpha`.
    pub alpha: f64,
    /// Blood-brain partition coefficient (ml/g).
    pub lambda: f64,
    /// Arterial blood T1 (s).
    pub t1b: f64,
}

/// simasl's `np.divide(n, d, out=zeros, where=d != 0)`: zero where the denominator is zero.
#[inline]
fn div0(n: f64, d: f64) -> f64 {
    if d != 0.0 {
        n / d
    } else {
        0.0
    }
}

/// `delta_m` for one voxel at signal time `t` (s), the difference in longitudinal magnetization
/// the label delivers.
///
/// `f_ml_100g_min` is the perfusion rate exactly as the phantom stores it; the `/ 6000` to
/// per-second happens here (`gkm_filter.py:105`). `dt` is the arterial transit time, `t1t` the
/// tissue T1, `m0` the tissue equilibrium magnetization. The result is what simasl's masked
/// array holds for this voxel: `0` where the bolus has not arrived (`0 < t <= dt`), and also `0`
/// for `t <= 0`, where simasl's chained comparison makes every mask false and the zero-initialised
/// output is never written.
pub fn delta_m(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64) -> f64 {
    let (f, m0b, t1p) = gkm_constants(k, f_ml_100g_min, t1t, m0);
    gkm_body(k, f, m0b, dt, t1p, t)
}

/// `f` (per second), `M0b` and `T1'`, with simasl's guards.
#[inline]
fn gkm_constants(k: &Kinetic, f_ml_100g_min: f64, t1t: f64, m0: f64) -> (f64, f64, f64) {
    let f = f_ml_100g_min / 6000.0;
    // M0b and f/lambda, with simasl's scalar guard on lambda (gkm_filter.py:135-147).
    let (m0b, flow_over_lambda) =
        if k.lambda != 0.0 { (m0 / k.lambda, f / k.lambda) } else { (0.0, 0.0) };
    // 1/T1' = 1/T1t + f/lambda, both divisions guarded (gkm_filter.py:149-153).
    let denom = div0(1.0, t1t) + flow_over_lambda;
    let t1p = div0(1.0, denom);
    (f, m0b, t1p)
}

/// [`delta_m`] with `T1'` given rather than derived from the tissue `T1` (P4 addendum, part A:
/// the intravascular part is the GKM with `T1'` replaced by `T1''`). The same delivery masks,
/// branches and guards; the arithmetic of [`delta_m`] is this function's.
pub fn delta_m_with_t1p(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1p: f64, m0: f64, t: f64) -> f64 {
    let f = f_ml_100g_min / 6000.0;
    let m0b = if k.lambda != 0.0 { m0 / k.lambda } else { 0.0 };
    gkm_body(k, f, m0b, dt, t1p, t)
}

#[inline]
fn gkm_body(k: &Kinetic, f: f64, m0b: f64, dt: f64, t1p: f64, t: f64) -> f64 {
    // Delivery state. simasl builds three masks and writes the arriving and arrived results
    // into a zero-initialised array; the not-arrived mask only ever writes zero.
    let arriving = dt < t && t < dt + k.tau;
    let arrived = t >= dt + k.tau;
    if !(arriving || arrived) {
        return 0.0;
    }

    match k.label_type {
        LabelType::Pasl => {
            // k = 1/T1b - 1/T1', with the scalar guard on T1b (gkm_filter.py:168-170).
            let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
            // The arterial decay factor is replaced by ZERO (not the quotient) for T1b <= 0
            // (gkm_filter.py:205, :218): `exp(-t/T1b) if t1b > 0 else 0`.
            let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
            if arriving {
                // q_pasl_arriving, numerator and denominator computed separately so that
                // t == dt cannot divide by zero (it is masked out anyway) (gkm_filter.py:173-185).
                let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * t).exp());
                let q = div0(num, kk * (t - dt));
                2.0 * m0b * f * (t - dt) * k.alpha * decay * q
            } else {
                let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * (dt + k.tau)).exp());
                let q = div0(num, kk * k.tau);
                2.0 * m0b * f * k.alpha * k.tau * decay * q
            }
        }
        LabelType::Casl | LabelType::Pcasl => {
            // Here the T1b test is `!= 0`, not `> 0` (gkm_filter.py:252, :265).
            let decay = if k.t1b != 0.0 { (-dt / k.t1b).exp() } else { 0.0 };
            if arriving {
                let q = 1.0 - (-div0(t - dt, t1p)).exp();
                2.0 * m0b * f * t1p * k.alpha * decay * q
            } else {
                let q = 1.0 - (-div0(k.tau, t1p)).exp();
                2.0 * m0b * f * t1p * k.alpha * decay * (-div0(t - k.tau - dt, t1p)).exp() * q
            }
        }
    }
}

/// The sub-bolus of parcels `a..b` (in `[0, tau]`; P4 addendum, "the one idea"): (P)CASL
/// parcels labeled in `[a, b]` are a GKM bolus of duration `b - a` started `a` later (the
/// transit decay is the same for every parcel); PASL parcels arriving in `[ATT + a, ATT + b]`
/// are a GKM bolus with `ATT' = ATT + a` and no shift. The uncut sub-bolus (`a = 0`,
/// `b = tau`) calls [`delta_m`] with the original arguments, so the P1 path is the P1 code.
#[allow(clippy::too_many_arguments)]
pub fn delta_m_sub(k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, a: f64, b: f64) -> f64 {
    if a == 0.0 && b == k.tau {
        return delta_m(k, f, dt, t1t, m0, t);
    }
    let ks = Kinetic { tau: b - a, ..*k };
    match k.label_type {
        LabelType::Pasl => delta_m(&ks, f, dt + a, t1t, m0, t),
        LabelType::Casl | LabelType::Pcasl => delta_m(&ks, f, dt, t1t, m0, t - a),
    }
}

/// `T1''` of the intravascular part: `1/T1'' = 1/T1' + 1/tau_ex`, with `T1'` guarded as
/// [`delta_m`] guards it. `None` where `T1'` is the guarded zero, which only happens where
/// `f/lambda` is zero and the GKM is therefore zero too.
fn t1pp(k: &Kinetic, f_ml_100g_min: f64, t1t: f64, m0: f64, tau_ex: f64) -> Option<(f64, f64, f64)> {
    let (f, m0b, t1p) = gkm_constants(k, f_ml_100g_min, t1t, m0);
    if t1p == 0.0 {
        return None;
    }
    Some((f, m0b, 1.0 / (1.0 / t1p + 1.0 / tau_ex)))
}

/// The intravascular part of [`delta_m`] (P4 addendum, part A): each parcel resident for `s`
/// has not yet exchanged with probability `exp(-s/tau_ex)`, which makes it the GKM with `T1'`
/// replaced by `T1''`. (P)CASL is [`delta_m_with_t1p`]'s arithmetic. PASL is not: its branch
/// forms `exp(kk t)` and `exp(-kk dt)` separately, and with `T1''` short `kk` is large and
/// negative, so those overflow to `0 * (inf - inf)`; here the exponents are combined, which is
/// the same quantity and finite for every `tau_ex`.
pub fn delta_m_iv(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64, tau_ex: f64) -> f64 {
    let Some((f, m0b, t1pp)) = t1pp(k, f_ml_100g_min, t1t, m0, tau_ex) else { return 0.0 };
    match k.label_type {
        LabelType::Casl | LabelType::Pcasl => gkm_body(k, f, m0b, dt, t1pp, t),
        LabelType::Pasl => {
            // The part never exceeds the whole delta_m computes. Where delta_m's own
            // kk = 1/T1b - 1/T1' is zero (its quotient guard) or within rounding of it (its
            // difference of exponentials cancels), delta_m is zero or noise; it stays as simasl
            // computes it, and T1'' moves kk off zero, so unclamped the part would be a positive
            // share of a zero whole and the extravascular rest negative.
            let iv = pasl_stable(k, f, m0b, dt, t1pp, t);
            iv.min(delta_m(k, f_ml_100g_min, dt, t1t, m0, t)).max(0.0)
        }
    }
}

/// [`delta_m_iv`] of the sub-bolus `a..b`, by the shifts of [`delta_m_sub`].
#[allow(clippy::too_many_arguments)]
pub fn delta_m_iv_sub(k: &Kinetic, f: f64, dt: f64, t1t: f64, m0: f64, t: f64, a: f64, b: f64, tau_ex: f64) -> f64 {
    if a == 0.0 && b == k.tau {
        return delta_m_iv(k, f, dt, t1t, m0, t, tau_ex);
    }
    let ks = Kinetic { tau: b - a, ..*k };
    match k.label_type {
        LabelType::Pasl => delta_m_iv(&ks, f, dt + a, t1t, m0, t, tau_ex),
        LabelType::Casl | LabelType::Pcasl => delta_m_iv(&ks, f, dt, t1t, m0, t - a, tau_ex),
    }
}

/// The PASL branch of [`gkm_body`] with `exp(kk t) (exp(-kk x) - exp(-kk y))` evaluated as
/// `exp(kk (t - y)) expm1(kk (y - x))`: the same masks and guards. Combining the exponents keeps
/// it finite at large `|kk|`; `expm1` keeps it accurate at small `|kk|`, where `tau_ex` makes
/// `T1''` nearly `T1b` and the difference of two exponentials would cancel to noise.
fn pasl_stable(k: &Kinetic, f: f64, m0b: f64, dt: f64, t1p: f64, t: f64) -> f64 {
    let arriving = dt < t && t < dt + k.tau;
    let arrived = t >= dt + k.tau;
    if !(arriving || arrived) {
        return 0.0;
    }
    let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
    let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
    if arriving {
        let num = (kk * (t - dt)).exp_m1();
        let q = div0(num, kk * (t - dt));
        2.0 * m0b * f * (t - dt) * k.alpha * decay * q
    } else {
        let num = (kk * (t - dt - k.tau)).exp() * (kk * k.tau).exp_m1();
        let q = div0(num, kk * k.tau);
        2.0 * m0b * f * k.alpha * k.tau * decay * q
    }
}

/// The arterial (macrovascular) difference magnetization (P4 addendum, part B), with the
/// parcel factor `g = 1`: `2 alpha M0b aBV exp(-aATT/T1b)` for (P)CASL and
/// `2 alpha M0b aBV exp(-t/T1b)` for PASL, inside `aATT <= t < aATT + tau`, zero outside, with
/// the GKM's `lambda` and `T1b` guards. Returns the value and the sub-bolus coordinate of the
/// parcel the voxel's arteries hold (`t - aATT` for both labeling types), `None` outside.
pub fn arterial_dm(k: &Kinetic, abv: f64, aatt: f64, m0: f64, t: f64) -> (f64, Option<f64>) {
    if !(aatt <= t && t < aatt + k.tau) {
        return (0.0, None);
    }
    let m0b = if k.lambda != 0.0 { m0 / k.lambda } else { 0.0 };
    let decay = match k.label_type {
        LabelType::Pasl => if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 },
        LabelType::Casl | LabelType::Pcasl => if k.t1b != 0.0 { (-aatt / k.t1b).exp() } else { 0.0 },
    };
    (2.0 * k.alpha * m0b * abv * decay, Some(t - aatt))
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_PCASL: Kinetic =
        Kinetic { label_type: LabelType::Pcasl, tau: 1.8, alpha: 0.85, lambda: 0.9, t1b: 1.65 };
    const K_PASL: Kinetic =
        Kinetic { label_type: LabelType::Pasl, tau: 0.7, alpha: 0.98, lambda: 0.9, t1b: 1.65 };
    // GM in the ASLDRO 3T phantom.
    const F: f64 = 60.0;
    const DT: f64 = 0.8;
    const T1T: f64 = 1.33;
    const M0: f64 = 74.622;

    fn t1_prime(f_ml: f64, t1t: f64, lambda: f64) -> f64 {
        1.0 / (1.0 / t1t + (f_ml / 6000.0) / lambda)
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs())
    }

    #[test]
    fn not_arrived_is_exactly_zero() {
        for t in [0.5, DT, 0.0, -1.0] {
            assert_eq!(delta_m(&K_PCASL, F, DT, T1T, M0, t), 0.0, "t = {t}");
            assert_eq!(delta_m(&K_PASL, F, DT, T1T, M0, t), 0.0, "t = {t}");
        }
    }

    #[test]
    fn pcasl_arrived_matches_closed_form() {
        // The simasl default: PLD 1.8 + tau 1.8.
        let t = 3.6;
        let t1p = t1_prime(F, T1T, 0.9);
        let m0b = M0 / 0.9;
        let f = F / 6000.0;
        let want = 2.0 * m0b * f * t1p * 0.85 * (-DT / 1.65).exp()
            * (-(t - 1.8 - DT) / t1p).exp()
            * (1.0 - (-1.8 / t1p).exp());
        let got = delta_m(&K_PCASL, F, DT, T1T, M0, t);
        assert!(close(got, want, 1e-12), "{got} vs {want}");
        assert!(got > 0.0);
    }

    #[test]
    fn pcasl_arriving_at_the_bolus_midpoint() {
        let t = DT + 0.9;
        let t1p = t1_prime(F, T1T, 0.9);
        let want = 2.0 * (M0 / 0.9) * (F / 6000.0) * t1p * 0.85 * (-DT / 1.65).exp()
            * (1.0 - (-(t - DT) / t1p).exp());
        assert!(close(delta_m(&K_PCASL, F, DT, T1T, M0, t), want, 1e-12));
    }

    #[test]
    fn pasl_arriving_and_arrived_match_the_k_form() {
        let t1p = t1_prime(F, T1T, 0.9);
        let kk = 1.0 / 1.65 - 1.0 / t1p;
        let m0b = M0 / 0.9;
        let f = F / 6000.0;
        // arriving: dt < 1.2 < dt + 0.7
        let t = 1.2;
        let q = (kk * t).exp() * ((-kk * DT).exp() - (-kk * t).exp()) / (kk * (t - DT));
        let want = 2.0 * m0b * f * (t - DT) * 0.98 * (-t / 1.65).exp() * q;
        assert!(close(delta_m(&K_PASL, F, DT, T1T, M0, t), want, 1e-12));
        // arrived: 2.0 >= dt + 0.7
        let t = 2.0;
        let q = (kk * t).exp() * ((-kk * DT).exp() - (-kk * (DT + 0.7)).exp()) / (kk * 0.7);
        let want = 2.0 * m0b * f * 0.98 * 0.7 * (-t / 1.65).exp() * q;
        assert!(close(delta_m(&K_PASL, F, DT, T1T, M0, t), want, 1e-12));
    }

    #[test]
    fn zero_perfusion_is_zero_everywhere() {
        for t in [0.5, 1.2, 2.0, 3.6, 10.0] {
            assert_eq!(delta_m(&K_PCASL, 0.0, DT, T1T, M0, t), 0.0);
            assert_eq!(delta_m(&K_PASL, 0.0, DT, T1T, M0, t), 0.0);
        }
    }

    #[test]
    fn zero_t1b_is_zero_in_both_branches() {
        // simasl replaces the whole arterial exponential with zero, not the quotient
        // (gkm_filter.py:205, :252), so this is 0, not the exp(0) = 1 a guarded division gives.
        let pc = Kinetic { t1b: 0.0, ..K_PCASL };
        let pa = Kinetic { t1b: 0.0, ..K_PASL };
        assert_eq!(delta_m(&pc, F, DT, T1T, M0, 3.6), 0.0);
        assert_eq!(delta_m(&pa, F, DT, T1T, M0, 2.0), 0.0);
        assert_eq!(delta_m(&pa, F, DT, T1T, M0, 1.2), 0.0);
    }

    #[test]
    fn pasl_k_exactly_zero_takes_the_quotient_guard_not_the_limit() {
        // Fidelity, not physics: with 1/T1b == 1/T1' exactly, k == 0, every q denominator is
        // zero, and simasl's np.divide(where=) yields q = 0, hence delta_m = 0 -- not the finite
        // 0/0 limit a textbook would take. Choose t1b so the subtraction is exactly zero.
        let t1p = t1_prime(F, T1T, 0.9);
        let k = Kinetic { t1b: 1.0 / (1.0 / t1p), ..K_PASL };
        let kk = 1.0 / k.t1b - 1.0 / t1p;
        assert_eq!(kk, 0.0, "the test premise: k must be exactly zero");
        assert_eq!(delta_m(&k, F, DT, T1T, M0, 1.2), 0.0);
        assert_eq!(delta_m(&k, F, DT, T1T, M0, 2.0), 0.0);
        // and the intravascular part follows the whole: zero, not a positive part of a zero
        // (final review: 0.00098 at exchange_time 0.5, every sub-bolus too)
        for tau_ex in [0.5, 3.0] {
            for t in [1.2, 2.0] {
                assert_eq!(delta_m_iv(&k, F, DT, T1T, M0, t, tau_ex), 0.0, "tau_ex {tau_ex} t {t}");
                assert_eq!(delta_m_iv_sub(&k, F, DT, T1T, M0, t, 0.1, 0.4, tau_ex), 0.0);
            }
        }
    }

    #[test]
    fn pasl_intravascular_part_never_exceeds_a_cancelled_whole() {
        // Near kk = 0 but not at it, delta_m's difference of exponentials cancels to zero or
        // noise (re-review: T1b = 2.9032258064516134 with T1t = 3 gives kk ~ -5.6e-17,
        // delta_m = 0 and an unclamped part of 0.0013). The part stays within [0, delta_m]
        // for the whole and every sub-bolus, a few ULPs and a little further either side.
        let (t1t, m0, t1b0) = (3.0, 1.0, 2.9032258064516134);
        let k0 = Kinetic { label_type: LabelType::Pasl, tau: 0.7, alpha: 0.85, lambda: 0.9, t1b: t1b0 };
        assert_eq!(delta_m(&k0, F, DT, t1t, m0, 2.0), 0.0, "the test premise: the whole cancels to zero");
        assert_eq!(delta_m_iv(&k0, F, DT, t1t, m0, 2.0, 0.5), 0.0);
        let mut steps: Vec<f64> = (-20..=20).map(|j| j as f64).collect();
        steps.extend([-1e6, -1e3, 1e3, 1e6]);
        for s in steps {
            let k = Kinetic { t1b: f64::from_bits((t1b0.to_bits() as i64 + s as i64) as u64), ..k0 };
            for tau_ex in [0.05, 0.5, 3.0] {
                for t in [1.0, 1.4, 2.0, 3.0] {
                    let dm = delta_m(&k, F, DT, t1t, m0, t);
                    let iv = delta_m_iv(&k, F, DT, t1t, m0, t, tau_ex);
                    assert!(iv >= 0.0 && iv <= dm.max(0.0), "t1b {} tau_ex {tau_ex} t {t}: iv {iv} dm {dm}", k.t1b);
                    for (a, b) in [(0.0, 0.3), (0.3, 0.7), (0.1, 0.4)] {
                        let dm = delta_m_sub(&k, F, DT, t1t, m0, t, a, b);
                        let iv = delta_m_iv_sub(&k, F, DT, t1t, m0, t, a, b, tau_ex);
                        assert!(iv >= 0.0 && iv <= dm.max(0.0), "sub {a}..{b} t {t}: iv {iv} dm {dm}");
                    }
                }
            }
        }
    }

    #[test]
    fn zero_tissue_t1_with_zero_flow_gives_zero_not_nan() {
        // t1p becomes 0 (guarded 1/0), every /t1p is guarded to 0, exp(-0) = 1, q = 0.
        let got = delta_m(&K_PCASL, 0.0, DT, 0.0, M0, 3.6);
        assert_eq!(got, 0.0);
        assert!(!delta_m(&K_PCASL, F, DT, 0.0, M0, 3.6).is_nan());
    }

    /// Fixture cases from simasl's GkmFilter (tools/gen_gkm_fixtures.py). Plain text, pure std.
    struct Case {
        k: Kinetic,
        t: f64,
        f: Vec<f64>,
        dt: Vec<f64>,
        t1t: Vec<f64>,
        m0: Vec<f64>,
        want: Vec<f64>,
    }

    fn parse_fixtures(text: &str) -> Vec<Case> {
        let mut cases = Vec::new();
        let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let floats = |l: &str| -> Vec<f64> {
            l.split_whitespace().skip(1).map(|t| t.parse::<f64>().unwrap()).collect()
        };
        while let Some(head) = lines.next() {
            let toks: Vec<&str> = head.split_whitespace().collect();
            assert_eq!(toks[0], "case", "unexpected line: {head}");
            let scalars = lines.next().unwrap();
            let s: Vec<&str> = scalars.split_whitespace().collect();
            assert_eq!(s[0], "scalars");
            let label_type = match s[1] {
                "pasl" => LabelType::Pasl,
                "casl" => LabelType::Casl,
                "pcasl" => LabelType::Pcasl,
                other => panic!("unknown label type {other}"),
            };
            let num = |i: usize| s[i].parse::<f64>().unwrap();
            let k = Kinetic { label_type, tau: num(2), alpha: num(4), lambda: num(5), t1b: num(6) };
            let t = num(3);
            let mut take = |name: &str| -> Vec<f64> {
                let l = lines.next().unwrap();
                assert!(l.starts_with(name), "expected {name}, got {l}");
                floats(l)
            };
            let f = take("perfusion_rate");
            let dt = take("transit_time");
            let t1t = take("t1_tissue");
            let m0 = take("m0");
            let want = take("delta_m");
            cases.push(Case { k, t, f, dt, t1t, m0, want });
        }
        cases
    }

    #[test]
    fn matches_simasl_gkm_fixtures_at_1e9() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gkm.txt"))
            .expect("tests/fixtures/gkm.txt is committed; regenerate with tools/gen_gkm_fixtures.py");
        let cases = parse_fixtures(&text);
        assert!(cases.len() >= 8, "too few cases: {}", cases.len());
        for (ci, c) in cases.iter().enumerate() {
            for i in 0..c.want.len() {
                let got = delta_m(&c.k, c.f[i], c.dt[i], c.t1t[i], c.m0[i], c.t);
                let want = c.want[i];
                if want == 0.0 {
                    assert_eq!(got, 0.0, "case {ci} voxel {i}: expected an exact zero, got {got:e}");
                } else {
                    let tol = 1e-9 * want.abs();
                    assert!((got - want).abs() <= tol,
                            "case {ci} voxel {i}: {got:.17e} vs {want:.17e} (rel {:.3e})",
                            (got - want).abs() / want.abs());
                }
            }
        }
    }

    // ------------------------------------------------------------------ P4

    /// `delta_m` exactly as it was before the P4 refactor (`p2-complete`): the bit-identity reference.
    fn delta_m_ref(k: &Kinetic, f_ml_100g_min: f64, dt: f64, t1t: f64, m0: f64, t: f64) -> f64 {
        let f = f_ml_100g_min / 6000.0;
        // M0b and f/lambda, with simasl's scalar guard on lambda (gkm_filter.py:135-147).
        let (m0b, flow_over_lambda) =
            if k.lambda != 0.0 { (m0 / k.lambda, f / k.lambda) } else { (0.0, 0.0) };
        // 1/T1' = 1/T1t + f/lambda, both divisions guarded (gkm_filter.py:149-153).
        let denom = div0(1.0, t1t) + flow_over_lambda;
        let t1p = div0(1.0, denom);

        // Delivery state. simasl builds three masks and writes the arriving and arrived results
        // into a zero-initialised array; the not-arrived mask only ever writes zero.
        let arriving = dt < t && t < dt + k.tau;
        let arrived = t >= dt + k.tau;
        if !(arriving || arrived) {
            return 0.0;
        }

        match k.label_type {
            LabelType::Pasl => {
                // k = 1/T1b - 1/T1', with the scalar guard on T1b (gkm_filter.py:168-170).
                let kk = (if k.t1b != 0.0 { 1.0 / k.t1b } else { 0.0 }) - div0(1.0, t1p);
                // The arterial decay factor is replaced by ZERO (not the quotient) for T1b <= 0
                // (gkm_filter.py:205, :218): `exp(-t/T1b) if t1b > 0 else 0`.
                let decay = if k.t1b > 0.0 { (-t / k.t1b).exp() } else { 0.0 };
                if arriving {
                    // q_pasl_arriving, numerator and denominator computed separately so that
                    // t == dt cannot divide by zero (it is masked out anyway) (gkm_filter.py:173-185).
                    let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * t).exp());
                    let q = div0(num, kk * (t - dt));
                    2.0 * m0b * f * (t - dt) * k.alpha * decay * q
                } else {
                    let num = (kk * t).exp() * ((-kk * dt).exp() - (-kk * (dt + k.tau)).exp());
                    let q = div0(num, kk * k.tau);
                    2.0 * m0b * f * k.alpha * k.tau * decay * q
                }
            }
            LabelType::Casl | LabelType::Pcasl => {
                // Here the T1b test is `!= 0`, not `> 0` (gkm_filter.py:252, :265).
                let decay = if k.t1b != 0.0 { (-dt / k.t1b).exp() } else { 0.0 };
                if arriving {
                    let q = 1.0 - (-div0(t - dt, t1p)).exp();
                    2.0 * m0b * f * t1p * k.alpha * decay * q
                } else {
                    let q = 1.0 - (-div0(k.tau, t1p)).exp();
                    2.0 * m0b * f * t1p * k.alpha * decay * (-div0(t - k.tau - dt, t1p)).exp() * q
                }
            }
        }
    }

    const KS: [(f64, f64, f64, f64); 4] = [(60.0, 0.8, 1.33, 74.622), (20.0, 1.2, 0.83, 60.0), (0.0, 1000.0, 3.0, 90.0), (45.0, 0.5, 1.1, 1.0)];

    fn kinds() -> Vec<Kinetic> {
        vec![
            K_PCASL, K_PASL,
            Kinetic { label_type: LabelType::Casl, ..K_PCASL },
            Kinetic { lambda: 0.0, ..K_PCASL }, Kinetic { lambda: 0.0, ..K_PASL },
            Kinetic { t1b: 0.0, ..K_PCASL }, Kinetic { t1b: 0.0, ..K_PASL },
            Kinetic { t1b: -1.0, ..K_PCASL }, Kinetic { t1b: -1.0, ..K_PASL },
        ]
    }

    #[test]
    fn delta_m_keeps_its_bits_after_the_refactor() {
        let mut n = 0;
        for k in kinds() {
            for &(f, dt, t1t, m0) in &KS {
                for i in 0..=6000 {
                    let t = i as f64 * 1e-3;
                    let (a, b) = (delta_m(&k, f, dt, t1t, m0, t), delta_m_ref(&k, f, dt, t1t, m0, t));
                    assert_eq!(a.to_bits(), b.to_bits(), "{k:?} f {f} dt {dt} t {t}: {a:e} vs {b:e}");
                    n += 1;
                }
            }
        }
        // the zero-T1 tissue case and t on every mask edge
        for k in kinds() {
            for t in [0.8, 0.8 + k.tau, 0.0, -1.0] {
                assert_eq!(delta_m(&k, 60.0, 0.8, 0.0, 74.6, t).to_bits(), delta_m_ref(&k, 60.0, 0.8, 0.0, 74.6, t).to_bits());
            }
        }
        assert!(n > 200_000);
    }

    /// A deterministic pseudo-random partition of `[0, tau]` with 2..6 cuts, plus `extra` cuts.
    fn partition(seed: u64, tau: f64, extra: &[f64]) -> Vec<f64> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let ncut = 2 + (next() * 5.0) as usize;
        let mut cuts: Vec<f64> = (0..ncut).map(|_| next() * tau).collect();
        cuts.extend(extra.iter().copied().filter(|c| *c > 0.0 && *c < tau));
        cuts.push(0.0);
        cuts.push(tau);
        cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cuts.dedup();
        cuts
    }

    /// `x` and its representable neighbours.
    fn around(x: f64) -> [f64; 3] {
        // wrapping: below 0.0 the neighbour is a NaN pattern, which the partition filters out
        [f64::from_bits(x.to_bits().wrapping_sub(1)), x, f64::from_bits(x.to_bits().wrapping_add(1))]
    }

    #[test]
    fn sub_boluses_sum_to_the_whole_bolus() {
        let mut worst = 0.0f64;
        for k in [K_PCASL, K_PASL, Kinetic { label_type: LabelType::Casl, ..K_PCASL }] {
            for &(f, dt, t1t, m0) in &KS[..2] {
                let peak = (0..=6000).map(|i| delta_m(&k, f, dt, t1t, m0, i as f64 * 1e-3).abs()).fold(0.0f64, f64::max);
                let mut times: Vec<f64> = (0..=600).map(|i| i as f64 * 1e-2).collect();
                for e in [dt, dt + k.tau] {
                    times.extend(around(e));
                }
                for (pi, &t) in times.iter().enumerate() {
                    // cuts at the representable neighbours of the delivery edges in the
                    // sub-bolus coordinate, as well as random ones
                    let edge = t - dt;
                    let extra: Vec<f64> = around(edge).iter().chain(around(edge - k.tau).iter()).copied().collect();
                    let cuts = partition(pi as u64 + 17, k.tau, &extra);
                    let sum: f64 = cuts.windows(2).map(|w| delta_m_sub(&k, f, dt, t1t, m0, t, w[0], w[1])).sum();
                    let whole = delta_m(&k, f, dt, t1t, m0, t);
                    if t <= dt {
                        assert_eq!(whole, 0.0);
                        assert_eq!(sum, 0.0, "{k:?} t {t}: not arrived must stay an exact zero");
                    }
                    let e = (sum - whole).abs() / peak;
                    worst = worst.max(e);
                    assert!(e <= 1e-12, "{k:?} f {f} t {t}: {sum:e} vs {whole:e} ({e:e} of peak)");
                }
            }
        }
        println!("sub-bolus partition identity: worst {worst:.2e} of peak");
        // the uncut sub-bolus is delta_m itself, bit for bit
        for t in [1.0, 2.0, 3.6, 5.0] {
            assert_eq!(delta_m_sub(&K_PCASL, 60.0, 0.8, 1.33, 74.6, t, 0.0, K_PCASL.tau).to_bits(),
                       delta_m(&K_PCASL, 60.0, 0.8, 1.33, 74.6, t).to_bits());
            assert_eq!(delta_m_sub(&K_PASL, 60.0, 0.8, 1.33, 74.6, t, 0.0, K_PASL.tau).to_bits(),
                       delta_m(&K_PASL, 60.0, 0.8, 1.33, 74.6, t).to_bits());
        }
    }

    #[test]
    fn intravascular_part_is_partition_invariant_and_bounded() {
        for k in [K_PCASL, K_PASL] {
            for tau_ex in [0.05, 0.5, 3.0] {
                let peak = (0..=600).map(|i| delta_m(&k, F, DT, T1T, M0, i as f64 * 1e-2)).fold(0.0f64, f64::max);
                for i in 0..=600 {
                    let t = i as f64 * 1e-2;
                    let dm = delta_m(&k, F, DT, T1T, M0, t);
                    let iv = delta_m_iv(&k, F, DT, T1T, M0, t, tau_ex);
                    assert!(iv.is_finite() && iv >= 0.0 && iv <= dm * (1.0 + 1e-12),
                            "{k:?} tau_ex {tau_ex} t {t}: iv {iv} dm {dm}");
                    let cuts = partition(i as u64 + 5, k.tau, &[]);
                    let sum: f64 = cuts.windows(2).map(|w| delta_m_iv_sub(&k, F, DT, T1T, M0, t, w[0], w[1], tau_ex)).sum();
                    assert!((sum - iv).abs() <= 1e-12 * peak, "{k:?} tau_ex {tau_ex} t {t}: {sum:e} vs {iv:e}");
                    if t <= DT {
                        assert_eq!(iv, 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn intravascular_part_limits_are_finite_in_both_branches() {
        for k in [K_PCASL, K_PASL] {
            // slow exchange: everything stays intravascular
            for t in [1.0, 1.4, 2.0, 3.6] {
                let dm = delta_m(&k, F, DT, T1T, M0, t);
                let iv = delta_m_iv(&k, F, DT, T1T, M0, t, 1e9);
                assert!((iv - dm).abs() <= 1e-8 * dm, "{k:?} t {t}: {iv} vs {dm}");
            }
            // fast exchange: just-arrived label has not exchanged yet; a second later it has
            let t = DT + 1e-7;
            let share = delta_m_iv(&k, F, DT, T1T, M0, t, 1e-6) / delta_m(&k, F, DT, T1T, M0, t);
            assert!((share - 0.95).abs() < 0.01, "{k:?}: share {share} at ATT + 1e-7");
            // both delivery branches at tau_ex = 1e-6 are finite (the shared PASL form gives NaN)
            for t in [DT + 0.3, DT + k.tau + 0.5, DT + 1.0] {
                let iv = delta_m_iv(&k, F, DT, T1T, M0, t, 1e-6);
                let dm = delta_m(&k, F, DT, T1T, M0, t);
                assert!(iv.is_finite(), "{k:?} t {t}: {iv}");
                if t >= DT + 1.0 {
                    assert!(iv / dm < 2e-6, "{k:?} t {t}: share {}", iv / dm);
                }
            }
        }
        // the PASL shared form really is the trap the stable form avoids
        let t1p = t1_prime(F, T1T, 0.9);
        let t1pp = 1.0 / (1.0 / t1p + 1e6);
        assert!(delta_m_with_t1p(&K_PASL, F, DT, t1pp, M0, DT + 1e-7).is_nan());
    }

    #[test]
    fn stable_pasl_form_equals_the_shared_form_where_that_is_finite() {
        let t1p = t1_prime(F, T1T, 0.9);
        for tau_ex in [0.2, 1.0, 10.0] {
            let t1pp = 1.0 / (1.0 / t1p + 1.0 / tau_ex);
            for i in 0..=400 {
                let t = i as f64 * 1e-2;
                let a = delta_m_iv(&K_PASL, F, DT, T1T, M0, t, tau_ex);
                let b = delta_m_with_t1p(&K_PASL, F, DT, t1pp, M0, t);
                assert!(a == b || (a - b).abs() <= 1e-12 * b.abs(), "t {t}: {a:e} vs {b:e}");
            }
        }
        // and delta_m_with_t1p at the GKM's own T1' is delta_m
        for t in [1.2, 2.0, 3.6] {
            assert_eq!(delta_m_with_t1p(&K_PCASL, F, DT, t1p, M0, t).to_bits(), delta_m(&K_PCASL, F, DT, T1T, M0, t).to_bits());
        }
    }

    #[test]
    fn pasl_intravascular_part_is_accurate_where_t1pp_nears_t1b() {
        // T1t = 3 puts the tau_ex that makes T1'' = T1b (kk = 0) in range: 1/tau_ex = 1/T1b - 1/T1'.
        // There the arrived branch's difference of two exponentials cancelled to noise, and the
        // computed part exceeded delta_m (Codex review: iv 0.01296 against delta_m 0.00567).
        let (t1t, m0, t) = (3.0, 1.0, 2.0);
        let t1p = t1_prime(F, t1t, 0.9);
        let tau_star = 1.0 / (1.0 / K_PASL.t1b - 1.0 / t1p);
        let dm = delta_m(&K_PASL, F, DT, t1t, m0, t);
        let iv = delta_m_iv(&K_PASL, F, DT, t1t, m0, t, 3.8223938223938227);
        assert!((iv - 0.004536224).abs() < 1e-8, "{iv} (delta_m {dm})");
        // the limit kk -> 0: q -> 1, so iv -> 2 M0b f alpha tau exp(-t/T1b)
        let limit = 2.0 * (m0 / 0.9) * (F / 6000.0) * K_PASL.alpha * K_PASL.tau * (-t / K_PASL.t1b).exp();
        let near = delta_m_iv(&K_PASL, F, DT, t1t, m0, t, tau_star * (1.0 + 1e-13));
        assert!((near - limit).abs() <= 1e-9 * limit, "{near} vs {limit}");
        // across the crossing, in both branches: bounded, and nondecreasing in tau_ex (slower
        // exchange keeps more label intravascular), which cancellation noise would break. A kk
        // of exactly zero takes the GKM's quotient guard (zero, which the spec keeps for every
        // P4 formula); it is skipped, and can happen at most once in the sweep.
        for t in [DT + 0.3, t, 3.0] {
            let dm = delta_m(&K_PASL, F, DT, t1t, m0, t);
            let (mut prev, mut zeros) = (0.0f64, 0);
            for j in -200..=200 {
                let tau_ex = tau_star * (1.0 + j as f64 * 1e-12);
                let iv = delta_m_iv(&K_PASL, F, DT, t1t, m0, t, tau_ex);
                assert!(iv >= 0.0 && iv <= dm * (1.0 + 1e-12), "t {t} tau_ex {tau_ex}: iv {iv} dm {dm}");
                if iv == 0.0 {
                    zeros += 1;
                    continue;
                }
                assert!(iv >= prev * (1.0 - 1e-12), "t {t} tau_ex {tau_ex}: {iv} after {prev}");
                prev = iv;
            }
            assert!(zeros <= 1, "t {t}: {zeros} guarded zeros");
        }
    }

    #[test]
    fn intravascular_guards_give_the_gkm_zeros() {
        for k in [Kinetic { lambda: 0.0, ..K_PCASL }, Kinetic { lambda: 0.0, ..K_PASL }, Kinetic { t1b: 0.0, ..K_PCASL },
                  Kinetic { t1b: 0.0, ..K_PASL }, Kinetic { t1b: -1.0, ..K_PASL }] {
            for t in [1.2, 2.0, 3.6] {
                assert_eq!(delta_m(&k, F, DT, T1T, M0, t), 0.0);
                assert_eq!(delta_m_iv(&k, F, DT, T1T, M0, t, 0.5), 0.0, "{k:?} t {t}");
            }
        }
        // zero flow and zero tissue T1: T1' is the guarded zero
        assert_eq!(delta_m_iv(&K_PCASL, 0.0, DT, 0.0, M0, 3.6, 0.5), 0.0);
    }

    #[test]
    fn arterial_term_follows_its_closed_forms() {
        let (abv, aatt) = (0.02, 0.5);
        for t in [0.5, 1.0, 2.29] {
            let (v, a) = arterial_dm(&K_PCASL, abv, aatt, M0, t);
            let want = 2.0 * 0.85 * (M0 / 0.9) * abv * (-aatt / 1.65f64).exp();
            assert!(close(v, want, 1e-14), "t {t}");
            assert_eq!(a, Some(t - aatt));
        }
        for t in [0.5, 0.9, 1.19] {
            let (v, a) = arterial_dm(&K_PASL, abv, aatt, M0, t);
            let want = 2.0 * 0.98 * (M0 / 0.9) * abv * (-t / 1.65f64).exp();
            assert!(close(v, want, 1e-14), "t {t}");
            assert_eq!(a, Some(t - aatt));
        }
        // the window: [aatt, aatt + tau)
        assert_eq!(arterial_dm(&K_PCASL, abv, aatt, M0, aatt - 1e-12), (0.0, None));
        assert_eq!(arterial_dm(&K_PCASL, abv, aatt, M0, aatt + 1.8), (0.0, None));
        assert!(arterial_dm(&K_PCASL, abv, aatt, M0, aatt).0 > 0.0);
        // guards
        assert_eq!(arterial_dm(&Kinetic { lambda: 0.0, ..K_PCASL }, abv, aatt, M0, 1.0).0, 0.0);
        assert_eq!(arterial_dm(&Kinetic { t1b: 0.0, ..K_PCASL }, abv, aatt, M0, 1.0).0, 0.0);
        assert_eq!(arterial_dm(&Kinetic { t1b: -1.0, ..K_PASL }, abv, aatt, M0, 1.0).0, 0.0);
        assert!(arterial_dm(&Kinetic { t1b: -1.0, ..K_PCASL }, abv, aatt, M0, 1.0).0 > 0.0,
                "(P)CASL tests != 0, as the GKM does");
    }
}
