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
}
