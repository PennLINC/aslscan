//! Post-excitation magnetization for the spin-echo, inversion-recovery and gradient-echo
//! contrasts, per voxel. Seconds throughout.
//!
//! Gradient echo (P5 addendum, part A) has two tissue forms: the model's spoiled steady state,
//! and simasl's form, which keeps transverse coherence through `exp(-TR/T2)` and is used under
//! compat only (the two differ by percent at long T2). Its transverse factor is `exp(-TE/T2*)`,
//! which the acquisition stage applies (`EchoFormation::Gradient`), so it too is divided out of
//! the fixtures (`tests/fixtures/mrsignal_ge.txt`).
//!
//! This departs from simasl's `MriSignalFilter` (`src/asldro/filters/mri_signal_filter.py`) in
//! exactly one way, on purpose: **no transverse relaxation**. simasl folds `exp(-TE/T2)` into
//! its signal equation (`mri_signal_filter.py:246`) because it has no readout model and the TE
//! decay has nowhere else to live. Here the acquisition stage applies `exp(-trf/T2 - |t|/T2')`
//! per phase-encode line (`mrsim_acq::kspace`), a term that already spans excitation to every
//! sampled line, so applying it here too would relax twice. The fixtures in
//! `tests/fixtures/mrsignal.txt` and `mrsignal_ir.txt` are simasl's output with that factor
//! divided out.
//!
//! The blood compartment carries `delta_m` from the kinetic model and gets **no** steady-state
//! recovery term: `delta_m` is already a magnetization difference, and running it through
//! `M0 (1 - exp(-TR/T1))` would attribute to it a saturation history it does not have. It gets
//! the flip-angle factor only, which for a spin echo's 90-degree excitation is exactly 1. Under
//! inversion recovery simasl applies `sin(fa)` and nothing else to `mag_enc`
//! (`mri_signal_filter.py:271-281`): the inversion pulse does not invert the label in the
//! oracle's model, and faithfulness wins (P3 addendum, part B).

/// The signal equation in use. The IR parameters live on the protocol, not here, so this stays
/// `Copy + Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contrast {
    SpinEcho,
    InversionRecovery,
    /// P5 addendum, part A: a gradient-echo EPI readout (the acquisition stage forms the echo
    /// with `EchoFormation::Gradient`).
    GradientEcho,
}

impl Contrast {
    pub fn as_str(&self) -> &'static str {
        match self {
            Contrast::SpinEcho => "se",
            Contrast::InversionRecovery => "ir",
            Contrast::GradientEcho => "ge",
        }
    }
}

/// Parse the overlay's `acq_contrast` (case-insensitive, as simasl lower-cases it).
pub fn parse_contrast(s: &str) -> Result<Contrast, String> {
    match s.to_ascii_lowercase().as_str() {
        "se" => Ok(Contrast::SpinEcho),
        "ir" => Ok(Contrast::InversionRecovery),
        "ge" => Ok(Contrast::GradientEcho),
        other => Err(format!("acq_contrast {other:?}: expected \"se\", \"ir\" or \"ge\"")),
    }
}

/// Transverse magnetization immediately after the 90-degree excitation of a spin-echo sequence
/// with repetition time `tr` (s): `m0 * (1 - exp(-tr / t1))`. A zero T1 takes simasl's guard
/// (`np.divide(where=t1 != 0)` makes the exponent 0, `mri_signal_filter.py:170-173`), so the
/// signal is `m0 * (1 - 1) = 0`, not `m0`.
pub fn tissue_se(m0: f64, t1: f64, tr: f64) -> f64 {
    let exponent = if t1 != 0.0 { tr / t1 } else { 0.0 };
    m0 * (1.0 - (-exponent).exp())
}

/// The blood compartment's post-excitation magnetization: `delta_m` times the flip-angle factor,
/// which is 1 for the 90-degree spin-echo excitation.
pub fn blood_se(delta_m: f64) -> f64 {
    delta_m
}

/// The inversion-recovery preparation and excitation (simasl's `inversion_time`,
/// `excitation_flip_angle`, `inversion_flip_angle`). Seconds and degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrParams {
    pub inversion_time: f64,
    pub excitation_flip_deg: f64,
    pub inversion_flip_deg: f64,
}

/// simasl's inversion-recovery steady state (`mri_signal_filter.py:252-283`, from Tofts 2009
/// eq. 7), transverse factor left to the acquisition stage:
///
/// ```text
/// sin(fa) * m0 * (1 - (1 - cos(fa_inv)) exp(-TI/T1) - cos(fa_inv) exp(-TR/T1))
///                / (1 - cos(fa) cos(fa_inv) exp(-TR/T1))
/// ```
///
/// Both exponentials take simasl's zero-T1 guard (`exp(0) = 1`), and a zero denominator gives
/// 0 (`np.divide(where=denominator != 0)`). A periodic steady state: the same
/// inversion-excitation cycle every TR.
pub fn tissue_ir(m0: f64, t1: f64, tr: f64, p: &IrParams) -> f64 {
    let fa = p.excitation_flip_deg.to_radians();
    let fa_inv = p.inversion_flip_deg.to_radians();
    let e_tr = (-(if t1 != 0.0 { tr / t1 } else { 0.0 })).exp();
    let e_ti = (-(if t1 != 0.0 { p.inversion_time / t1 } else { 0.0 })).exp();
    let numerator = m0 * (1.0 - (1.0 - fa_inv.cos()) * e_ti - fa_inv.cos() * e_tr);
    let denominator = 1.0 - fa.cos() * fa_inv.cos() * e_tr;
    let quotient = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    fa.sin() * quotient
}

/// The blood compartment under inversion recovery: `sin(fa) * delta_m`, as simasl applies to
/// `mag_enc`. The label is not inverted by the preparation in this model (see the module doc).
pub fn blood_ir(delta_m: f64, p: &IrParams) -> f64 {
    p.excitation_flip_deg.to_radians().sin() * delta_m
}

/// `exp(-tr/t)` with simasl's zero-time guard (`np.divide(where=t != 0)`: a zero time gives a zero
/// exponent, `exp(0) = 1`).
fn guarded_decay(tr: f64, t: f64) -> f64 {
    (-(if t != 0.0 { tr / t } else { 0.0 })).exp()
}

/// The gradient-echo tissue signal of the model (P5 addendum, part A): the spoiled steady state
/// `sin(fa) m0 (1 - E1) / (1 - cos(fa) E1)`, `E1 = exp(-TR/T1)`, transverse factor left to the
/// acquisition stage, with simasl's zero-T1 guard and its zero-denominator guard. EPI spoils
/// between repetitions, so no transverse coherence is carried; [`tissue_ge_simasl`] is the
/// oracle's form, used under compat only.
pub fn tissue_ge_spoiled(m0: f64, t1: f64, tr: f64, flip_deg: f64) -> f64 {
    let fa = flip_deg.to_radians();
    let e1 = guarded_decay(tr, t1);
    let numerator = m0 * (1.0 - e1);
    let denominator = 1.0 - fa.cos() * e1;
    let quotient = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    fa.sin() * quotient
}

/// simasl's gradient-echo signal (`mri_signal_filter.py:188-225`), transverse factor
/// `exp(-TE/T2*)` left to the acquisition stage:
/// `sin(fa) m0 (1 - E1) / (1 - cos(fa) E1 - E2 (E1 - cos(fa)))`, `E2 = exp(-TR/T2)`, which keeps
/// transverse coherence across repetitions. Both exponentials take simasl's zero-time guard (so a
/// zero T2 gives `E2 = 1`), and a zero denominator gives 0. Compat only (`[compat] asldro`).
pub fn tissue_ge_simasl(m0: f64, t1: f64, t2: f64, tr: f64, flip_deg: f64) -> f64 {
    let fa = flip_deg.to_radians();
    let (e1, e2) = (guarded_decay(tr, t1), guarded_decay(tr, t2));
    let numerator = m0 * (1.0 - e1);
    let denominator = 1.0 - fa.cos() * e1 - e2 * (e1 - fa.cos());
    let quotient = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    fa.sin() * quotient
}

/// The blood compartment under gradient echo: `sin(fa) * delta_m`, as simasl applies to `mag_enc`.
pub fn blood_ge(delta_m: f64, flip_deg: f64) -> f64 {
    flip_deg.to_radians().sin() * delta_m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn se_closed_form() {
        let want = 74.622 * (1.0 - (-4.0f64 / 1.33).exp());
        let got = tissue_se(74.622, 1.33, 4.0);
        assert!((got - want).abs() <= 1e-15 * want, "{got} vs {want}");
    }

    #[test]
    fn zero_t1_gives_zero_signal_not_m0() {
        assert_eq!(tissue_se(74.622, 0.0, 4.0), 0.0);
    }

    #[test]
    fn every_contrast_parses() {
        assert_eq!(parse_contrast("ge").unwrap(), Contrast::GradientEcho);
        assert_eq!(parse_contrast("GE").unwrap().as_str(), "ge");
        assert!(parse_contrast("bogus").unwrap_err().contains("\"ge\""));
        assert_eq!(parse_contrast("se").unwrap(), Contrast::SpinEcho);
        assert_eq!(parse_contrast("SE").unwrap(), Contrast::SpinEcho);
        assert_eq!(parse_contrast("ir").unwrap(), Contrast::InversionRecovery);
        assert_eq!(parse_contrast("IR").unwrap().as_str(), "ir");
    }

    #[test]
    fn blood_is_passthrough_for_spin_echo_and_sin_fa_for_ir() {
        assert_eq!(blood_se(-0.37), -0.37);
        assert_eq!(blood_se(0.0), 0.0);
        let p = IrParams { inversion_time: 1.0, excitation_flip_deg: 30.0, inversion_flip_deg: 180.0 };
        assert!((blood_ir(2.0, &p) - 1.0).abs() < 1e-12);
        let p90 = IrParams { excitation_flip_deg: 90.0, ..p };
        assert_eq!(blood_ir(-0.37, &p90), -0.37);
    }

    /// No inversion (`fa_inv = 0`): `sin(fa) m0 (1 - E) / (1 - cos(fa) E)`, which is the P1
    /// saturation recovery only at `fa = 90`.
    #[test]
    fn ir_without_inversion_keeps_its_denominator() {
        let (m0, t1, tr) = (74.622f64, 1.33f64, 1.33f64);
        let e = (-tr / t1).exp();
        let p = IrParams { inversion_time: 0.5, excitation_flip_deg: 60.0, inversion_flip_deg: 0.0 };
        let got = tissue_ir(m0, t1, tr, &p);
        let fa = 60f64.to_radians();
        let want = fa.sin() * m0 * (1.0 - e) / (1.0 - fa.cos() * e);
        assert!((got - want).abs() <= 1e-12 * want, "{got} vs {want}");
        assert!((got - fa.sin() * tissue_se(m0, t1, tr)).abs() > 0.05 * want, "must differ from saturation recovery");
        let p90 = IrParams { excitation_flip_deg: 90.0, ..p };
        let got90 = tissue_ir(m0, t1, tr, &p90);
        assert!((got90 - tissue_se(m0, t1, tr)).abs() <= 1e-12 * got90);
    }

    #[test]
    fn ir_guards_match_simasl() {
        let p = IrParams { inversion_time: 1.0, excitation_flip_deg: 90.0, inversion_flip_deg: 180.0 };
        // zero T1: both exponentials are exp(0) = 1, numerator m0 (1 - 2 + 1) = 0
        assert_eq!(tissue_ir(74.622, 0.0, 4.0, &p), 0.0);
        // zero denominator: fa = 0, fa_inv = 0, T1 -> infinity is not reachable with finite
        // inputs, but fa = 0 and fa_inv = 0 with t1 = 0 gives 1 - 1 = 0 exactly
        let p0 = IrParams { inversion_time: 1.0, excitation_flip_deg: 0.0, inversion_flip_deg: 0.0 };
        assert_eq!(tissue_ir(74.622, 0.0, 4.0, &p0), 0.0);
    }

    /// One fixture case: (scalars, t1, m0, want, t2).
    type Case = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

    /// Parse a fixture file: `case`, `scalars ...`, then named float rows.
    fn read_fixture(name: &str) -> Vec<Case> {
        let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("{path} is committed; regenerate with tools/gen_mrsignal_fixtures.py"));
        let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let floats = |l: &str| -> Vec<f64> { l.split_whitespace().skip(1).map(|t| t.parse::<f64>().unwrap()).collect() };
        let mut cases = Vec::new();
        while let Some(head) = lines.next() {
            assert!(head.starts_with("case"), "unexpected line: {head}");
            let s = lines.next().unwrap();
            assert!(s.starts_with("scalars"));
            let scalars = floats(s);
            let mut take = |name: &str| -> Vec<f64> {
                let l = lines.next().unwrap();
                assert!(l.starts_with(name), "expected {name}, got {l}");
                floats(l)
            };
            let t1 = take("t1");
            let t2 = take("t2"); // simasl's gradient-echo steady state reads it; recorded for all
            let m0 = take("m0");
            let want = take("signal_no_te");
            cases.push((scalars, t1, m0, want, t2));
        }
        cases
    }

    /// simasl MriSignalFilter fixtures with exp(-TE/T2) divided out (tools/gen_mrsignal_fixtures.py).
    #[test]
    fn matches_simasl_spin_echo_with_the_transverse_factor_divided_out() {
        let cases = read_fixture("mrsignal.txt");
        assert!(cases.len() >= 4, "too few cases: {}", cases.len());
        for (ci, (scalars, t1, m0, want, _)) in cases.iter().enumerate() {
            let tr = scalars[1];
            for i in 0..want.len() {
                let got = tissue_se(m0[i], t1[i], tr);
                let tol = 1e-12 * want[i].abs().max(1e-300);
                assert!((got - want[i]).abs() <= tol, "case {ci} voxel {i}: {got:.17e} vs {:.17e}", want[i]);
            }
        }
    }

    #[test]
    fn matches_simasl_inversion_recovery_with_the_transverse_factor_divided_out() {
        let cases = read_fixture("mrsignal_ir.txt");
        assert!(cases.len() >= 16, "too few cases: {}", cases.len());
        let mut saw_no_inversion = false;
        for (ci, (scalars, t1, m0, want, _)) in cases.iter().enumerate() {
            let (tr, ti, fa, fa_inv) = (scalars[1], scalars[2], scalars[3], scalars[4]);
            saw_no_inversion |= fa_inv == 0.0;
            let p = IrParams { inversion_time: ti, excitation_flip_deg: fa, inversion_flip_deg: fa_inv };
            for i in 0..want.len() {
                let got = tissue_ir(m0[i], t1[i], tr, &p);
                let tol = 1e-12 * want[i].abs().max(1e-300);
                assert!((got - want[i]).abs() <= tol, "case {ci} voxel {i}: {got:.17e} vs {:.17e}", want[i]);
            }
        }
        assert!(saw_no_inversion, "the fixture must include an fa_inv = 0 case");
    }

    /// simasl's gradient-echo fixtures with exp(-TE/T2*) divided out: its own form at 1e-12
    /// everywhere, and the model's spoiled form wherever simasl's E2 = exp(-TR/T2) is below 1e-16
    /// (TR 10 s, nonzero T2), where the two must agree.
    #[test]
    fn matches_simasl_gradient_echo_with_the_transverse_factor_divided_out() {
        let cases = read_fixture("mrsignal_ge.txt");
        assert!(cases.len() >= 16, "too few cases: {}", cases.len());
        let mut spoiled_checked = 0usize;
        for (ci, (scalars, t1, m0, want, t2)) in cases.iter().enumerate() {
            let (tr, fa) = (scalars[1], scalars[2]);
            for i in 0..want.len() {
                let tol = 1e-12 * want[i].abs().max(1e-300);
                let got = tissue_ge_simasl(m0[i], t1[i], t2[i], tr, fa);
                assert!((got - want[i]).abs() <= tol, "case {ci} voxel {i}: {got:.17e} vs {:.17e}", want[i]);
                if t2[i] > 0.0 && (-tr / t2[i]).exp() < 1e-16 {
                    let spoiled = tissue_ge_spoiled(m0[i], t1[i], tr, fa);
                    assert!((spoiled - want[i]).abs() <= tol, "spoiled, case {ci} voxel {i}: {spoiled:.17e} vs {:.17e}", want[i]);
                    spoiled_checked += 1;
                }
            }
        }
        assert!(spoiled_checked > 100, "the spoiled form must be checked on many voxels: {spoiled_checked}");
    }

    #[test]
    fn gradient_echo_closed_forms_and_guards() {
        let (m0, t1, tr) = (74.622f64, 1.33f64, 4.0f64);
        let e = (-tr / t1).exp();
        for fa in [90.0f64, 60.0, 30.0, -30.0] {
            let r = fa.to_radians();
            let want = r.sin() * m0 * (1.0 - e) / (1.0 - r.cos() * e);
            let got = tissue_ge_spoiled(m0, t1, tr, fa);
            assert!((got - want).abs() <= 1e-12 * want.abs(), "{fa}: {got} vs {want}");
        }
        // at 90 degrees the spoiled gradient echo is the spin echo's saturation recovery
        assert!((tissue_ge_spoiled(m0, t1, tr, 90.0) - tissue_se(m0, t1, tr)).abs() <= 1e-12 * m0);
        // a long T2 makes simasl's coherent form differ (the review's 3.7 percent at T2 = 2 s)
        let (s, c) = (tissue_ge_spoiled(m0, 3.0, 4.0, 60.0), tissue_ge_simasl(m0, 3.0, 2.0, 4.0, 60.0));
        assert!((c / s - 1.0).abs() > 0.03, "{s} vs {c}");
        // guards: zero T1 is zero signal; zero T2 gives simasl's E2 = 1
        assert_eq!(tissue_ge_spoiled(m0, 0.0, tr, 60.0), 0.0);
        assert_eq!(tissue_ge_simasl(m0, 0.0, 0.1, tr, 60.0), 0.0);
        assert_eq!(blood_ge(-0.37, 90.0), -0.37 * 90f64.to_radians().sin());
        assert!((blood_ge(2.0, 30.0) - 1.0).abs() < 1e-12);
    }
}
