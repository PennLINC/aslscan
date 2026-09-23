//! Post-excitation magnetization for the spin-echo contrast, per voxel. Seconds throughout.
//!
//! This departs from simasl's `MriSignalFilter` (`src/asldro/filters/mri_signal_filter.py`) in
//! exactly one way, on purpose: **no transverse relaxation**. simasl folds `exp(-TE/T2)` into
//! its signal equation (`mri_signal_filter.py:246`) because it has no readout model and the TE
//! decay has nowhere else to live. Here the acquisition stage applies `exp(-trf/T2 - |t|/T2')`
//! per phase-encode line (`mrsim_acq::kspace`), a term that already spans excitation to every
//! sampled line, so applying it here too would relax twice. The fixtures in
//! `tests/fixtures/mrsignal.txt` are simasl's output with that factor divided out.
//!
//! The blood compartment carries `delta_m` from the kinetic model and gets **no** steady-state
//! recovery term: `delta_m` is already a magnetization difference, and running it through
//! `M0 (1 - exp(-TR/T1))` would attribute to it a saturation history it does not have. It gets
//! the flip-angle factor only, which for a spin echo's 90-degree excitation is exactly 1.

/// The signal equation in use. P1 accepts spin echo only; see [`parse_contrast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contrast {
    SpinEcho,
}

/// Parse the overlay's `acq_contrast` (case-insensitive, as simasl lower-cases it). Only `"se"`
/// is accepted in P1. `"ge"` needs a gradient-echo echo-formation model in the acquisition stage
/// (unrefocused off-resonance phase, monotonic T2* decay) and waits for P5; `"ir"` is inversion
/// preparation before an otherwise unchanged readout and waits for P3 with background suppression.
pub fn parse_contrast(s: &str) -> Result<Contrast, String> {
    match s.to_ascii_lowercase().as_str() {
        "se" => Ok(Contrast::SpinEcho),
        "ge" => Err("acq_contrast \"ge\": P1 simulates spin-echo readouts only; gradient echo needs \
                     a different echo-formation model and arrives with P5"
            .to_string()),
        "ir" => Err("acq_contrast \"ir\": P1 simulates spin-echo readouts only; inversion recovery \
                     arrives with P3 alongside background suppression"
            .to_string()),
        other => Err(format!("acq_contrast {other:?}: expected \"se\" (P1 spin-echo only)")),
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
/// which is 1 for the 90-degree spin-echo excitation. Kept as a function so the P3/P5 contrasts
/// have one place to put `sin(flip_angle)`.
pub fn blood_se(delta_m: f64) -> f64 {
    delta_m
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
    fn ge_and_ir_are_rejected_naming_the_restriction() {
        let ge = parse_contrast("ge").unwrap_err();
        assert!(ge.contains("spin-echo") && ge.contains("P5"), "{ge}");
        let ir = parse_contrast("ir").unwrap_err();
        assert!(ir.contains("spin-echo") && ir.contains("P3"), "{ir}");
        assert!(parse_contrast("bogus").is_err());
        assert_eq!(parse_contrast("se").unwrap(), Contrast::SpinEcho);
        assert_eq!(parse_contrast("SE").unwrap(), Contrast::SpinEcho);
    }

    #[test]
    fn blood_is_passthrough_in_p1() {
        assert_eq!(blood_se(-0.37), -0.37);
        assert_eq!(blood_se(0.0), 0.0);
    }

    /// simasl MriSignalFilter fixtures with exp(-TE/T2) divided out (tools/gen_mrsignal_fixtures.py).
    #[test]
    fn matches_simasl_spin_echo_with_the_transverse_factor_divided_out() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mrsignal.txt"))
            .expect("tests/fixtures/mrsignal.txt is committed; regenerate with tools/gen_mrsignal_fixtures.py");
        let mut lines = text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty());
        let floats = |l: &str| -> Vec<f64> {
            l.split_whitespace().skip(1).map(|t| t.parse::<f64>().unwrap()).collect()
        };
        let mut n_cases = 0;
        while let Some(head) = lines.next() {
            assert!(head.starts_with("case"), "unexpected line: {head}");
            let s: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
            assert_eq!(s[0], "scalars");
            let (te, tr) = (s[1].parse::<f64>().unwrap(), s[2].parse::<f64>().unwrap());
            let _ = te; // divided out by the generator; recorded for the reader
            let mut take = |name: &str| -> Vec<f64> {
                let l = lines.next().unwrap();
                assert!(l.starts_with(name), "expected {name}, got {l}");
                floats(l)
            };
            let t1 = take("t1");
            let _t2 = take("t2");
            let m0 = take("m0");
            let want = take("signal_no_te");
            for i in 0..want.len() {
                let got = tissue_se(m0[i], t1[i], tr);
                let tol = 1e-12 * want[i].abs().max(1e-300);
                assert!((got - want[i]).abs() <= tol,
                        "case {n_cases} voxel {i}: {got:.17e} vs {:.17e}", want[i]);
            }
            n_cases += 1;
        }
        assert!(n_cases >= 4, "too few cases: {n_cases}");
    }
}
