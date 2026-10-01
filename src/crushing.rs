//! Vascular crushing (P4 addendum, part C): the surviving fraction of the arterial signal under
//! crusher gradients. Pure std.
//!
//! Model: within a voxel the arteries run in all directions and carry a laminar (Poiseuille)
//! profile, whose volume-weighted speed is uniform on `[0, v_max]`; a speed `v` at a random
//! direction has a projected velocity uniform on `[-v, v]`, and a spin moving at `u` along the
//! crusher acquires the phase `pi u / VENC` (the simulator's declared convention: BIDS defines
//! `VascularCrushingVENC` only as a strength in cm/s). Averaging the phase factor over
//! directions gives `sin(pi v/VENC) / (pi v/VENC)`, and over speeds
//! `c = Si(pi r) / (pi r)`, `r = v_max / VENC`.

use std::f64::consts::{FRAC_PI_2, PI};

/// The sine integral `Si(x) = integral from 0 to x of sin(u)/u du`, for every finite `x`.
/// Power series for `|x| <= 2`; beyond, `pi/2 + Im E1(i x)` with `E1` from its complex
/// continued fraction by the modified Lentz method (Numerical Recipes' `cisi`), which converges
/// faster as `x` grows, so no asymptotic switch is needed.
pub fn si(x: f64) -> f64 {
    assert!(x.is_finite(), "Si of a non-finite argument");
    if x < 0.0 {
        return -si(-x);
    }
    if x == 0.0 {
        return 0.0;
    }
    if x <= 2.0 {
        // sum (-1)^n x^(2n+1) / ((2n+1) (2n+1)!)
        let x2 = x * x;
        let mut term = x; // x^(2n+1) / (2n+1)! with its sign
        let mut sum = x;
        let mut n = 0usize;
        loop {
            n += 1;
            let k = (2 * n) as f64;
            term *= -x2 / (k * (k + 1.0));
            let add = term / (k + 1.0);
            sum += add;
            if add.abs() < 1e-17 * sum.abs() {
                return sum;
            }
            assert!(n < 100, "Si series did not converge at {x}");
        }
    }
    // E1(i x) by the continued fraction 1/(1 + i x -) 1^2/(3 + i x -) 2^2/(5 + i x -) ...
    const FPMIN: f64 = 1e-300;
    let cmul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
    let cdiv = |a: (f64, f64), b: (f64, f64)| {
        let d = b.0 * b.0 + b.1 * b.1;
        ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
    };
    let mut b = (1.0, x);
    let mut c = (1.0 / FPMIN, 0.0);
    let mut d = cdiv((1.0, 0.0), b);
    let mut h = d;
    let mut i = 2usize;
    loop {
        let a = -(((i - 1) * (i - 1)) as f64);
        b.0 += 2.0;
        // d = 1 / (a d + b); c = b + a / c
        d = cdiv((1.0, 0.0), (a * d.0 + b.0, a * d.1 + b.1));
        let ac = cdiv((a, 0.0), c);
        c = (b.0 + ac.0, b.1 + ac.1);
        let del = cmul(c, d);
        h = cmul(h, del);
        if (del.0 - 1.0).abs() + del.1.abs() < 4.0 * f64::EPSILON {
            break;
        }
        i += 1;
        assert!(i < 100_000, "Si continued fraction did not converge at {x}");
    }
    // E1(i x) = (cos x - i sin x) h
    let e1 = cmul((x.cos(), -x.sin()), h);
    FRAC_PI_2 + e1.1
}

/// The surviving fraction of the arterial signal for an arterial speed range `v_max` (cm/s) and
/// a crusher `VENC` (cm/s): `1` with crushing off (`venc == 0`) or no flow, else
/// `Si(pi r) / (pi r)`.
pub fn survival(v_max: f64, venc: f64) -> f64 {
    if venc == 0.0 || v_max == 0.0 {
        return 1.0;
    }
    let r = v_max / venc;
    assert!(r.is_finite() && r > 0.0, "crushing ratio v_max / VENC = {v_max} / {venc} is not finite");
    let x = PI * r;
    si(x) / x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scipy.special.sici(x)[0]` (scipy 1.5.3).
    const SCIPY: [(f64, f64); 6] = [
        (0.5, 0.49310741804306674),
        (1.0, 0.9460830703671831),
        (PI, 1.8519370519824658),
        (4.0, 1.758203138949053),
        (10.0, 1.658347594218874),
        (20.0, 1.5482417010434397),
    ];

    #[test]
    fn si_matches_scipy() {
        for (x, want) in SCIPY {
            let got = si(x);
            assert!((got - want).abs() <= 1e-12, "Si({x}) = {got:.17} vs {want:.17}");
            assert_eq!(si(-x), -got);
        }
        assert_eq!(si(0.0), 0.0);
    }

    #[test]
    fn si_is_continuous_across_the_method_switch_and_right_for_large_x() {
        let (lo, hi) = (si(2.0 - 1e-9), si(2.0 + 1e-9));
        // the slope there is sin(2)/2 ~ 0.45, so the two sides differ by ~9e-10
        assert!((hi - lo - 2e-9 * (2.0f64.sin() / 2.0)).abs() < 1e-12, "{lo:.17} {hi:.17}");
        for x in [1e3f64, 1e6] {
            // Si(x) = pi/2 - cos(x)/x - sin(x)/x^2 + O(x^-3)
            let asym = FRAC_PI_2 - x.cos() / x - x.sin() / (x * x);
            assert!((si(x) - asym).abs() < 3.0 / (x * x * x), "Si({x}) = {} vs {asym}", si(x));
        }
    }

    #[test]
    fn survival_values_and_monotonicity() {
        assert_eq!(survival(10.0, 0.0), 1.0);
        assert_eq!(survival(0.0, 4.0), 1.0);
        assert!((survival(4.0, 4.0) - 0.5894898722360835).abs() < 1e-12);
        let mut prev = 1.0;
        for i in 1..=3000 {
            let r = i as f64 * 1e-3;
            let c = survival(r, 1.0);
            assert!(c < prev, "not decreasing at r = {r}: {c} >= {prev}");
            assert!(c > 0.0);
            prev = c;
        }
        // to zero as the crusher strengthens
        assert!(survival(1000.0, 1.0) < 1e-3);
    }
}
