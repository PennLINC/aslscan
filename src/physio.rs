//! Physiological noise (P4 addendum, part E). Pure std, seconds on the series' clock.
//!
//! Three processes: cardiac and respiratory phase (initial phase uniform, periods drawn
//! independently from a normal of mean `1/f` and coefficient of variation `cv`, truncated at
//! `±3 sd`, the phase linear within each period) and a drift (an Ornstein-Uhlenbeck process of
//! unit variance on a `0.05` s grid, linearly interpolated). Two factors, each
//! `1 + a_c sin(phi_c) + a_r sin(phi_r) + a_d x`: the tissue factor at a time, and the label
//! factor averaged over a window (exactly, for the piecewise-linear phase and drift).
//!
//! The streams are normative, so a dataset can be regenerated from its sidecar: the series seed
//! XOR [`PHYSIO_SEED_SALT`] is the base, and the cardiac, respiratory and drift streams are
//! seeded from the base XOR `1`, `2` and `3`. [`Physio::new`] is the only place the salt is
//! applied; callers pass the series seed as it is.

use std::f64::consts::PI;

use crate::rng::Normal;

/// "PHYSIO", so that turning physiological noise on leaves the acquisition and motion draws
/// unchanged.
pub const PHYSIO_SEED_SALT: u64 = 0x5048_5953_494F;

/// The drift grid step (s).
pub const DRIFT_STEP: f64 = 0.05;

/// A quasi-periodic phase: period boundaries from `0`, the phase advancing `2 pi` per period.
#[derive(Debug, Clone)]
pub struct PhaseProcess {
    pub phase0: f64,
    /// Period boundaries `b_0 = 0 < b_1 < ...`, past the horizon.
    pub bounds: Vec<f64>,
}

impl PhaseProcess {
    /// Draws, in order: the initial phase (one uniform), then one truncated normal per period
    /// (resampled while `|z| > 3`) until the boundaries pass `horizon`.
    pub fn new(f: f64, cv: f64, rng: &mut Normal, horizon: f64) -> PhaseProcess {
        assert!(f > 0.0 && (0.0..=0.3).contains(&cv) && horizon >= 0.0, "phase process f {f} cv {cv}");
        let phase0 = 2.0 * PI * rng.unit();
        let mut bounds = vec![0.0];
        while *bounds.last().unwrap() <= horizon {
            let mut z = rng.draw();
            while z.abs() > 3.0 {
                z = rng.draw();
            }
            let period = 1.0 / f + (cv / f) * z;
            bounds.push(bounds.last().unwrap() + period);
        }
        PhaseProcess { phase0, bounds }
    }

    /// The period containing `t` (clamped to the generated range).
    fn segment(&self, t: f64) -> usize {
        let n = self.bounds.len() - 1;
        match self.bounds.binary_search_by(|b| b.partial_cmp(&t).unwrap()) {
            Ok(i) => i.min(n - 1),
            Err(i) => i.saturating_sub(1).min(n - 1),
        }
    }

    pub fn phase(&self, t: f64) -> f64 {
        let k = self.segment(t);
        let (b0, b1) = (self.bounds[k], self.bounds[k + 1]);
        self.phase0 + 2.0 * PI * (k as f64 + (t - b0) / (b1 - b0))
    }

    /// The average of `sin(phase)` over `[t0, t1]`, exact for the piecewise-linear phase; at
    /// `t0 == t1` the value there.
    pub fn mean_sin(&self, t0: f64, t1: f64) -> f64 {
        if t1 <= t0 {
            return self.phase(t0).sin();
        }
        let mut acc = 0.0;
        let mut s = t0;
        while s < t1 {
            let k = self.segment(s);
            // to the end of this period, or to t1; past the last generated period the phase
            // continues linearly, so the rest is one piece
            let end = self.bounds[k + 1];
            let e = if end > s { end.min(t1) } else { t1 };
            let w = 2.0 * PI / (self.bounds[k + 1] - self.bounds[k]);
            let ps = self.phase(s);
            acc += (ps.cos() - (ps + w * (e - s)).cos()) / w;
            s = e;
        }
        acc / (t1 - t0)
    }
}

/// An Ornstein-Uhlenbeck drift of unit variance and time constant `tau_d`, sampled on the
/// [`DRIFT_STEP`] grid from `0` (exact discretization, `x_0` stationary) and linearly
/// interpolated.
#[derive(Debug, Clone)]
pub struct OuDrift {
    pub x: Vec<f64>,
}

impl OuDrift {
    pub fn new(tau_d: f64, rng: &mut Normal, horizon: f64) -> OuDrift {
        assert!(tau_d > 0.0 && tau_d.is_finite(), "drift time {tau_d}");
        let n = (horizon / DRIFT_STEP).ceil() as usize + 2;
        let a = (-DRIFT_STEP / tau_d).exp();
        let b = (1.0 - a * a).sqrt();
        let mut x = Vec::with_capacity(n);
        x.push(rng.draw());
        for k in 1..n {
            let z = rng.draw();
            x.push(x[k - 1] * a + b * z);
        }
        OuDrift { x }
    }

    fn at_grid(&self, t: f64) -> (usize, f64) {
        let last = (self.x.len() - 1) as f64;
        let u = (t / DRIFT_STEP).clamp(0.0, last);
        let k = (u.floor() as usize).min(self.x.len() - 2);
        (k, u - k as f64)
    }

    pub fn value(&self, t: f64) -> f64 {
        let (k, fr) = self.at_grid(t);
        self.x[k] + (self.x[k + 1] - self.x[k]) * fr
    }

    /// The average of the interpolant over `[t0, t1]`, exact (trapezoids between the grid
    /// points inside the window and the interpolated ends); at `t0 == t1` the value there.
    pub fn mean(&self, t0: f64, t1: f64) -> f64 {
        if t1 <= t0 {
            return self.value(t0);
        }
        // Walk the grid intervals by integer index: recomputing the index from a grid time
        // (`floor(n * step / step)`) lands on n - 1 for many n, which would merge the rest of the
        // window into one trapezoid.
        let last = self.x.len() - 1;
        let mut k = ((t0 / DRIFT_STEP).floor().max(0.0) as usize).min(last - 1);
        if ((k + 1) as f64 * DRIFT_STEP) <= t0 && k + 1 < last {
            k += 1;
        }
        let mut acc = 0.0;
        let mut s = t0;
        while s < t1 {
            // the end of interval k, or t1; past the grid the interpolant is constant (clamped)
            let e = if k + 1 < last { ((k + 1) as f64 * DRIFT_STEP).min(t1) } else { t1 };
            let e = e.max(s);
            acc += 0.5 * (self.value(s) + self.value(e)) * (e - s);
            if e >= t1 {
                break;
            }
            s = e;
            k += 1;
        }
        acc / (t1 - t0)
    }
}

/// `[physio]`: amplitudes as `[cardiac, respiratory, drift]` for the tissue and the label
/// factor, the two frequencies (Hz) and coefficients of variation, and the drift time (s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysioParams {
    pub tissue: [f64; 3],
    pub label: [f64; 3],
    pub cardiac_frequency: f64,
    pub cardiac_cv: f64,
    pub respiratory_frequency: f64,
    pub respiratory_cv: f64,
    pub drift_time: f64,
}

impl Default for PhysioParams {
    fn default() -> Self {
        PhysioParams {
            tissue: [0.0; 3],
            label: [0.0; 3],
            cardiac_frequency: 1.0,
            cardiac_cv: 0.1,
            respiratory_frequency: 0.25,
            respiratory_cv: 0.2,
            drift_time: 30.0,
        }
    }
}

/// The realized processes for one series.
#[derive(Debug, Clone)]
pub struct Physio {
    pub params: PhysioParams,
    pub cardiac: PhaseProcess,
    pub respiratory: PhaseProcess,
    pub drift: OuDrift,
}

impl Physio {
    /// `series_seed` is the series' own seed, unsalted: the salt is applied here and nowhere
    /// else.
    pub fn new(params: PhysioParams, series_seed: u64, horizon: f64) -> Physio {
        let base = series_seed ^ PHYSIO_SEED_SALT;
        let cardiac = PhaseProcess::new(params.cardiac_frequency, params.cardiac_cv, &mut Normal::new(base ^ 1), horizon);
        let respiratory =
            PhaseProcess::new(params.respiratory_frequency, params.respiratory_cv, &mut Normal::new(base ^ 2), horizon);
        let drift = OuDrift::new(params.drift_time, &mut Normal::new(base ^ 3), horizon);
        Physio { params, cardiac, respiratory, drift }
    }

    fn combine(a: &[f64; 3], c: f64, r: f64, d: f64) -> f64 {
        1.0 + a[0] * c + a[1] * r + a[2] * d
    }

    /// The tissue factor at `t`.
    pub fn tissue_factor(&self, t: f64) -> f64 {
        Self::combine(&self.params.tissue, self.cardiac.phase(t).sin(), self.respiratory.phase(t).sin(), self.drift.value(t))
    }

    /// The label factor averaged over `[t0, t1]` ((P)CASL's labeling window), with the three
    /// window averages it is made of.
    pub fn label_factor_window(&self, t0: f64, t1: f64) -> (f64, [f64; 3]) {
        let m = [self.cardiac.mean_sin(t0, t1), self.respiratory.mean_sin(t0, t1), self.drift.mean(t0, t1)];
        (Self::combine(&self.params.label, m[0], m[1], m[2]), m)
    }

    /// The label factor at `t` (PASL's inversion).
    pub fn label_factor_at(&self, t: f64) -> (f64, [f64; 3]) {
        self.label_factor_window(t, t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periods_have_the_requested_statistics_and_stay_positive() {
        for (f, cv) in [(1.0, 0.1), (0.25, 0.2), (1.2, 0.3)] {
            let p = PhaseProcess::new(f, cv, &mut Normal::new(11), 10_200.0 / f);
            let periods: Vec<f64> = p.bounds.windows(2).map(|w| w[1] - w[0]).collect();
            let n = periods.len() as f64;
            let mean = periods.iter().sum::<f64>() / n;
            let sd = (periods.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
            assert!(periods.len() >= 10_000);
            assert!((mean * f - 1.0).abs() < 0.02, "f {f}: mean {mean}");
            // truncation at 3 sd removes 0.27% of the mass: the sd is 0.987 of cv / f
            assert!((sd * f / cv - 0.987).abs() < 0.02, "f {f} cv {cv}: sd {sd}");
            assert!(periods.iter().all(|x| *x > 0.0));
        }
    }

    #[test]
    fn mean_sin_is_exact_against_quadrature() {
        let p = PhaseProcess::new(1.0, 0.1, &mut Normal::new(3), 50.0);
        // composite Simpson on each smooth piece (split at the period boundaries, where the
        // phase has a kink), 2 000 intervals per piece: its own error is ~1e-14
        let quad = |t0: f64, t1: f64| {
            let mut cuts = vec![t0];
            cuts.extend(p.bounds.iter().copied().filter(|b| *b > t0 && *b < t1));
            cuts.push(t1);
            let mut acc = 0.0;
            for w in cuts.windows(2) {
                let (a, b) = (w[0], w[1]);
                let n = 2000;
                let h = (b - a) / n as f64;
                let mut s = (p.phase(a).sin() + p.phase(b).sin()) / 3.0;
                for i in 1..n {
                    let x = a + i as f64 * h;
                    // the phase is linear inside the piece; evaluate from its left end so a
                    // point that rounds onto the boundary stays on this piece's line
                    s += if i % 2 == 1 { 4.0 } else { 2.0 } / 3.0 * p.phase(x.min(b - 1e-15)).sin();
                }
                acc += s * h;
            }
            acc / (t1 - t0)
        };
        let b = p.bounds[3];
        for (t0, t1) in [(0.1, 0.4), (0.3, 7.9), (b, b + 1.8), (2.0, 20.0)] {
            let (a, q) = (p.mean_sin(t0, t1), quad(t0, t1));
            assert!((a - q).abs() < 1e-10, "[{t0}, {t1}]: {a} vs {q}");
        }
        assert_eq!(p.mean_sin(5.0, 5.0), p.phase(5.0).sin());
    }

    #[test]
    fn drift_is_unit_variance_with_the_requested_correlation() {
        let d = OuDrift::new(30.0, &mut Normal::new(5), 200_000.0);
        let n = d.x.len() as f64;
        let mean = d.x.iter().sum::<f64>() / n;
        let var = d.x.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        assert!((var - 1.0).abs() < 0.05, "var {var}");
        let lag = (30.0 / DRIFT_STEP) as usize;
        let c = d.x.iter().zip(&d.x[lag..]).map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>() / (n - lag as f64) / var;
        assert!((c - (-1.0f64).exp()).abs() < 0.05, "lag correlation {c}");
        // the window mean is exact for the interpolant over many grid intervals, including
        // windows that start on grid times where floor(n * step / step) rounds to n - 1 (n = 43
        // is the first: 2.15 s; 4.0 s is a labeling start at TR 4 s)
        let w = OuDrift::new(30.0, &mut Normal::new(8), 40.0);
        for (t0, t1) in [(4.0, 5.8), (2.15, 3.95), (8.0, 9.8), (0.013, 1.987), (4.0, 4.0 + 1e-9)] {
            let n = 200_000;
            let h = (t1 - t0) / n as f64;
            // composite Simpson per grid interval is exact for a piecewise-linear integrand; a
            // fine midpoint rule over the whole window is within ~h of it, enough at this n
            let quad = (0..n).map(|i| w.value(t0 + (i as f64 + 0.5) * h)).sum::<f64>() / n as f64;
            let got = w.mean(t0, t1);
            assert!((got - quad).abs() < 1e-6, "[{t0}, {t1}]: {got} vs {quad}");
        }
        // the window mean is exact for the interpolant: a hand-built two-point case
        let two = OuDrift { x: vec![1.0, 3.0, 3.0] };
        assert!((two.mean(0.0, DRIFT_STEP) - 2.0).abs() < 1e-15);
        assert!((two.mean(0.025, 0.075) - (0.5 * (2.0 + 3.0) * 0.025 + 3.0 * 0.025) / 0.05).abs() < 1e-12);
    }

    #[test]
    fn factors_combine_the_processes() {
        let params = PhysioParams { tissue: [0.02, 0.01, 0.005], label: [0.03, 0.02, 0.01], ..Default::default() };
        let ph = Physio::new(params, 9, 100.0);
        let t = 12.3;
        let want = 1.0 + 0.02 * ph.cardiac.phase(t).sin() + 0.01 * ph.respiratory.phase(t).sin() + 0.005 * ph.drift.value(t);
        assert_eq!(ph.tissue_factor(t), want);
        let (lf, m) = ph.label_factor_window(10.0, 11.8);
        assert_eq!(lf, 1.0 + 0.03 * m[0] + 0.02 * m[1] + 0.01 * m[2]);
        assert!(m[0].abs() <= 1.0 && m[1].abs() <= 1.0);
    }

    /// The normative stream: a fixed series seed reproduces these values. A change here changes
    /// every dataset generated with physiological noise.
    #[test]
    fn the_reference_stream_is_reproduced() {
        let ph = Physio::new(PhysioParams::default(), 2026, 30.0);
        let got: Vec<f64> = ph.cardiac.bounds[1..4].iter().chain(ph.respiratory.bounds[1..3].iter()).copied()
            .chain([ph.cardiac.phase0, ph.respiratory.phase0, ph.drift.x[0], ph.drift.x[1], ph.drift.x[100]])
            .collect();
        println!("reference stream: {got:?}");
        let want: [f64; 10] = REFERENCE;
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.to_bits(), w.to_bits(), "{got:?}");
        }
    }

    const REFERENCE: [f64; 10] = [
        0.9134230619383962, 1.8615433500644096, 2.8011501445176545, 3.59985759881823, 7.248621944674101,
        2.4757249083954083, 4.127296359699489, 0.20600214428028393, 0.2490356266877675, 0.7008211512605311,
    ];
}
