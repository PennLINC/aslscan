//! A seedable PRNG for the within-volume motion events (SplitMix64, the same generator
//! `mrsim_acq::motion` uses for its own draws, which it does not export). Pure std; only our own
//! reproducibility matters, so nothing here needs to match another implementation.

/// SplitMix64 state.
#[derive(Debug, Clone)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-amp, amp]`.
    pub fn signed(&mut self, amp: f64) -> f64 {
        2.0 * amp * self.unit() - amp
    }
}

/// Standard normal draws by Box-Muller from a [`SplitMix64`] (P4 addendum, part E): each pair of
/// uniforms gives two normals, the cosine one returned and the sine one cached for the next
/// call, so both values of every pair are used, in order. A separate type so that
/// `SplitMix64` keeps its one-field form and its constructor sites.
#[derive(Debug, Clone)]
pub struct Normal {
    pub rng: SplitMix64,
    cached: Option<f64>,
}

impl Normal {
    pub fn new(seed: u64) -> Normal {
        Normal { rng: SplitMix64(seed), cached: None }
    }

    /// A uniform in `[0, 1)` from the same stream; it does not touch the cached normal.
    pub fn unit(&mut self) -> f64 {
        self.rng.unit()
    }

    pub fn draw(&mut self) -> f64 {
        if let Some(z) = self.cached.take() {
            return z;
        }
        // (0, 1] for the logarithm
        let u1 = 1.0 - self.rng.unit();
        let u2 = self.rng.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        self.cached = Some(r * th.sin());
        r * th.cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_moments_and_pair_order() {
        let mut n = Normal::new(42);
        let k = 1_000_000;
        let (mut s, mut s2) = (0.0f64, 0.0f64);
        for _ in 0..k {
            let z = n.draw();
            s += z;
            s2 += z * z;
        }
        let mean = s / k as f64;
        let var = s2 / k as f64 - mean * mean;
        assert!(mean.abs() < 3e-3 && (var - 1.0).abs() < 3e-3, "mean {mean} var {var}");
        // the second value of a pair costs no draw
        let mut a = Normal::new(7);
        let _ = a.draw();
        let state = a.rng.0;
        let _ = a.draw();
        assert_eq!(a.rng.0, state, "the cached value must not advance the stream");
        let _ = a.draw();
        assert_ne!(a.rng.0, state);
    }

    #[test]
    fn deterministic_and_in_range() {
        let mut a = SplitMix64(7);
        let mut b = SplitMix64(7);
        for _ in 0..1000 {
            let u = a.unit();
            assert_eq!(u, b.unit());
            assert!((0.0..1.0).contains(&u));
            let s = a.signed(2.5);
            assert_eq!(s, b.signed(2.5));
            assert!(s.abs() <= 2.5);
        }
        let mut c = SplitMix64(8);
        assert_ne!(a.unit(), c.unit());
    }
}
