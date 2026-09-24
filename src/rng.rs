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

#[cfg(test)]
mod tests {
    use super::*;

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
