//! The bolus-position suppression model (P4 addendum, part D). Pure std.
//!
//! A parcel of label is indexed by its sub-bolus coordinate `a` in `[0, tau]`: its labeling
//! time for (P)CASL, its arrival time less `ATT` for PASL. A suppression pulse at `p`
//! multiplies a parcel's factor by `1 - 2 epsilon` when the parcel is inside the pulse's region
//! at `p`, which is when `a + delta <= p`, `delta` being the region's entry offset
//! ([`entry_offset`]). So each pulse cuts the bolus at `a = p - delta`, and between cuts the
//! factor is constant: the bolus is a sum of sub-boluses (`kinetic::delta_m_sub`), each weighted
//! by its factor. The tissue's own timeline (P3) is unchanged: tissue is always inside.

use crate::kinetic::LabelType;

/// Where a pulse acts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Region {
    /// Everywhere the label is, from its labeling.
    Global,
    /// On the label once it has entered the pulsed slab, `d` seconds after labeling.
    Slab(f64),
    /// On delivered label only (the slab entry is the arrival in the voxel).
    Arrival,
}

/// The entry offset `delta`: parcel `a` is inside the region from `a + delta`. `att` is the
/// voxel's `ATT` (or `aATT` for the arterial compartment). `None`: inside from the start (a
/// PASL bolus is labeled whole at `t = 0`, so a global pulse acts on every parcel).
pub fn entry_offset(label_type: LabelType, region: Region, att: f64) -> Option<f64> {
    match (region, label_type) {
        (Region::Global, LabelType::Pasl) => None,
        (Region::Global, _) => Some(0.0),
        (Region::Slab(d), _) => Some(d),
        (Region::Arrival, _) => Some(att),
    }
}

/// The cuts strictly inside `(0, tau)`, ascending and deduplicated: `p - delta` per pulse. A
/// pulse whose cut is at or below 0 acts on no parcel and one at or above `tau` on every parcel;
/// neither cuts.
pub fn cuts(pulses: &[f64], tau: f64, delta: Option<f64>) -> Vec<f64> {
    let Some(d) = delta else { return Vec::new() };
    let mut c: Vec<f64> = pulses.iter().map(|p| p - d).filter(|c| *c > 0.0 && *c < tau).collect();
    c.sort_by(|a, b| a.partial_cmp(b).expect("finite cuts"));
    c.dedup();
    c
}

/// Whether pulse `p` acts on the parcels of a sub-bolus whose upper end is `b`: its cut is at
/// or beyond `b` (cuts are sub-bolus boundaries, so a cut is never strictly inside one).
fn acts_on(p: f64, b: f64, delta: Option<f64>) -> bool {
    match delta {
        None => true,
        Some(d) => p - d >= b,
    }
}

/// The sub-boluses `(a, b, factor)` partitioning `[0, tau]`, the factor being the product of
/// `1 - 2 epsilon` over the pulses that act on the sub-bolus, multiplied in pulse order from
/// `1.0` as `longitudinal::label_factor` multiplies them, so an uncut bolus whose every parcel
/// sees every pulse has P3's factor bit for bit. `pulses` must be sorted (as
/// `longitudinal::Suppression` keeps them).
pub fn subbolus_factors(pulses: &[f64], epsilon: f64, tau: f64, delta: Option<f64>) -> Vec<(f64, f64, f64)> {
    let mut bounds = vec![0.0];
    bounds.extend(cuts(pulses, tau, delta));
    bounds.push(tau);
    bounds
        .windows(2)
        .map(|w| {
            let f: f64 = pulses.iter().filter(|&&p| acts_on(p, w[1], delta)).map(|_| 1.0 - 2.0 * epsilon).product();
            (w[0], w[1], f)
        })
        .collect()
}

/// The factor of the single parcel `a` the arterial compartment holds: the product over the
/// pulses with `a + delta <= p`, in pulse order.
pub fn arterial_factor(pulses: &[f64], epsilon: f64, a: f64, delta: Option<f64>) -> f64 {
    pulses
        .iter()
        .filter(|&&p| match delta {
            None => true,
            Some(d) => a + d <= p,
        })
        .map(|_| 1.0 - 2.0 * epsilon)
        .product()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kinetic::{delta_m, delta_m_sub, Kinetic};
    use crate::longitudinal::{label_factor, Suppression};

    const ASL002: [f64; 2] = [2.05, 3.276];
    const K: Kinetic = Kinetic { label_type: LabelType::Pcasl, tau: 1.8, alpha: 0.85, lambda: 0.9, t1b: 1.65 };
    // the ASLDRO 3 T phantom's GM
    const F: f64 = 60.0;
    const ATT: f64 = 0.8;
    const T1T: f64 = 1.33;
    const M0: f64 = 74.622;

    #[test]
    fn global_after_labeling_is_the_global_bolus_factor_bit_for_bit() {
        let sets: [&[f64]; 4] = [&ASL002, &[1.9, 2.5, 3.0], &[2.2], &[1.81, 1.82, 2.9, 3.1]];
        for pulses in sets {
            for eps in [1.0, 0.95, 0.7] {
                let s = Suppression::new(pulses.to_vec(), eps, false);
                for lt in [LabelType::Pcasl, LabelType::Pasl] {
                    let sb = subbolus_factors(&s.pulse_times, eps, 1.8, entry_offset(lt, Region::Global, ATT));
                    assert_eq!(sb.len(), 1, "{lt:?} {pulses:?}: global after labeling must not cut");
                    assert_eq!((sb[0].0, sb[0].1), (0.0, 1.8));
                    assert_eq!(sb[0].2.to_bits(), label_factor(&s).to_bits(), "{lt:?} {pulses:?} eps {eps}");
                }
            }
        }
    }

    /// `F = 2 (exp(1.25/T1') - 1) / (exp(1.8/T1') - 1) - 1` (P4 addendum, part D).
    fn closed_form() -> f64 {
        let t1p = 1.0 / (1.0 / T1T + (F / 6000.0) / 0.9);
        2.0 * ((1.25 / t1p).exp() - 1.0) / ((1.8 / t1p).exp() - 1.0) - 1.0
    }

    #[test]
    fn arrival_slab_gives_the_corrected_asl002_factor() {
        let delta = entry_offset(LabelType::Pcasl, Region::Arrival, ATT);
        let sb = subbolus_factors(&ASL002, 1.0, 1.8, delta);
        assert_eq!(sb.len(), 2);
        assert!((sb[0].1 - 1.25).abs() < 1e-12 && sb[0].2 == 1.0 && sb[1].2 == -1.0, "{sb:?}");
        // the first slice's readout: PLD 2.0 + tau 1.8
        let t = 3.8;
        let suppressed: f64 = sb.iter().map(|&(a, b, f)| f * delta_m_sub(&K, F, ATT, T1T, M0, t, a, b)).sum();
        let ratio = suppressed / delta_m(&K, F, ATT, T1T, M0, t);
        let cf = closed_form();
        assert!((cf - 0.0821045).abs() < 5e-8, "closed form {cf}");
        assert!((ratio - cf).abs() < 1e-6, "{ratio} vs {cf}");
        // an independent parcel sum over the GKM kernel, split AT the cut (a uniform rule across
        // it misses by ~1.3e-5) with 20 000 midpoints per piece
        let t1p = 1.0 / (1.0 / T1T + (F / 6000.0) / 0.9);
        let piece = |a: f64, b: f64| {
            let n = 20_000;
            let h = (b - a) / n as f64;
            (0..n).map(|i| (-(t - (a + (i as f64 + 0.5) * h) - ATT) / t1p).exp()).sum::<f64>() * h
        };
        let quad = (piece(0.0, 1.25) - piece(1.25, 1.8)) / (piece(0.0, 1.25) + piece(1.25, 1.8));
        assert!((quad - cf).abs() < 1e-6, "quadrature {quad} vs {cf}");
        println!("asl002 GM, slab entry at arrival: factor {ratio:.7} (closed form {cf:.7}, quadrature {quad:.7})");
    }

    #[test]
    fn cuts_follow_the_entry_offsets() {
        // PASL global never cuts
        assert!(cuts(&[0.3, 0.9], 0.7, entry_offset(LabelType::Pasl, Region::Global, ATT)).is_empty());
        assert_eq!(subbolus_factors(&[0.3, 0.9], 1.0, 0.7, None), vec![(0.0, 0.7, 1.0)]);
        // a slab entry d cuts at p - d
        assert_eq!(cuts(&[1.0, 2.0], 1.8, Some(0.4)), vec![0.6, 1.6]);
        // pulses whose cut falls outside (0, tau) do not cut; duplicates collapse
        assert_eq!(cuts(&[0.2, 0.9, 0.9, 5.0], 1.8, Some(0.4)), vec![0.5]);
        // a cut exactly at a boundary is not a cut
        assert!(cuts(&[1.8 + 0.4], 1.8, Some(0.4)).is_empty());
        assert!(cuts(&[0.4], 1.8, Some(0.4)).is_empty());
        // during PCASL labeling, a slab pulse acts only on parcels already in the slab
        let sb = subbolus_factors(&[1.0], 1.0, 1.8, Some(0.4));
        assert_eq!(sb, vec![(0.0, 0.6, -1.0), (0.6, 1.8, 1.0)]);
    }

    #[test]
    fn arterial_factor_on_both_sides_of_a_cut() {
        let delta = Some(0.5);
        assert_eq!(arterial_factor(&[1.0, 2.0], 1.0, 0.4, delta), 1.0); // 0.9 <= 1.0 and 2.0: both
        assert_eq!(arterial_factor(&[1.0, 2.0], 1.0, 0.6, delta), -1.0); // only the second
        assert_eq!(arterial_factor(&[1.0, 2.0], 1.0, 1.6, delta), 1.0); // neither
        assert_eq!(arterial_factor(&[1.0], 0.95, 5.0, None), 1.0 - 2.0 * 0.95);
    }

    #[test]
    fn zero_efficiency_pulses_cut_but_do_not_scale() {
        let sb = subbolus_factors(&[1.0, 1.5], 0.0, 1.8, Some(0.0));
        assert_eq!(sb.len(), 3);
        assert!(sb.iter().all(|s| s.2 == 1.0));
        // the weighted sum is the uncut bolus to rounding
        let t = 3.6;
        let s: f64 = sb.iter().map(|&(a, b, f)| f * delta_m_sub(&K, F, ATT, T1T, M0, t, a, b)).sum();
        let whole = delta_m(&K, F, ATT, T1T, M0, t);
        assert!((s - whole).abs() <= 1e-12 * whole);
    }
}
