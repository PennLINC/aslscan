//! Hadamard time-encoded labeling (P6 addendum, part A), pure std: the Sylvester matrices, the
//! encoding and its labeling weights, and the decoding.
//!
//! A cycle of order `H` acquires `H` raw volumes; raw volume `i` labels sub-bolus `j` (of
//! `H - 1`) where `h_ij = -1` and leaves it as control where `h_ij = +1`, with `h` the Sylvester
//! matrix of order `H` without its all-ones first column. Its blood is `-sum_j w_ij dM_j` with the
//! weight `w_ij = (1 - h_ij) / 2`, and the tissue `T` is common to the cycle, so
//! `S_i = T - sum_j w_ij dM_j`. Every encoding column sums to zero and the columns are orthogonal
//! (`sum_i h_ij h_ik = H delta_jk`), so `D_j = (2 / H) sum_i h_ij S_i = dM_j`: the tissue cancels
//! and the sub-bolus is recovered with the sign of a `deltam` volume.

/// The orders supported.
pub const ORDERS: [usize; 4] = [4, 8, 16, 32];

/// The Sylvester-Hadamard matrix of `order` (a power of two): `H_1 = [1]`,
/// `H_2n = [[H_n, H_n], [H_n, -H_n]]`. Row `i`, column `j`; row 0 and column 0 are all ones.
pub fn sylvester(order: usize) -> Vec<Vec<i8>> {
    assert!(order.is_power_of_two(), "a Sylvester matrix has a power-of-two order, not {order}");
    let mut h = vec![vec![1i8]];
    while h.len() < order {
        let n = h.len();
        let mut next = vec![vec![0i8; 2 * n]; 2 * n];
        for i in 0..n {
            for j in 0..n {
                next[i][j] = h[i][j];
                next[i][j + n] = h[i][j];
                next[i + n][j] = h[i][j];
                next[i + n][j + n] = -h[i][j];
            }
        }
        h = next;
    }
    h
}

/// The encoding of `order`: the Sylvester matrix without its first (all-ones) column, `order`
/// rows (raw volumes) by `order - 1` columns (sub-boli).
pub fn encoding(order: usize) -> Vec<Vec<i8>> {
    sylvester(order).into_iter().map(|row| row[1..].to_vec()).collect()
}

/// The labeling weights of one encoding row: `(1 - h) / 2`, 1 where the sub-bolus is labeled.
pub fn weights(row: &[i8]) -> Vec<u8> {
    row.iter().map(|&h| ((1 - h as i32) / 2) as u8).collect()
}

/// Decode one cycle: `images[i]` is raw volume `i` (complex, as `(re, im)`), all the same length;
/// returns sub-bolus `j`'s `D_j = (2 / H) sum_i h_ij S_i` for `j` in `0..H - 1`, in `f64`.
pub fn decode(images: &[&[(f64, f64)]], order: usize) -> Vec<Vec<(f64, f64)>> {
    assert_eq!(images.len(), order, "a cycle of order {order} has {order} raw volumes, not {}", images.len());
    let len = images.first().map_or(0, |v| v.len());
    assert!(images.iter().all(|v| v.len() == len), "the raw volumes of a cycle differ in length");
    let h = encoding(order);
    let scale = 2.0 / order as f64;
    (0..order - 1)
        .map(|j| {
            (0..len)
                .map(|x| {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for (i, img) in images.iter().enumerate() {
                        let s = h[i][j] as f64;
                        re += s * img[x].0;
                        im += s * img[x].1;
                    }
                    (scale * re, scale * im)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_encodings_are_orthogonal_with_zero_column_sums() {
        for order in ORDERS {
            let h = sylvester(order);
            assert!(h[0].iter().all(|&x| x == 1) && h.iter().all(|r| r[0] == 1));
            for a in 0..order {
                for b in 0..order {
                    let dot: i32 = (0..order).map(|i| h[i][a] as i32 * h[i][b] as i32).sum();
                    assert_eq!(dot, if a == b { order as i32 } else { 0 }, "order {order}, columns {a} {b}");
                }
            }
            let e = encoding(order);
            assert_eq!((e.len(), e[0].len()), (order, order - 1));
            for j in 0..order - 1 {
                assert_eq!(e.iter().map(|r| r[j] as i32).sum::<i32>(), 0, "order {order}, column {j}");
            }
            // every sub-bolus is labeled in half the raw volumes; raw volume 0 labels nothing
            for j in 0..order - 1 {
                assert_eq!(e.iter().map(|r| weights(r)[j] as usize).sum::<usize>(), order / 2);
            }
            assert!(weights(&e[0]).iter().all(|&w| w == 0));
        }
        assert_eq!(weights(&[1, -1, -1, 1]), vec![0, 1, 1, 0]);
    }

    /// Synthetic data: tissue `T` (complex, per voxel) plus the encoded blood of sub-boli `dM_j`.
    fn encoded(order: usize, t: &[(f64, f64)], dm: &[Vec<(f64, f64)>], h: &[Vec<i8>]) -> Vec<Vec<(f64, f64)>> {
        (0..order)
            .map(|i| {
                let w = weights(&h[i]);
                (0..t.len())
                    .map(|x| {
                        let (mut re, mut im) = t[x];
                        for j in 0..order - 1 {
                            re -= w[j] as f64 * dm[j][x].0;
                            im -= w[j] as f64 * dm[j][x].1;
                        }
                        (re, im)
                    })
                    .collect()
            })
            .collect()
    }

    type Image = Vec<(f64, f64)>;

    fn data(order: usize, n: usize) -> (Image, Vec<Image>) {
        let t: Vec<(f64, f64)> = (0..n).map(|x| (100.0 + x as f64, -30.0 + 0.5 * x as f64)).collect();
        let dm = (0..order - 1)
            .map(|j| (0..n).map(|x| (1.0 + 0.1 * j as f64 + 0.01 * x as f64, 0.2 * ((j + x) as f64).sin())).collect())
            .collect();
        (t, dm)
    }

    fn worst(a: &[Vec<(f64, f64)>], b: &[Vec<(f64, f64)>]) -> f64 {
        a.iter().zip(b).flat_map(|(u, v)| u.iter().zip(v).map(|(p, q)| (p.0 - q.0).hypot(p.1 - q.1))).fold(0.0, f64::max)
    }

    #[test]
    fn decoding_recovers_each_sub_bolus() {
        for order in ORDERS {
            let (t, dm) = data(order, 17);
            let h = encoding(order);
            let s = encoded(order, &t, &dm, &h);
            let refs: Vec<&[(f64, f64)]> = s.iter().map(|v| v.as_slice()).collect();
            let d = decode(&refs, order);
            assert!(worst(&d, &dm) < 1e-12, "order {order}: {}", worst(&d, &dm));
        }
    }

    /// The general formula: with a tissue term that differs between raw volumes (`C_i`), the
    /// decoded volume carries `(2 / H) sum_i h_ij C_i` beside `dM_j`.
    #[test]
    fn a_varying_tissue_term_decodes_to_its_own_combination() {
        let order = 8;
        let (t, dm) = data(order, 9);
        let h = encoding(order);
        let mut s = encoded(order, &t, &dm, &h);
        let c: Vec<f64> = (0..order).map(|i| 0.3 * i as f64 - 1.0).collect();
        for (i, v) in s.iter_mut().enumerate() {
            for z in v.iter_mut() {
                z.0 += c[i];
            }
        }
        let refs: Vec<&[(f64, f64)]> = s.iter().map(|v| v.as_slice()).collect();
        let d = decode(&refs, order);
        for j in 0..order - 1 {
            let leak = 2.0 / order as f64 * (0..order).map(|i| h[i][j] as f64 * c[i]).sum::<f64>();
            for x in 0..9 {
                assert!((d[j][x].0 - dm[j][x].0 - leak).abs() < 1e-12 && (d[j][x].1 - dm[j][x].1).abs() < 1e-12);
            }
        }
    }

    /// Negative controls: a flipped sign or two swapped columns in the encoding do not decode.
    #[test]
    fn a_wrong_encoding_does_not_decode() {
        let order = 8;
        let (t, dm) = data(order, 11);
        let mut h = encoding(order);
        h[3][2] = -h[3][2];
        let s = encoded(order, &t, &dm, &h);
        let refs: Vec<&[(f64, f64)]> = s.iter().map(|v| v.as_slice()).collect();
        assert!(worst(&decode(&refs, order), &dm) > 0.1);
        let mut h = encoding(order);
        for r in h.iter_mut() {
            r.swap(0, 4);
        }
        let s = encoded(order, &t, &dm, &h);
        let refs: Vec<&[(f64, f64)]> = s.iter().map(|v| v.as_slice()).collect();
        assert!(worst(&decode(&refs, order), &dm) > 0.1);
    }
}
