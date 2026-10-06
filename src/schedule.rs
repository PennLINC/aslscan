//! The schedule (P6 addendum, "The one idea underneath"): what the scanner does, in three index
//! levels, separate from what is written.
//!
//! - A **preparation** is one labeling (or none, for an m0scan row): its start on the series clock,
//!   its labeling window, the suppression pulse set and crushing VENC that apply to it.
//! - A **raw volume** is one volume of the acquisition call. In 2D it holds one preparation; a
//!   segmented 3D volume holds `NumberShots` of them (P5 part D); under Look-Locker several raw
//!   volumes (the readouts) share one preparation.
//! - An **output volume** is what the dataset lists: a raw volume, or a decoded one (Hadamard).
//!
//! `Protocol.rows` stays the input/output rows (what `aslcontext.tsv` lists). Every protocol
//! without a P6 feature has the [`Schedule::identity`] layout, and the series runs today's code
//! for it (the legacy dispatch, `series.rs`); the identity schedule exists so the P6 path and its
//! tests can state that layout, not so the legacy path can read it.

use crate::kinetic::LabelType;
use crate::protocol::Protocol;
use crate::rows::{Row, RowKind};

/// One labeling (P6: "preparation").
#[derive(Debug, Clone, PartialEq)]
pub struct Preparation {
    /// The raw volume this preparation is read in (the first, under Look-Locker).
    pub raw: usize,
    /// Its shot within that raw volume (0 in 2D).
    pub shot: usize,
    /// Start of the repetition on the series clock (s): the labeling starts here.
    pub start_s: f64,
    /// The labeling window on the series clock (s): `[start, start + tau]` for (P)CASL, the
    /// instant `[start, start]` for PASL and m0scan.
    pub labeling_window: [f64; 2],
    /// Index into `SuppressionSpec::per_row` of the pulse set this labeling runs (when
    /// suppression is on).
    pub suppression: usize,
    /// The crushing VENC (cm/s) of this labeling, when crushing is on.
    pub venc: Option<f64>,
}

/// One volume of the acquisition call.
#[derive(Debug, Clone, PartialEq)]
pub struct RawVolume {
    /// Its first preparation.
    pub prep: usize,
    /// How many preparations it holds (`NumberShots` in segmented 3D, else 1).
    pub n_preps: usize,
    /// Its readout index within the preparation (0 except under Look-Locker).
    pub readout: usize,
    /// The Hadamard or Look-Locker cycle it belongs to.
    pub cycle: Option<usize>,
    /// Hadamard: its row of the encoding (`crate::hadamard::encoding`), which every one of its
    /// preparations repeats.
    pub encoding_row: Option<usize>,
}

/// One output volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// A raw volume, written as acquired.
    Raw(usize),
    /// Sub-bolus `subbolus` of Hadamard cycle `cycle`, decoded.
    Decoded { cycle: usize, subbolus: usize },
}

/// A Hadamard or Look-Locker cycle: a run of raw volumes that belong together.
#[derive(Debug, Clone, PartialEq)]
pub struct Cycle {
    /// The raw volumes, `raws[start..end]`.
    pub raws: std::ops::Range<usize>,
    /// The input/output rows it produces (Hadamard: its decoded rows, in sub-bolus order).
    pub rows: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    /// One row per raw volume, with its own timing and kind.
    pub raw_rows: Vec<Row>,
    pub preps: Vec<Preparation>,
    pub raws: Vec<RawVolume>,
    pub outputs: Vec<Output>,
    pub cycles: Vec<Cycle>,
}

impl Schedule {
    /// The protocol's schedule: Hadamard's under `[hadamard]`, else the identity.
    pub fn new(p: &Protocol) -> Schedule {
        match &p.hadamard {
            Some(h) => Schedule::hadamard(p, h),
            None => Schedule::identity(p),
        }
    }

    /// Hadamard (P6 addendum, part A, "The schedule"): an `m0scan` row is one raw volume, as today;
    /// a cycle is `H` raw volumes in encoding-row order, each a `label` row at
    /// `t = tau_tot + PLD`, `tau = tau_tot` (its blood is the encoded sum of sub-boli) and each
    /// `NumberShots` preparations (3D) repeating its row; the clock advances `NumberShots x tr`
    /// per raw volume. The outputs follow the input rows: an `m0scan` row is its raw volume, a
    /// `deltam` row its cycle's decoded sub-bolus. Pulse sets and VENC are the cycle's (the
    /// protocol checked they are one per cycle), taken from its first row.
    pub fn hadamard(p: &Protocol, h: &crate::protocol::HadamardSpec) -> Schedule {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
        let mut sched = Schedule { raw_rows: Vec::new(), preps: Vec::new(), raws: Vec::new(), outputs: Vec::new(), cycles: Vec::new() };
        let mut clock = 0.0;
        let mut push_raw = |sched: &mut Schedule, row: Row, source_row: usize, labeled: bool, cycle: Option<usize>, enc: Option<usize>| {
            let r = sched.raws.len();
            sched.raws.push(RawVolume { prep: sched.preps.len(), n_preps: shots, readout: 0, cycle, encoding_row: enc });
            for s in 0..shots {
                let start = clock + s as f64 * row.tr;
                sched.preps.push(Preparation {
                    raw: r,
                    shot: s,
                    start_s: start,
                    labeling_window: [start, if labeled { start + row.tau } else { start }],
                    suppression: source_row,
                    venc: p.crushing.as_ref().map(|c| c.venc[source_row]),
                });
            }
            clock += shots as f64 * row.tr;
            sched.raw_rows.push(row);
            r
        };
        let mut v = 0;
        while v < p.rows.len() {
            if let Some(c) = h.cycles.iter().position(|cy| cy.rows[0] == v) {
                let cy = &h.cycles[c];
                let tr = p.rows[v].tr;
                let first = sched.raws.len();
                for e in 0..h.order {
                    let row = Row { kind: RowKind::Label, t: h.tau_tot + h.pld, tau: h.tau_tot, tr };
                    push_raw(&mut sched, row, v, true, Some(c), Some(e));
                }
                sched.cycles.push(Cycle { raws: first..first + h.order, rows: cy.rows.clone() });
                for j in 0..cy.rows.len() {
                    sched.outputs.push(Output::Decoded { cycle: c, subbolus: j });
                }
                v += cy.rows.len();
            } else {
                // an m0scan row (the protocol allows nothing else outside a cycle)
                let r = push_raw(&mut sched, p.rows[v].clone(), v, false, None, None);
                sched.outputs.push(Output::Raw(r));
                v += 1;
            }
        }
        sched
    }

    /// Today's layout: one raw volume per row, `NumberShots` preparations per raw volume in
    /// segmented 3D (one in 2D), each on the series clock of `Protocol::row_start` (shot `s` of
    /// row `v` starts at `row_start[v] + s * tr`), one output per raw volume.
    pub fn identity(p: &Protocol) -> Schedule {
        let shots = p.readout.as_ref().map_or(1, |r| r.number_shots.0);
        let mut preps = Vec::with_capacity(p.rows.len() * shots);
        let mut raws = Vec::with_capacity(p.rows.len());
        for (v, row) in p.rows.iter().enumerate() {
            raws.push(RawVolume { prep: preps.len(), n_preps: shots, readout: 0, cycle: None, encoding_row: None });
            for s in 0..shots {
                let start = p.row_start[v] + s as f64 * row.tr;
                let end = match p.label_type {
                    LabelType::Pasl => start,
                    _ => start + row.tau,
                };
                preps.push(Preparation {
                    raw: v,
                    shot: s,
                    start_s: start,
                    labeling_window: [start, end],
                    suppression: v,
                    venc: p.crushing.as_ref().map(|c| c.venc[v]),
                });
            }
        }
        Schedule {
            raw_rows: p.rows.clone(),
            preps,
            outputs: (0..p.rows.len()).map(Output::Raw).collect(),
            raws,
            cycles: Vec::new(),
        }
    }

    /// The preparations of raw volume `r`.
    pub fn preps_of(&self, r: usize) -> &[Preparation] {
        let rv = &self.raws[r];
        &self.preps[rv.prep..rv.prep + rv.n_preps]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{parse, Overlay};
    use serde_json::{json, Value};

    /// H8, two cycles of equal sub-boli (0.25 s, PLD 0.2 s), an m0scan row first at TR 6 s.
    fn h8(three_d: bool) -> Protocol {
        let mut ld = vec![0.0];
        let mut pld = vec![0.0];
        for _ in 0..2 {
            for j in 0..7 {
                ld.push(0.25);
                pld.push(0.2 + 0.25 * (6 - j) as f64);
            }
        }
        let mut tr = vec![6.0];
        tr.extend(vec![4.0; 14]);
        let mut s: Value = json!({
            "ArterialSpinLabelingType": "PCASL", "LabelingDuration": ld, "PostLabelingDelay": pld,
            "BackgroundSuppression": false, "M0Type": "Included", "RepetitionTimePreparation": tr,
            "EchoTime": 0.012, "MagneticFieldStrength": 3, "AcquisitionVoxelSize": [3.5, 3.5, 5],
            "MRAcquisitionType": "2D", "SliceTiming": [0.0, 0.05], "PhaseEncodingDirection": "j-",
            "TotalReadoutTime": 0.016
        });
        if three_d {
            let o = s.as_object_mut().unwrap();
            o.remove("SliceTiming");
            o.remove("TotalReadoutTime");
            s["MRAcquisitionType"] = json!("3D");
            s["PulseSequenceType"] = json!("3Dgrase");
            s["EffectiveEchoSpacing"] = json!(0.0003);
            s["NumberShots"] = json!(4);
            s["FlipAngle"] = json!(150);
        }
        let ov: Overlay = toml::from_str("[hadamard]\norder = 8\n").unwrap();
        let ctx = format!("volume_type\nm0scan\n{}", "deltam\n".repeat(14));
        parse(&s, &ctx, Some(&ov), None).unwrap()
    }

    #[test]
    fn the_hadamard_schedule() {
        for (three_d, shots) in [(false, 1usize), (true, 4)] {
            let p = h8(three_d);
            let s = Schedule::new(&p);
            // an m0scan raw volume, then two cycles of eight
            assert_eq!((s.raws.len(), s.raw_rows.len(), s.preps.len(), s.outputs.len()), (17, 17, 17 * shots, 15));
            assert_eq!(s.cycles.iter().map(|c| c.raws.clone()).collect::<Vec<_>>(), vec![1..9, 9..17]);
            assert_eq!(s.cycles[1].rows, (8..15).collect::<Vec<_>>());
            assert_eq!(s.outputs[0], Output::Raw(0));
            assert_eq!(s.outputs[1], Output::Decoded { cycle: 0, subbolus: 0 });
            assert_eq!(s.outputs[14], Output::Decoded { cycle: 1, subbolus: 6 });
            assert_eq!((s.raws[0].cycle, s.raws[0].encoding_row, s.raw_rows[0].kind), (None, None, RowKind::M0scan));
            let mut clock = 0.0;
            for (r, raw) in s.raws.iter().enumerate() {
                let row = &s.raw_rows[r];
                if r > 0 {
                    // every encoded raw volume is a label row read at tau_tot + PLD
                    assert_eq!((raw.cycle, raw.encoding_row), (Some((r - 1) / 8), Some((r - 1) % 8)));
                    assert_eq!((row.kind, row.tau, row.tr), (RowKind::Label, 1.75, 4.0));
                    assert!((row.t - 1.95).abs() < 1e-12);
                    // the readout at the raw t: the first sub-bolus's own PLD + tau
                    let first = s.cycles[raw.cycle.unwrap()].rows[0];
                    assert!((row.t - p.rows[first].t).abs() < 1e-12);
                }
                assert_eq!(raw.n_preps, shots);
                for (k, prep) in s.preps_of(r).iter().enumerate() {
                    assert_eq!((prep.raw, prep.shot), (r, k));
                    assert_eq!(prep.start_s, clock + k as f64 * row.tr);
                    let end = if r == 0 { prep.start_s } else { prep.start_s + 1.75 };
                    assert_eq!(prep.labeling_window, [prep.start_s, end]);
                    assert_eq!(prep.suppression, if r == 0 { 0 } else { 1 + 7 * ((r - 1) / 8) });
                }
                clock += shots as f64 * row.tr;
            }
            // the m0scan row at its own 6 s
            assert_eq!(s.preps_of(1)[0].start_s, shots as f64 * 6.0);
        }
    }
}
