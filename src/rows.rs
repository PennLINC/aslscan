//! The volume list: one [`Row`] per `_aslcontext.tsv` line, timing already resolved. Pure std,
//! so the naming and TSV code in `bids` can use it without the `io` feature; `protocol` (which
//! produces rows) re-exports both types.

/// One `aslcontext.tsv` row kind. `cbf` is rejected at parse time: a CBF map is a quantified
/// output, not an acquired volume, and producing one means choosing a quantification model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    M0scan,
    Control,
    Label,
    Deltam,
}

impl RowKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RowKind::M0scan => "m0scan",
            RowKind::Control => "control",
            RowKind::Label => "label",
            RowKind::Deltam => "deltam",
        }
    }
}

/// One volume of the series, with its timing already resolved from scalar-or-array fields.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub kind: RowKind,
    /// Kinetic signal time `t` (s) before the per-slice offset: PLD + tau for (P)CASL, PLD for
    /// PASL (BIDS measures the PASL PLD from the middle of the labeling pulse, which is already
    /// the GKM's clock). Zero for m0scan rows.
    pub t: f64,
    /// Bolus duration `tau` (s) for this row. Zero for m0scan rows.
    pub tau: f64,
    /// Repetition time the tissue steady state is evaluated at (s).
    pub tr: f64,
}
