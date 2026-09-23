//! # aslscan
//!
//! An ASL digital reference object simulator: a BIDS ASL protocol and a BIDS-derivatives phantom
//! in, the Buxton general kinetic model and the spin-echo signal equation per phantom voxel, then
//! the shared `mrsim-acq` acquisition stage (EPI distortion, relaxation, ringing, coils, noise)
//! once for the whole series, and a validator-clean BIDS ASL dataset with ground truth out.
//!
//! ```text
//! asl.json + aslcontext.tsv (+ TOML overlay) -> protocol
//! BIDS phantom maps                          -> phantom
//!        kinetic (delta_m)  ->  mrsignal (tissue, blood)  ->  resample  ->  series
//!        -> mrsim_acq::simulate_acquisition_oversampled -> bids
//! ```
//!
//! Units: seconds everywhere upstream of the acquisition stage; `protocol` and `phantom` are the
//! only places that convert to the milliseconds `mrsim_acq::kspace::Acquisition` and `T2Volume`
//! take. simasl (ASLDRO v2.2.0) is the numerical oracle for `kinetic` and `mrsignal`.
//!
//! ## Build shape
//! The default build is pure std: `kinetic`, `mrsignal`, `resample` and the `bids` naming code
//! test offline. `io` adds the loaders, the series driver and NIfTI/JSON/TOML; `cli` the binary.

/// The Buxton general kinetic model, per voxel, as simasl computes it.
pub mod kinetic;
/// Post-excitation magnetization (spin echo only in P1); no transverse relaxation here.
pub mod mrsignal;
/// Box-overlap averaging between the phantom, acquisition and simulation grids.
pub mod resample;
/// BIDS ASL sidecar + aslcontext + TOML overlay -> `Protocol` (feature `io`).
#[cfg(feature = "io")]
pub mod protocol;
/// BIDS-derivatives phantom maps -> `Phantom`, T2' derivation, class/voxel resolution (feature `io`).
#[cfg(feature = "io")]
pub mod phantom;
