//! # FRI Protocol Implementation
//!
//! Fast Reed-Solomon Interactive Oracle Proof for low-degree testing.
//! Proves that a committed polynomial has degree below a target bound.
//!
//! ## Domain Convention
//!
//! This FRI implementation treats inputs as evaluations over the unshifted two-adic subgroup.
//! If the PCS evaluates over a coset `gK`, the shift is absorbed into the polynomial:
//! `Q'(X) = Q(g·X)`. The low-degree test is run on `Q'` using subgroup points.

mod fold;
mod proof;
pub(crate) mod prover;
pub(crate) mod verifier;

pub use fold::FriFold;
pub use proof::{FriRoundTranscript, FriTranscript};
pub use verifier::FriError;

/// FRI protocol parameters.
///
/// Controls the trade-off between proof size, prover time, and verifier time.
///
/// Higher `log_blowup` increases soundness per query (fewer queries needed) but introduces
/// larger Merkle trees — increasing both proof size (longer authentication paths) and prover time
/// (LDE over a larger domain). Higher arity reduces the number of FRI rounds (fewer Merkle
/// tree commitments) but increases per-query proof size (each opening reveals `arity`
/// siblings). `log_final_degree` reduces the number of rounds and therefore the number of
/// Merkle commitments; if too large, the final polynomial's coefficients dominate the proof
/// size.
#[derive(Clone, Copy, Debug)]
pub struct FriParams {
    /// Log₂ of the blowup factor (LDE domain size / polynomial degree).
    ///
    /// Higher values increase soundness but also proof size and prover time.
    /// Typical values: 2-4 (blowup factors of 4-16).
    log_blowup: u8,

    /// The FRI folding strategy.
    ///
    /// Determines the folding arity (2, 4, or 8).
    pub fold: FriFold,

    /// Log₂ of the final polynomial degree.
    ///
    /// Folding stops when degree reaches `2^log_final_degree`.
    /// Final polynomial coefficients are sent in descending degree order
    /// `[cₙ, ..., c₁, c₀]` for direct Horner evaluation by the verifier.
    log_final_degree: u8,

    /// Grinding bits before each folding challenge.
    folding_pow_bits: u8,
}

impl FriParams {
    /// Create new FRI parameters.
    pub fn new(log_blowup: u8, fold: FriFold, log_final_degree: u8, folding_pow_bits: u8) -> Self {
        Self {
            log_blowup,
            fold,
            log_final_degree,
            folding_pow_bits,
        }
    }

    /// Log₂ of the blowup factor.
    #[inline]
    pub fn log_blowup(&self) -> usize {
        self.log_blowup as usize
    }

    /// Log₂ of the final polynomial degree.
    #[inline]
    pub fn log_final_degree(&self) -> usize {
        self.log_final_degree as usize
    }

    /// Grinding bits before each folding challenge.
    #[inline]
    pub fn folding_pow_bits(&self) -> usize {
        self.folding_pow_bits as usize
    }

    /// Compute the number of folding rounds for a given initial evaluation domain size.
    ///
    /// Each round reduces the domain by `2^log_folding_factor`. We fold until the domain
    /// size reaches `2^(log_final_degree + log_blowup)`, at which point the polynomial
    /// degree is at most `2^log_final_degree`.
    ///
    /// Uses `div_ceil` to round up, ensuring we always reach the target degree even if
    /// the domain size doesn't divide evenly by the folding factor.
    #[inline]
    pub fn num_rounds(&self, log_domain_size: usize) -> usize {
        // Final domain size = final_degree × blowup = 2^(log_final_degree + log_blowup)
        let log_max_final_size = self.log_final_degree() + self.log_blowup();
        // Number of times we need to divide by 2^log_folding_factor
        log_domain_size
            .saturating_sub(log_max_final_size)
            .div_ceil(self.fold.log_arity())
    }

    /// Compute the final polynomial degree after folding.
    ///
    /// After `num_rounds` folding rounds, the domain shrinks from `2^log_domain_size`
    /// to `2^(log_domain_size - num_rounds × log_folding_factor)`. The polynomial
    /// degree is then `domain_size / blowup`.
    ///
    /// Due to `div_ceil` in `num_rounds`, the actual final degree may be smaller than
    /// `2^log_final_degree` when the folding doesn't divide evenly.
    #[inline]
    pub fn final_poly_degree(&self, log_domain_size: usize) -> usize {
        let num_rounds = self.num_rounds(log_domain_size);
        // log of final domain size after folding
        let log_final_size = log_domain_size - num_rounds * self.fold.log_arity();
        // degree = domain_size / blowup = 2^(log_final_size - log_blowup)
        1 << log_final_size.saturating_sub(self.log_blowup())
    }
}

#[cfg(test)]
mod tests;
