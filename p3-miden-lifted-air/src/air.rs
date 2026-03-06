//! The `LiftedAir` super-trait for AIR definitions in the lifted STARK system.
//!
//! # Panic safety of `eval()`
//!
//! [`LiftedAir::eval`] is generic over `AB: LiftedAirBuilder`, so it cannot branch
//! on the concrete builder type. All builders expose data through the same trait
//! methods — [`main()`](crate::AirBuilder::main),
//! [`permutation()`](crate::PermutationAirBuilder::permutation),
//! [`public_values()`](crate::AirBuilder::public_values),
//! [`permutation_randomness()`](crate::PermutationAirBuilder::permutation_randomness),
//! [`permutation_values()`](crate::PermutationAirBuilder::permutation_values), and
//! [`periodic_values()`](crate::PeriodicAirBuilder::periodic_values) — which return
//! matrices or slices.
//!
//! If the symbolic evaluation in [`LiftedAir::log_quotient_degree`] succeeds (i.e.
//! does not panic), it proves that the AIR's `eval()` only accesses indices within
//! the declared dimensions. Any concrete builder constructed with matching dimensions
//! is therefore safe from out-of-bounds panics.
//!
//! Use [`LiftedAir::is_valid_builder`] to verify that a concrete builder's
//! dimensions match the AIR before calling `eval()`.

use p3_air::{BaseAir, WindowAccess};
use p3_field::{ExtensionField, Field};
use p3_util::log2_ceil_usize;
use thiserror::Error;

use crate::auxiliary::{ReducedAuxValues, ReductionError, VarLenPublicInputs};
use crate::{
    AirLayout, AirWithPeriodicColumns, LiftedAirBuilder, SymbolicAirBuilder, SymbolicExpressionExt,
};

/// Super-trait for AIR definitions used by the lifted STARK prover/verifier.
///
/// Inherits from upstream traits for width, public values, and periodic columns.
/// Adds Miden-specific auxiliary trace support. Every `LiftedAir` must provide
/// an auxiliary trace (even if it is a minimal 1-column dummy).
///
/// # Type Parameters
/// - `F`: Base field
/// - `EF`: Extension field (for aux trace challenges and aux values)
pub trait LiftedAir<F: Field, EF>: Sync + BaseAir<F> + AirWithPeriodicColumns<F> {
    /// Number of extension-field challenges required for the auxiliary trace.
    fn num_randomness(&self) -> usize;

    /// Number of extension-field columns in the auxiliary trace.
    fn aux_width(&self) -> usize;

    /// Number of extension-field aux values committed to the Fiat-Shamir transcript.
    ///
    /// These are the values returned by [`AuxBuilder::build_aux_trace`](crate::AuxBuilder::build_aux_trace)
    /// alongside the aux trace matrix. Their count may differ from [`aux_width`](Self::aux_width)
    /// (the number of aux trace columns).
    ///
    /// These values are exposed to AIR constraints as *permutation values* via
    /// [`PermutationAirBuilder::permutation_values`](crate::PermutationAirBuilder::permutation_values).
    fn num_aux_values(&self) -> usize;

    /// Number of variable-length public inputs this AIR expects.
    ///
    /// Each input is a slice of base-field elements that
    /// [`reduced_aux_values`](Self::reduced_aux_values) reduces to a single value.
    /// The prover validates that witnesses provide exactly this many slices.
    ///
    /// Implementors of [`reduced_aux_values`](Self::reduced_aux_values) should verify
    /// that `var_len_public_inputs` contains exactly this many slices, returning
    /// [`ReductionError`] otherwise.
    fn num_var_len_public_inputs(&self) -> usize {
        0
    }

    /// Reduce this AIR's aux values to a [`ReducedAuxValues`] contribution.
    ///
    /// Called by the verifier (with concrete field values, not symbolic expressions)
    /// to compute each AIR's contribution to the global cross-AIR bus identity check.
    /// The verifier accumulates contributions across all AIRs and checks that the
    /// combined result is identity (prod=1, sum=0).
    ///
    /// # Arguments
    /// - `aux_values`: prover-supplied aux values (from the proof)
    /// - `challenges`: extension-field challenges (same as used for aux trace building)
    /// - `public_values`: this AIR's public values (base field)
    /// - `var_len_public_inputs`: reducible inputs for the cross-AIR identity check
    ///
    /// # Errors
    ///
    /// The verifier validates instance dimensions (public values length,
    /// var-len public inputs count) before calling this method, so
    /// implementations can assume correct input counts. However, the
    /// *length of each individual var-len slice* is not validated upfront —
    /// implementations that index into these slices must check lengths
    /// themselves or use the `Result` return type to report errors.
    ///
    /// Default: returns identity (correct for AIRs without buses).
    fn reduced_aux_values(
        &self,
        _aux_values: &[EF],
        _challenges: &[EF],
        _public_values: &[F],
        _var_len_public_inputs: VarLenPublicInputs<'_, F>,
    ) -> Result<ReducedAuxValues<EF>, ReductionError>
    where
        EF: ExtensionField<F>,
    {
        Ok(ReducedAuxValues::identity())
    }

    /// Validate that this AIR satisfies the [`LiftedAir`] contract.
    ///
    /// The lifted STARK protocol relies on several structural properties of the AIR
    /// that can be checked statically (i.e. without a witness). This method verifies
    /// the subset that is machine-checkable; the full list of trust assumptions is
    /// documented in the module docs of `p3-miden-lifted-stark`. Both the prover and
    /// verifier call this before proceeding, so a malformed AIR is caught early.
    ///
    /// # Checked properties
    ///
    /// - **No preprocessed trace** — the lifted STARK protocol does not support
    ///   preprocessed (fixed) columns; their presence is an error.
    /// - **Positive auxiliary width** — every lifted AIR must declare at least one
    ///   auxiliary column (`aux_width() > 0`).
    /// - **Well-formed periodic columns** — each periodic column must be non-empty
    ///   and have a power-of-two length.
    fn validate(&self) -> Result<(), AirValidationError> {
        if self.preprocessed_trace().is_some() {
            return Err(AirValidationError::PreprocessedTrace);
        }
        if self.aux_width() == 0 {
            return Err(AirValidationError::ZeroAuxWidth);
        }
        for (i, col) in self.periodic_columns().iter().enumerate() {
            if col.is_empty() || !col.len().is_power_of_two() {
                return Err(AirValidationError::InvalidPeriodicColumn {
                    index: i,
                    length: col.len(),
                });
            }
        }
        Ok(())
    }

    /// Evaluate all AIR constraints using the provided builder.
    fn eval<AB: LiftedAirBuilder<F = F>>(&self, builder: &mut AB);

    /// Log₂ of the number of quotient chunks, inferred from symbolic constraint analysis.
    ///
    /// Evaluates the AIR on a [`SymbolicAirBuilder`](crate::SymbolicAirBuilder) to determine
    /// the maximum constraint degree M, then returns `log2_ceil(M - 1)` (padded so M ≥ 2).
    ///
    /// Uses `SymbolicAirBuilder<F>` (i.e. `EF = F`) which is sufficient for degree
    /// computation since extension-field operations have the same degree structure.
    ///
    /// # Why `M − 1` chunks?
    ///
    /// Let N be the trace height (so trace columns are polynomials of degree < N).
    /// Symbolic evaluation assigns each constraint a *degree multiple* M, meaning the
    /// resulting numerator polynomial C(X) has degree bounded by roughly M·(N − 1).
    ///
    /// In a STARK, the constraint numerator is divisible by the trace vanishing
    /// polynomial `Z_H(X) = Xᴺ − 1`, so the quotient polynomial
    /// `Q(X) = C(X) / Z_H(X)` has
    ///
    /// `deg(Q) ≤ deg(C) − N ≤ M·(N − 1) − N < (M − 1)·N`.
    ///
    /// We commit to Q(X) by splitting it into D chunks of degree < N. The bound above
    /// shows that D = M − 1 chunks suffice; we then round D up to a power of two and
    /// return `log2(D)`.
    ///
    /// We clamp M ≥ 2 so that D ≥ 1. If M = 1 then `deg(C) < N`, and divisibility by
    /// `Z_H` would force C(X) to be the zero polynomial (i.e. the constraint carries no
    /// information about the trace).
    fn log_quotient_degree(&self) -> usize
    where
        Self: Sized,
    {
        let layout = AirLayout {
            preprocessed_width: self.preprocessed_trace().map_or(0, |t| t.width),
            main_width: self.width(),
            num_public_values: self.num_public_values(),
            permutation_width: self.aux_width(),
            num_permutation_challenges: self.num_randomness(),
            num_permutation_values: self.num_aux_values(),
            num_periodic_columns: self.periodic_columns().len(),
        };
        let mut builder = SymbolicAirBuilder::<F>::new(layout);
        self.eval(&mut builder);

        let base_degree = builder
            .base_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max()
            .unwrap_or(0);
        let ext_degree = builder
            .extension_constraints()
            .iter()
            .map(|c: &SymbolicExpressionExt<F, F>| c.degree_multiple())
            .max()
            .unwrap_or(0);
        let constraint_degree = base_degree.max(ext_degree).max(2);

        log2_ceil_usize(constraint_degree - 1)
    }

    /// Number of quotient chunks: `2^log_quotient_degree()`.
    fn constraint_degree(&self) -> usize
    where
        Self: Sized,
    {
        1 << self.log_quotient_degree()
    }

    /// Check that a builder's dimensions match this AIR.
    ///
    /// Verifies every data-carrying accessor on [`LiftedAirBuilder`]: main trace,
    /// preprocessed trace, aux trace, public values, randomness, aux values, and
    /// periodic values.
    ///
    /// This guards the invariant that makes [`eval`](Self::eval) panic-free: if
    /// the symbolic evaluation in [`log_quotient_degree`](Self::log_quotient_degree)
    /// succeeds and this check passes, then `eval()` cannot panic from
    /// out-of-bounds access on the builder's accessors.
    fn is_valid_builder<AB: LiftedAirBuilder<F = F>>(
        &self,
        builder: &AB,
    ) -> Result<(), AirValidationError> {
        let check =
            |part: TracePart, expected: usize, actual: usize| -> Result<(), AirValidationError> {
                if actual != expected {
                    return Err(AirValidationError::BuilderMismatch {
                        part,
                        expected,
                        actual,
                    });
                }
                Ok(())
            };

        let main = builder.main();
        // Check current and next slices of the main trace.
        check(TracePart::Main, self.width(), main.current_slice().len())?;
        check(TracePart::Main, self.width(), main.next_slice().len())?;

        // Check current and next slices of the aux trace.
        let perm = builder.permutation();
        check(TracePart::Aux, self.aux_width(), perm.current_slice().len())?;
        check(TracePart::Aux, self.aux_width(), perm.next_slice().len())?;

        check(
            TracePart::PublicValues,
            self.num_public_values(),
            builder.public_values().len(),
        )?;
        check(
            TracePart::Randomness,
            self.num_randomness(),
            builder.permutation_randomness().len(),
        )?;
        check(
            TracePart::AuxValues,
            self.num_aux_values(),
            builder.permutation_values().len(),
        )?;
        check(
            TracePart::PeriodicValues,
            self.periodic_columns().len(),
            builder.periodic_values().len(),
        )?;

        Ok(())
    }
}

/// Which part of the trace a builder mismatch refers to.
#[derive(Copy, Clone, Debug)]
pub enum TracePart {
    Main,
    Aux,
    PublicValues,
    Randomness,
    AuxValues,
    PeriodicValues,
}

/// Errors from AIR validation.
///
/// Returned by [`LiftedAir::validate`],
/// [`AirInstance::validate`](crate::AirInstance::validate), and
/// [`validate_instances`](crate::validate_instances).
#[derive(Debug, Error)]
pub enum AirValidationError {
    #[error("no instances provided")]
    Empty,
    #[error("instances not in ascending height order")]
    NotAscending,
    #[error("periodic column {index}: length must be positive power of two, got {length}")]
    InvalidPeriodicColumn { index: usize, length: usize },
    #[error("preprocessed traces are not supported")]
    PreprocessedTrace,
    #[error("{part:?} dimension mismatch: expected {expected}, got {actual}")]
    BuilderMismatch {
        part: TracePart,
        expected: usize,
        actual: usize,
    },
    #[error("aux width must be positive")]
    ZeroAuxWidth,
    #[error("trace height {height} is not a power of two")]
    InvalidTraceHeight { height: usize },
    #[error("trace width mismatch: expected {expected}, got {actual}")]
    WidthMismatch { expected: usize, actual: usize },
    #[error("public values length mismatch: expected {expected}, got {actual}")]
    PublicValuesMismatch { expected: usize, actual: usize },
    #[error("var-len public inputs count mismatch: expected {expected}, got {actual}")]
    VarLenPublicInputsMismatch { expected: usize, actual: usize },
    #[error("trace height {trace_height} is less than max periodic column length {max_period}")]
    TraceHeightBelowPeriod {
        trace_height: usize,
        max_period: usize,
    },
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::RowWindow;
    use p3_air::{AirBuilder, ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder};
    use p3_field::PrimeCharacteristicRing;
    use p3_miden_dev_utils::configs::baby_bear_poseidon2::{EF, F};

    use super::{AirValidationError, LiftedAir, TracePart};
    use crate::{AirWithPeriodicColumns, LiftedAirBuilder};

    struct DummyAir {
        width: usize,
        aux_width: usize,
        num_public_values: usize,
        num_randomness: usize,
        num_aux_values: usize,
        periodic: Vec<Vec<F>>,
    }

    impl p3_air::BaseAir<F> for DummyAir {
        fn width(&self) -> usize {
            self.width
        }

        fn num_public_values(&self) -> usize {
            self.num_public_values
        }
    }

    impl AirWithPeriodicColumns<F> for DummyAir {
        fn periodic_columns(&self) -> &[Vec<F>] {
            &self.periodic
        }
    }

    impl LiftedAir<F, EF> for DummyAir {
        fn num_randomness(&self) -> usize {
            self.num_randomness
        }

        fn aux_width(&self) -> usize {
            self.aux_width
        }

        fn num_aux_values(&self) -> usize {
            self.num_aux_values
        }

        fn eval<AB: LiftedAirBuilder<F = F>>(&self, _builder: &mut AB) {}
    }

    struct DummyBuilder<'a> {
        main: RowWindow<'a, F>,
        preprocessed: RowWindow<'a, F>,
        aux: RowWindow<'a, EF>,
        public_values: &'a [F],
        randomness: &'a [EF],
        permutation_values: &'a [EF],
        periodic_values: &'a [F],
    }

    impl<'a> AirBuilder for DummyBuilder<'a> {
        type F = F;
        type Expr = F;
        type Var = F;
        type M = RowWindow<'a, F>;
        type PublicVar = F;

        fn main(&self) -> Self::M {
            self.main
        }

        fn preprocessed(&self) -> &Self::M {
            &self.preprocessed
        }

        fn public_values(&self) -> &[Self::PublicVar] {
            self.public_values
        }

        fn is_first_row(&self) -> Self::Expr {
            F::ZERO
        }

        fn is_last_row(&self) -> Self::Expr {
            F::ZERO
        }

        fn is_transition_window(&self, _size: usize) -> Self::Expr {
            F::ZERO
        }

        fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {}
    }

    impl<'a> ExtensionBuilder for DummyBuilder<'a> {
        type EF = EF;
        type ExprEF = EF;
        type VarEF = EF;

        fn assert_zero_ext<I>(&mut self, _x: I)
        where
            I: Into<Self::ExprEF>,
        {
        }
    }

    impl<'a> PermutationAirBuilder for DummyBuilder<'a> {
        type MP = RowWindow<'a, EF>;
        type RandomVar = EF;
        type PermutationVar = EF;

        fn permutation(&self) -> Self::MP {
            self.aux
        }

        fn permutation_randomness(&self) -> &[Self::RandomVar] {
            self.randomness
        }

        fn permutation_values(&self) -> &[Self::PermutationVar] {
            self.permutation_values
        }
    }

    impl<'a> PeriodicAirBuilder for DummyBuilder<'a> {
        type PeriodicVar = F;

        fn periodic_values(&self) -> &[Self::PeriodicVar] {
            self.periodic_values
        }
    }

    fn build_builder(
        main_len: usize,
        aux_len: usize,
        public_len: usize,
        randomness_len: usize,
        aux_values_len: usize,
        periodic_len: usize,
    ) -> DummyBuilder<'static> {
        let main_current = Box::leak(vec![F::ZERO; main_len].into_boxed_slice());
        let main_next = Box::leak(vec![F::ZERO; main_len].into_boxed_slice());
        let aux_current = Box::leak(vec![EF::ZERO; aux_len].into_boxed_slice());
        let aux_next = Box::leak(vec![EF::ZERO; aux_len].into_boxed_slice());
        let pre_current: &'static [F] = &[];
        let pre_next: &'static [F] = &[];
        let public_values = Box::leak(vec![F::ZERO; public_len].into_boxed_slice());
        let randomness = Box::leak(vec![EF::ZERO; randomness_len].into_boxed_slice());
        let permutation_values = Box::leak(vec![EF::ZERO; aux_values_len].into_boxed_slice());
        let periodic_values = Box::leak(vec![F::ZERO; periodic_len].into_boxed_slice());

        DummyBuilder {
            main: RowWindow::from_two_rows(main_current, main_next),
            preprocessed: RowWindow::from_two_rows(pre_current, pre_next),
            aux: RowWindow::from_two_rows(aux_current, aux_next),
            public_values,
            randomness,
            permutation_values,
            periodic_values,
        }
    }

    #[test]
    fn is_valid_builder_accepts_matching_dimensions() {
        let air = DummyAir {
            width: 4,
            aux_width: 3,
            num_public_values: 2,
            num_randomness: 2,
            num_aux_values: 1,
            periodic: vec![vec![F::ZERO; 2], vec![F::ONE; 2]],
        };

        let builder = build_builder(4, 3, 2, 2, 1, 2);
        air.is_valid_builder(&builder)
            .expect("builder should validate");
    }

    #[test]
    fn is_valid_builder_reports_mismatched_dimensions() {
        let air = DummyAir {
            width: 4,
            aux_width: 3,
            num_public_values: 2,
            num_randomness: 2,
            num_aux_values: 1,
            periodic: vec![vec![F::ZERO; 2], vec![F::ONE; 2]],
        };

        let cases = [
            (TracePart::Main, 3, 3, 2, 2, 1, 2),
            (TracePart::Aux, 4, 2, 2, 2, 1, 2),
            (TracePart::PublicValues, 4, 3, 1, 2, 1, 2),
            (TracePart::Randomness, 4, 3, 2, 3, 1, 2),
            (TracePart::AuxValues, 4, 3, 2, 2, 2, 2),
            (TracePart::PeriodicValues, 4, 3, 2, 2, 1, 3),
        ];

        for (part, main_len, aux_len, public_len, randomness_len, aux_values_len, periodic_len) in
            cases
        {
            let builder = build_builder(
                main_len,
                aux_len,
                public_len,
                randomness_len,
                aux_values_len,
                periodic_len,
            );
            let err = air
                .is_valid_builder(&builder)
                .expect_err("expected builder mismatch");
            match err {
                AirValidationError::BuilderMismatch {
                    part: actual_part,
                    expected,
                    actual,
                } => {
                    assert!(
                        core::mem::discriminant(&actual_part) == core::mem::discriminant(&part)
                    );
                    match part {
                        TracePart::Main => {
                            assert_eq!(expected, air.width);
                            assert_eq!(actual, main_len);
                        }
                        TracePart::Aux => {
                            assert_eq!(expected, air.aux_width);
                            assert_eq!(actual, aux_len);
                        }
                        TracePart::PublicValues => {
                            assert_eq!(expected, air.num_public_values);
                            assert_eq!(actual, public_len);
                        }
                        TracePart::Randomness => {
                            assert_eq!(expected, air.num_randomness);
                            assert_eq!(actual, randomness_len);
                        }
                        TracePart::AuxValues => {
                            assert_eq!(expected, air.num_aux_values);
                            assert_eq!(actual, aux_values_len);
                        }
                        TracePart::PeriodicValues => {
                            assert_eq!(expected, air.periodic.len());
                            assert_eq!(actual, periodic_len);
                        }
                    }
                }
                _ => panic!("unexpected error: {err:?}"),
            }
        }
    }
}
