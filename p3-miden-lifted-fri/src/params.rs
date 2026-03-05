//! PCS parameters.

use crate::deep::DeepParams;
use crate::fri::FriParams;

/// Complete PCS parameters combining DEEP and FRI parameters.
///
/// Groups all parameters needed for `open` and `verify` into a single struct,
/// reducing the number of function arguments and ensuring consistent configuration.
#[derive(Clone, Copy, Debug)]
pub struct PcsParams {
    /// DEEP quotient parameters.
    pub deep: DeepParams,

    /// FRI protocol parameters.
    pub fri: FriParams,

    /// Number of query repetitions.
    num_queries: usize,

    /// Grinding bits before query index sampling.
    query_pow_bits: u8,
}

impl PcsParams {
    /// Create new PCS parameters.
    ///
    /// # Panics
    ///
    /// Panics if `num_queries == 0`.
    pub fn new(deep: DeepParams, fri: FriParams, num_queries: usize, query_pow_bits: u8) -> Self {
        assert!(num_queries > 0, "num_queries must be greater than 0");
        Self {
            deep,
            fri,
            num_queries,
            query_pow_bits,
        }
    }

    /// Number of query repetitions.
    #[inline]
    pub fn num_queries(&self) -> usize {
        self.num_queries
    }

    /// Grinding bits before query index sampling.
    #[inline]
    pub fn query_pow_bits(&self) -> usize {
        self.query_pow_bits as usize
    }
}
