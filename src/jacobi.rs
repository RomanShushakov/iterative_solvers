//! Jacobi (diagonal) preconditioner.
//!
//! The Jacobi preconditioner uses only the diagonal of `A`:
//!
//! ```text
//! M = diag(A)
//! z = M^{-1} r
//! ```
//!
//! For SPD matrices this is often a reasonable baseline preconditioner.
//! It is cheap to build and apply, but may converge slowly for ill-conditioned
//! problems.
//!
use extended_matrix::{CsrMatrix, FloatTrait};

/// Jacobi preconditioner for a CSR matrix: M = diag(A)
///
/// We store diag_inv[i] = 1 / A_ii.
/// Applying M^{-1} r is then: z[i] = diag_inv[i] * r[i].
#[derive(Clone, Debug)]
pub struct JacobiPreconditioner<V> {
    diag_inv: Vec<V>,
}

impl<V> JacobiPreconditioner<V>
where
    V: FloatTrait<Output = V> + Copy,
{
    pub fn create(a: &CsrMatrix<V>) -> Result<Self, String> {
        let n = a.get_n_rows();
        if a.get_n_rows() != a.get_n_cols() {
            return Err("JacobiPreconditioner::create: matrix is not square".to_string());
        }

        let mut diag_inv = vec![V::from(0.0_f32); n];

        for i in 0..n {
            let row_start = a.get_row_ptr()[i];
            let row_end = a.get_row_ptr()[i + 1];

            let mut diag_val_opt: Option<V> = None;

            for idx in row_start..row_end {
                let j = a.get_col_index()[idx];
                if j == i {
                    diag_val_opt = Some(a.get_values()[idx]);
                    break;
                }
            }

            let diag_val = diag_val_opt.ok_or_else(|| {
                format!(
                    "JacobiPreconditioner::create: no diagonal entry for row {}",
                    i
                )
            })?;

            // You may want checks here for zero/negative diag_val if needed.
            diag_inv[i] = V::from(1.0_f32) / diag_val;
        }

        Ok(Self { diag_inv })
    }

    /// z <- M^{-1} r   (element-wise scaling)
    pub fn apply(&self, r: &[V], z: &mut [V]) -> Result<(), String>
    where
        V: FloatTrait<Output = V> + Copy,
    {
        if r.len() != self.diag_inv.len() || z.len() != self.diag_inv.len() {
            return Err(format!(
                "JacobiPreconditioner::apply: length mismatch: r = {}, z = {}, n = {}",
                r.len(),
                z.len(),
                self.diag_inv.len()
            ));
        }

        for i in 0..r.len() {
            z[i] = self.diag_inv[i] * r[i];
        }

        Ok(())
    }
}
