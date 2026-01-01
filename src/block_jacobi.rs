//! Block-Jacobi preconditioner.
//!
//! Block-Jacobi generalizes Jacobi by replacing single diagonal entries with
//! small dense blocks on the diagonal:
//!
//! ```text
//! A ≈ block_diag(A_0, A_1, ..., A_{B-1})
//! M^{-1} r = solve(A_b, r_b) for each block b
//! ```
//!
//! In this crate the CPU-side preconditioner is used both for pure-CPU PCG,
//! and (in the WebGPU project) as a reference when building the GPU block data.
//!
//! **Important:** the current implementation is specialized to **6×6** blocks
//! (matching the GPU kernel used in the companion repo). Smaller “tail blocks”
//! at the end are supported by solving only the leading `m×m` portion.
//!
use extended_matrix::{CsrMatrix, FloatTrait};

#[derive(Clone, Debug)]
struct Block<V> {
    offset: usize, // starting row/col index in K_aa
    size: usize,   // block size
    lu: Vec<V>,    // LU factorization, row-major (size x size)
}

/// Block Jacobi preconditioner for a CSR matrix with variable block sizes.
///
/// Blocks are defined by `block_starts`, where
///   block i is rows/cols [ block_starts[i] .. block_starts[i+1] )
#[derive(Clone, Debug)]
pub struct BlockJacobiPreconditioner<V> {
    n: usize,
    blocks: Vec<Block<V>>,
}

impl<V> BlockJacobiPreconditioner<V>
where
    V: FloatTrait<Output = V> + Copy,
{
    /// Extract dense block A_block from CSR and factor it with LU (no pivoting).
    fn extract_block_and_factor(
        a: &CsrMatrix<V>,
        offset: usize,
        size: usize,
    ) -> Result<Vec<V>, String> {
        let zero = V::from(0.0_f32);
        let mut mat = vec![zero; size * size];

        // mat[i_local, j_local] = A[offset + i_local, offset + j_local]
        for i_local in 0..size {
            let i = offset + i_local;
            let row_start = a.get_row_ptr()[i];
            let row_end = a.get_row_ptr()[i + 1];

            for idx in row_start..row_end {
                let j = a.get_col_index()[idx];
                if j >= offset && j < offset + size {
                    let j_local = j - offset;
                    mat[i_local * size + j_local] = a.get_values()[idx];
                }
            }
        }

        Self::lu_factor(&mut mat, size)?;
        Ok(mat)
    }

    /// In-place LU factorization (no pivoting) for small dense matrices.
    fn lu_factor(mat: &mut [V], n: usize) -> Result<(), String> {
        let zero = V::from(0.0_f32);

        for k in 0..n {
            let a_kk = mat[k * n + k];
            if a_kk == zero {
                return Err(format!(
                    "BlockJacobiPreconditioner::lu_factor: zero pivot at k = {}",
                    k
                ));
            }

            // L(i,k) for i > k
            for i in (k + 1)..n {
                mat[i * n + k] = mat[i * n + k] / a_kk;
            }

            // Update trailing submatrix
            for i in (k + 1)..n {
                let l_ik = mat[i * n + k];
                if l_ik != zero {
                    for j in (k + 1)..n {
                        let a_kj = mat[k * n + j];
                        mat[i * n + j] = mat[i * n + j] - l_ik * a_kj;
                    }
                }
            }
        }

        Ok(())
    }

    /// Solve LU x = b, given LU in `mat` (row-major n x n).
    fn lu_solve(mat: &[V], n: usize, b: &[V], x: &mut [V]) -> Result<(), String> {
        let zero = V::from(0.0_f32);

        if b.len() != n || x.len() != n {
            return Err(format!(
                "BlockJacobiPreconditioner::lu_solve: length mismatch: b = {}, x = {}, n = {}",
                b.len(),
                x.len(),
                n
            ));
        }

        let mut y = vec![zero; n];

        // Forward: L y = b (L has unit diagonal)
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum = sum - mat[i * n + j] * y[j];
            }
            y[i] = sum;
        }

        // Backward: U x = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum = sum - mat[i * n + j] * x[j];
            }
            let u_ii = mat[i * n + i];
            x[i] = sum / u_ii;
        }

        Ok(())
    }

    /// Build Block Jacobi preconditioner from CSR matrix `a`.
    ///
    /// `block_starts` must be:
    ///   - sorted ascending,
    ///   - first element = 0,
    ///   - last element <= n_rows (if < n_rows, a final block [last..n_rows) is added).
    /// Build a block-Jacobi preconditioner by extracting diagonal blocks from a CSR matrix.
    ///
    /// `block_starts` defines block boundaries (length `num_blocks + 1`).
    /// Each block is factorized with an in-place LU (no pivoting).
    pub fn create_from_csr_with_blocks(
        a: &CsrMatrix<V>,
        block_starts: &[usize],
    ) -> Result<Self, String> {
        let n = a.get_n_rows();
        if a.get_n_cols() != n {
            return Err(
                "BlockJacobiPreconditioner::create_from_csr_with_blocks: matrix is not square"
                    .to_string(),
            );
        }
        if block_starts.is_empty() {
            return Err(
                "BlockJacobiPreconditioner::create_from_csr_with_blocks: block_starts is empty"
                    .to_string(),
            );
        }
        if block_starts[0] != 0 {
            return Err(
                "BlockJacobiPreconditioner::create_from_csr_with_blocks: first block_start must be 0"
                    .to_string(),
            );
        }
        for w in block_starts.windows(2) {
            if w[0] > w[1] {
                return Err(
                    "BlockJacobiPreconditioner::create_from_csr_with_blocks: block_starts must be sorted"
                        .to_string(),
                );
            }
        }

        let mut blocks: Vec<Block<V>> = Vec::new();

        // blocks i: [block_starts[i] .. block_starts[i+1])
        for i in 0..(block_starts.len() - 1) {
            let offset = block_starts[i];
            let next = block_starts[i + 1];
            if offset >= n || next > n {
                return Err(format!(
                    "BlockJacobiPreconditioner::create_from_csr_with_blocks: block range out of bounds: [{}, {}) with n={}",
                    offset, next, n
                ));
            }
            let size = next - offset;
            if size == 0 {
                continue;
            }
            let lu = Self::extract_block_and_factor(a, offset, size)?;
            blocks.push(Block { offset, size, lu });
        }

        // Trailing block if needed
        let last_start = *block_starts.last().unwrap();
        if last_start < n {
            let offset = last_start;
            let size = n - offset;
            let lu = Self::extract_block_and_factor(a, offset, size)?;
            blocks.push(Block { offset, size, lu });
        }

        Ok(Self { n, blocks })
    }

    /// Apply block Jacobi preconditioner: z <- M^{-1} r
    pub fn apply(&self, r: &[V], z: &mut [V]) -> Result<(), String> {
        if r.len() != self.n || z.len() != self.n {
            return Err(format!(
                "BlockJacobiPreconditioner::apply: length mismatch: r = {}, z = {}, n = {}",
                r.len(),
                z.len(),
                self.n
            ));
        }

        let zero = V::from(0.0_f32);
        for zi in z.iter_mut() {
            *zi = zero;
        }

        for block in &self.blocks {
            let off = block.offset;
            let m = block.size;

            let r_block = &r[off..off + m];
            let z_block = &mut z[off..off + m];

            Self::lu_solve(&block.lu, m, r_block, z_block)?;
        }

        Ok(())
    }
}
