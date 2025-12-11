use extended_matrix::{CsrMatrix, FloatTrait};

/// Incomplete Cholesky (IC(0)) preconditioner built from a CSR matrix.
///
/// Internally stores a dense lower-triangular factor L (n x n),
/// but only uses the sparsity pattern of A when computing entries.
/// This is a simple, CPU-only IC(0) suitable for small/medium systems.
#[derive(Clone, Debug)]
pub struct Ichol0Preconditioner<V> {
    n: usize,
    /// Row-major dense storage, only lower triangle + diag are used.
    /// L[i,j] (i >= j) is stored at index i*n + j.
    l: Vec<V>,
}

impl<V> Ichol0Preconditioner<V>
where
    V: FloatTrait<Output = V> + Copy + PartialOrd,
{
    /// Build IC(0) from a symmetric positive definite CSR matrix A.
    pub fn new_from_csr(a: &CsrMatrix<V>) -> Result<Self, String> {
        let n = a.n_rows;
        if a.n_cols != n {
            return Err("Ichol0Preconditioner::new_from_csr: matrix is not square".to_string());
        }

        let zero = V::from(0.0_f32);
        let mut l = vec![zero; n * n];

        // 1) Build a symmetric dense copy of A into l.
        //
        //    l starts as A, then we overwrite it in-place with the IC(0) factor L.
        for i in 0..n {
            let row_start = a.row_ptr[i];
            let row_end = a.row_ptr[i + 1];

            for idx in row_start..row_end {
                let j = a.col_index[idx];
                let val = a.values[idx];

                // A(i,j)
                l[i * n + j] = val;
                // assume A is symmetric; keep it symmetric in dense form.
                l[j * n + i] = val;
            }
        }

        // 2) Incomplete Cholesky factorization:
        //    Only compute L(i,k) where A(i,k) != 0 (IC(0) pattern).
        //
        //    Algorithm:
        //    for k = 0..n-1:
        //       L[k,k] = sqrt( A[k,k] - sum_{p<k} L[k,p]^2 )
        //       for i = k+1..n-1:
        //           if A[i,k] != 0:
        //               L[i,k] = ( A[i,k] - sum_{p<k} L[i,p]*L[k,p] ) / L[k,k]
        //
        //    Note: positions with A(i,k) == 0 keep L(i,k) = 0 -> IC(0).

        // Small diagonal shift to avoid breakdown if the matrix is not
        // perfectly SPD numerically.
        let shift = V::from(1.0e-8_f32);

        for k in 0..n {
            // diag
            let mut diag = l[k * n + k];
            for p in 0..k {
                let l_kp = l[k * n + p];
                diag = diag - l_kp * l_kp;
            }

            if diag <= zero {
                // add a small safety shift
                diag = diag + shift;
            }

            let diag_sqrt = diag.my_sqrt();
            l[k * n + k] = diag_sqrt;

            // below-diagonal entries in column k
            for i in (k + 1)..n {
                let mut val = l[i * n + k]; // this currently holds A(i,k)

                if val != zero {
                    for p in 0..k {
                        let l_ip = l[i * n + p];
                        let l_kp = l[k * n + p];
                        val = val - l_ip * l_kp;
                    }
                    l[i * n + k] = val / diag_sqrt;
                } else {
                    // No entry in A(i,k) => keep zero (IC(0))
                    // l[i * n + k] = 0;
                }
            }

            // (optional) clear upper triangle in row k
            for j in (k + 1)..n {
                l[k * n + j] = zero;
            }
        }

        Ok(Self { n, l })
    }

    /// Apply preconditioner: z <- M^{-1} r, where M = L L^T.
    ///
    /// Implemented via:
    ///   1) Forward solve   L y = r
    ///   2) Backward solve  L^T z = y
    pub fn apply(&self, r: &[V], z: &mut [V]) -> Result<(), String> {
        if r.len() != self.n || z.len() != self.n {
            return Err(format!(
                "Ichol0Preconditioner::apply: length mismatch: r = {}, z = {}, n = {}",
                r.len(),
                z.len(),
                self.n
            ));
        }

        let n = self.n;
        let l = &self.l;
        let zero = V::from(0.0_f32);

        let mut y = vec![zero; n];

        // Forward: L y = r
        for i in 0..n {
            let mut sum = r[i];
            for j in 0..i {
                let lij = l[i * n + j];
                if lij != zero {
                    sum = sum - lij * y[j];
                }
            }
            let lii = l[i * n + i];
            y[i] = sum / lii;
        }

        // Backward: L^T z = y
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                let lji = l[j * n + i]; // L[j,i]
                if lji != zero {
                    sum = sum - lji * z[j];
                }
            }
            let lii = l[i * n + i];
            z[i] = sum / lii;
        }

        Ok(())
    }
}
