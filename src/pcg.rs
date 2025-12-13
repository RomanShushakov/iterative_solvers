use extended_matrix::{CsrMatrix, FloatTrait};

use crate::block_jacobi::BlockJacobiPreconditioner;
use crate::jacobi::JacobiPreconditioner;
use crate::linalg::{axpy, dot, scale};

/// PCG with Jacobi preconditioner on a CSR matrix.
///
/// Solves A x = b for SPD A.
/// - `a`: CSR matrix
/// - `b`: right-hand side
/// - `x`: initial guess on input, solution on output
///
/// Returns Ok(iterations) on convergence.
pub fn pcg_jacobi_csr<V>(
    a: &CsrMatrix<V>,
    b: &[V],
    x: &mut [V],
    max_iter: usize,
    rel_tol: V,
    abs_tol: V,
) -> Result<usize, String>
where
    V: FloatTrait<Output = V> + Copy,
{
    let n = a.n_rows;
    if a.n_cols != n {
        return Err("PCG: matrix A is not square".to_string());
    }
    if b.len() != n || x.len() != n {
        return Err(format!(
            "PCG: dimension mismatch: A is {}x{}, b len {}, x len {}",
            a.n_rows,
            a.n_cols,
            b.len(),
            x.len()
        ));
    }

    // Build Jacobi preconditioner
    let mut m = JacobiPreconditioner::new(a)?;

    // r = b - A x
    let ax = a.spmv(x).map_err(|e| format!("PCG: A*x failed: {}", e))?;
    let mut r = vec![V::from(0.0_f32); n];
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }

    // z = M^{-1} r
    let mut z = vec![V::from(0.0_f32); n];
    m.apply(&r, &mut z)?;

    // p = z
    let mut p = z.clone();

    // Scalars
    let mut rz_old = dot(&r, &z)?;
    let b_norm2 = dot(b, b)?; // squared norm

    // Handle zero RHS
    if b_norm2 == V::from(0.0_f32) {
        // b = 0 => x ≈ 0 is solution
        return Ok(0);
    }

    let rel_tol2 = rel_tol * rel_tol;
    let abs_tol2 = abs_tol * abs_tol;

    let mut iterations = 0usize;

    for k in 0..max_iter {
        iterations = k + 1;

        // Ap = A p
        let ap = a.spmv(&p).map_err(|e| format!("PCG: A*p failed: {}", e))?;

        let p_ap = dot(&p, &ap)?;
        if p_ap == V::from(0.0_f32) {
            return Err("PCG: dot(p,Ap) is zero (breakdown)".to_string());
        }

        let alpha = rz_old / p_ap;

        // x = x + alpha p
        axpy(x, alpha, &p)?;

        // r = r - alpha Ap
        axpy(&mut r, alpha * V::from(-1f32), &ap)?;

        // Check convergence using squared norms
        let r_norm2 = dot(&r, &r)?;
        if r_norm2 <= abs_tol2 || r_norm2 <= rel_tol2 * b_norm2 {
            return Ok(iterations);
        }

        // z = M^{-1} r
        m.apply(&r, &mut z)?;

        let rz_new = dot(&r, &z)?;
        if rz_old == V::from(0.0_f32) {
            return Err("PCG: rz_old is zero (breakdown)".to_string());
        }

        let beta = rz_new / rz_old;

        // p = z + beta p
        scale(&mut p, beta)?;
        axpy(&mut p, V::from(1.0_f32), &z)?;

        rz_old = rz_new;
    }

    Err(format!("PCG: did not converge in {} iterations", max_iter))
}

/// PCG with Block Jacobi preconditioner on a CSR matrix.
///
/// Blocks are defined by `block_starts` in local K_aa indexing.
pub fn pcg_block_jacobi_csr<V>(
    a: &CsrMatrix<V>,
    b: &[V],
    x: &mut [V],
    max_iter: usize,
    rel_tol: V,
    abs_tol: V,
    block_starts: &[usize],
) -> Result<usize, String>
where
    V: FloatTrait<Output = V> + Copy,
{
    let n = a.n_rows;
    if a.n_cols != n {
        return Err("PCG(BlockJacobi): matrix A is not square".to_string());
    }
    if b.len() != n || x.len() != n {
        return Err(format!(
            "PCG(BlockJacobi): dimension mismatch: A is {}x{}, b len {}, x len {}",
            a.n_rows,
            a.n_cols,
            b.len(),
            x.len()
        ));
    }

    let m = BlockJacobiPreconditioner::new_from_csr_with_blocks(a, block_starts)
        .map_err(|e| format!("PCG(BlockJacobi): building preconditioner failed: {}", e))?;

    let zero = V::from(0.0_f32);

    // r = b - A x
    let ax = a
        .spmv(x)
        .map_err(|e| format!("PCG(BlockJacobi): A*x failed: {}", e))?;
    let mut r = vec![zero; n];
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }

    // z = M^{-1} r
    let mut z = vec![zero; n];
    m.apply(&r, &mut z)?;

    // p = z
    let mut p = z.clone();

    let mut rz_old = dot(&r, &z)?;
    let b_norm2 = dot(b, b)?;
    if b_norm2 == zero {
        return Ok(0);
    }

    let rel_tol2 = rel_tol * rel_tol;
    let abs_tol2 = abs_tol * abs_tol;
    let mut iterations = 0usize;

    for k in 0..max_iter {
        iterations = k + 1;

        let ap = a
            .spmv(&p)
            .map_err(|e| format!("PCG(BlockJacobi): A*p failed: {}", e))?;

        let p_ap = dot(&p, &ap)?;
        if p_ap == zero {
            return Err("PCG(BlockJacobi): dot(p,Ap) is zero (breakdown)".to_string());
        }

        let alpha = rz_old / p_ap;

        axpy(x, alpha, &p)?;
        axpy(&mut r, alpha * V::from(-1f32), &ap)?;

        let r_norm2 = dot(&r, &r)?;
        if r_norm2 <= abs_tol2 || r_norm2 <= rel_tol2 * b_norm2 {
            return Ok(iterations);
        }

        m.apply(&r, &mut z)?;
        let rz_new = dot(&r, &z)?;
        if rz_old == zero {
            return Err("PCG(BlockJacobi): rz_old is zero (breakdown)".to_string());
        }
        let beta = rz_new / rz_old;

        scale(&mut p, beta)?;
        axpy(&mut p, V::from(1.0_f32), &z)?;

        rz_old = rz_new;
    }

    Err(format!(
        "PCG(BlockJacobi): did not converge in {} iterations",
        max_iter
    ))
}
