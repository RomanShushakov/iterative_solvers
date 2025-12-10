#![allow(unused_imports)]

use extended_matrix::{SquareMatrix, Position, BasicOperationsTrait, CsrMatrix};
use crate::pcg_jacobi_csr;

const ABS_TOL: f64 = 1e-10;

fn mat2x2(a11: f64, a12: f64, a21: f64, a22: f64) -> SquareMatrix<f64> {
    SquareMatrix::create(2, &[a11, a12, a21, a22])
}

#[test]
fn test_pcg_jacobi_csr_2x2() {
    // Same system as before:
    //
    // A = [[4, 1],
    //      [1, 3]]
    // b = [1, 2]^T
    // Exact solution: x = [1/11, 7/11]
    let a_dense = mat2x2(4.0, 1.0, 1.0, 3.0);
    let a_csr = CsrMatrix::from_square_matrix(&a_dense).unwrap();

    let b = vec![1.0_f64, 2.0_f64];
    let mut x = vec![0.0_f64, 0.0_f64];

    let max_iter = 100;
    let rel_tol = 1e-10_f64;
    let abs_tol = 1e-12_f64;

    let iters = pcg_jacobi_csr(&a_csr, &b, &mut x, max_iter, rel_tol, abs_tol).unwrap();
    println!("PCG CSR converged in {} iterations", iters);

    let x0_exact = 1.0_f64 / 11.0_f64;
    let x1_exact = 7.0_f64 / 11.0_f64;

    assert!((x[0] - x0_exact).abs() < ABS_TOL);
    assert!((x[1] - x1_exact).abs() < ABS_TOL);
}
