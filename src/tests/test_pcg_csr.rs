#![allow(unused_imports)]

use extended_matrix::{SquareMatrix, Position, BasicOperationsTrait, CsrMatrix};
use crate::{pcg::pcg_ichol0_csr, pcg_jacobi_csr};

const ABS_TOL: f64 = 1e-10;

fn mat2x2(a11: f64, a12: f64, a21: f64, a22: f64) -> SquareMatrix<f64> {
    SquareMatrix::create(2, &[a11, a12, a21, a22])
}

#[test]
fn test_pcg_jacobi_csr() {
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

#[test]
fn test_pcg_ic_csr() {
    let a_dense = mat2x2(4.0, 1.0, 1.0, 3.0);
    let a_csr = CsrMatrix::from_square_matrix(&a_dense).unwrap();

    let b = vec![1.0_f64, 2.0_f64];
    let mut x = vec![0.0_f64, 0.0_f64];

    let max_iter = 10;
    let rel_tol = 1e-12_f64;
    let abs_tol = 1e-12_f64;

    let iters_ic = pcg_ichol0_csr(&a_csr, &b, &mut x, max_iter, rel_tol, abs_tol).unwrap();
    println!("PCG(IC) converged in {} iterations", iters_ic);

    let x0_exact = 1.0_f64 / 11.0_f64;
    let x1_exact = 7.0_f64 / 11.0_f64;

    assert!((x[0] - x0_exact).abs() < ABS_TOL);
    assert!((x[1] - x1_exact).abs() < ABS_TOL);
}
