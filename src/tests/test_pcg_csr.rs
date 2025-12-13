#![allow(unused_imports)]

use crate::{pcg_block_jacobi_csr, pcg_jacobi_csr};
use extended_matrix::{BasicOperationsTrait, CsrMatrix, Position, SquareMatrix};

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

fn mat4_blockdiag_2x2() -> SquareMatrix<f64> {
    // A = blockdiag([[4,1],[1,3]], [[2,0],[0,5]])
    let mut m = SquareMatrix::create(
        4,
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );

    *m.get_mut_element_value(&Position(0, 0)).unwrap() = 4.0;
    *m.get_mut_element_value(&Position(0, 1)).unwrap() = 1.0;
    *m.get_mut_element_value(&Position(1, 0)).unwrap() = 1.0;
    *m.get_mut_element_value(&Position(1, 1)).unwrap() = 3.0;

    *m.get_mut_element_value(&Position(2, 2)).unwrap() = 2.0;
    *m.get_mut_element_value(&Position(3, 3)).unwrap() = 5.0;

    m
}

#[test]
fn test_pcg_block_jacobi_4x4_block_2x2() {
    let a_dense = mat4_blockdiag_2x2();
    let a_csr = CsrMatrix::from_square_matrix(&a_dense).unwrap();

    let b = vec![1.0_f64, 2.0_f64, 3.0_f64, 4.0_f64];
    let mut x = vec![0.0_f64; 4];

    let max_iter = 20;
    let rel_tol = 1e-12_f64;
    let abs_tol = 1e-12_f64;

    // Two blocks: [0..2) and [2..4)
    let block_starts = vec![0usize, 2usize, 4usize];

    let iters = pcg_block_jacobi_csr(
        &a_csr,
        &b,
        &mut x,
        max_iter,
        rel_tol,
        abs_tol,
        &block_starts,
    )
    .unwrap();

    println!("PCG(BlockJacobi) converged in {} iterations", iters);

    // Check that A x ≈ b
    let y = a_csr.spmv(&x).unwrap();
    for i in 0..4 {
        assert!((y[i] - b[i]).abs() < ABS_TOL);
    }
}

#[test]
fn test_pcg_block_jacobi_1x1_degenerate() {
    // A = [10], b = [5] → x = [0.5]
    let mut m = SquareMatrix::create(1, &[1.0]);
    *m.get_mut_element_value(&Position(0, 0)).unwrap() = 10.0;
    let a_csr = CsrMatrix::from_square_matrix(&m).unwrap();

    let b = vec![5.0_f64];
    let mut x = vec![0.0_f64];

    let max_iter = 10;
    let rel_tol = 1e-12_f64;
    let abs_tol = 1e-12_f64;

    let block_starts = vec![0usize, 1usize]; // single 1x1 block

    let iters = pcg_block_jacobi_csr(
        &a_csr,
        &b,
        &mut x,
        max_iter,
        rel_tol,
        abs_tol,
        &block_starts,
    )
    .unwrap();

    println!("PCG(BlockJacobi, 1x1) converged in {} iterations", iters);
    assert!((x[0] - 0.5).abs() < ABS_TOL);
}
