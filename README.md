# iterative_solvers

![Rust](https://img.shields.io/badge/Rust-stable-orange)
![Linear Algebra](https://img.shields.io/badge/Linear%20Algebra-iterative%20solvers-blue)
![CSR](https://img.shields.io/badge/Sparse%20Matrices-CSR-lightgrey)
![Status](https://img.shields.io/badge/status-learning%20%2F%20building-lightgrey)

A small Rust library implementing classic iterative methods for solving large linear systems.

This project grew out of a long-standing interest in numerical methods and linear algebra, shaped by earlier work with finite-element analysis (FEA) tools and large sparse systems. It is intentionally focused on clarity and correctness rather than feature completeness or aggressive optimization.

The crate aims to provide readable, well-structured reference implementations of iterative solvers that can be reused, extended, or studied.

---

## Overview

The library implements iterative methods for linear systems of the form:

```
A · x = b
```

with an emphasis on cases where the system is large or sparse and direct solvers are impractical.

The implementations follow standard textbook formulations closely, keeping the relationship between the mathematical algorithm and the code explicit.

---

## Implemented methods

Currently, the crate includes implementations of:

- **Conjugate Gradient (CG)**  
  For symmetric positive definite systems.

- **Preconditioned Conjugate Gradient (PCG)**  
  With explicitly modeled preconditioning steps.

- **Jacobi and Gauss–Seidel–style iterations**

- **Block-based variants**, where applicable, to reflect typical FEM-style decompositions.

The focus is on solver structure, iteration flow, and convergence logic rather than on specialized matrix formats or hardware acceleration.

---

## Design principles

This crate is guided by a few core principles:

- **Algorithmic transparency**  
  The code mirrors the mathematical formulation of each method as directly as possible.

- **Minimal abstraction**  
  Control flow and data movement are explicit; there is no heavy framework or hidden indirection.

- **Numerical correctness first**  
  Priority is given to correctness, stability, and clear handling of convergence and breakdown conditions.

- **Composable components**  
  Vector operations, preconditioners, and stopping criteria are kept modular where reasonable.

These choices make the crate suitable both for experimentation and as a reference implementation.

---

## Intended use

This library may be useful if you are:

- experimenting with iterative solvers or preconditioners,
- learning or teaching numerical linear algebra,
- building small research or engineering tools involving linear systems,
- prototyping solver logic before integrating with more specialized backends.

It is **not** intended to replace highly optimized or production-grade solver libraries.

---

## Quick example

```rust
use extended_matrix::CsrMatrix;
use iterative_solvers_smpl::pcg::pcg_block_jacobi_csr;

// Build or load A in CSR form (example code omitted here)
let a: CsrMatrix<f32> = /* ... */;

// Right-hand side and initial guess
let b: Vec<f32> = /* ... */;
let mut x = vec![0.0; a.get_n_rows()];

// Block boundaries (num_blocks + 1 entries).
// Example: uniform 6-sized blocks:
let mut block_starts = Vec::new();
let n = a.get_n_rows();
let mut i = 0;
while i < n {
    block_starts.push(i);
    i += 6;
}
block_starts.push(n);

// Solve
let iters = pcg_block_jacobi_csr(
    &a,
    &b,
    &mut x,
    200,    // max_iter
    1e-6,   // rel_tol
    1e-12,  // abs_tol
    &block_starts,
).expect("PCG failed");

println!("Converged in {iters} iterations");
```

See the crate documentation for concrete APIs and usage patterns.

---

## Project context

This crate is part of a broader personal effort to better understand numerical algorithms end-to-end — from linear algebra and solver theory to concrete implementations in Rust.

Related projects explore similar ideas in GPU-accelerated and WebAssembly/WebGPU contexts, but this library deliberately stays CPU-centric and straightforward.

---

## Status

This is a **stable, experimental library**:

- APIs may evolve,
- additional solvers or preconditioners may be added,
- performance tuning is secondary to clarity and correctness.

Feedback and discussion are welcome.

---

## License

MIT
