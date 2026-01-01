//! # iterative_solvers_smpl
//!
//! A small, dependency-light collection of iterative linear solvers and preconditioners,
//! written in Rust and built on top of [`extended_matrix`] for CSR storage.
//!
//! This crate started as part of a larger *finite-element-analysis* (FEA) learning project,
//! where the focus was on understanding:
//! - sparse matrix formats (CSR),
//! - iterative methods (PCG),
//! - preconditioners (Jacobi and block-Jacobi),
//! - and the practical “plumbing” required to run these methods in real code.
//!
//! The implementations here are intentionally straightforward and readable, with
//! explicit data flow and plenty of comments—good for learning, experiments, and
//! as a base for further optimization.
//!
//! ## What’s included
//! - **PCG (Preconditioned Conjugate Gradient)** for SPD systems in CSR form
//! - **Jacobi** preconditioner (diagonal scaling)
//! - **Block-Jacobi** preconditioner (small dense LU per block; currently specialized to 6×6 blocks)
//! - A tiny `linalg` module (`dot`, `axpy`, `scale`) used by the solvers
//!
//! ## What’s *not* included (yet)
//! - Pivoting, reordering, or advanced preconditioners (ILU, AMG, etc.)
//! - GPU backends (those live in the companion WebGPU/WASM repos)
//!
//! See the repository README for a quick tour and examples.

pub mod block_jacobi;
pub mod jacobi;
pub mod linalg;
pub mod pcg;
mod tests;

pub use jacobi::JacobiPreconditioner;
pub use pcg::{pcg_block_jacobi_csr, pcg_jacobi_csr};
