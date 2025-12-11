pub mod jacobi;
pub mod linalg;
pub mod pcg;
pub mod ic;
mod tests;

pub use jacobi::JacobiPreconditioner;
pub use pcg::pcg_jacobi_csr;
