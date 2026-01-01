//! Lightweight linear-algebra helpers.
//!
//! These are tiny building blocks used by the iterative solvers.
//! They are deliberately “low ceremony” (plain slices) so they can be reused
//! in CPU code and in contexts where you may later swap the backend.
//!
//! Conventions:
//! - Vectors are `&[f32]` / `&mut [f32]`
//! - Length mismatches are treated as errors (returned as `Result`)
//! - Operations are written in a BLAS-like style (`axpy`, `dot`, `scale`)
//!
//! **Tip:** if you later want SIMD or threaded execution, this file is the
//! natural place to introduce it behind feature flags.
//!
use extended_matrix::FloatTrait;

/// Dot product: <x, y>
pub fn dot<V>(x: &[V], y: &[V]) -> Result<V, String>
where
    V: FloatTrait<Output = V> + Copy,
{
    if x.len() != y.len() {
        return Err(format!(
            "dot: length mismatch: x = {}, y = {}",
            x.len(),
            y.len()
        ));
    }

    let mut s = V::from(0.0_f32);
    for i in 0..x.len() {
        s = s + x[i] * y[i];
    }
    Ok(s)
}

/// y <- y + alpha * x
pub fn axpy<V>(y: &mut [V], alpha: V, x: &[V]) -> Result<(), String>
where
    V: FloatTrait<Output = V> + Copy,
{
    if x.len() != y.len() {
        return Err(format!(
            "axpy: length mismatch: x = {}, y = {}",
            x.len(),
            y.len()
        ));
    }

    for i in 0..x.len() {
        y[i] = y[i] + alpha * x[i];
    }

    Ok(())
}

/// x <- alpha * x
pub fn scale<V>(x: &mut [V], alpha: V) -> Result<(), String>
where
    V: FloatTrait<Output = V> + Copy,
{
    for xi in x.iter_mut() {
        *xi = *xi * alpha;
    }
    Ok(())
}
