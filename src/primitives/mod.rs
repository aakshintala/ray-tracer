pub mod point;
pub mod vector;

pub trait Tuple {
    fn new(x: f64, y: f64, z: f64) -> Self;
    fn zero() -> Self;
}

pub const EPSILON: f64 = 1e-6;

pub fn approx_eq(a: f64, b: f64) -> bool {
    (a - b) < EPSILON
}
