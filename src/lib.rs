#![allow(dead_code)]

mod canvas;
mod primitives;
mod ray;
mod sphere;
mod utils;

pub use canvas::{Canvas, Color};
pub use primitives::{Matrix, Point, Vector};
pub use ray::Ray;
pub use sphere::Sphere;
pub use utils::clamp;
