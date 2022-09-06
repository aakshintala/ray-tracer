#![allow(dead_code)]

mod canvas;
mod primitives;
mod utils;

pub use canvas::{Canvas, Color};
pub use primitives::{Matrix, Point, Vector};
pub use utils::clamp;
