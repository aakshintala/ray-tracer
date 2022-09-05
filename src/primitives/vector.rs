use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

use super::point::Point;
use crate::utils::approx_eq;

const VECTOR_W: f64 = 0.0;

#[derive(Clone, Copy, Debug)]
pub struct Vector {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vector {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Vector {
            x,
            y,
            z,
            w: VECTOR_W,
        }
    }

    pub const fn zero() -> Self {
        Vector {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: VECTOR_W,
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let magnitude = self.magnitude();
        Vector::new(self.x / magnitude, self.y / magnitude, self.z / magnitude)
    }

    pub fn dot_product(&self, rhs: &Vector) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross_product(&self, rhs: &Vector) -> Vector {
        Vector::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        approx_eq(self.x, other.x) && approx_eq(self.y, other.y) && approx_eq(self.z, other.z)
    }
}

impl Add<&Vector> for Vector {
    type Output = Vector;

    fn add(self, rhs: &Vector) -> Self::Output {
        Vector::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Add<&Point> for Vector {
    type Output = Point;

    fn add(self, rhs: &Point) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub<&Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: &Vector) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Neg for Vector {
    type Output = Vector;

    fn neg(self) -> Self::Output {
        Vector::new(0.0 - self.x, 0.0 - self.y, 0.0 - self.z)
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        Vector::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vector {
    type Output = Vector;

    fn div(self, rhs: f64) -> Self::Output {
        Vector::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => unreachable!(),
        }
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut f64 {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_init_correct() {
        let vector = Vector::new(4.0, -4.0, 3.0);
        assert_eq!(vector.x, 4.0);
        assert_eq!(vector.y, -4.0);
        assert_eq!(vector.z, 3.0);
        assert_eq!(vector.w, VECTOR_W);
    }

    #[test]
    fn vector_init_zero_correct() {
        let vector = Vector::zero();
        assert_eq!(vector.x, 0.0);
        assert_eq!(vector.y, 0.0);
        assert_eq!(vector.z, 0.0);
        assert_eq!(vector.w, VECTOR_W);
    }

    #[test]
    fn vector_equality_zero() {
        let vector1 = Vector::zero();
        let vector2 = Vector::zero();
        assert_eq!(vector1, vector2);
    }

    #[test]
    fn vector_equality_new() {
        let vector1 = Vector::new(4.0, -4.0, 3.0);
        let vector2 = Vector::new(4.0, -4.0, 3.0);
        assert_eq!(vector1, vector2);
    }

    #[test]
    fn add_vector_and_vector() {
        let vector1 = Vector::new(4.0, -4.0, 3.0);
        let vector2 = Vector::new(1.0, -1.0, 1.0);
        let result = Vector::new(5.0, -5.0, 4.0);
        assert_eq!(vector1 + &vector2, result);
    }

    #[test]
    fn add_vector_and_point() {
        let vector = Vector::new(4.0, -4.0, 3.0);
        let point = Point::new(1.0, -1.0, 1.0);
        let result = Point::new(5.0, -5.0, 4.0);
        assert_eq!(vector + &point, result);
    }

    #[test]
    fn sub_vector_and_vector() {
        let vector1 = Vector::new(4.0, -4.0, 3.0);
        let vector2 = Vector::new(4.0, -4.0, 3.0);
        let result = Vector::zero();
        assert_eq!(vector1 - &vector2, result);
    }

    #[test]
    fn neg_vector() {
        let vector = Vector::new(4.0, -4.0, 3.0);
        let neg_vector = Vector::new(-4.0, 4.0, -3.0);
        assert_eq!(-vector, neg_vector);
    }

    #[test]
    fn multiply_by_scalar() {
        let vector = Vector::new(-2.0, 2.0, -1.5);
        let scaled_vector = Vector::new(-4.0, 4.0, -3.0);
        assert_eq!(vector * 2.0, scaled_vector);
    }

    #[test]
    fn divide_by_scalar() {
        let vector = Vector::new(4.0, -4.0, 3.0);
        let scaled_vector = Vector::new(2.0, -2.0, 1.5);
        assert_eq!(vector / 2.0, scaled_vector);
    }

    #[test]
    fn magnitude_unit() {
        let x_unit = Vector::new(1.0, 0.0, 0.0);
        assert_eq!(x_unit.magnitude(), 1.0);
        let y_unit = Vector::new(0.0, 1.0, 0.0);
        assert_eq!(y_unit.magnitude(), 1.0);
        let z_unit = Vector::new(0.0, 0.0, 1.0);
        assert_eq!(z_unit.magnitude(), 1.0);
    }

    #[test]
    fn magnitude_non_unit() {
        let vector = Vector::new(1.0, 2.0, 3.0);
        let expected: f64 = 14.0;
        assert_eq!(vector.magnitude(), expected.sqrt());

        let vector = Vector::new(-1.0, -2.0, -3.0);
        let expected: f64 = 14.0;
        assert_eq!(vector.magnitude(), expected.sqrt());
    }

    #[test]
    fn normalize1() {
        let vector = Vector::new(1.0, 2.0, 3.0);
        let magnitude = vector.magnitude();
        let expected = Vector::new(1.0 / magnitude, 2.0 / magnitude, 3.0 / magnitude);
        assert_eq!(vector.normalize(), expected);
    }

    #[test]
    fn normalize2() {
        let vector = Vector::new(4.0, 0.0, 0.0);
        let expected = Vector::new(1.0, 0.0, 0.0);
        assert_eq!(vector.normalize(), expected);
    }

    #[test]
    fn magnitude_of_normalized_vector() {
        let vector = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(vector.normalize().magnitude(), 1.0);
    }

    #[test]
    fn dot_product() {
        let vector1 = Vector::new(1.0, 2.0, 3.0);
        let vector2 = Vector::new(2.0, 3.0, 4.0);
        assert_eq!(vector1.dot_product(&vector2), 20.0);
    }

    #[test]
    fn cross_product() {
        let vector1 = Vector::new(1.0, 2.0, 3.0);
        let vector2 = Vector::new(2.0, 3.0, 4.0);
        assert_eq!(
            vector1.cross_product(&vector2),
            Vector::new(-1.0, 2.0, -1.0)
        );

        assert_eq!(vector2.cross_product(&vector1), Vector::new(1.0, -2.0, 1.0))
    }
}
