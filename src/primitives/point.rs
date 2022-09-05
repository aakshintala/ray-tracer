use std::ops::{Add, Index, IndexMut, Sub};

use super::vector::Vector;
use crate::utils::approx_eq;

const POINT_W: f64 = 1.0;
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Point {
            x,
            y,
            z,
            w: POINT_W,
        }
    }

    pub const fn zero() -> Self {
        Point::new(0.0, 0.0, 0.0)
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        approx_eq(self.x, other.x) && approx_eq(self.y, other.y) && approx_eq(self.z, other.z)
    }
}

impl Add<Vector> for Point {
    type Output = Point;

    fn add(self, rhs: Vector) -> Self::Output {
        Point::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub<Vector> for Point {
    type Output = Point;

    fn sub(self, rhs: Vector) -> Self::Output {
        Point::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Sub<Point> for Point {
    type Output = Vector;

    fn sub(self, rhs: Point) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Index<usize> for Point {
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

impl IndexMut<usize> for Point {
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
    fn point_init_correct() {
        let point = Point::new(4.0, -4.0, 3.0);
        assert_eq!(point.x, 4.0);
        assert_eq!(point.y, -4.0);
        assert_eq!(point.z, 3.0);
    }

    #[test]
    fn point_init_zero_correct() {
        let point = Point::zero();
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);
    }

    #[test]
    fn point_equality_zero() {
        let point1 = Point::zero();
        let point2 = Point::zero();
        assert_eq!(point1, point2);
    }

    #[test]
    fn point_equality_new() {
        let point1 = Point::new(4.0, -4.0, 3.0);
        let point2 = Point::new(4.0, -4.0, 3.0);
        assert_eq!(point1, point2);
    }

    #[test]
    fn add_point_and_vector() {
        let point = Point::new(4.0, -4.0, 3.0);
        let vector = Vector::new(1.0, -1.0, 1.0);
        let result = Point::new(5.0, -5.0, 4.0);
        assert_eq!(point + vector, result);
    }

    #[test]
    fn sub_point_and_point() {
        let point1 = Point::new(4.0, -4.0, 3.0);
        let point2 = Point::new(4.0, -4.0, 3.0);
        let result = Vector::zero();
        assert_eq!(point1 - point2, result);
    }

    #[test]
    fn sub_point_and_zero_point() {
        let point1 = Point::new(4.0, -4.0, 3.0);
        let zero_point = Point::zero();
        let result1 = Vector::new(-4.0, 4.0, -3.0);
        assert_eq!(zero_point - point1, result1);

        let result2 = Vector::new(4.0, -4.0, 3.0);
        assert_eq!(point1 - zero_point, result2);
    }

    #[test]
    fn sub_point_and_vector() {
        let point = Point::new(4.0, -4.0, 3.0);
        let vector = Vector::new(1.0, -1.0, 1.0);
        let result = Point::new(3.0, -3.0, 2.0);
        assert_eq!(point - vector, result);

        let zero_vector = Vector::zero();
        assert_eq!(point - zero_vector, point);
    }
}
