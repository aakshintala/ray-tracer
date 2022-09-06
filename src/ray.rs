use crate::{Point, Vector};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point,
    pub direction: Vector,
}

impl Ray {
    pub fn new(origin: Point, direction: Vector) -> Self {
        Ray { origin, direction }
    }

    pub fn position(&self, time: f64) -> Point {
        self.origin + self.direction * time
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Point, Vector};

    #[test]
    pub fn ray_creation() {
        let origin = Point::new(1.0, 2.0, 3.0);
        let direction = Vector::new(4.0, 5.0, 6.0);

        let ray = Ray::new(origin, direction);
        assert_eq!(ray.origin, origin);
        assert_eq!(ray.direction, direction);
    }

    #[test]
    pub fn computing_a_point_from_a_distance() {
        let ray = Ray::new(Point::new(2.0, 3.0, 4.0), Vector::new(1.0, 0.0, 0.0));
        assert_eq!(ray.position(0.0), Point::new(2.0, 3.0, 4.0));
        assert_eq!(ray.position(1.0), Point::new(3.0, 3.0, 4.0));
        assert_eq!(ray.position(-1.0), Point::new(1.0, 3.0, 4.0));
        assert_eq!(ray.position(2.5), Point::new(4.5, 3.0, 4.0));
    }
}
