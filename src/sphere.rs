use crate::{Point, Ray};

pub trait Intersectable<T> {
    fn intersect(&self, _: T) -> Option<[f64; 2]>;
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    center: Point,
    radius: f64,
}

impl Sphere {
    pub fn new() -> Self {
        Sphere {
            center: Point::new(0.0, 0.0, 0.0),
            radius: 1.0,
        }
    }
}

impl Intersectable<Ray> for Sphere {
    fn intersect(&self, ray: Ray) -> Option<[f64; 2]> {
        let sphere_to_ray = ray.origin - self.center;

        let a = ray.direction.dot_product(&ray.direction);
        let b = ray.direction.dot_product(&sphere_to_ray) * 2.0;
        let c = sphere_to_ray.dot_product(&sphere_to_ray) - 1.0;

        let discriminant = b * b - (4.0 * a * c);
        if discriminant < 0.0 {
            None
        } else {
            let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
            let t2 = (-b + discriminant.sqrt()) / (2.0 * a);
            let result = if t1 < t2 { [t1, t2] } else { [t2, t1] };
            Some(result)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Point, Ray, Vector};

    #[test]
    pub fn sphere_creation() {
        let sphere = Sphere::new();
        assert_eq!(sphere.center, Point::new(0.0, 0.0, 0.0));
    }

    #[test]
    pub fn ray_intersects_sphere_at_2_points() {
        let ray = Ray::new(Point::new(0.0, 0.0, -5.0), Vector::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new();

        let xs = sphere.intersect(ray).unwrap();
        assert_eq!(xs[0], 4.0);
        assert_eq!(xs[1], 6.0);
    }

    #[test]
    pub fn ray_intersects_sphere_as_tangent() {
        let ray = Ray::new(Point::new(0.0, 1.0, -5.0), Vector::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new();

        let xs = sphere.intersect(ray).unwrap();
        assert_eq!(xs[0], 5.0);
        assert_eq!(xs[1], 5.0);
    }

    #[test]
    pub fn ray_does_not_intersect_sphere() {
        let ray = Ray::new(Point::new(0.0, 2.0, -5.0), Vector::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new();

        assert!(sphere.intersect(ray).is_none());
    }

    #[test]
    pub fn ray_originates_inside_sphere() {
        let ray = Ray::new(Point::new(0.0, 0.0, 0.0), Vector::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new();

        let xs = sphere.intersect(ray).unwrap();
        assert_eq!(xs[0], -1.0);
        assert_eq!(xs[1], 1.0);
    }

    #[test]
    pub fn ray_originates_behind_sphere() {
        let ray = Ray::new(Point::new(0.0, 0.0, 5.0), Vector::new(0.0, 0.0, 1.0));
        let sphere = Sphere::new();

        let xs = sphere.intersect(ray).unwrap();
        assert_eq!(xs[0], -6.0);
        assert_eq!(xs[1], -4.0);
    }
}
