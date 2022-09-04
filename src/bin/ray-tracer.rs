use lib_ray_tracer::{Point, Vector};

fn main() {
    let point = Point::new(1.0, 1.0, 1.0);
    let vector = Vector::new(-1.0, 1.0, -1.0);
    println!("Hello, world!");
    println!("Here's a point for you! {:#?}", point);
    println!("Have a vector too while you're at it! {:#?}", vector);
}
