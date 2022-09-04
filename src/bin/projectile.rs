use lib_ray_tracer::{Point, Vector};

struct Environment {
    gravity: Vector,
    wind: Vector,
}

struct Projectile {
    pub position: Point,
    pub velocity: Vector,
}

impl Projectile {
    pub fn tick(&mut self, env: &Environment) {
        self.position = self.position + self.velocity;
        self.velocity = self.velocity + env.gravity + env.wind;
    }
}

pub fn main() {
    let mut projectile = Projectile {
        position: Point::new(0.0, 1.0, 0.0),
        velocity: Vector::new(0.0, 1.0, 0.0),
    };

    let env = Environment {
        gravity: Vector::new(0.0, -0.9, 0.0),
        wind: Vector::new(-0.01, 0.2, 0.0),
    };

    let mut num_ticks = 0;
    println!("Start = {:?}", projectile.position);
    while projectile.position.y > 0.0 {
        projectile.tick(&env);
        println!("At Tick {num_ticks} = {:?}", projectile.position);
        num_ticks += 1;
    }
    println!("End = {:?}", projectile.position);
    println!("Total number of ticks = {num_ticks}");
}
