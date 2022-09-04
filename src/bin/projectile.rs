use lib_ray_tracer::{Canvas, Color, Point, Vector};

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
fn clamp(value: f64, max: usize) -> usize {
    if value < 0.0 {
        0
    } else if value as usize >= max {
        max - 1
    } else {
        max - value as usize - 1
    }
}
fn write_projectile_position(canvas: &mut Canvas, position: &Point) {
    let red = Color::new(1.0, 0.0, 0.0);

    for i in [-1.0, 0.0, 1.0] {
        let x = clamp(position.x + i, canvas.width);
        for j in [-1.0, 0.0, 1.0] {
            let y = clamp(position.y + j, canvas.height);
            canvas.write_pixel(x, y, &red);
        }
    }
}

pub fn main() {
    let mut projectile = Projectile {
        position: Point::new(0.0, 1.0, 0.0),
        velocity: Vector::new(1.0, 1.8, 0.0).normalize() * 11.25,
    };

    let env = Environment {
        gravity: Vector::new(0.0, -0.1, 0.0),
        wind: Vector::new(-0.01, 0.0, 0.0),
    };

    let mut canvas = Canvas::new(900, 550);

    write_projectile_position(&mut canvas, &projectile.position);
    while projectile.position.y > 0.0 {
        projectile.tick(&env);
        write_projectile_position(&mut canvas, &projectile.position);
    }

    canvas.export("images/projectile.png").unwrap();
}
