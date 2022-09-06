use std::f64::consts::PI;

use lib_ray_tracer::{clamp, Canvas, Color, Matrix, Point};

fn write_hour_marker(canvas: &mut Canvas, position: &Point) {
    let red = Color::new(1.0, 0.0, 0.0);

    // Center the point at the center of the canvas and then multiply it by the radius of the clock
    // face to get it to it's final position.
    let position_x: f64 =
        (canvas.width / 2) as f64 + (position.x * 3.0 * (canvas.width / 8) as f64);
    let position_y: f64 =
        (canvas.height / 2) as f64 + (position.y * 3.0 * (canvas.width / 8) as f64);
    for i in [-2.0, -1.0, 0.0, 1.0, 2.0] {
        let x = clamp(position_x + i, canvas.width);
        for j in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            let y = clamp(position_y + j, canvas.height);
            canvas.write_pixel(x, y, &red);
        }
    }
}

fn create_canvas() -> Canvas {
    const CANVAS_HEIGHT: usize = 600;
    const CANVAS_WIDTH: usize = 600;
    Canvas::new(CANVAS_WIDTH, CANVAS_HEIGHT)
}

fn draw_clock() {
    let mut canvas = create_canvas();
    // Origin = (w/2, h/2)
    let twelve_pm = Point::new(0.0, 1.0, 0.0);

    // Use rotation to keep moving that point in the X,Y plane by PI/6 radians.
    let marker = twelve_pm.clone();
    for i in 0..12 {
        let rotation_matrix = Matrix::<4>::rotation_z(i as f64 * PI / 6.0);
        write_hour_marker(&mut canvas, &(rotation_matrix * marker));
    }

    canvas.export("images/clock_face.png").unwrap();
}

fn draw_circle() {
    let mut canvas = create_canvas();
    // Origin = (w/2, h/2)
    let start_point = Point::new(0.0, 1.0, 0.0);

    // Use rotation to keep moving that point in the X,Y plane by PI/6 radians.
    let marker = start_point.clone();
    for i in 0..365 {
        let rotation_matrix = Matrix::<4>::rotation_z(i as f64 * PI / 180.0);
        write_hour_marker(&mut canvas, &(rotation_matrix * marker));
    }

    canvas.export("images/circle.png").unwrap();
}

pub fn main() {
    draw_clock();

    draw_circle();
}
