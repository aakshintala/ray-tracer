use std::ops::{Add, Mul, Sub};

use crate::utils::approx_eq;

#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn clamp_to_unit_range(primary: f64) -> f64 {
        if primary < 0.0 {
            0.0
        } else if primary <= 1.0 {
            primary
        } else {
            1.0
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            r: Self::clamp_to_unit_range(self.r) * factor,
            g: Self::clamp_to_unit_range(self.g) * factor,
            b: Self::clamp_to_unit_range(self.b) * factor,
        }
    }
}

impl Add<Color> for Color {
    type Output = Color;

    fn add(self, rhs: Color) -> Self::Output {
        Color::new(self.r + rhs.r, self.g + rhs.g, self.b + rhs.b)
    }
}

impl Sub<Color> for Color {
    type Output = Color;

    fn sub(self, rhs: Color) -> Self::Output {
        Color::new(self.r - rhs.r, self.g - rhs.g, self.b - rhs.b)
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, rhs: f64) -> Self::Output {
        Color::new(self.r * rhs, self.g * rhs, self.b * rhs)
    }
}

impl Mul<Color> for Color {
    type Output = Color;

    fn mul(self, rhs: Color) -> Self::Output {
        Color::new(self.r * rhs.r, self.g * rhs.g, self.b * rhs.b)
    }
}

impl PartialEq<Color> for Color {
    fn eq(&self, other: &Color) -> bool {
        approx_eq(self.r, other.r) && approx_eq(self.g, other.g) && approx_eq(self.b, other.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn color_init_and_equality() {
        let white = Color::new(1.0, 1.0, 1.0);
        let another_white = Color::new(1.0, 1.0, 1.0);
        assert_eq!(white, another_white);
    }

    #[test]
    pub fn color_init_and_equality_random() {
        let color1 = Color::new(-0.5, 16.0, -0.00004);
        let color2 = Color::new(-0.5, 16.0, -0.00004);
        assert_eq!(color1, color2);
    }

    #[test]
    pub fn add_colors() {
        let color1 = Color::new(0.9, 0.6, 0.75);
        let color2 = Color::new(0.7, 0.1, 0.25);
        let expected = Color::new(1.6, 0.7, 1.0);
        assert_eq!(color1 + color2, expected);
    }

    #[test]
    pub fn sub_colors() {
        let color1 = Color::new(0.9, 0.6, 0.75);
        let color2 = Color::new(0.7, 0.1, 0.25);
        let expected = Color::new(0.2, 0.5, 0.5);
        assert_eq!(color1 - color2, expected);
    }

    #[test]
    pub fn multiply_color_by_scalar() {
        let color1 = Color::new(0.1, 0.25, 0.25);
        let expected = Color::new(0.2, 0.5, 0.5);
        assert_eq!(color1 * 2.0, expected);
    }

    #[test]
    pub fn multiply_color_by_color() {
        let color1 = Color::new(0.9, 0.6, 0.75);
        let color2 = Color::new(1.0, 0.1, -2.0);
        let expected = Color::new(0.9, 0.06, -1.5);
        assert_eq!(color1 * color2, expected);
    }
}
