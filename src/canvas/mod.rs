mod color;

pub use color::Color;

#[derive(Debug)]
pub enum CanvasError {
    PixelNotFound(usize, usize),
}

#[derive(Clone, Debug)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub(self) pixels: Vec<Color>,
}

impl Canvas {
    /// Returns a blank canvas of the specified `width` * `height`.
    /// All pixels in the canvas will be initialized to black `Color {0.0, 0.0, 0.0}`
    pub fn new(width: usize, height: usize) -> Self {
        let num_pixels: usize = width * height;
        Canvas {
            width,
            height,
            pixels: vec![Color::new(0.0, 0.0, 0.0); num_pixels],
        }
    }

    pub fn pixel_at(&self, x: usize, y: usize) -> Result<Color, CanvasError> {
        if x >= self.width || y >= self.height {
            return Err(CanvasError::PixelNotFound(x, y));
        }

        Ok(self.pixels[x * self.height + y].clone())
    }

    pub fn write_pixel(&mut self, x: usize, y: usize, new_color: &Color) {
        if x < self.width && y < self.height {
            println!("{x}, {y}, {0}", (x * self.height) + y);
            self.pixels[(x * self.height) + y] = new_color.clone();
        }
    }

    pub fn export(&self, path: &str) -> image::ImageResult<()> {
        let mut img = image::ImageBuffer::new(self.width as u32, self.height as u32);

        for (x, y, pixel) in img.enumerate_pixels_mut() {
            let color = self.pixel_at(x as usize, y as usize).unwrap();
            let scaled_color = color.scale(255.0);
            *pixel = image::Rgb([
                scaled_color.r as u8,
                scaled_color.g as u8,
                scaled_color.b as u8,
            ]);
        }

        img.save(path)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn blank_canvas() {
        let canvas = Canvas::new(10, 20);
        let num_pixels: usize = 10 * 20;
        let expected_pixels = vec![Color::new(0.0, 0.0, 0.0); num_pixels];
        assert_eq!(canvas.width, 10);
        assert_eq!(canvas.height, 20);
        assert_eq!(canvas.pixels, expected_pixels);
    }

    #[test]
    pub fn set_pixel() {
        let mut canvas = Canvas::new(10, 20);
        let red: Color = Color::new(1.0, 0.0, 0.0);
        assert_eq!(canvas.width, 10);
        assert_eq!(canvas.height, 20);
        canvas.write_pixel(2, 3, &red);
        assert_eq!(canvas.pixel_at(2, 3,).unwrap(), red);
    }
}
