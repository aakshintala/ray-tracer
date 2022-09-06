const EPSILON: f64 = 0.00001;

pub fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON
}

#[cfg(test)]
mod test {
    use super::approx_eq;

    #[test]
    pub fn test_approx_eq() {
        assert_eq!(approx_eq(-2120.0, 0.0), false);
    }
}

pub fn clamp(value: f64, max: usize) -> usize {
    if value < 0.0 {
        0
    } else if value as usize >= max {
        max - 1
    } else {
        max - value as usize - 1
    }
}
