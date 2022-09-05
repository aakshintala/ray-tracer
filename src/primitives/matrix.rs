use std::ops::{Index, IndexMut, Mul};

use crate::utils::approx_eq;
use crate::Vector;

#[derive(Clone, Debug)]
pub struct Matrix<const N: usize> {
    data: [[f64; N]; N],
}

impl<const N: usize> Matrix<N> {
    pub const fn new() -> Self {
        Self {
            data: [[0.0; N]; N],
        }
    }

    pub const fn new_with_data(data: [[f64; N]; N]) -> Self {
        Self { data }
    }

    pub fn identity() -> Self {
        let mut data = [[0.0; N]; N];

        for i in 0..N {
            for j in 0..N {
                if i == j {
                    data[i][j] = 1.0;
                }
            }
        }

        Self { data }
    }

    pub fn transpose(&self) -> Matrix<N> {
        let mut result = Matrix::<N>::new();

        for row in 0..N {
            for col in 0..N {
                result[row][col] = self[col][row];
            }
        }

        result
    }
}

impl Matrix<2> {
    pub fn determinant(&self) -> f64 {
        let determinant = (self.data[0][0] * self.data[1][1]) - (self.data[0][1] * self.data[1][0]);
        determinant
    }
}

impl Matrix<3> {
    fn submatrix(&self, row: usize, col: usize) -> Matrix<2> {
        let mut submatrix = Matrix::<2>::new();

        let mut submatrix_row = 0;
        for r in 0..3 {
            if r == row {
                continue;
            }
            let mut submatrix_col = 0;
            for c in 0..3 {
                if c == col {
                    continue;
                }
                submatrix[submatrix_row][submatrix_col] = self.data[r][c];
                submatrix_col += 1;
            }
            submatrix_row += 1;
        }
        submatrix
    }

    fn minor(&self, row: usize, col: usize) -> f64 {
        self.submatrix(row, col).determinant()
    }

    fn determinant(&self) -> f64 {
        let mut determinant: f64 = 0.0;
        for column in 0..3 {
            determinant += self.cofactor(0, column) * self[0][column];
        }

        determinant
    }

    fn cofactor(&self, row: usize, col: usize) -> f64 {
        let minor = self.minor(row, col);

        if (row + col) % 2 == 0 {
            minor
        } else {
            -minor
        }
    }
}

impl Matrix<4> {
    fn submatrix(&self, row: usize, col: usize) -> Matrix<3> {
        let mut submatrix = Matrix::<3>::new();

        let mut submatrix_row = 0;
        for r in 0..4 {
            if r == row {
                continue;
            }
            let mut submatrix_col = 0;
            for c in 0..4 {
                if c == col {
                    continue;
                }
                submatrix[submatrix_row][submatrix_col] = self.data[r][c];
                submatrix_col += 1;
            }
            submatrix_row += 1;
        }
        submatrix
    }

    fn minor(&self, row: usize, col: usize) -> f64 {
        self.submatrix(row, col).determinant()
    }

    fn determinant(&self) -> f64 {
        let mut determinant: f64 = 0.0;
        for column in 0..4 {
            determinant += self.cofactor(0, column) * self[0][column];
        }

        determinant
    }

    fn cofactor(&self, row: usize, col: usize) -> f64 {
        let minor = self.minor(row, col);

        if (row + col) % 2 == 0 {
            minor
        } else {
            -minor
        }
    }

    pub fn is_invertible(&self) -> bool {
        !approx_eq(self.determinant(), 0.0)
    }

    pub fn inverse(&self) -> Matrix<4> {
        if !self.is_invertible() {
            panic!("Matrix is not invertible.");
        }
        let mut matrix = Matrix::new();
        let determinant = self.determinant();
        for row in 0..4 {
            for column in 0..4 {
                let cofactor = self.cofactor(row, column);
                // transposed storage
                matrix[column][row] = cofactor / determinant;
            }
        }
        matrix
    }
}

impl<const N: usize> PartialEq<Matrix<N>> for Matrix<N> {
    fn eq(&self, other: &Matrix<N>) -> bool {
        let mut equal = true;
        for i in 0..N {
            for j in 0..N {
                equal &= approx_eq(self.data[i][j], other.data[i][j]);
            }
        }
        equal
    }
}

impl<const N: usize> Index<usize> for Matrix<N> {
    type Output = [f64; N];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize> IndexMut<usize> for Matrix<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize> Mul<&Matrix<N>> for Matrix<N> {
    type Output = Matrix<N>;

    fn mul(self, rhs: &Matrix<N>) -> Self::Output {
        let mut new_matrix = Matrix::new();
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    new_matrix[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        new_matrix
    }
}

impl Mul<&Vector> for Matrix<4> {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        let mut result = Vector::zero();
        for i in 0..4 {
            for j in 0..4 {
                result[i] += self[i][j] * rhs[j];
            }
        }
        result
    }
}

impl Mul<&Matrix<4>> for Vector {
    type Output = Vector;

    fn mul(self, rhs: &Matrix<4>) -> Self::Output {
        let mut result = Vector::zero();
        for i in 0..4 {
            for j in 0..4 {
                result[i] += rhs[i][j] * self[j];
            }
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn new_matrix2() {
        let mut matrix2 = Matrix::<2>::new();
        assert_eq!(matrix2[0][0], 0.0);
        matrix2[1][0] = 1.0;
        assert_eq!(matrix2[1][0], 1.0);
    }
    #[test]
    pub fn new_matrix3() {
        let mut matrix3 = Matrix::<3>::new();
        assert_eq!(matrix3[0][0], 0.0);
        matrix3[1][2] = 1.0;
        assert_eq!(matrix3[1][2], 1.0);
    }

    #[test]
    pub fn new_matrix4() {
        let mut matrix4 = Matrix::<4>::new();
        assert_eq!(matrix4[0][0], 0.0);
        matrix4[3][3] = 1.0;
        assert_eq!(matrix4[3][3], 1.0);
    }

    #[test]
    pub fn identity_matrix2() {
        let matrix2: Matrix<2> = Matrix::<2>::identity();
        const EXPECTED: Matrix<2> = Matrix::<2>::new_with_data([[1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(matrix2, EXPECTED);
    }
    #[test]
    pub fn identity_matrix3() {
        let matrix3: Matrix<3> = Matrix::<3>::identity();
        const EXPECTED: Matrix<3> =
            Matrix::<3>::new_with_data([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(matrix3, EXPECTED);
    }

    #[test]
    pub fn identity_matrix4() {
        let matrix4: Matrix<4> = Matrix::<4>::identity();
        const EXPECTED: Matrix<4> = Matrix::<4>::new_with_data([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(matrix4, EXPECTED);
    }

    #[test]
    pub fn matrix4_matrix4_multiplication() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        const M2: Matrix<4> = Matrix::<4>::new_with_data([
            [-2.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, -1.0],
            [4.0, 3.0, 6.0, 5.0],
            [1.0, 2.0, 7.0, 8.0],
        ]);
        const EXPECTED: Matrix<4> = Matrix::<4>::new_with_data([
            [20.0, 22.0, 50.0, 48.0],
            [44.0, 54.0, 114.0, 108.0],
            [40.0, 58.0, 110.0, 102.0],
            [16.0, 26.0, 46.0, 42.0],
        ]);
        assert_eq!(M1 * &M2, EXPECTED);
    }

    #[test]
    pub fn matrix4_vector_multiplication() {
        const MATRIX1: Matrix<4> = Matrix::<4>::new_with_data([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 4.0, 2.0],
            [8.0, 6.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        const VECTOR1: Vector = Vector::new(1.0, 2.0, 3.0);
        const EXPECTED: Vector = Vector::new(14.0, 22.0, 32.0);
        assert_eq!(MATRIX1 * &VECTOR1, EXPECTED);
    }

    #[test]
    pub fn matrix4_indentity4_multiplication() {
        const MATRIX1: Matrix<4> = Matrix::<4>::new_with_data([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0],
        ]);
        let matrix2: Matrix<4> = Matrix::<4>::identity();
        assert_eq!(MATRIX1.clone() * &matrix2, MATRIX1);
    }

    #[test]
    pub fn indentity4_vector_multiplication() {
        let matrix1: Matrix<4> = Matrix::<4>::identity();
        const VECTOR1: Vector = Vector::new(1.0, 2.0, 3.0);
        assert_eq!(matrix1 * &VECTOR1, VECTOR1);
    }

    #[test]
    pub fn transpose_identity_matrix() {
        let matrix1: Matrix<4> = Matrix::<4>::identity();
        let matrix2: Matrix<4> = matrix1.transpose();
        assert_eq!(matrix1, matrix2);
    }

    #[test]
    pub fn transpose_matrix4() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [0.0, 9.0, 3.0, 0.0],
            [9.0, 8.0, 0.0, 8.0],
            [1.0, 8.0, 5.0, 3.0],
            [0.0, 0.0, 5.0, 8.0],
        ]);
        const M2: Matrix<4> = Matrix::<4>::new_with_data([
            [0.0, 9.0, 1.0, 0.0],
            [9.0, 8.0, 8.0, 0.0],
            [3.0, 0.0, 5.0, 5.0],
            [0.0, 8.0, 3.0, 8.0],
        ]);
        assert_eq!(M1.transpose(), M2);
    }

    #[test]
    pub fn transpose_matrix3() {
        const M1: Matrix<3> =
            Matrix::<3>::new_with_data([[0.0, 9.0, 3.0], [9.0, 8.0, 0.0], [1.0, 8.0, 5.0]]);
        const M2: Matrix<3> =
            Matrix::<3>::new_with_data([[0.0, 9.0, 1.0], [9.0, 8.0, 8.0], [3.0, 0.0, 5.0]]);
        assert_eq!(M1.transpose(), M2);
    }

    #[test]
    pub fn transpose_matrix2() {
        const M1: Matrix<2> = Matrix::<2>::new_with_data([[0.0, 3.0], [9.0, 8.0]]);
        const M2: Matrix<2> = Matrix::<2>::new_with_data([[0.0, 9.0], [3.0, 8.0]]);
        assert_eq!(M1.transpose(), M2);
    }

    pub fn test_determinant2() {
        const M1: Matrix<2> = Matrix::<2>::new_with_data([[1.0, 5.0], [-3.0, 2.0]]);
        assert_eq!(M1.determinant(), 17.0);
    }

    #[test]
    fn a_submatrix_of_a_3x3_matrix_is_a_2x2_matrix() {
        const M1: Matrix<3> =
            Matrix::<3>::new_with_data([[1.0, 5.0, 0.0], [-3.0, 2.0, 7.0], [0.0, 6.0, -3.0]]);
        let submatrix = M1.submatrix(0, 2);
        const EXPECTED: Matrix<2> = Matrix::<2>::new_with_data([[-3.0, 2.0], [0.0, 6.0]]);

        assert_eq!(submatrix, EXPECTED);
    }

    #[test]
    fn a_submatrix_of_a_4x4_matrix_is_a_3x3_matrix() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [-6.0, 1.0, 1.0, 6.0],
            [-8.0, 5.0, 8.0, 6.0],
            [-1.0, 0.0, 8.0, 2.0],
            [-7.0, 1.0, -1.0, 1.0],
        ]);
        let submatrix = M1.submatrix(2, 1);
        const EXPECTED: Matrix<3> =
            Matrix::<3>::new_with_data([[-6.0, 1.0, 6.0], [-8.0, 8.0, 6.0], [-7.0, -1.0, 1.0]]);

        assert_eq!(submatrix, EXPECTED);
    }

    #[test]
    fn calculating_a_minor_of_a_3x3_matrix() {
        const M1: Matrix<3> =
            Matrix::<3>::new_with_data([[3.0, 5.0, 0.0], [2.0, -1.0, -7.0], [6.0, -1.0, 5.0]]);
        let submatrix = M1.submatrix(0, 0);
        const EXPECTED: Matrix<2> = Matrix::new_with_data([[-1.0, -7.0], [-1.0, 5.0]]);
        assert_eq!(submatrix, EXPECTED);

        assert_eq!(M1.minor(0, 0), -12.0);
        assert_eq!(M1.minor(1, 0), 25.0);
    }

    #[test]
    fn calculating_a_cofactor_of_a_3x3_matrix() {
        const M1: Matrix<3> =
            Matrix::<3>::new_with_data([[3.0, 5.0, 0.0], [2.0, -1.0, -7.0], [6.0, -1.0, 5.0]]);
        assert_eq!(M1.cofactor(0, 0), -12.0);
        assert_eq!(M1.cofactor(1, 0), -25.0);
    }

    #[test]
    fn calculating_the_determinant_of_3x3_matrix() {
        const M1: Matrix<3> =
            Matrix::<3>::new_with_data([[1.0, 2.0, 6.0], [-5.0, 8.0, -4.0], [2.0, 6.0, 4.0]]);

        assert_eq!(M1.cofactor(0, 0), 56.0);
        assert_eq!(M1.cofactor(0, 1), 12.0);
        assert_eq!(M1.cofactor(0, 2), -46.0);
        assert_eq!(M1.determinant(), -196.0);
    }

    #[test]
    fn calculating_the_determinant_of_4x4_matrix() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [-2.0, -8.0, 3.0, 5.0],
            [-3.0, 1.0, 7.0, 3.0],
            [1.0, 2.0, -9.0, 6.0],
            [-6.0, 7.0, 7.0, -9.0],
        ]);

        assert_eq!(M1.cofactor(0, 0), 690.0);
        assert_eq!(M1.cofactor(0, 1), 447.0);
        assert_eq!(M1.cofactor(0, 2), 210.0);
        assert_eq!(M1.determinant(), -4071.0);
    }

    #[test]
    fn testing_an_invertible_matrix_for_invertibility() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [6.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, 7.0, 6.0],
            [4.0, -9.0, 3.0, -7.0],
            [9.0, 1.0, 7.0, -6.0],
        ]);

        assert_eq!(M1.cofactor(0, 0), -668.0);
        assert_eq!(M1.cofactor(0, 1), 112.0);
        assert_eq!(M1.cofactor(0, 2), 620.0);
        assert_eq!(M1.cofactor(0, 3), -260.0);

        assert_eq!(M1.determinant(), -2120.0);

        assert_eq!(M1.is_invertible(), true);
    }

    #[test]
    fn testing_a_noninvertible_matrix_for_invertibility() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [-4.0, 2.0, -2.0, -3.0],
            [9.0, 6.0, 2.0, 6.0],
            [0.0, -5.0, 1.0, -5.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);

        assert_eq!(M1.is_invertible(), false);
    }

    #[test]
    fn calculating_the_inverse_of_a_matrix() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [-5.0, 2.0, 6.0, -8.0],
            [1.0, -5.0, 1.0, 8.0],
            [7.0, 7.0, -6.0, -7.0],
            [1.0, -3.0, 7.0, 4.0],
        ]);

        let inverse = M1.inverse();

        assert_eq!(M1.determinant(), 532.0);
        assert_eq!(M1.cofactor(2, 3), -160.0);
        assert_eq!(inverse[3][2], -160.0 / 532.0);
        assert_eq!(M1.cofactor(3, 2), 105.0);
        assert_eq!(inverse[2][3], 105.0 / 532.0);

        const EXPECTED: Matrix<4> = Matrix::<4>::new_with_data([
            [0.21805, 0.45113, 0.24060, -0.04511],
            [-0.80827, -1.45677, -0.44361, 0.52068],
            [-0.07895, -0.22368, -0.05263, 0.19737],
            [-0.52256, -0.81391, -0.30075, 0.30639],
        ]);

        assert_eq!(inverse, EXPECTED);
    }

    #[test]
    fn calculating_the_inverse_of_another_matrix() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [8.0, -5.0, 9.0, 2.0],
            [7.0, 5.0, 6.0, 1.0],
            [-6.0, 0.0, 9.0, 6.0],
            [-3.0, 0.0, -9.0, -4.0],
        ]);

        let inverse = M1.inverse();

        const EXPECTED: Matrix<4> = Matrix::<4>::new_with_data([
            [-0.15385, -0.15385, -0.28205, -0.53846],
            [-0.07692, 0.12308, 0.02564, 0.03077],
            [0.35897, 0.35897, 0.43590, 0.92308],
            [-0.69231, -0.69231, -0.76923, -1.92308],
        ]);

        assert_eq!(inverse, EXPECTED);
    }

    #[test]
    fn calculating_the_inverse_of_a_third_matrix() {
        const MATRIX1: Matrix<4> = Matrix::<4>::new_with_data([
            [9.0, 3.0, 0.0, 9.0],
            [-5.0, -2.0, -6.0, -3.0],
            [-4.0, 9.0, 6.0, 4.0],
            [-7.0, 6.0, 6.0, 2.0],
        ]);

        let inverse = MATRIX1.inverse();

        const EXPECTED: Matrix<4> = Matrix::<4>::new_with_data([
            [-0.04074, -0.07778, 0.14444, -0.22222],
            [-0.07778, 0.03333, 0.36667, -0.33333],
            [-0.02901, -0.14630, -0.10926, 0.12963],
            [0.17778, 0.06667, -0.26667, 0.33333],
        ]);

        assert_eq!(inverse, EXPECTED);
    }

    #[test]
    fn multiplying_a_product_by_its_inverse() {
        const M1: Matrix<4> = Matrix::<4>::new_with_data([
            [3.0, -9.0, 7.0, 3.0],
            [3.0, -8.0, 2.0, -9.0],
            [-4.0, 4.0, 4.0, 1.0],
            [-6.0, 5.0, -1.0, 1.0],
        ]);
        const M2: Matrix<4> = Matrix::<4>::new_with_data([
            [8.0, 2.0, 2.0, 2.0],
            [3.0, -1.0, 7.0, 0.0],
            [7.0, 0.0, 5.0, 4.0],
            [6.0, -2.0, 0.0, 5.0],
        ]);

        let c = M1 * &M2;
        let b_inverse = M2.inverse();

        assert_eq!(c * &b_inverse, M1);
    }
}
