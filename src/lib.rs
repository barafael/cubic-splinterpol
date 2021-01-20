#![cfg_attr(not(test), no_std)]

use vecmat::{
    prelude::*,
    vector::{self, Vector16},
    Matrix, Vector,
};

const NUM_ELEMENTS: usize = 16;

pub fn splinterpol(xs: &[f32], ys: &[f32]) -> Matrix<f32, 15, 2> {
    let mut diagonal = [0f32; NUM_ELEMENTS - 2];
    calc_diagonal(&xs, &mut diagonal);

    let a: Matrix<f32, 14, 14> = calc_matrix_a(&xs, &diagonal);

    let r: Vector<f32, 14> = calc_r(&xs, &ys);

    let c: Vector<f32, 14> = solve_system(&a, &r);
    let c: Vector<f32, 16> = {
        let mut vector: Vector<f32, 16> = Default::default();
        for (i, v) in c.iter().enumerate() {
            vector[i + 1] = *v;
        }
        vector
    };

    let mut a: Vector<f32, 15> = Default::default();
    a.iter_mut().enumerate().for_each(|(i, val)| {
        *val = ys[i];
    });

    Default::default()
}

fn solve_system(a: &Matrix<f32, 14, 14>, b: &Vector<f32, 14>) -> Vector<f32, 14> {
    let inv = a.inv();
    inv.dot(*b)
}

fn cubic_spline(a: f32, b: f32, c: f32, d: f32, vec: &mut [f32], start: f32, step_size: f32) {
    for (i, v) in vec.iter_mut().enumerate() {
        let x = start + i as f32 * step_size;
        *v = a + b * x + c * (x * x) + d * (x * x * x);
    }
}

fn h(i: usize, vals: &[f32]) -> f32 {
    vals[i + 1] - vals[i]
}

fn calc_diagonal(xs: &[f32], result: &mut [f32]) {
    assert_eq!(NUM_ELEMENTS, xs.len());
    let mut diagonal = [0f32; NUM_ELEMENTS - 2];
    for i in 0..(NUM_ELEMENTS - 2) {
        diagonal[i] = 2f32 * (h(i, &xs) + h(i + 1, &xs));
    }
    result.copy_from_slice(&diagonal);
}

fn calc_matrix_a(xs: &[f32], diagonal: &[f32]) -> Matrix<f32, 14, 14> {
    assert_eq!(14, diagonal.len());
    let mut a: Matrix<f32, 14, 14> = Default::default();
    for i in 1..14 {
        let hi = h(i, &xs);
        a[(i - 1, i)] = hi;
        a[(i, i - 1)] = hi;
        a[(i - 1, i - 1)] = diagonal[i - 1];
    }
    a[(13, 13)] = diagonal[13];
    a
}

fn calc_r(xs: &[f32], ys: &[f32]) -> Vector<f32, 14> {
    let mut r: Vector<f32, 14> = Default::default();
    for i in 0..14 {
        let div1 = (ys[i + 2] - ys[i + 1]) / (h(i + 1, &xs));
        let div2 = (ys[i + 1] - ys[i]) / (h(i, &xs));
        r[i] = 3f32 * (div1 - div2);
    }
    r
}

fn calc_b(xs: &[f32], ys: &[f32], cs: &Vector<f32, 16>) -> Vector<f32, 15> {
    let mut b: Vector<f32, 15> = Default::default();
    for i in 0..15 {
        let div_1 = (ys[i + 1] - ys[i + 0]) / (h(i + 0, &xs));
        let div_2 = (2f32 * cs[i + 0] + cs[i + 1]) / 3f32;
        b[i] = div_1 - div_2 * h(i + 0, &xs);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_mul() {
        let c: Matrix<f32, 2, 4> = {
            let mut a: Matrix<f32, 2, 1> = Default::default();
            let mut b: Matrix<f32, 1, 4> = Default::default();

            a[(0, 0)] = 1f32;
            a[(1, 0)] = 2f32;

            b[(0, 0)] = 1f32;
            b[(0, 1)] = 2f32;
            b[(0, 2)] = 3f32;
            b[(0, 3)] = 4f32;

            a.dot(b)
        };
        let expected: Matrix<f32, 2, 4> =
            Matrix::from([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]);
        assert_eq!(expected, c);
    }

    #[test]
    fn do_cubic_spline() {
        let mut xs = [0f32; 64];
        cubic_spline(4.0, 2.0, 2.0, 1.5, &mut xs, 0.0, 0.05);
        let expected = vec![
            4.0, 4.1051874, 4.2215, 4.350063, 4.492, 4.6484375, 4.8205, 5.009312, 5.2160006,
            5.4416876, 5.6875, 5.9545627, 6.244, 6.556938, 6.8945003, 7.2578125, 7.6480002,
            8.066188, 8.5135, 8.991062, 9.5, 10.041438, 10.6165, 11.226313, 11.872001, 12.5546875,
            13.275501, 14.035563, 14.836, 15.677938, 16.5625, 17.490814, 18.464, 19.483187,
            20.5495, 21.664063, 22.828003, 24.042439, 25.3085, 26.627316, 28.0, 29.427689,
            30.911507, 32.452568, 34.052002, 35.710938, 37.4305, 39.21182, 41.056, 42.964188,
            44.9375, 46.97706, 49.084003, 51.25944, 53.5045, 55.820313, 58.208, 60.668694,
            63.203506, 65.81357, 68.5, 71.26394, 74.10651, 77.028824,
        ];
        assert_eq!(expected, xs);
    }

    #[test]
    fn diagonal_test() {
        let mut xs = [0f32; 16];
        xs.iter_mut().enumerate().for_each(|(i, v)| {
            *v = i as f32;
        });
        xs[4] = 4.5f32;
        let mut diagonal = [0f32; 14];
        calc_diagonal(&xs, &mut diagonal);
        let expected = [
            4f32, 4f32, 5f32, 4f32, 3f32, 4f32, 4f32, 4f32, 4f32, 4f32, 4f32, 4f32, 4f32, 4f32,
        ];
        assert_eq!(expected, diagonal);
    }

    #[test]
    fn diagonal_test_2() {
        let mut xs = [0f32; 16];
        xs.iter_mut().enumerate().for_each(|(i, v)| {
            *v = i as f32;
        });

        xs[0] = 0.5f32;
        xs[4] = 4.5f32;
        xs[11] = 11.5f32;

        let mut diagonal = [0f32; 14];
        calc_diagonal(&xs, &mut diagonal);
        let expected = [
            3f32, 4f32, 5f32, 4f32, 3f32, 4f32, 4f32, 4f32, 4f32, 5f32, 4f32, 3f32, 4f32, 4f32,
        ];
        assert_eq!(expected, diagonal);
    }

    #[test]
    fn calc_r_test() {
        let mut xs = [0f32; 16];
        xs.iter_mut().enumerate().for_each(|(i, v)| {
            *v = i as f32;
        });

        xs[0] = 0.5f32;
        xs[4] = 4.5f32;
        xs[11] = 11.5f32;

        let ys = [
            0f32, 0f32, 1f32, 2f32, 4f32, 7f32, 9f32, 10f32, 8f32, 6f32, 3f32, 2f32, 2f32, 1f32,
            1f32, 0f32,
        ];

        let r = calc_r(&xs, &ys);
        let expected = [
            3f32, 0f32, 1f32, 14f32, -12f32, -3f32, -9f32, 0f32, -3f32, 7f32, 2f32, -3f32, 3f32,
            -3f32,
        ];
        for (r, expected) in (&r).iter().zip(&expected) {
            assert!(r - expected < 0.0001);
        }
    }

    #[test]
    fn calc_b_test() {
        let xs = [
            0f32, 1f32, 3f32, 6f32, 8f32, 9f32, 10f32, 12f32, 13f32, 14f32, 16f32, 17f32, 18f32,
            19f32, 20f32, 21f32,
        ];

        let ys = [
            0f32, 1f32, -2f32, 4f32, 1f32, -1f32, 0f32, 0f32, 1f32, 2f32, 4f32, 5f32, 4f32, 3f32,
            2f32, 0f32,
        ];
        let cs: Vector<f32, 16> = Vector::from_array([
            0f32, -1.8847, 1.9041, -1.5906, -0.15336, 2.6013, -1.2517, 0.95437, -0.22289,
            -0.062811, 0.29988, -1.6737, 0.39473, 0.094739, -0.77368, 0f32,
        ]);
        let b = calc_b(&xs, &ys, &cs);
        let expected = Vector::from_array([
            1.6282333,
            -0.25646675,
            -0.21759987,
            0.72303987,
            -2.76486,
            -0.31696665,
            1.0326867,
            0.43804997,
            1.1695304,
            0.883828,
            1.35798,
            -0.015776694,
            -1.294733,
            -0.805266,
            -1.4842134,
        ]);
        assert_eq!(b, expected);
    }

    #[test]
    fn solve_system_test() {
        let a = Matrix::from([[1f32, 2f32], [3f32, 4f32]]);
        let b = Vector::from([1f32, 2f32]);
        let inv = a.inv();

        let c = inv.dot(b);
        assert_eq!(b, a.dot(c))
    }

    #[test]
    fn calc_a_test() {
        let mut xs = [0f32; 16];
        xs.iter_mut().enumerate().for_each(|(i, v)| {
            *v = i as f32;
        });

        xs[0] = 0.5f32;
        xs[4] = 4.5f32;
        xs[11] = 11.5f32;

        let mut diag = [0f32; 14];
        calc_diagonal(&xs, &mut diag);

        let a = calc_matrix_a(&xs, &diag);
        let expected: [[f32; 14]; 14] = [
            [
                3.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                1.0f32, 4.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 1.0f32, 5.0f32, 1.5f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 1.5f32, 4.0f32, 0.5f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.5f32, 3.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 4.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 4.0f32, 1.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 4.0f32, 1.0f32, 0.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 4.0f32, 1.0f32,
                0.0f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 5.0f32,
                1.5f32, 0.0f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 1.5f32,
                4.0f32, 0.5f32, 0.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.5f32, 3.0f32, 1.0f32, 0.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 1.0f32, 4.0f32, 1.0f32,
            ],
            [
                0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32,
                0.0f32, 0.0f32, 1.0f32, 4.0f32,
            ],
        ];
        for i in 0..14 {
            for j in 0..14 {
                assert_eq!(expected[i][j], a[(i, j)]);
            }
        }
    }
}
