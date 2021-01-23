//! A library for using cubic spline interpolation on no_std.

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(not(test), no_std)]

mod plot_spline;
mod thomas_algorithm;

/// The possible errors of this crate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Slice of invalid length passed
    InvalidSliceLength,
}

/// Given xs and ys of same length n, calculate the coefficients of n-1 cubic
/// polynomials.
pub fn splinterpol<const N: usize>(
    xs: &[f32],
    ys: &[f32],
    coefficients: &mut [(f32, f32, f32, f32)],
) -> Result<(), Error> {
    // Array size const expression workaround
    let mut diagonal = [0f32; N];
    let mut diagonal = &mut diagonal[0..N - 2];

    calc_diagonal::<N>(&xs, &mut diagonal).unwrap();

    let mut r = [0f32; N];
    let mut r = &mut r[0..N - 2];

    if let Err(e) = calc_r::<N>(&xs, &ys, &mut r) {
        return Err(e);
    }

    let mut sub_diagonal = [0f32; N];
    let mut sub_diagonal = &mut sub_diagonal[0..N - 3];

    if let Err(e) = calc_subdiagonal(&xs, &mut sub_diagonal) {
        return Err(e);
    }

    let c = {
        let mut c = [0f32; N];
        let mut c_body = &mut c[1..N - 1];
        if let Err(e) = thomas_algorithm::thomas_algorithm_symmetric(
            &sub_diagonal,
            &mut diagonal,
            &mut r,
            &mut c_body,
        ) {
            return Err(e);
        }
        c
    };

    let mut b = [0f32; N];
    let mut b = &mut b[0..N - 1];

    if let Err(e) = calc_b::<N>(&xs, &ys, &c, &mut b) {
        return Err(e);
    }

    let mut d = [0f32; N];
    let mut d = &mut d[0..N - 1];

    if let Err(e) = calc_d::<N>(&xs, &c, &mut d) {
        return Err(e);
    }

    for i in 0..N - 1 {
        coefficients[i].0 = ys[i];
        coefficients[i].1 = b[i];
        coefficients[i].2 = c[i];
        coefficients[i].3 = d[i];
    }
    Ok(())
}

fn calc_subdiagonal(vals: &[f32], sub: &mut [f32]) -> Result<(), Error> {
    if vals.len() != sub.len() + 3 {
        return Err(Error::InvalidSliceLength);
    }
    let n = vals.len();
    for i in 0..(n - 3) {
        sub[i] = vals[i + 2] - vals[i + 1];
    }
    Ok(())
}

fn cubic_spline(a: f32, b: f32, c: f32, d: f32, vec: &mut [f32], step_size: f32) {
    for (i, v) in vec.iter_mut().enumerate() {
        let x = i as f32 * step_size;
        let value = a + b * x + c * (x * x) + d * (x * x * x);
        *v = value;
    }
}

fn h(i: usize, vals: &[f32]) -> f32 {
    vals[i + 1] - vals[i]
}

fn calc_diagonal<const N: usize>(xs: &[f32], result: &mut [f32]) -> Result<(), Error> {
    if xs.len() != N {
        return Err(Error::InvalidSliceLength);
    }
    for i in 0..N - 2 {
        result[i] = 2f32 * (h(i, &xs) + h(i + 1, &xs));
    }
    Ok(())
}

fn calc_r<const N: usize>(xs: &[f32], ys: &[f32], r: &mut [f32]) -> Result<(), Error> {
    if r.len() != N - 2 {
        return Err(Error::InvalidSliceLength);
    }
    if xs.len() != N {
        return Err(Error::InvalidSliceLength);
    }
    if ys.len() != N {
        return Err(Error::InvalidSliceLength);
    }
    for i in 0..N - 2 {
        let div1 = (ys[i + 2] - ys[i + 1]) / (h(i + 1, &xs));
        let div2 = (ys[i + 1] - ys[i]) / (h(i, &xs));
        r[i] = 3f32 * (div1 - div2);
    }
    Ok(())
}

fn calc_b<const N: usize>(xs: &[f32], ys: &[f32], cs: &[f32], b: &mut [f32]) -> Result<(), Error> {
    if cs.len() != N {
        return Err(Error::InvalidSliceLength);
    }
    if b.len() != N - 1 {
        return Err(Error::InvalidSliceLength);
    }
    for i in 0..N - 1 {
        let div_1 = (ys[i + 1] - ys[i]) / (h(i, &xs));
        let div_2 = (2f32 * cs[i] + cs[i + 1]) / 3f32;
        b[i] = div_1 - div_2 * h(i, &xs);
    }
    Ok(())
}

fn calc_d<const N: usize>(xs: &[f32], cs: &[f32], d: &mut [f32]) -> Result<(), Error> {
    if xs.len() != N {
        return Err(Error::InvalidSliceLength);
    }
    if cs.len() != N {
        return Err(Error::InvalidSliceLength);
    }
    if d.len() != N - 1 {
        return Err(Error::InvalidSliceLength);
    }
    for i in 0..N - 1 {
        d[i] = (cs[i + 1] - cs[i]) / (3f32 * h(i, &xs));
    }
    Ok(())
}

/// Plot given coefficients into the buffer according to the intervals given in xs
pub fn plot_coeffs_into(
    buffer: &mut [f32],
    coefficients: &[(f32, f32, f32, f32)],
    xs: &[f32],
) -> Result<(), ()> {
    let x_range = xs.last().unwrap() - xs.first().unwrap();
    let step_size = x_range as f64 / buffer.len() as f64;
    let mut current_index = 0;
    for i in 0..coefficients.len() {
        let range = xs[i + 1] - xs[i];
        let ratio = range / x_range;
        // f32::round not available in no_std
        let buffer_ratio = {
            let r = buffer.len() as f32 * ratio;
            if r - ((r as u32) as f32) < 0.5 {
                r as u32
            } else {
                r as u32 + 1
            }
        };
        let mut upper = current_index + buffer_ratio as usize;
        if upper >= buffer.len() {
            upper = buffer.len()
        };
        let mut current_slice = &mut buffer[current_index..upper];
        cubic_spline(
            coefficients[i].0,
            coefficients[i].1,
            coefficients[i].2,
            coefficients[i].3,
            &mut current_slice,
            step_size as f32,
        );
        current_index += buffer_ratio as usize;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_coeffs() {
        let coeffs: [(f32, f32, f32, f32); 15] = [
            (0.0, -0.16381307, 0.0, 0.6552523),
            (0.0, 0.32762617, 0.98287845, -0.31050465),
            (1.0, 1.3618692, 0.051364563, -0.41323376),
            (2.0, 0.22489715, -1.1883367, 1.2848629),
            (4.0, 5.3327103, 4.593546, -6.517933),
            (7.0, 5.0378065, -5.1833534, 2.145547),
            (9.0, 1.1077404, 1.2532874, -1.3610278),
            (10.0, -0.46876848, -2.829796, 1.2985644),
            (8.0, -2.2326672, 1.0658972, -0.83323),
            (6.0, -2.6005628, -1.4337928, 1.0343556),
            (3.0, -2.3650815, 1.669274, -0.35799825),
            (2.0, 0.22625208, 0.05828173, -1.0215718),
            (2.0, -0.48164505, -1.4740759, 0.9557209),
            (1.0, -0.56263405, 1.3930869, -0.8304529),
            (1.0, -0.26781887, -1.0982717, 0.36609057),
        ];
        let mut buffer = [0f32; 100];
        let xs = [
            0.5f32, 1f32, 2f32, 3f32, 4.5f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 11.5f32, 12f32,
            13f32, 14f32, 15f32,
        ];
        plot_coeffs_into(&mut buffer, &coeffs, &xs).unwrap();
        dbg!(buffer);
    }

    #[test]
    fn test_splinterpol() {
        let xs = [
            0.5f32, 1f32, 2f32, 3f32, 4.5f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 11.5f32, 12f32,
            13f32, 14f32, 15f32,
        ];
        let ys = [
            0f32, 0f32, 1f32, 2f32, 4f32, 7f32, 9f32, 10f32, 8f32, 6f32, 3f32, 2f32, 2f32, 1f32,
            1f32, 0f32,
        ];
        let mut coeffs = [(0f32, 0f32, 0f32, 0f32); 15];
        splinterpol::<16>(&xs, &ys, &mut coeffs).unwrap();
        let expected: [(f32, f32, f32, f32); 15] = [
            (0.0, -0.16381307, 0.0, 0.6552523),
            (0.0, 0.32762617, 0.98287845, -0.31050465),
            (1.0, 1.3618692, 0.051364563, -0.41323376),
            (2.0, 0.22489715, -1.1883367, 1.2848629),
            (4.0, 5.3327103, 4.593546, -6.517933),
            (7.0, 5.0378065, -5.1833534, 2.145547),
            (9.0, 1.1077404, 1.2532874, -1.3610278),
            (10.0, -0.46876848, -2.829796, 1.2985644),
            (8.0, -2.2326672, 1.0658972, -0.83323),
            (6.0, -2.6005628, -1.4337928, 1.0343556),
            (3.0, -2.3650815, 1.669274, -0.35799825),
            (2.0, 0.22625208, 0.05828173, -1.0215718),
            (2.0, -0.48164505, -1.4740759, 0.9557209),
            (1.0, -0.56263405, 1.3930869, -0.8304529),
            (1.0, -0.26781887, -1.0982717, 0.36609057),
        ];
        assert_eq!(expected, coeffs);
    }

    #[test]
    fn test_splinterpol_8x8() {
        let xs = [0.5f32, 1f32, 2f32, 3f32, 4.5f32, 5f32, 6f32, 7f32];
        let ys = [0f32, 0f32, 1f32, 2f32, 4f32, 7f32, 9f32, 10f32];
        let mut coeffs = [(0f32, 0f32, 0f32, 0f32); 7];
        splinterpol::<8>(&xs, &ys, &mut coeffs).unwrap();

        let mut buffer = [0f32; 1000];
        plot_coeffs_into(&mut buffer, &coeffs, &xs).unwrap();

        let expected = [
            (0.0, -0.16399321, 0.0, 0.65597284),
            (0.0, 0.32798642, 0.98395926, -0.31194568),
            (1.0, 1.360068, 0.048122242, -0.40819016),
            (2.0, 0.2317419, -1.1764482, 1.273895),
            (4.0, 5.301188, 4.5560794, -6.316911),
            (7.0, 5.119584, -4.9192877, 1.7997031),
            (9.0, 0.6801188, 0.47982174, -0.15994059),
        ];
        assert_eq!(expected, coeffs);
    }

    #[test]
    fn plot_splinterpol() {
        use plotters::prelude::*;

        let xs = [
            0.5f32, 1f32, 2f32, 3f32, 4.5f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 11.5f32, 12f32,
            13f32, 14f32, 15f32,
        ];

        let coeffs: [(f32, f32, f32, f32); 15] = [
            (0.0, -0.16381307, 0.0, 0.6552523),
            (0.0, 0.32762617, 0.98287845, -0.31050465),
            (1.0, 1.3618692, 0.051364563, -0.41323376),
            (2.0, 0.22489715, -1.1883367, 1.2848629),
            (4.0, 5.3327103, 4.593546, -6.517933),
            (7.0, 5.0378065, -5.1833534, 2.145547),
            (9.0, 1.1077404, 1.2532874, -1.3610278),
            (10.0, -0.46876848, -2.829796, 1.2985644),
            (8.0, -2.2326672, 1.0658972, -0.83323),
            (6.0, -2.6005628, -1.4337928, 1.0343556),
            (3.0, -2.3650815, 1.669274, -0.35799825),
            (2.0, 0.22625208, 0.05828173, -1.0215718),
            (2.0, -0.48164505, -1.4740759, 0.9557209),
            (1.0, -0.56263405, 1.3930869, -0.8304529),
            (1.0, -0.26781887, -1.0982717, 0.36609057),
        ];

        let mut buffer = [0f32; 1000];
        plot_coeffs_into(&mut buffer, &coeffs, &xs).unwrap();

        let root = BitMapBackend::new("0.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("spline", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f32..1000f32, 0.0f32..15f32)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                buffer.iter().enumerate().map(|(i, v)| (i as f32, *v)),
                &RED,
            ))
            .unwrap();

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .unwrap()
    }

    #[test]
    fn test_calc_subdiagonal() {
        let xs = [
            0.5f32, 1f32, 2f32, 3f32, 4.5f32, 5f32, 6f32, 7f32, 8f32, 9f32, 10f32, 11.5f32, 12f32,
            13f32, 14f32, 15f32,
        ];
        let mut sub = [0f32; 13];
        calc_subdiagonal(&xs, &mut sub).unwrap();
        let expected = [
            1.0, 1.0, 1.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 0.5, 1.0, 1.0,
        ];
        assert_eq!(expected, sub);
    }

    #[test]
    fn do_cubic_spline() {
        let mut xs = [0f32; 64];
        cubic_spline(4.0, 2.0, 2.0, 1.5, &mut xs, 0.05);
        let expected = [
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
        calc_diagonal::<16>(&xs, &mut diagonal).unwrap();
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
        calc_diagonal::<16>(&xs, &mut diagonal).unwrap();
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

        let mut r = [0f32; 14];
        calc_r::<16>(&xs, &ys, &mut r).unwrap();
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
        let xs: [f32; 16] = [
            0.0, 1.0, 3.0, 6.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0,
        ];

        let ys: [f32; 16] = [
            0.0, 1.0, -2.0, 4.0, 1.0, -1.0, 0.0, 0.0, 1.0, 2.0, 4.0, 5.0, 4.0, 3.0, 2.0, 0.0,
        ];
        let cs: [f32; 16] = [
            0.0, -1.8847, 1.9041, -1.5906, -0.15336, 2.6013, -1.2517, 0.95437, -0.22289, -0.062811,
            0.29988, -1.6737, 0.39473, 0.094739, -0.77368, 0.0,
        ];
        let mut b = [0f32; 15];
        calc_b::<16>(&xs, &ys, &cs, &mut b).unwrap();
        let expected: [f32; 15] = [
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
        ];
        assert_eq!(expected, b);
    }

    #[test]
    fn calc_d_test_1() {
        let xs: [f32; 16] = [
            0.0, 1.0, 3.0, 6.0, 8.0, 9.0, 10.0, 12.0, 13.0, 14.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0,
        ];

        let cs = [
            0f32, -1.8847, 1.9041, -1.5906, -0.15336, 2.6013, -1.2517, 0.95437, -0.22289,
            -0.062811, 0.29988, -1.6737, 0.39473, 0.094739, -0.77368, 0f32,
        ];
        let mut d = [0f32; 15];
        calc_d::<16>(&xs, &cs, &mut d).unwrap();
        let expected: [f32; 15] = [
            -0.6282333,
            0.6314666,
            -0.3883,
            0.23954,
            0.91822,
            -1.2843333,
            0.3676783,
            -0.39242002,
            0.05335967,
            0.060448498,
            -0.65786,
            0.68947667,
            -0.09999701,
            -0.289473,
            0.25789332,
        ];
        assert_eq!(expected, d);
    }

    #[test]
    fn calc_d_test_2() {
        let xs: [f32; 16] = [
            0.5, 1.0, 2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.5, 12.0, 13.0, 14.0, 15.0,
        ];

        let cs = [
            0.0, 0.98288, 0.051365, -1.1883, 4.5935, -5.1834, 1.2533, -2.8298, 1.0659, -1.4338,
            1.6693, 0.058282, -1.4741, 1.3931, -1.0983, 0.0,
        ];

        let mut d = [0f32; 15];
        calc_d::<16>(&xs, &cs, &mut d).unwrap();
        let expected: [f32; 15] = [
            0.65525335,
            -0.310505,
            -0.4132217,
            1.2848445,
            -6.5179334,
            2.1455667,
            -1.3610333,
            1.2985667,
            -0.83323336,
            1.0343666,
            -0.35800397,
            -1.021588,
            0.9557333,
            -0.8304667,
            0.36609998,
        ];
        assert_eq!(expected, d);
    }
}
