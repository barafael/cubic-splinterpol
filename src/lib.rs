#![cfg_attr(not(test), no_std)]

mod plot;
mod thomas_algorithm;

const NUM_ELEMENTS: usize = 16;

pub fn splinterpol(
    xs: &[f32],
    ys: &[f32],
    coefficients: &mut [(f32, f32, f32, f32)],
) -> Result<(), ()> {
    let mut diagonal = [0f32; NUM_ELEMENTS - 2];
    calc_diagonal(&xs, &mut diagonal);

    let mut r = [0f32; 14];
    if let Err(()) = calc_r(&xs, &ys, &mut r) {
        return Err(());
    }

    let mut sub_diagonal = [0f32; 13];
    if let Err(()) = calc_subdiagonal(&xs, &mut sub_diagonal) {
        return Err(());
    }

    let c = {
        let mut c = [0f32; 16];
        let mut c_body = &mut c[1..15];
        if let Err(()) = thomas_algorithm::thomas_algorithm(
            &sub_diagonal,
            &mut diagonal,
            &sub_diagonal,
            &mut r,
            &mut c_body,
        ) {
            return Err(());
        }
        c
    };

    let mut b = [0f32; 15];
    if let Err(()) = calc_b(&xs, &ys, &c, &mut b) {
        return Err(());
    }

    let mut d = [0f32; 15];
    if let Err(()) = calc_d(&xs, &c, &mut d) {
        return Err(());
    }

    for i in 0..15 {
        coefficients[i].0 = ys[i];
        coefficients[i].1 = b[i];
        coefficients[i].2 = c[i];
        coefficients[i].3 = d[i];
    }
    Ok(())
}

fn calc_subdiagonal(vals: &[f32], sub: &mut [f32]) -> Result<(), ()> {
    assert_eq!(vals.len(), 3 + sub.len());
    let n = vals.len();
    for i in 0..(n - 3) {
        sub[i] = vals[i + 2] - vals[i + 1];
    }
    Ok(())
}

fn cubic_spline(a: f32, b: f32, c: f32, d: f32, vec: &mut [f32], start: f32, step_size: f32) {
    for (i, v) in vec.iter_mut().enumerate() {
        let x = i as f32 * step_size;
        let value = a + b * x + c * (x * x) + d * (x * x * x);
        *v = value;
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

fn calc_r(xs: &[f32], ys: &[f32], r: &mut [f32]) -> Result<(), ()> {
    if r.len() != 14 {
        return Err(());
    }
    if xs.len() != 16 {
        return Err(());
    }
    if ys.len() != 16 {
        return Err(());
    }
    for i in 0..14 {
        let div1 = (ys[i + 2] - ys[i + 1]) / (h(i + 1, &xs));
        let div2 = (ys[i + 1] - ys[i]) / (h(i, &xs));
        r[i] = 3f32 * (div1 - div2);
    }
    Ok(())
}

fn calc_b(xs: &[f32], ys: &[f32], cs: &[f32], b: &mut [f32]) -> Result<(), ()> {
    if cs.len() != 16 {
        return Err(());
    }
    if b.len() != 15 {
        return Err(());
    }
    for i in 0..15 {
        let div_1 = (ys[i + 1] - ys[i]) / (h(i, &xs));
        let div_2 = (2f32 * cs[i] + cs[i + 1]) / 3f32;
        b[i] = div_1 - div_2 * h(i, &xs);
    }
    Ok(())
}

fn calc_d(xs: &[f32], cs: &[f32], d: &mut [f32]) -> Result<(), ()> {
    if xs.len() != 16 {
        return Err(());
    }
    if cs.len() != 16 {
        return Err(());
    }
    if d.len() != 15 {
        return Err(());
    }
    for i in 0..15 {
        d[i] = (cs[i + 1] - cs[i]) / (3f32 * h(i, &xs));
    }
    Ok(())
}

fn plot_coeffs_into(buffer: &mut [f32], coefficients: &[(f32, f32, f32, f32)], xs: &[f32]) -> Result<(), ()> {
    let x_range = xs.last().unwrap() - xs.first().unwrap();
    let step_size = x_range as f64 / buffer.len() as f64;
    let mut current_index = 0;
    for i in 0..coefficients.len() {
        let range = xs[i+1] - xs[i];
        let ratio = range / x_range;
        let buffer_ratio = buffer.len() as f32 * ratio;
        let mut current_slice = &mut buffer[current_index..current_index+buffer_ratio.round() as usize];
        cubic_spline(coefficients[i].0, coefficients[i].1, coefficients[i].2, coefficients[i].3, &mut current_slice, xs[i], step_size as f32);
        current_index += buffer_ratio.round() as usize;
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
        splinterpol(&xs, &ys, &mut coeffs).unwrap();
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

        let mut buffer = [0f32; 100];
        plot_coeffs_into(&mut buffer, &coeffs, &xs).unwrap();

        let root = BitMapBackend::new("0.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("spline", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f32..100f32, 0.0f32..15f32)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                buffer.iter().enumerate().map(|(i, v)| {(i as f32, *v)}),
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
        cubic_spline(4.0, 2.0, 2.0, 1.5, &mut xs, 0.0, 0.05);
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

        let mut r = [0f32; 14];
        calc_r(&xs, &ys, &mut r).unwrap();
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
        calc_b(&xs, &ys, &cs, &mut b).unwrap();
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
        calc_d(&xs, &cs, &mut d).unwrap();
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
        calc_d(&xs, &cs, &mut d).unwrap();
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
