// possibly optimize for symmetrical matrix
pub fn thomas_algorithm(
    lower: &[f32],
    main: &mut [f32],
    upper: &[f32],
    r: &mut [f32],
    x: &mut [f32],
) -> Result<(), ()> {
    let n = main.len();
    if n < 4 {
        return Err(());
    }
    if lower.len() != n - 1 {
        return Err(());
    }
    if upper.len() != n - 1 {
        return Err(());
    }
    if r.len() != n {
        return Err(());
    }
    if x.len() != n {
        return Err(());
    }
    for i in 1..n {
        let mc = lower[i - 1] / main[i - 1];
        main[i] -= mc * upper[i - 1];
        r[i] -= mc * r[i - 1];
    }
    x.copy_from_slice(main);
    x[n - 1] = r[n - 1] / main[n - 1];

    for i in (0..=(n - 2)).rev() {
        x[i] = (r[i] - upper[i] * x[i + 1]) / main[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thomas_algorithm_14x14_test() {
        let mut main = [
            3.0f32, 4.0f32, 5.0f32, 4.0f32, 3.0f32, 4.0f32, 4.0f32, 4.0f32, 4.0f32, 5.0f32, 4.0f32,
            3.0f32, 4.0f32, 4.0f32,
        ];
        let upper = [
            1.0f32, 1.0f32, 1.5f32, 0.5f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.5f32, 0.5f32,
            1.0f32, 1.0f32,
        ];
        let lower = [
            1.0f32, 1.0f32, 1.5f32, 0.5f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.5f32, 0.5f32,
            1.0f32, 1.0f32,
        ];

        let mut r = [
            3f32, 0f32, 1f32, 14f32, -12f32, -3f32, -9f32, 0f32, -3f32, 7f32, 2f32, -3f32, 3f32,
            -3f32,
        ];

        let mut x = [0f32; 14];
        thomas_algorithm(&lower, &mut main, &upper, &mut r, &mut x).unwrap();

        let expected = [
            0.98287845,
            0.051364593,
            -1.1883368,
            4.593546,
            -5.1833534,
            1.2532874,
            -2.829796,
            1.0658972,
            -1.4337928,
            1.669274,
            0.05828173,
            -1.4740759,
            1.3930869,
            -1.0982717,
        ];
        assert_eq!(expected, x);
    }

    #[test]
    fn thomas_algorithm_4x4_test() {
        let lower = [3f32, 1f32, 3f32];
        let mut main = [10f32, 10f32, 7f32, 4f32];
        let upper = [2f32, 4f32, 5f32];

        let mut r = [3f32, 4f32, 5f32, 6f32];

        let mut x = [0f32; 4];
        thomas_algorithm(&lower, &mut main, &upper, &mut r, &mut x).unwrap();

        let expected = [0.14877588, 0.7561206, -1.0018834, 2.2514126];
        assert_eq!(expected, x);
    }

    #[test]
    fn thomas_method_test() {
        let lower = [3f32, 1f32, 1f32];
        let mut main = [1f32, 2f32, 2f32, 1f32];
        let upper = [2f32, 1f32, 3f32];
        let mut d = [1f32, 2f32, 3f32, 4f32];
        let mut x = [0f32; 4];
        thomas_algorithm(&lower, &mut main, &upper, &mut d, &mut x).unwrap();
        let expected = [-5.666666, 3.333333, 12.333332, -8.333332];
        assert_eq!(expected, x);
    }
}
