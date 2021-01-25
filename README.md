# splinterpol-baremetal
Bare metal cubic spline interpolation with Rust [no\_std]

Given 2d points on the plane, calculate piecewise polynomials which connect
the points in a most smooth way.

There can be an arbitrary (const) number of points. Due to limitations in
num-trait or my coding abilities, only f32 coordinates are supported.

![spline.png](https://github.com/barafael/splinterpol-baremetal/blob/main/16-points.png)

![elephant.png](https://github.com/barafael/splinterpol-baremetal/blob/main/elephant.png)
