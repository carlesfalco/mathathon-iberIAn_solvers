//! Minimal complex arithmetic. No external crates, so the whole reference builds offline.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

pub const fn c(re: f64, im: f64) -> C64 {
    C64 { re, im }
}

impl C64 {
    pub const ZERO: C64 = c(0.0, 0.0);
    pub const ONE: C64 = c(1.0, 0.0);

    pub fn conj(self) -> Self {
        c(self.re, -self.im)
    }
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.norm_sqr().sqrt()
    }
    pub fn scale(self, a: f64) -> Self {
        c(self.re * a, self.im * a)
    }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let (s, co) = theta.sin_cos();
        c(r * co, r * s)
    }
    pub fn div(self, o: C64) -> C64 {
        let d = o.norm_sqr();
        if d == 0.0 {
            C64::ZERO
        } else {
            (self * o.conj()).scale(1.0 / d)
        }
    }
    pub fn is_one(self) -> bool {
        (self.re - 1.0).abs() < 1e-15 && self.im.abs() < 1e-15
    }
}

impl Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        c(self.re + o.re, self.im + o.im)
    }
}
impl Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        c(self.re - o.re, self.im - o.im)
    }
}
impl Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        c(-self.re, -self.im)
    }
}
impl Mul for C64 {
    type Output = C64;
    fn mul(self, o: C64) -> C64 {
        c(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl Mul<f64> for C64 {
    type Output = C64;
    fn mul(self, a: f64) -> C64 {
        self.scale(a)
    }
}
impl Div<f64> for C64 {
    type Output = C64;
    fn div(self, a: f64) -> C64 {
        self.scale(1.0 / a)
    }
}
impl AddAssign for C64 {
    fn add_assign(&mut self, o: C64) {
        *self = *self + o;
    }
}
impl SubAssign for C64 {
    fn sub_assign(&mut self, o: C64) {
        *self = *self - o;
    }
}

/// `exp(+2*pi*i*m/n)` with the project's sign convention.
///
/// `updated/unified/core/kernels.py::fio1d` uses `K(x, xi) = exp(+2 pi i x xi)`.
/// NumPy's `fft` uses the opposite sign, so the reference this crate is checked
/// against is `(-1)^j * N * np.fft.ifft(g)` (see `exp/repro_e2.py` and
/// `updated/unified/showcase/models.py::check_dft`), not `np.fft.fft(g)`.
/// Each value is computed from `sin_cos` of a reduced argument, so no error
/// accumulates across a stage.
pub fn twiddle(m: i64, n: usize) -> C64 {
    let r = m.rem_euclid(n as i64) as f64;
    C64::from_polar(1.0, 2.0 * std::f64::consts::PI * r / (n as f64))
}
