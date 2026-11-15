import unittest
import numpy as np
from butterfly import butterfly_dft, butterfly_fio, dense_dft, dense_fio, fft_dft


class ButterflyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_fft_matches_dense(self):
        for n in (2, 4, 8, 16):
            f = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
            np.testing.assert_allclose(fft_dft(f), dense_dft(f), rtol=2e-14, atol=2e-14)

    def test_butterfly_matches_dense_and_converges(self):
        n = 16
        f = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
        exact = dense_dft(f)
        errors = []
        for q in (4, 6, 8, 10):
            approximate = butterfly_dft(f, q)
            errors.append(np.linalg.norm(approximate - exact) / np.linalg.norm(exact))
        self.assertLess(errors[-1], 1e-7)
        self.assertLess(errors[-1], errors[0])

    def test_error_is_stable_as_n_grows(self):
        errors = []
        for n in (16, 32, 64):
            f = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
            exact = fft_dft(f)
            approximate = butterfly_dft(f, q=8)
            errors.append(np.linalg.norm(approximate-exact) / np.linalg.norm(exact))
        self.assertLess(max(errors), 2e-5)
        self.assertLess(max(errors) / min(errors), 5.0)

    def test_non_dft_kernel_matches_dense(self):
        n = 32
        f = self.rng.normal(size=n) + 1j * self.rng.normal(size=n)
        phase = lambda x, p, scale: scale * (x*p + 0.1*x**2*p**2)
        exact = dense_fio(f, phase)
        approximate = butterfly_fio(f, q=10, phase=phase)
        error = np.linalg.norm(approximate-exact) / np.linalg.norm(exact)
        self.assertLess(error, 2e-7)


if __name__ == "__main__":
    unittest.main()
