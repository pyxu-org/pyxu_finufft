import numpy as np
import pytest
import pyxu.runtime as pxrt

import pyxu_finufft.operator as pxo


class TestNUFFT2:
    # NUFFT2 is just NUFFT1's adjoint with a sign flip.
    # We therefore assume de-facto that it works if NUFFT1 works.
    # All we check here is that the outputs are as expected, i.e. we didn't make a mistake in flipping the sign/etc.

    @pytest.mark.parametrize("D", [1, 2, 3])
    def test_coherent_with_nufft1(self, D):
        rng = np.random.default_rng()

        M = 200
        N = (51, 52, 53)[:D]
        x = rng.uniform(-np.pi, np.pi, size=(M, D))
        width = pxrt.Width(x.dtype)

        with pxrt.Precision(width):
            kwargs = dict(
                x=x,
                N=N,
                eps=1e-5,
                enable_warnings=False,
            )
            A = pxo.NUFFT1(isign=1, **kwargs)
            B = pxo.NUFFT2(isign=-1, **kwargs)

        # Helper functions
        norm = lambda x: np.sqrt(np.sum(x**2))
        rel_err = lambda a, b: norm(a - b) / norm(b)

        sh = (2, 1, 3)
        wR = rng.standard_normal(size=(*sh, M, 2))
        wC = wR[..., 0] + 1j * wR[..., 1]
        assert rel_err(A.apply(wR), B.adjoint(wR)) <= 1e-6
        assert rel_err(A.capply(wC), B.cadjoint(wC)) <= 1e-6

        L = 2 * np.r_[N] + 1
        vR = rng.standard_normal(size=(*sh, *L, 2))
        vC = vR[..., 0] + 1j * vR[..., 1]
        assert rel_err(A.adjoint(vR), B.apply(vR)) <= 1e-6
        assert rel_err(A.cadjoint(vC), B.capply(vC)) <= 1e-6
