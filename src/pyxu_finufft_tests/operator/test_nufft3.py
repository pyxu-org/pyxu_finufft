import itertools

import numpy as np
import pytest
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as conftest

import pyxu_finufft.operator as pxo


class TestNUFFT3(conftest.LinOpT):
    @classmethod
    def _metric(
        cls,
        a: pxt.NDArray,
        b: pxt.NDArray,
        as_dtype: pxt.DType,
    ) -> bool:
        # NUFFT is an approximate transform.
        # Based on [FINUFFT], results hold up to a small relative error.
        #
        # We choose a conservative threshold, irrespective of the `eps` parameter chosen by the
        # user. Additional tests below test explicitly if computed values correctly obey `eps`.
        eps_default = 1e-3

        cast = lambda x: pxu.to_NUMPY(x)
        lhs = np.linalg.norm((cast(a) - cast(b)).ravel())
        rhs = np.linalg.norm(cast(b).ravel())
        return ct.less_equal(lhs, eps_default * rhs, as_dtype=as_dtype).all()

    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
            ],
            pxrt.CWidth,
        )
    )
    def spec(
        self,
        x_spec,
        v_spec,
        isign,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        x_spec = xp.array(x_spec, dtype=width.real.value)
        v_spec = xp.array(v_spec, dtype=width.real.value)

        with pxrt.Precision(width.real):
            op = pxo.NUFFT3(
                x=x_spec,
                v=v_spec,
                isign=isign,
                eps=1e-5,  # tested manually -> works
                enable_warnings=False,
            )
        return op, ndi, width

    @pytest.fixture
    def dim_shape(self, x_spec) -> pxt.NDArrayShape:
        # size of inputs, and not the transform dimensions!
        return (len(x_spec), 2)

    @pytest.fixture
    def codim_shape(self, v_spec) -> pxt.NDArrayShape:
        return (len(v_spec), 2)

    @pytest.fixture
    def data_apply(
        self,
        x_spec,
        v_spec,
        isign,
    ) -> conftest.DataLike:
        M = len(x_spec)
        x = self._random_array((M,)) + 1j * self._random_array((M,))

        A = np.exp((1j * isign) * (v_spec @ x_spec.T))  # (N, M)
        y = A @ x  # (N,)

        return dict(
            in_=dict(arr=pxu.view_as_real(x)),
            out=pxu.view_as_real(y),
        )

    # Fixtures (internal) -----------------------------------------------------
    @pytest.fixture(params=[1, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def x_spec(self, space_dim) -> np.ndarray:
        # (M, D) canonical point cloud [NUMPY]
        M = 150
        rng = np.random.default_rng()

        x = np.zeros((M, space_dim))
        for d in range(space_dim):
            x[:, d] = rng.uniform(-5, 6, size=M)
        return x

    @pytest.fixture
    def v_spec(self, space_dim) -> np.ndarray:
        # (N, D) canonical point cloud [NUMPY]
        N = 151
        rng = np.random.default_rng()

        v = np.zeros((N, space_dim))
        for d in range(space_dim):
            v[:, d] = rng.uniform(-2, 3, size=N)
        return v

    @pytest.fixture(params=[1, -1])
    def isign(self, request) -> int:
        return request.param

    # Tests -------------------------------------------------------------------
