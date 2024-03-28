import numpy as np
import pytest
import pyxu.info.deps as pxd
import pyxu.runtime as pxrt

from pyxu_finufft import Flip


def allclose(x, y, xp):
    return xp.allclose(x, y)


@pytest.fixture(params=pxd.NDArrayInfo)
def ndi(request):
    ndi_ = request.param
    if ndi_.module() is None:
        pytest.skip(f"{ndi_} unsupported on this machine.")
    return ndi_


@pytest.fixture(params=pxrt.Width)
def width(request):
    return request.param


@pytest.fixture(
    params=[
        # arg_shape, #axis
        ((2,), 0),
        ((2,), -1),
        ((2, 3, 4), 0),
        ((2, 3, 4), 1),
        ((2, 3, 4), 2),
        ((2, 3, 4), -1),
    ]
)
def params(request):
    return request.param


@pytest.fixture
def arg_shape(params):
    return params[0]


@pytest.fixture
def axis(params):
    return params[1]


@pytest.fixture
def batch_shape():
    return 3, 2


@pytest.fixture(params=[0, 1, 2, 3, 4])
def seed(request):
    return request.param


@pytest.fixture
def data(arg_shape, batch_shape, seed, ndi, width, axis):
    # Make a batch of real & symmetric input matrices (PSD)
    rng = np.random.default_rng(seed=seed)
    input_shape = batch_shape + arg_shape
    x = rng.normal(size=input_shape)

    xp = ndi.module()
    axis = (len(arg_shape) - 1) if axis == -1 else axis
    axis += len(batch_shape)
    return {"in_": xp.array(x.reshape(*batch_shape, -1)), "out_gt": np.flip(x, axis=axis)}


def test_linop(data, axis, arg_shape, batch_shape, ndi):
    input_shape = batch_shape + arg_shape
    flip = Flip(arg_shape=arg_shape, axis=axis)
    out = flip(data["in_"]).reshape(*input_shape)
    assert np.allclose(out, data["out_gt"])
