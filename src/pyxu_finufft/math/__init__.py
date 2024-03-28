import numba
import numpy as np
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.util as pxu

__all__ = ["eigh"]


def eigh(arr: pxt.NDArray, arg_shape: pxt.NDArrayShape, normalize=True):
    r"""
    Batch computation of the eigenvalues and eigenvectors of a batch (:math:`\mathcal{N}`-shaped, where
    :math:`\mathcal{N} = (N_{1}, \cdots, N_{k})`) of complex Hermitian (conjugate symmetric) or a real symmetric
    matrices.

    This function leverages Numba's GUVectorize to improve performance.

    Returns two objects, an :math:`\mathcal{N}`-shaped batch of 1-D arrays containing the eigenvalues, and an
    :math:`\mathcal{N}`-shaped batch of 2-D square matrices of the corresponding eigenvectors (in columns).

    Closed form 2x2 from https://hal.science/hal-01501221/document
    Closed form 3x3 from https://www.wikiwand.com/en/Eigenvalue_algorithm


    Parameters
    ----------
    arr: (…, M, M) pxt.NDArray
    arg_shape: pxt.NDArrayShape
    normalize: bool

    Returns
    -------
    w: (…, M) pxt.NDArray
        The eigenvalues in ascending order, each repeated according to its multiplicity.
    v: (…, M, M) pxt.NDArray
        The column v[..., i] is the normalized eigenvector corresponding to the eigenvalue w[..., i].
    """
    assert len(np.unique(arg_shape))
    st_sh = arr.shape[:-1]
    st_size = np.prod(st_sh)
    N = pxd.NDArrayInfo
    xp = pxu.get_array_module(arr)

    if N.from_obj(arr) == N.DASK:
        chunksize = arr.chunksize
        arr = arr.reshape(*st_sh, *arg_shape)
        arr = arr.rechunk(chunks=(*((-1),) * len(st_sh), *((-1),) * len(arg_shape)))
    else:
        arr = arr.reshape(st_size, *arg_shape)

    w = xp.zeros_like(arr[..., 0])
    v = xp.zeros_like(arr)

    w, v = eigh_dispatcher(arr, w, v)

    if normalize:
        v /= xp.linalg.norm(v, axis=-2, keepdims=True)

    w = w.reshape((*st_sh, -1))
    v = v.reshape((*st_sh, -1))

    if N.from_obj(arr) == N.DASK:
        w = w.rechunk(chunks=(*chunksize[:-1], -1))
        v = v.rechunk(chunks=(*chunksize[:-1], -1))
    return w, v


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:]),
        (numba.float64[:, :], numba.float64[:]),
    ],
    "(m, m) -> (m)",
)
def compute_eigenvalues_2x2(arr, out):
    d = 4 * arr[0, 1] ** 2 + (arr[0, 0] - arr[1, 1]) ** 2
    d = d**0.5
    out[0] = (arr[0, 0] + arr[1, 1] - d) / 2
    out[1] = (arr[0, 0] + arr[1, 1] + d) / 2

    if out[0] > out[1]:
        out[0], out[1] = out[1], out[0]


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:], numba.float32[:, :]),
        (numba.float64[:, :], numba.float64[:], numba.float64[:, :]),
    ],
    "(m, m), (m) -> (m, m)",
)
def compute_eigenvectors_2x2(arr, eigval, out):
    p = out[1, 0] ** 2 + out[0, 1] ** 2
    if p == 0:
        # A is diagonal.
        # First eigenvector
        out[0, 0] = 1
        out[1, 0] = 0
        # Second eigenvector
        out[0, 1] = 0
        out[1, 1] = 1
    else:
        # First eigenvector
        out[0, 0] = (eigval[1] - arr[0, 0]) / arr[0, 1]
        out[1, 0] = 1
        # Second eigenvector
        out[0, 1] = (eigval[0] - arr[0, 0]) / arr[0, 1]
        out[1, 1] = 1


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:]),
        (numba.float64[:, :], numba.float64[:]),
    ],
    "(m, m) -> (m)",
)
def compute_eigenvalues_3x3(arr, out):
    a = arr[0, 0]
    d = arr[0, 1]
    f = arr[0, 2]
    b = arr[1, 1]
    e = arr[1, 2]
    c = arr[2, 2]

    p1 = d**2 + f**2 + e**2

    if p1 == 0:
        # A is diagonal.
        out[0] = a
        out[1] = b
        out[2] = c
    else:
        q = (a + b + c) / 3  # trace(A) is the sum of all diagonal values
        p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1
        p = p2 / 6
        p = p**0.5
        det_b = (a - q) * ((b - q) * (c - q) - e**2) - d * (d * (c - q) - e * f) + f * (d * e - (b - q) * f)
        r = det_b / (2 * p**3)

        # In exact arithmetic for a symmetric matrix -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if r <= -1:
            phi = np.pi / 3
        elif r >= 1:
            phi = 0
        else:
            phi = np.arccos(r) / 3

        # the eigenvalues satisfy eig3 <= eig2 <= eig1
        out[0] = q + 2 * p * np.cos(phi + (2 * np.pi / 3))
        out[2] = q + 2 * p * np.cos(phi)
        out[1] = 3 * q - out[0] - out[2]  # since trace(A) = eig1 + eig2 + eig3

    if out[0] > out[1]:
        out[0], out[1] = out[1], out[0]
    if out[1] > out[2]:
        out[1], out[2] = out[2], out[1]
    if out[0] > out[1]:
        out[0], out[1] = out[1], out[0]


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:], numba.float32[:, :]),
        (numba.float64[:, :], numba.float64[:], numba.float64[:, :]),
    ],
    "(m, m), (m) -> (m, m)",
)
def compute_eigenvectors_3x3(arr, eigval, out):
    # Step 4: Compute the eigenvectors

    d = arr[0, 1]
    f = arr[0, 2]
    b = arr[1, 1]
    e = arr[1, 2]
    c = arr[2, 2]

    p1 = d**2 + f**2 + e**2
    if p1 == 0:
        # A is diagonal.
        out[0, 0] = 1
        out[0, 1] = 0
        out[0, 2] = 0
        out[1, 0] = 0
        out[1, 1] = 1
        out[1, 2] = 0
        out[2, 0] = 0
        out[2, 1] = 0
        out[2, 2] = 1
    else:
        m1 = (d * (c - eigval[0]) - e * f) / (f * (b - eigval[0]) - d * e)
        m2 = (d * (c - eigval[1]) - e * f) / (f * (b - eigval[1]) - d * e)
        m3 = (d * (c - eigval[2]) - e * f) / (f * (b - eigval[2]) - d * e)

        out[0, 0] = (eigval[0] - c - e * m1) / f
        out[0, 1] = (eigval[1] - c - e * m2) / f
        out[0, 2] = (eigval[2] - c - e * m3) / f

        out[1, 0] = m1
        out[1, 1] = m2
        out[1, 2] = m3

        out[2, 0] = 1
        out[2, 1] = 1
        out[2, 2] = 1


def eigh_dispatcher(arr, w, v):
    xp = pxu.get_array_module(arr)
    N = pxd.NDArrayInfo
    d = arr.shape[-1]
    if N.from_obj(arr) == N.CUPY:
        import _gpu

        if d == 2:
            w = _gpu.compute_eigenvalues_2x2_gpu(arr)
            v = _gpu.compute_eigenvectors_2x2_gpu(arr, w)
        elif d == 3:
            w = _gpu.compute_eigenvalues_3x3_gpu(arr)
            v = _gpu.compute_eigenvectors_3x3_gpu(arr, w)
        else:
            w, v = xp.linalg.eigh(arr)
    elif N.from_obj(arr) in [N.NUMPY, N.DASK]:
        if d == 2:
            w = compute_eigenvalues_2x2(arr)
            v = compute_eigenvectors_2x2(arr, w)
        elif d == 3:
            w = compute_eigenvalues_3x3(arr)
            v = compute_eigenvectors_3x3(arr, w)
        else:
            if N.from_obj(arr) == N.NUMPY:
                w, v = xp.linalg.eigh(arr)
            if N.from_obj(arr) == N.DASK:
                meta = np.array([], dtype=arr.dtype)[(np.newaxis,) * (arr.ndim - 1)]
                tuple_array = arr.map_blocks(np.linalg.eigh, dtype=arr.dtype, meta=meta)
                tuple_array = tuple_array.persist()
                get_w = lambda x: x[0][..., np.newaxis]
                get_v = lambda x: x[1]
                w = tuple_array.map_blocks(
                    get_w,
                    dtype=arr.dtype,
                    meta=np.array([], dtype=arr.dtype)[(np.newaxis,) * (arr.ndim - 2)],
                )[..., 0]
                v = tuple_array.map_blocks(get_v, dtype=arr.dtype, meta=meta)
    else:
        raise NotImplementedError
    return w, v
