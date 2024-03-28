import numba
import numpy as np


@numba.guvectorize(
    [
        (numba.float32[:, :], numba.float32[:]),
        (numba.float64[:, :], numba.float64[:]),
    ],
    "(m, m) -> (m)",
    target="cuda",
)
def compute_eigenvalues_2x2_gpu(arr, out):
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
    target="cuda",
)
def compute_eigenvectors_2x2_gpu(arr, eigval, out):
    p = arr[0, 1] ** 2 + arr[1, 0] ** 2
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
    target="cuda",
)
def compute_eigenvalues_3x3_gpu(arr, out):
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
        q = (a + b + c) / 3
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
    target="cuda",
)
def compute_eigenvectors_3x3_gpu(arr, eigval, out):
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
