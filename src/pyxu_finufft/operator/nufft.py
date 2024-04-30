import collections.abc as cabc
import operator
import warnings

import numpy as np
import pyxu.abc as pxa
import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu

isign_default = 1
eps_default = 1e-4
enable_warnings_default = True

__all__ = [
    "NUFFT1",
    "NUFFT2",
    "NUFFT3",
]


class _NUFFT(pxa.LinOp):
    # Super-class of NUFFT[123] containing common methods.

    def __init__(
        self,
        dim_shape: pxt.NDArrayShape,
        codim_shape: pxt.NDArrayShape,
    ):
        super().__init__(
            dim_shape=dim_shape,
            codim_shape=codim_shape,
        )

    # Helpers (Internal) ------------------------------------------------------
    def _cast_warn(self, arr: pxt.NDArray) -> pxt.NDArray:
        cwidth = pxrt.Width(self._x.dtype).complex
        cdtype = cwidth.value
        if arr.dtype == cdtype:
            out = arr
        else:
            if self._enable_warnings:
                msg = "Computation may not be performed at the requested precision."
                warnings.warn(msg, pxw.PrecisionWarning)
            out = arr.astype(dtype=cdtype)
        return out

    @staticmethod
    def _as_seq(x, N, _type=None) -> tuple:
        if isinstance(x, cabc.Iterable):
            _x = tuple(x)
        else:
            _x = (x,)
        if len(_x) == 1:
            _x *= N  # broadcast
        assert len(_x) == N

        if _type is None:
            return _x
        else:
            return tuple(map(_type, _x))

    def _transform(self, x: pxt.NDArray, mode: str) -> pxt.NDArray:
        # Parameters
        # ----------
        # x: NDArray [complex]
        #     (N_stack, <core_in>) array to transform.
        # mode: "fw", "bw"
        #     Transform direction.
        #
        # Returns
        # -------
        # y: NDArray [complex]
        #     (N_stack, <core_out>) transformed array.
        xp = pxu.get_array_module(x)
        N_stack, sh_in = x.shape[0], x.shape[1:]
        if mode == "fw":
            plan = self._pfw
            sh_out = self.codim_shape[:-1]
        elif mode == "bw":
            plan = self._pbw
            sh_out = self.dim_shape[:-1]
        else:
            raise NotImplementedError

        # Pad stack-dims to be a multiple of n_trans
        Q, r = divmod(N_stack, plan.n_trans)
        if r > 0:
            data = xp.zeros(shape=(N_stack + r, *sh_in), dtype=x.dtype)
            data[:N_stack] = x
            Q += 1
        else:
            data = x

        # Apply FINUFFT plan per batch.
        data = data.reshape(Q, plan.n_trans, *sh_in)
        out = xp.zeros((Q, plan.n_trans, *sh_out), dtype=x.dtype)
        for q in range(Q):
            plan.execute(data[q], out[q])

        # Remove pad/reshape output
        out = out.reshape(-1, *sh_out)[:N_stack]
        return out


class NUFFT1(_NUFFT):
    r"""
    Type-1 Non-Uniform FFT :math:`\mathbf{A}: \mathbb{C}^{M} \to \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.

    NUFFT1 approximates, up to a requested relative accuracy :math:`\varepsilon > 0`, the following exponential sum:

    .. math::

       v_{\mathbf{n}} = (\mathbf{A} \mathbf{w})_{n} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \langle \mathbf{n},
       \mathbf{x}_{m} \rangle},

    where

    * :math:`s \in \pm 1` defines the sign of the transform;
    * :math:`\mathbf{n} \in \{ -N_{1}, \ldots N_{1} \} \times\cdots\times \{ -N_{D}, \ldots, N_{D} \}`, with
      :math:`L_{d} = 2 N_{d} + 1`;
    * :math:`\{\mathbf{x}_{m}\}_{m=1}^{M} \in [-\pi, \pi]^{D}` are non-uniform support points;
    * :math:`\mathbf{w} \in \mathbb{C}^{M}` are weights associated with :math:`\{\mathbf{x}\}_{m=1}^{M}`.

    .. rubric:: Implementation Notes

    * :py:class:`~pyxu_finufft.operator.NUFFT1` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    * :py:class:`~pyxu_finufft.operator.NUFFT1` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `x`. Only NUMPY and CUPY backends are supported.  (DASK arrays are
      not supported because FFTW planning is not thread safe, hence cannot be guaranteed to not seg-fault.)
    """

    def __init__(
        self,
        x: pxt.NDArray,
        N: tuple[int],
        *,
        isign: int = isign_default,
        eps: float = eps_default,
        enable_warnings: bool = enable_warnings_default,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\mathbf{x}_{m} \in [-\pi,\pi]^{D}`.
        N: int, tuple[int]
            Number of coefficients [-N,...,N] to compute per dimension.
        isign: 1, -1
            Sign :math:`s` of the transform.
        eps: float
            Requested relative accuracy :math:`\varepsilon > 0`.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            Most useful are ``n_trans``, ``nthreads`` and ``debug``.
            (``modeord`` is not supported.)
        """
        # Put all variables in canonical form & validate ----------------------
        #   x: (M, D) array (NUMPY/CUPY)
        #   N: (D,) int
        #   isign: {-1, +1}
        #   eps: float
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape
        N = self._as_seq(N, D, int)
        assert all(n > 0 for n in N)
        isign = isign // abs(isign)
        assert 1e-17 <= eps <= 5e-3

        # Initialize Operator -------------------------------------------------
        L = [2 * n + 1 for n in N]
        super().__init__(
            dim_shape=(M, 2),
            codim_shape=(*L, 2),
        )
        self._x = x.astype(  # finufft.Plan.setpts() warns if dimensions are not contiguous.
            order="F",
            dtype=pxrt.getPrecision().value,
        )
        self._kwargs = dict(
            N=N,
            isign=isign,
            eps=eps,
            **kwargs,
        )
        self._kwargs.pop("modeord", None)  # unsupported parameter
        self._enable_warnings = bool(enable_warnings)
        self.lipschitz = np.sqrt(np.prod(L) * M)

        # Backend-Specific Metadata -------------------------------------------
        ndi = pxd.NDArrayInfo.from_obj(self._x)
        if ndi == pxd.NDArrayInfo.DASK:
            msg = "FFTW planning is not thread-safe."
            raise NotImplementedError(msg)
        else:
            self._pfw = self._plan_fw(x=self._x, **self._kwargs)
            self._pbw = self._plan_bw(x=self._x, **self._kwargs)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., L1,...,LD,2) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., M)
        y = self.capply(x)  # (..., L1,...,LD)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., L1,...,LD,2)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.

        Returns
        -------
        out: NDArray
            (..., L1,...,LD) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.
        """
        arr = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)

        sh = arr.shape[:-1]
        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError
        else:  # NUMPY/CUPY
            N_stack = int(np.prod(sh))
            out = self._transform(
                arr.reshape(N_stack, *self.dim_shape[:-1]),
                mode="fw",
            ).reshape(*sh, *self.codim_shape[:-1])
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., L1,...,LD,2) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` viewed as a
            real array. (See :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., L1,...,LD)
        y = self.cadjoint(x)  # (..., M)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., M,2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., L1,...,LD) weights :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}`.

        Returns
        -------
        out: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.
        """
        arr = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)

        sh = arr.shape[: -(self.codim_rank - 1)]
        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError
        else:  # NUMPY/CUPY
            N_stack = int(np.prod(sh))
            out = self._transform(
                arr.reshape(N_stack, *self.codim_shape[:-1]),
                mode="bw",
            ).reshape(*sh, *self.dim_shape[:-1])
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        N = self._kwargs["N"]
        isign = self._kwargs["isign"]

        # Perform computation in `x`-backend ... ------------------------------
        xp = pxu.get_array_module(self._x)

        A = xp.stack(  # (L1,...,LD, D)
            xp.meshgrid(
                *[xp.arange(-n, n + 1) for n in N],
                indexing="ij",
            ),
            axis=-1,
        )
        B = xp.exp((1j * isign) * xp.tensordot(A, self._x, axes=[[-1], [-1]]))  # (L1,...,LD, M)

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        C = xp.array(
            pxu.to_NUMPY(pxu.as_real_op(B, dim_rank=1)),
            dtype=pxrt.Width(dtype).value,
        )
        return C

    # Helpers (Internal) ------------------------------------------------------
    @staticmethod
    def _plan_fw(**kwargs):
        kwargs = kwargs.copy()

        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        cwidth = pxrt.Width(x.dtype).complex
        kwargs.update(
            nufft_type=1,
            dtype=cwidth.value,
            isign=kwargs.pop("isign"),
        )
        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi == pxd.NDArrayInfo.NUMPY:
            kwargs["n_modes_or_dim"] = tuple(2 * n + 1 for n in N)
        else:  # CUPY
            kwargs["n_modes"] = tuple(2 * n + 1 for n in N)

        _, D = x.shape
        plan = _get_planner(ndi)(**kwargs)
        plan.setpts(**dict(zip("xyz"[:D], x.T[:D])))
        return plan

    @staticmethod
    def _plan_bw(**kwargs):
        kwargs = kwargs.copy()

        x, N = [kwargs.pop(_) for _ in ("x", "N")]
        cwidth = pxrt.Width(x.dtype).complex
        kwargs.update(
            nufft_type=2,
            dtype=cwidth.value,
            isign=-kwargs.pop("isign"),
        )
        ndi = pxd.NDArrayInfo.from_obj(x)
        if ndi == pxd.NDArrayInfo.NUMPY:
            kwargs["n_modes_or_dim"] = tuple(2 * n + 1 for n in N)
        else:  # CUPY
            kwargs["n_modes"] = tuple(2 * n + 1 for n in N)

        _, D = x.shape
        plan = _get_planner(ndi)(**kwargs)
        plan.setpts(**dict(zip("xyz"[:D], x.T[:D])))
        return plan


def NUFFT2(
    x: pxt.NDArray,
    N: tuple[int],
    *,
    isign: int = isign_default,
    eps: float = eps_default,
    enable_warnings: bool = enable_warnings_default,
    **kwargs,
) -> pxt.OpT:
    r"""
    Type-2 Non-Uniform FFT :math:`\mathbf{A}: \mathbb{C}^{L_{1} \times\cdots\times L_{D}} \to \mathbb{C}^{M}`.

    NUFFT2 approximates, up to a requested relative accuracy :math:`\varepsilon > 0`, the following exponential sum:

    .. math::

       \mathbf{w}_{m} = (\mathbf{A} \mathbf{v})_{m} = \sum_{\mathbf{n}} v_{\mathbf{n}} e^{j \cdot s \cdot 2\pi \langle \mathbf{n}, \mathbf{x}_{m} / \mathbf{T} \rangle},

    where

    * :math:`s \in \pm 1` defines the sign of the transform;
    * :math:`\mathbf{n} \in \{ -N_{1}, \ldots N_{1} \} \times\cdots\times \{ -N_{D}, \ldots, N_{D} \}`, with
      :math:`L_{d} = 2 * N_{d} + 1`;
    * :math:`\{\mathbf{x}_{m}\}_{m=1}^{M} \in [-\pi, \pi]^{D}` are non-uniform support points;
    * :math:`\mathbf{v} \in \mathbb{C}^{L_{1} \times\cdots\times L_{D}}` are weights.

    .. rubric:: Implementation Notes

    * :py:func:`~pyxu_finufft.operator.NUFFT2` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as `x`.  A warning is emitted if inputs must be cast to the support dtype.
    * :py:class:`~pyxu_finufft.operator.NUFFT2` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as `x`. Only NUMPY and CUPY backends are supported.  (DASK arrays are
      not supported because FFTW planning is not thread safe, hence cannot be guaranteed to not seg-fault.)


    Parameters
    ----------
    x: NDArray
        (M, D) support points :math:`\mathbf{x}_{m} \in [-\pi,\pi]^{D}`.
    N: int, tuple[int]
        Number of coefficients [-N,...,N] to compute per dimension.
    isign: 1, -1
        Sign :math:`s` of the transform.
    eps: float
        Requested relative accuracy :math:`\varepsilon > 0`.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    **kwargs
        Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
        Most useful are ``n_trans``, ``nthreads`` and ``debug``.
        (``modeord`` is not supported.)
    """
    op1 = NUFFT1(
        x=x,
        N=N,
        isign=-isign,
        eps=eps,
        enable_warnings=enable_warnings,
        **kwargs,
    )
    op2 = op1.T
    op2._name = "NUFFT2"

    # Expose c[apply,adjoint]()
    op2.capply = op1.cadjoint
    op2.cadjoint = op1.capply

    return op2


class NUFFT3(_NUFFT):
    r"""
    Type-3 Non-Uniform FFT :math:`\mathbf{A}: \mathbb{C}^{M} \to \mathbb{C}^{N}`.

    NUFFT3 approximates, up to a requested relative accuracy :math:`\varepsilon > 0`, the following exponential sum:

    .. math::

       z_{n} = (\mathbf{A} \mathbf{w})_{n} = \sum_{m=1}^{M} w_{m} e^{j \cdot s \langle \mathbf{v}_{n}, \mathbf{x}_{m}
       \rangle},

    where

    * :math:`s \in \pm 1` defines the sign of the transform;
    * :math:`\{ \mathbf{x}_{m} \in \mathbb{R}^{D} \}_{m=1}^{M}` are non-uniform support points;
    * :math:`\{ \mathbf{v}_{n} \in \mathbb{R}^{D} \}_{n=1}^{N}` are non-uniform frequencies;
    * :math:`\mathbf{w} \in \mathbb{C}^{M}` are weights associated with :math:`\{\mathbf{x}\}_{m=1}^{M}`.

    .. rubric:: Implementation Notes

    * :py:class:`~pyxu_finufft.operator.NUFFT3` is not **precision-agnostic**: it will only work on NDArrays with the
      same dtype as (`x`,`v`).  A warning is emitted if inputs must be cast to their dtype.
    * :py:class:`~pyxu_finufft.operator.NUFFT3` instances are **not arraymodule-agnostic**: they will only work with
      NDArrays belonging to the same array module as (`x`,`v`). Only NUMPY backends are supported.  (DASK arrays are not
      supported because FFTW planning is not thread safe, hence cannot be guaranteed to not seg-fault; and cuFINUFFT
      does not yet support the type-3 transform.)
    """

    def __init__(
        self,
        x: pxt.NDArray,
        v: pxt.NDArray,
        *,
        isign: int = isign_default,
        eps: float = eps_default,
        enable_warnings: bool = enable_warnings_default,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        x: NDArray
            (M, D) support points :math:`\mathbf{x}_{m} \in \mathbb{R}^{D}`.
        v: NDArray
            (N, D) frequencies :math:`\mathbf{v}_{n} \in \mathbb{R}^{D}`.
        isign: 1, -1
            Sign :math:`s` of the transform.
        eps: float
            Requested relative accuracy :math:`\varepsilon > 0`.
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.
        **kwargs
            Extra kwargs to `finufft.Plan <https://finufft.readthedocs.io/en/latest/python.html#finufft.Plan>`_.
            Most useful are ``n_trans``, ``nthreads`` and ``debug``.
        """
        # Put all variables in canonical form & validate ----------------------
        #   x: (M, D) array (NUMPY/CUPY)
        #   v: (N, D) array (NUMPY/CUPY)
        #   isign: {-1, +1}
        #   eps: float
        if x.ndim == 1:
            x = x[:, np.newaxis]
        M, D = x.shape
        if v.ndim == 1:
            v = v[:, np.newaxis]
        N, _ = v.shape
        assert operator.eq(
            pxd.NDArrayInfo.from_obj(x),
            pxd.NDArrayInfo.from_obj(v),
        ), "[x,v] Must belong to the same array backend."
        assert D == v.shape[1], "[x,v] Must have same dimension D."
        isign = isign // abs(isign)
        assert 1e-17 <= eps <= 5e-3

        # Initialize Operator -------------------------------------------------
        super().__init__(
            dim_shape=(M, 2),
            codim_shape=(N, 2),
        )
        self._x = x.astype(  # finufft.Plan.setpts() warns if dimensions are not contiguous.
            order="F",
            dtype=pxrt.getPrecision().value,
        )
        self._v = v.astype(  # finufft.Plan.setpts() warns if dimensions are not contiguous.
            order="F",
            dtype=pxrt.getPrecision().value,
        )
        self._kwargs = dict(
            isign=isign,
            eps=eps,
            **kwargs,
        )
        self._enable_warnings = bool(enable_warnings)
        self.lipschitz = np.sqrt(N * M)

        # Backend-Specific Metadata -------------------------------------------
        ndi = pxd.NDArrayInfo.from_obj(self._x)
        if ndi == pxd.NDArrayInfo.DASK:
            msg = "FFTW planning is not thread-safe."
            raise NotImplementedError(msg)
        elif ndi == pxd.NDArrayInfo.CUPY:
            msg = "Not supported by cuFINUFFT."
            raise NotImplementedError(msg)
        else:
            self._pfw = self._plan_fw(x=self._x, v=self._v, **self._kwargs)
            self._pbw = self._plan_bw(x=self._x, v=self._v, **self._kwargs)

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., N,2) weights :math:`\mathbf{z} \in \mathbb{C}^{N}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., M)
        y = self.capply(x)  # (..., N)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., N,2)
        return out

    def capply(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.

        Returns
        -------
        out: NDArray
            (..., N) weights :math:`\mathbf{z} \in \mathbb{C}^{N}`.
        """
        arr = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)

        sh = arr.shape[:-1]
        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError
        elif ndi == pxd.NDArrayInfo.CUPY:
            raise NotImplementedError
        else:  # NUMPY
            N_stack = int(np.prod(sh))
            out = self._transform(
                arr.reshape(N_stack, *self.dim_shape[:-1]),
                mode="fw",
            ).reshape(*sh, *self.codim_shape[:-1])
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N,2) weights :math:`\mathbf{z} \in \mathbb{C}^{N}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)

        Returns
        -------
        out: NDArray
            (..., M,2) weights :math:`\mathbf{w} \in \mathbb{C}^{M}` viewed as a real array. (See
            :py:func:`~pyxu.util.view_as_real`.)
        """
        x = pxu.view_as_complex(pxu.require_viewable(arr))  # (..., N)
        y = self.cadjoint(x)  # (..., M)
        out = pxu.view_as_real(pxu.require_viewable(y))  # (..., M,2)
        return out

    def cadjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            (..., N) weights :math:`\mathbf{z} \in \mathbb{C}^{N}`.

        Returns
        -------
        out: NDArray
            (..., M) weights :math:`\mathbf{w} \in \mathbb{C}^{M}`.
        """
        arr = self._cast_warn(arr)
        ndi = pxd.NDArrayInfo.from_obj(arr)

        sh = arr.shape[: -(self.codim_rank - 1)]
        if ndi == pxd.NDArrayInfo.DASK:
            raise NotImplementedError
        elif ndi == pxd.NDArrayInfo.CUPY:
            raise NotImplementedError
        else:  # NUMPY
            N_stack = int(np.prod(sh))
            out = self._transform(
                arr.reshape(N_stack, *self.codim_shape[:-1]),
                mode="bw",
            ).reshape(*sh, *self.dim_shape[:-1])
        return out

    def asarray(self, **kwargs) -> pxt.NDArray:
        isign = self._kwargs["isign"]

        # Perform computation in `x`-backend ... ------------------------------
        xp = pxu.get_array_module(self._x)
        B = xp.exp((1j * isign) * (self._v @ self._x.T))  # (N, M)

        # ... then abide by user's backend/precision choice. ------------------
        xp = kwargs.get("xp", pxd.NDArrayInfo.NUMPY.module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)
        C = xp.array(
            pxu.to_NUMPY(pxu.as_real_op(B, dim_rank=1)),
            dtype=pxrt.Width(dtype).value,
        )
        return C

    # Helpers (Internal) ------------------------------------------------------
    @staticmethod
    def _plan_fw(**kwargs):
        kwargs = kwargs.copy()

        x, v = [kwargs.pop(_) for _ in ("x", "v")]
        _, D = x.shape
        cwidth = pxrt.Width(x.dtype).complex
        kwargs.update(
            nufft_type=3,
            n_modes_or_dim=D,
            dtype=cwidth.value,
            isign=kwargs.pop("isign"),
        )

        ndi = pxd.NDArrayInfo.from_obj(x)
        plan = _get_planner(ndi)(**kwargs)
        pts = dict(
            zip(
                "xyz"[:D] + "stu"[:D],
                (*x.T[:D], *v.T[:D]),
            )
        )
        plan.setpts(**pts)
        return plan

    @staticmethod
    def _plan_bw(**kwargs):
        kwargs = kwargs.copy()

        x, v = [kwargs.pop(_) for _ in ("x", "v")]
        _, D = x.shape
        cwidth = pxrt.Width(x.dtype).complex
        kwargs.update(
            nufft_type=3,
            n_modes_or_dim=D,
            dtype=cwidth.value,
            isign=-kwargs.pop("isign"),
        )

        ndi = pxd.NDArrayInfo.from_obj(x)
        plan = _get_planner(ndi)(**kwargs)
        pts = dict(
            zip(
                "xyz"[:D] + "stu"[:D],
                (*v.T[:D], *x.T[:D]),
            )
        )
        plan.setpts(**pts)
        return plan


def _get_planner(ndi: pxd.NDArrayInfo):
    # Load [cu]finufft.Plan based on supplied backend.

    if ndi == pxd.NDArrayInfo.NUMPY:
        import finufft as fi
    elif ndi == pxd.NDArrayInfo.CUPY:
        import cufinufft as fi
    else:
        raise NotImplementedError
    return fi.Plan
