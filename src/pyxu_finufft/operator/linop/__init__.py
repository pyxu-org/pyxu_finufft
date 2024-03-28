import collections.abc as cabc

import numpy as np

import pyxu.abc.operator as pxo
import pyxu.runtime as pxrt
import pyxu.util as pxu
import pyxu.info.ptype as pxt

__all__ = ["Flip"]


class Flip(pxo.LinOp):

    def __init__(self, arg_shape: pxt.NDArrayShape, axis: pxt.NDArrayAxis = None) -> pxt.OpT:
        r"""
        Reverse the order of elements in an array along the given axis.

        The shape of the array is preserved, but the elements are reordered.

        Parameters
        ----------
        arg_shape: pxt.NDArrayShape
            Shape of the data to be flipped.
        axis: pxt.NDArrayAxis
            Axis or axes along which the input array is flipped.
            The default, axis=None, will flip all the in all the axis of the input array.
            If axis is negative it counts from the last to the first axis.
        """

        def as_array(obj) -> np.ndarray:
            if isinstance(obj, cabc.Sequence):
                pass
            else:
                obj = [obj]
            return np.array(obj, dtype=int)

        self.arg_shape = as_array(arg_shape)
        assert np.all(self.arg_shape > 0)
        dim = np.prod(self.arg_shape).item()
        N_dim = len(self.arg_shape)

        if axis is None:
            axis = np.arange(N_dim)
        axis = np.unique(as_array(axis))  # drop potential duplicates
        assert np.all((-N_dim <= axis) & (axis < N_dim))  # all axes in valid range
        self.axis = (axis + N_dim) % N_dim  # get rid of negative axes

        super().__init__(shape=(dim, dim))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        sh = arr.shape[:-1]
        arr = arr.reshape(*sh, *self.arg_shape)
        out = arr.copy()
        xp = pxu.get_array_module(arr)
        for ax in self.axis:
            out = xp.flip(out, axis=len(sh) + ax)

        out = out.reshape(*sh, self.codim)
        return out

    @pxrt.enforce_precision(i="arr")
    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self.apply(arr)
