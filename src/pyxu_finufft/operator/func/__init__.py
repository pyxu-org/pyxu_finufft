import pyxu.info.ptype as pxt
import pyxu.operator.linop as pxl

__all__ = ["NullFunc"]


def NullFunc(dim: pxt.Integer) -> pxt.OpT:
    """
    Null functional (modified from base Pyxu class).
    This functional maps any input vector on the null scalar.

    The plugin modification adds a print at init time.
    """
    op = pxl.NullFunc(dim)
    op._name = "ModifiedNullFunc"
    print(
        "The modified NullFunc exemplifies how to overload a base class. ",
        "To overload a Pyxu base class, an underscore needs to be added in front of the class name, "
        "in the setup.cfg file section [options.entry_points]).",
    )
    return op
