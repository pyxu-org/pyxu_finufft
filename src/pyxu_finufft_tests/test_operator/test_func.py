from pyxu_finufft import NullFunc


def test_nullfunc():
    assert NullFunc(1)._name == "ModifiedNullFunc"
