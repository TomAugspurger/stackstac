import numpy as np
import xarray as xr
import stackstac


def test_mosaic():
    x1 = np.array([np.nan, np.nan, 3, 4]).reshape((1, 2, 2))
    x2 = np.array([1, np.nan, np.nan, 4]).reshape((1, 2, 2))
    x = xr.DataArray(np.concatenate([x1, x2]), dims=("time", "y", "x"))

    result = stackstac.mosaic(x)
    expected = xr.DataArray(
        np.concatenate([
            np.array([1, np.nan, 3, 4]).reshape(2, 2)
        ]), dims=("y", "x")
    )
    xr.testing.assert_equal(result, expected)


def test_mosaic_na_value():
    x1 = np.array([0, 0, 3, 4]).reshape((1, 2, 2))
    x2 = np.array([1, 0, 0, 4]).reshape((1, 2, 2))
    x = xr.DataArray(np.concatenate([x1, x2]), dims=("time", "y", "x"))

    result = stackstac.mosaic(x, na_value=0)
    expected = xr.DataArray(
        np.concatenate([
            np.array([1, 0, 3, 4]).reshape(2, 2)
        ]), dims=("y", "x")
    )
    xr.testing.assert_equal(result, expected)

