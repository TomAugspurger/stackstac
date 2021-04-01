import json
from pathlib import Path
import xarray as xr

import stackstac


HERE = Path(__file__).parent

with open(HERE / "data/naip-items.json") as f:
    naip_items = json.load(f)


def test_multiband():
    result = stackstac.stack(naip_items["features"])
    assert result.shape == (4, 196870, 83790)