import sys
from pathlib import Path

import pandas as pd

# Add src/ to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car_transformers import BrandTargetEncoder


def test_brand_target_encoder():
    X = pd.DataFrame({
        "brand": ["Ford", "Ford", "BMW"]
    })

    y = pd.Series([10000, 20000, 30000])

    encoder = BrandTargetEncoder()
    encoder.fit(X, y)

    result = encoder.transform(
        pd.DataFrame({
            "brand": ["Ford", "BMW", "Toyota"]
        })
    )

    assert "brand_encoded" in result.columns
    assert result.loc[0, "brand_encoded"] == 15000
    assert result.loc[1, "brand_encoded"] == 30000
    assert result.loc[2, "brand_encoded"] == 20000