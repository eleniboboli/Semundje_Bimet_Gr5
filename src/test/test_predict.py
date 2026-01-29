import numpy as np
from PIL import Image
from src.predict import single_prediction


def test_single_prediction_returns_int(tmp_path, model, device):
    # create random image
    arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    p = tmp_path / "x.jpg"
    img.save(p)

    idx = single_prediction(model, str(p), device=device)
    assert isinstance(idx, int)
    assert 0 <= idx < 5
