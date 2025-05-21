# tests/test_inference_time.py
import pytest
from time import perf_counter

@pytest.mark.parametrize("model_fixture", ["current_model", "baseline_model"])
def test_inference_time(model_fixture, request, test_data):
    model = request.getfixturevalue(model_fixture)
    X = test_data.drop("Survived", axis=1)

    # 推論時間を計測
    start = perf_counter()
    _ = model.predict(X)
    elapsed = perf_counter() - start

    # ここでは 0.2 秒以内であればOKとする例
    assert elapsed < 0.2, f"Inference too slow ({model_fixture}): {elapsed:.3f}s"

