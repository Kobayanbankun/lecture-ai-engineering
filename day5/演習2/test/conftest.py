import pickle
import pytest
from time import perf_counter
import pandas as pd

@pytest.fixture(scope="session")
def test_data():
    # テスト用データセット（サンプル）を読み込む
    return pd.read_csv("data/Titanic.csv").sample(n=100, random_state=0)

@pytest.fixture(scope="session")
def current_model():
    with open("models/titanic_model.pkl", "rb") as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def baseline_model():
    with open("models/titanic_model_v1.pkl", "rb") as f:
        return pickle.load(f)
