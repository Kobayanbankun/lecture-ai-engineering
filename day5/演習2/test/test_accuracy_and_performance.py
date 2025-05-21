import numpy as np

def test_accuracy_not_regress(current_model, baseline_model, test_data):
    X = test_data.drop("Survived", axis=1)
    y = test_data["Survived"]

    # 現行モデル精度
    y_pred_curr = current_model.predict(X)
    acc_curr = np.mean(y_pred_curr == y)

    # 基準モデル精度
    y_pred_base = baseline_model.predict(X)
    acc_base = np.mean(y_pred_base == y)

    # "改善していなくても過度に劣化していない"ことを保証（1%未満の差ならOK）
    assert acc_curr >= acc_base - 0.01, (
        f"Accuracy regressed too much: "
        f"current={acc_curr:.3f}, baseline={acc_base:.3f}"
    )