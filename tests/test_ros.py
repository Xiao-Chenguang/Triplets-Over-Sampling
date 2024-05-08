import numpy as np
from tos import TOS


def test_smote():
    x = np.random.rand(1000, 2)
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.06

    tps = TOS(k_neighbors=5)
    x_res, y_res = tps.fit_resample(x, y)

    gen_data = x_res[y.shape[0] :]
    valid_cnt = (gen_data[:, 0] - 0.5) ** 2 + (gen_data[:, 1] - 0.5) ** 2 <= 0.06

    assert valid_cnt.sum() > 0.9 * gen_data.shape[0]
    print(f"test passed! with {valid_cnt.sum() / gen_data.shape[0]} valid samples.")


if __name__ == "__main__":
    test_smote()
