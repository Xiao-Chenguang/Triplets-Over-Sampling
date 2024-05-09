import numpy as np

from tos import TOS


def test_neighbors():
    x = np.random.rand(1000, 2)
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.06

    tps = TOS(k_neighbors=10)
    x_res, y_res = tps.fit_resample(x, y)

    gen_data = x_res[y.shape[0] :]
    valid_cnt = (gen_data[:, 0] - 0.5) ** 2 + (gen_data[:, 1] - 0.5) ** 2 <= 0.06

    npos = y_res.sum()
    nneg = y_res.shape[0] - npos
    assert npos == nneg, f"npos={npos}, nneg={nneg}"
    assert valid_cnt.sum() > 0.85 * gen_data.shape[0], f"{valid_cnt.sum()} valid."


def test_none_neighbors():
    x = np.random.rand(1000, 2)
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.06

    tps = TOS()
    x_res, y_res = tps.fit_resample(x, y)

    gen_data = x_res[y.shape[0] :]
    valid_cnt = (gen_data[:, 0] - 0.5) ** 2 + (gen_data[:, 1] - 0.5) ** 2 <= 0.06

    npos = y_res.sum()
    nneg = y_res.shape[0] - npos
    assert npos == nneg, f"npos={npos}, nneg={nneg}"
    assert valid_cnt.sum() > 0.85 * gen_data.shape[0], f"{valid_cnt.sum()} valid."


def test_strategy():
    x = np.random.rand(1000, 2)
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.06

    tps = TOS(sampling_strategy={0: 1000, 1: 1000})
    x_res, y_res = tps.fit_resample(x, y)
    npos = y_res.sum()
    nneg = y_res.shape[0] - npos
    assert npos == 1000, f"npos={npos}"
    assert npos == nneg, f"npos={npos}, nneg={nneg}"

    x_res, y_res = x_res[y.shape[0] :], y_res[y.shape[0] :]
    gen_data = x_res[y_res == 1]
    valid_cnt = (gen_data[:, 0] - 0.5) ** 2 + (gen_data[:, 1] - 0.5) ** 2 <= 0.06

    assert valid_cnt.sum() > 0.85 * gen_data.shape[0], f"{valid_cnt.sum()} valid."


def test_len_lim():
    x = np.random.rand(1000, 2)
    y = (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.06

    tps = TOS(len_limit=False)
    x_res, y_res = tps.fit_resample(x, y)

    gen_data = x_res[y.shape[0] :]
    valid_cnt = (gen_data[:, 0] - 0.5) ** 2 + (gen_data[:, 1] - 0.5) ** 2 <= 0.06

    npos = y_res.sum()
    nneg = y_res.shape[0] - npos
    assert npos == nneg, f"npos={npos}, nneg={nneg}"
    assert valid_cnt.sum() > 0.85 * gen_data.shape[0], f"{valid_cnt.sum()} valid."
