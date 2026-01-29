from src.data import make_samplers


def test_make_samplers_sizes():
    train_s, val_s, test_s, sizes = make_samplers(dataset_len=100, seed=1)
    train_n, val_n, test_n = sizes

    assert train_n > 0
    assert val_n > 0
    assert test_n > 0
    assert train_n + val_n + test_n == 100
