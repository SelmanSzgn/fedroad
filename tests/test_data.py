import random

from fedroad.data import create_class_indices, sample_local_data


def test_class_indices(fake):
    cidx = create_class_indices(fake)
    assert sorted(cidx) == list(range(10))
    for k, idx in cidx.items():
        assert len(idx) == 20
        assert all(fake.targets[i] == k for i in idx)


def test_sample_uses_at_most_n_classes(fake):
    cidx = create_class_indices(fake)
    sub = sample_local_data(fake, cidx, 30, 2)
    assert len(sub) == 30
    assert len({fake.targets[i] for i in sub.indices}) <= 2


def test_sample_capped_by_available_data(fake):
    cidx = create_class_indices(fake)
    sub = sample_local_data(fake, cidx, 500, 1)
    assert len(sub) == 20


def test_sample_is_seeded(fake):
    cidx = create_class_indices(fake)
    random.seed(1)
    a = sample_local_data(fake, cidx, 10, 3).indices
    random.seed(1)
    b = sample_local_data(fake, cidx, 10, 3).indices
    assert a == b
