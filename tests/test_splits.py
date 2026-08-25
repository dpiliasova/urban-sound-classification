from statistics import mean, pstdev

import pytest

from urban_sound.splits import ALL_FOLDS, next_validation_fold, official_fold_ids


def test_official_split_is_complete_and_disjoint():
    train, validation, test = official_fold_ids(test_fold=10, val_fold=9)
    assert train == (1, 2, 3, 4, 5, 6, 7, 8)
    assert set(train) | {validation, test} == set(ALL_FOLDS)
    assert not set(train) & {validation, test}


@pytest.mark.parametrize("test_fold,expected", [(1, 2), (9, 10), (10, 1)])
def test_validation_fold_wraps(test_fold, expected):
    assert next_validation_fold(test_fold) == expected


@pytest.mark.parametrize("test_fold,val_fold", [(0, 1), (11, 1), (1, 1)])
def test_invalid_fold_combinations_fail(test_fold, val_fold):
    with pytest.raises(ValueError):
        official_fold_ids(test_fold, val_fold)


def test_recorded_fold_summary_is_consistent():
    accuracies = [
        0.7858,
        0.8514,
        0.7222,
        0.8626,
        0.8226,
        0.8019,
        0.8174,
        0.7717,
        0.8456,
        0.8292,
    ]
    assert mean(accuracies) == pytest.approx(0.81104)
    assert pstdev(accuracies) == pytest.approx(0.04018258826905008)
