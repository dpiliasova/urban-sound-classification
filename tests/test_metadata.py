import pandas as pd
import pytest

from urban_sound.splits import (
    assert_source_disjoint,
    limit_split,
    source_overlap_ids,
    split_metadata,
)


def _metadata() -> pd.DataFrame:
    records = []
    for fold in range(1, 11):
        for class_id in range(3):
            records.append(
                {
                    "slice_file_name": f"fold-{fold}-class-{class_id}.wav",
                    "fold": fold,
                    "classID": class_id,
                    "class": f"class_{class_id}",
                    "fsID": fold * 100 + class_id,
                }
            )
    return pd.DataFrame(records)


def test_metadata_split_preserves_official_folds():
    train, validation, test = split_metadata(_metadata(), test_fold=10, val_fold=9)
    assert set(train["fold"]) == set(range(1, 9))
    assert set(validation["fold"]) == {9}
    assert set(test["fold"]) == {10}


def test_source_overlap_is_rejected():
    first = pd.DataFrame({"fsID": [1, 2]})
    second = pd.DataFrame({"fsID": [2, 3]})
    with pytest.raises(ValueError, match="leakage"):
        assert_source_disjoint(first, second)


def test_source_overlap_can_be_reported_without_changing_official_folds():
    first = pd.DataFrame({"fsID": [1, 2]})
    second = pd.DataFrame({"fsID": [2, 3]})
    assert source_overlap_ids(first, second) == {2}


def test_smoke_limit_retains_classes_when_space_allows():
    limited = limit_split(_metadata(), maximum=9)
    assert len(limited) == 9
    assert set(limited["classID"]) == {0, 1, 2}
