"""Official-fold splitting and leakage checks for UrbanSound8K."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

ALL_FOLDS = tuple(range(1, 11))
REQUIRED_COLUMNS = {"slice_file_name", "fold", "classID", "class"}


def official_fold_ids(test_fold: int, val_fold: int) -> tuple[tuple[int, ...], int, int]:
    """Return train, validation, and test folds without random reassignment."""
    if test_fold not in ALL_FOLDS or val_fold not in ALL_FOLDS:
        raise ValueError("test_fold and val_fold must be integers from 1 to 10")
    if test_fold == val_fold:
        raise ValueError("Validation and test folds must be different")

    train_folds = tuple(fold for fold in ALL_FOLDS if fold not in {test_fold, val_fold})
    return train_folds, val_fold, test_fold


def next_validation_fold(test_fold: int) -> int:
    """Use the next official fold as validation, wrapping 10 to 1."""
    if test_fold not in ALL_FOLDS:
        raise ValueError("test_fold must be an integer from 1 to 10")
    return 1 if test_fold == 10 else test_fold + 1


def validate_metadata(metadata: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    invalid_folds = set(metadata["fold"].unique()).difference(ALL_FOLDS)
    if invalid_folds:
        raise ValueError(f"Unexpected fold identifiers: {sorted(invalid_folds)}")


def split_metadata(
    metadata: pd.DataFrame, test_fold: int, val_fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split metadata using only the official fold assignments."""
    validate_metadata(metadata)
    train_folds, val_fold, test_fold = official_fold_ids(test_fold, val_fold)
    train = metadata[metadata["fold"].isin(train_folds)].reset_index(drop=True)
    validation = metadata[metadata["fold"] == val_fold].reset_index(drop=True)
    test = metadata[metadata["fold"] == test_fold].reset_index(drop=True)
    assert_source_disjoint(train, validation, test)
    return train, validation, test


def assert_source_disjoint(*splits: pd.DataFrame) -> None:
    """Check that source recordings do not cross splits when `fsID` is available."""
    if not splits or any("fsID" not in split.columns for split in splits):
        return

    seen: set[int] = set()
    for split in splits:
        source_ids = set(int(value) for value in split["fsID"].unique())
        overlap = seen.intersection(source_ids)
        if overlap:
            preview = sorted(overlap)[:5]
            raise ValueError(f"Source-recording leakage detected for fsID values: {preview}")
        seen.update(source_ids)


def limit_split(metadata: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    """Deterministically limit a split for smoke tests, retaining every class if possible."""
    if maximum is None or len(metadata) <= maximum:
        return metadata.reset_index(drop=True)
    if maximum <= 0:
        raise ValueError("maximum must be positive")

    class_ids: Iterable[int] = sorted(int(value) for value in metadata["classID"].unique())
    class_ids = tuple(class_ids)
    per_class = max(1, maximum // len(class_ids))
    sampled = metadata.groupby("classID", group_keys=False).head(per_class)
    remaining = maximum - len(sampled)
    if remaining > 0:
        import pandas as pd

        unused = metadata.drop(index=sampled.index).head(remaining)
        sampled = pd.concat([sampled, unused])
    return sampled.head(maximum).reset_index(drop=True)
