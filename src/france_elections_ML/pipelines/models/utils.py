import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import polars as pl


def ID_to_float(df, ID):
    return df.assign(
        **{
            ID: lambda df: df[ID]
            .str.replace("A", "00")
            .str.replace("B", "01")
            .str.replace(r"[^0-9]", "0")
            .astype(float)
        }
    )


class GroupbyDataset(torch.utils.data.Dataset):
    def __init__(self, df, groupby, groups=None):
        self.df = df
        self.groupby_column = groupby
        self.groupby = pd.Series(df[groupby].to_numpy())
        self.groups = groups or (self.groupby).drop_duplicates().tolist()

    def __len__(self):
        return len(self.groups)

    def _compute_group_slice(self, group):
        indices = np.argwhere(self.groupby.isin([group]).tolist())
        idx_min = int(np.min(indices))
        idx_max = int(np.max(indices))
        if idx_max - idx_min + 1 != len(indices):
            raise ValueError(
                "groupby column might have gaps or might not be sorted"
            )
        return slice(idx_min, idx_max + 1)

    def _get_group_by_idx(self, idx):
        df_group = self.df[self._compute_group_slice(self.groups[idx])]
        if isinstance(df_group, pl.DataFrame):
            df_group = df_group.to_pandas()
        return (
            (df_group)
            .pipe(ID_to_float, self.groupby_column)
            .values.astype(float)
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_group_by_idx(idx)
        if isinstance(idx, slice):
            return np.concatenate(
                [self.__getitem__(i) for i in range(*idx.indices(len(self)))]
            )
        raise TypeError(f"Invalid argument type: {type(idx)}")


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self._cache_dict = dict()

    def _cache(self, dataset, s):
        slice_ID = f"{s.start:04d}:{s.stop:04d}"
        if slice_ID not in self._cache_dict:
            log = logging.getLogger(__name__)
            log.info("Caching data for slice %s", slice_ID)
            self._cache_dict[slice_ID] = dataset[s]
        return self._cache_dict[slice_ID]

    def __getitem__(self, s):
        if isinstance(s, int):
            return self._cache(self.dataset, slice(s, s + 1))
        return self._cache(self.dataset, s)

    def __len__(self):
        return len(self.dataset)


class SlicedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_slices):
        self.dataset = dataset
        self.n_slices = n_slices
        self.slices = [
            slice(split[0], split[-1] + 1)
            for split in np.array_split(range(len(dataset)), n_slices)
        ]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        return self.dataset[self.slices[i]]


torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


def get_last_checkpoint(log_folder="lightning_logs"):
    checkpoints = sorted(Path(log_folder).rglob("**/*.ckpt"))
    return str(checkpoints[-1]) if checkpoints else None


def get_df_columns(df):
    if isinstance(df, pd.DataFrame) or isinstance(df, pl.DataFrame):
        return list(df.columns)
    raise ValueError("Unknown dataframe type: %s", type(df))


class MasterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df_features,
        df_targets,
        ID_column,
        n_slices=50,
    ):
        self.features = get_df_columns(df_features)
        self.targets = get_df_columns(df_targets)

        IDs = sorted(
            set(df_targets[ID_column].unique())
            & set(df_features[ID_column].unique())
        )
        features_dataset = GroupbyDataset(df_features, ID_column, IDs)
        targets_dataset = GroupbyDataset(df_targets, ID_column, IDs)

        concat_dataset = ConcatDataset(features_dataset, targets_dataset)
        cached_dataset = CachedDataset(concat_dataset)
        self.dataset = SlicedDataset(cached_dataset, n_slices=n_slices)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


def train_test_split(dataset, test_size=0.1, num_workers=0):
    test_len = round(len(dataset) * test_size)
    train_len = len(dataset) - test_len
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, num_workers=num_workers
    )
    return train_loader, test_loader
