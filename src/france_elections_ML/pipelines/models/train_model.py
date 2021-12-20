# %%
import os

import pytorch_lightning as pl

from .aggregate_model import AggregateModel
from .utils import MasterDataset, train_test_split


def version_name_from_params(params_dict):
    sorted_items = sorted(params_dict.items(), key=lambda x: x[0])
    return "__".join(f"{k}={v}" for k, v in sorted_items)


# %%
def train_model(df_features, df_targets, name, kwargs):
    model_dir = kwargs.pop("dir", "./data/06_models")
    version = version_name_from_params(kwargs)
    seed = kwargs.pop("seed", 7)
    pl.utilities.seed.seed_everything(seed, workers=True)

    df_targets_non_null = df_targets.loc[lambda df: ~(df == 0).any(axis=1)]
    dataset = MasterDataset(
        df_features,
        df_targets_non_null,
        ID_column="code_census_tract",
    )

    # %%
    train_loader, test_loader = train_test_split(dataset, test_size=0.1)

    # %%
    not_feature_or_target = ["code_census_tract", "population"]
    n_features = len(set(dataset.features) - set(not_feature_or_target))
    n_targets = len(set(dataset.targets) - set(not_feature_or_target))
    model = AggregateModel(n_features=n_features, n_targets=n_targets, **kwargs)

    trainer = pl.Trainer(
        logger=[
            pl.loggers.NeptuneLogger(
                tags=[name],
                capture_stdout=False,
                capture_stderr=False,
                name="zou",
            ),
        ],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_last=True,
                save_top_k=3,
                filename="{epoch}-{val_loss:.3g}",
            ),
        ],
        max_epochs=20,
        default_root_dir=model_dir
    )

    trainer.fit(model, train_loader, test_loader)

    # %%
    return True
