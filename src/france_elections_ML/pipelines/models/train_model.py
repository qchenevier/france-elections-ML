# %%
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl

from .aggregate_model import AggregateModel
from .utils import MasterDataset, train_test_split


def version_name_from_params(params_dict):
    sorted_items = sorted(params_dict.items(), key=lambda x: x[0])
    return "__".join(f"{k}={v}" for k, v in sorted_items)


def find_checkpoint(checkpoint_dir):
    return next(Path(checkpoint_dir).rglob("*.ckpt"))


def train_model(df_features, df_targets, name, kwargs):
    model_dir = "./data/tmp"
    version = version_name_from_params(kwargs)
    checkpoint_dir = os.path.join(model_dir, name, version)
    seed = kwargs.pop("seed", 7)
    pl.utilities.seed.seed_everything(seed, workers=True)
    max_epochs = kwargs.pop("max_epochs", 10)

    df_targets_non_null = df_targets.loc[lambda df: ~(df == 0).any(axis=1)]
    dataset = MasterDataset(
        df_features,
        df_targets_non_null,
        ID_column="code_census_tract",
    )
    train_loader, test_loader = train_test_split(dataset, test_size=0.1)

    not_feature_or_target = ["code_census_tract", "population"]
    n_features = len(set(dataset.features) - set(not_feature_or_target))
    n_targets = len(set(dataset.targets) - set(not_feature_or_target))
    model = AggregateModel(n_features=n_features, n_targets=n_targets, **kwargs)

    trainer = pl.Trainer(
        logger=[pl.loggers.NeptuneLogger(tags=[name], name=name)],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val_loss_sqrt_per_capita",
                mode="min",
                save_last=False,
                save_top_k=1,
                filename="{epoch}-{val_loss:.3g}",
            ),
        ],
        max_epochs=max_epochs,
        enable_progress_bar=False,  # to prevent neptune logger overflow bug
    )
    trainer.fit(model, train_loader, test_loader)

    best_model = AggregateModel.load_from_checkpoint(
        find_checkpoint(checkpoint_dir)
    )
    shutil.rmtree(model_dir)
    return best_model
