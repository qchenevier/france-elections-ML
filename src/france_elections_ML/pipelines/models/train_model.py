# %%
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl

from .aggregate_model import AggregateModel
from .utils import MasterDataset, train_val_split


def version_name_from_params(params_dict):
    sorted_items = sorted(params_dict.items(), key=lambda x: x[0])
    return "__".join(f"{k}={v}" for k, v in sorted_items)


def find_checkpoint(checkpoint_dir):
    return next(Path(checkpoint_dir).rglob("*.ckpt"))


def train_model(df_features, df_targets, parameters):
    features = parameters.get("features", "minimal")
    seed = parameters.get("seed", 7)
    max_epochs = parameters.get("max_epochs", 10)
    version = version_name_from_params(parameters)

    model_dir = "./data/tmp"
    checkpoint_dir = os.path.join(model_dir, features, version)
    pl.utilities.seed.seed_everything(seed, workers=True)

    df_targets_non_null = df_targets.loc[lambda df: ~(df == 0).any(axis=1)]
    dataset = MasterDataset(
        df_features,
        df_targets_non_null,
        ID_column="code_census_tract",
    )
    train_loader, val_loader, test_loader = train_val_split(
        dataset, val_size=0.1
    )

    not_feature_or_target = ["code_census_tract", "population"]
    n_features = len(set(dataset.features) - set(not_feature_or_target))
    n_targets = len(set(dataset.targets) - set(not_feature_or_target))
    model = AggregateModel(
        n_features=n_features,
        n_targets=n_targets,
        **parameters,
    )

    trainer = pl.Trainer(
        logger=[pl.loggers.NeptuneLogger()],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val_loss_per_capita",
                mode="min",
                save_last=False,
                save_top_k=1,
                filename="{epoch}-{val_loss:.3g}",
            ),
        ],
        max_epochs=max_epochs,
        enable_progress_bar=False,  # to prevent neptune logger overflow bug
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    best_model = AggregateModel.load_from_checkpoint(
        find_checkpoint(checkpoint_dir)
    )
    shutil.rmtree(model_dir)
    return best_model
