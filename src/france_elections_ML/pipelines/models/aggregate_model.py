import torch
import pandas as pd
import pytorch_lightning as pl


class AggregateModel(pl.LightningModule):
    def __init__(
        self,
        n_features,
        n_targets,
        lr=1e-3,
        hidden_activation="ReLU",
        output_activation="ReLU",
        hidden_layers=0,
        hidden_size_factor=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_layers = hidden_layers
        self.hidden_size_factor = hidden_size_factor
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.build_model()
        self.loss_fn = torch.nn.MSELoss(reduction="mean")
        self.lr = lr

    def build_model(self):
        hidden_layer_dim = int(self.n_features * self.hidden_size_factor)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features, hidden_layer_dim),
            torch.nn.__getattribute__(self.hidden_activation)(),
            *[
                layer
                for i in range(self.hidden_layers)
                for layer in [
                    torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
                    torch.nn.__getattribute__(self.hidden_activation)(),
                ]
            ],
            torch.nn.Linear(hidden_layer_dim, self.n_targets),
            torch.nn.__getattribute__(self.output_activation)(),
        )

    def forward(self, df_features):
        group = df_features[:, [0]]
        group_index = pd.Series(group.reshape(-1)).astype("category").cat.codes
        weights = torch.tensor(df_features[:, [1]]).view(-1, 1)
        X = torch.tensor(df_features[:, 2:])
        index = (
            torch.tensor(group_index, dtype=torch.int64)
            .view(-1, 1)
            .repeat(1, self.n_targets)
        )
        y_pred_shape = (index.max() + 1, self.n_targets)
        y_pred_unweighted = self.model(X)
        y_pred_before_aggregation = y_pred_unweighted * weights
        y_pred = (
            (torch)
            .zeros(
                *y_pred_shape,
                dtype=y_pred_before_aggregation.dtype,
            )
            .scatter_add(0, index, y_pred_before_aggregation)
        )
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def _step(self, batch):
        df_features_batch, df_targets_batch = batch
        df_features = df_features_batch[0, :, :]
        df_targets = df_targets_batch[0, :, :]
        y_pred = self.forward(df_features)
        y_true = torch.tensor(df_targets[:, 1:], dtype=y_pred.dtype)
        loss = self.loss_fn(y_pred, y_true)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch)
        self.log("train_loss", loss)
        df_features_train = train_batch[0][0, :, :]
        population = df_features_train[:, 1]
        self.log(
            "train_loss_sqrt_per_capita", torch.sqrt(loss) / population.sum()
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch)
        self.log("val_loss", loss, prog_bar=True)
        df_features_val = val_batch[0][0, :, :]
        population = df_features_val[:, 1]
        self.log(
            "val_loss_sqrt_per_capita", torch.sqrt(loss) / population.sum()
        )
        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        return loss.backward()
