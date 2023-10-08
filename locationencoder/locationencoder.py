from torch import optim, nn
import lightning.pytorch as pl

import locationencoder.pe as PE
import locationencoder.nn as NN
from utils.losses import get_loss_fn

from sklearn.metrics import (
    accuracy_score, 
    jaccard_score,
    mean_absolute_error
)

def get_positional_encoding(name, hparams=None):
    if name == "direct":
        return PE.Direct()
    elif name == "cartesian3d":
        return PE.Cartesian3D()
    elif name == "sphericalharmonics":

        # default to analytic
        if "harmonics_calculation" not in hparams.keys():
            hparams["harmonics_calculation"] = "analytic"

        if "harmonics_calculation" in hparams.keys() and hparams['harmonics_calculation'] == "discretized":
            return PE.DiscretizedSphericalHarmonics(legendre_polys=hparams['legendre_polys'])
        else:
            return PE.SphericalHarmonics(legendre_polys=hparams['legendre_polys'],
                                         harmonics_calculation=hparams['harmonics_calculation'])
    elif name == "theory":
        return PE.Theory(min_radius=hparams['min_radius'],
                         max_radius=hparams['max_radius'],
                         frequency_num=hparams['frequency_num'])
    elif name == "wrap":
        return PE.Wrap()
    elif name in ["grid", "spherec", "spherecplus", "spherem", "spheremplus"]:
        return PE.GridAndSphere(min_radius=hparams['min_radius'],
                       max_radius=hparams['max_radius'],
                       frequency_num=hparams['frequency_num'],
                       name=name)
    else:
        raise ValueError(f"{name} not a known positional encoding.")

def get_neural_network(name, input_dim, hparams=None):
    if name == "linear":
        return nn.Linear(input_dim, hparams['num_classes'])
    elif name ==  "siren":
        return NN.SirenNet(
                dim_in=input_dim,
                dim_hidden=hparams['dim_hidden'],
                num_layers=hparams['num_layers'],
                dim_out=hparams['num_classes'],
                dropout=hparams['dropout'] if "dropout" in hparams.keys() else False
            )
    elif name == "fcnet":
        return NN.FCNet(
                num_inputs=input_dim,
                num_classes=hparams['num_classes'],
                dim_hidden=hparams['dim_hidden']
            )
    else:
        raise ValueError(f"{name} not a known neural networks.")

def get_param(hparams, key, default=False):
    """
    Convenience function that indexes the hyperparameter dict but returns a default value if not defined rather than
    an error
    """
    return hparams[key] if key in hparams.keys() else default

# define the LightningModule
class LocationEncoder(pl.LightningModule):
    def __init__(self, positional_encoding_name, neural_network_name, hparams):
        super().__init__()

        self.learning_rate = hparams["optimizer"]["lr"]
        self.weight_decay = hparams["optimizer"]["wd"]
        self.regression = get_param(hparams, "regression")

        self.loss_fn = get_loss_fn(presence_only=get_param(hparams, "presence_only_loss"), 
                                   loss_weight=get_param(hparams, "loss_weight"),
                                   regression=self.regression)

        self.positional_encoder = get_positional_encoding(
            positional_encoding_name, hparams
        )
        self.neural_network = get_neural_network(
            neural_network_name,
            input_dim=self.positional_encoder.embedding_dim,
            hparams=hparams
        )

        # this enables LocationEncoder.load_from_checkpoint(path)
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        lonlats, label = batch
        return self.loss_fn(self, lonlats, label)

    def forward(self, lonlats):
        embedding = self.positional_encoder(lonlats)
        return self.neural_network(embedding)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return {"val_loss":loss}

    def predict_step(self, batch, batch_idx):
        lonlats, label = batch
        prediction_logits = self.forward(lonlats)
        return prediction_logits, lonlats, label

    def test_step(self, batch, batch_idx):
        lonlats, label = batch
        prediction_logits = self.forward(lonlats)
        
        loss = self.loss_fn(self, lonlats, label)

        # check if binary
        non_binary_task = self.regression
        if (prediction_logits.size(1) == 1) and not (non_binary_task):
            y_pred = (prediction_logits.squeeze() > 0).cpu()
            average = "binary"
        elif self.regression:
            y_pred = prediction_logits.cpu()
        else: # take argmax
            y_pred = prediction_logits.argmax(-1).cpu()
            average = "macro"

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        if self.regression:
            MAE = mean_absolute_error(y_true=label.cpu(), y_pred = y_pred)
            self.log("test_MAE", MAE, on_step=False, on_epoch=True)
            
            test_results = {"test_loss":loss,
                            "test_MAE":MAE}
            
        else:
            accuracy = accuracy_score(y_true=label.cpu(), y_pred= y_pred).astype("float32")
            IoU = jaccard_score(y_true=label.cpu(),  y_pred = y_pred, average=average).astype("float32")
            self.log("test_accuracy", accuracy, on_step=False, on_epoch=True)
            self.log("test_IoU", IoU, on_step=False, on_epoch=True)
            
            test_results = {"test_loss":loss,
                          "test_accuracy":accuracy}

        return test_results

    def configure_optimizers(self):
        optimizer = optim.Adam([{"params": self.neural_network.parameters()},
                                {"params": self.positional_encoder.parameters(), "weight_decay":0}],
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer

