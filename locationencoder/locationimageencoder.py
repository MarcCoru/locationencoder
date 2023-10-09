import lightning.pytorch as pl
from .locationencoder import LocationEncoder, get_param, get_loss_fn
from .imageencoder import ImageEncoder
from torch import optim
from torch import nn
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np

class LocationImageEncoder(pl.LightningModule):
    def __init__(self, imageencoder_checkpoint, locationencoder_checkpoint, use_logits=False,
                 lr=0.0045,
                 momentum=0.9,
                 weight_decay=1e-4,
                 num_classes=8142):
        super().__init__()

        self.locationencoder = LocationEncoder.load_from_checkpoint(locationencoder_checkpoint)
        self.imageencoder = ImageEncoder.load_from_checkpoint(imageencoder_checkpoint, num_classes=num_classes)
        self.imageencoder.freeze()

        self.loss_fn = nn.CrossEntropyLoss()

        self.use_logits = use_logits
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.save_hyperparameters()


    def common_step(self, batch, batch_idx):
        images, _, label, _, lonlats = batch
        logits = self((images, lonlats))
        return self.loss_fn(logits, label), logits

    def forward(self, inputs):
        images, lonlats = inputs
        if self.use_logits:
            # Directly use logits
            image_logits = images
        else:
            image_logits = self.imageencoder(images)
        location_logits = self.locationencoder(lonlats)

        return image_logits + location_logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        _, _, labels, _, _ = batch
        loss, output = self.common_step(batch, batch_idx)

        prec1 = top_k_accuracy_score(labels.cpu(), output.cpu(), labels=np.arange(output.shape[1]), k=1)
        acc = accuracy_score(y_true=labels.cpu(), y_pred=output.argmax(1).cpu()) # should be same as prec1
        prec3 = top_k_accuracy_score(labels.cpu(), output.cpu(), labels=np.arange(output.shape[1]), k=3)

        self.log("val_prec1", prec1)
        self.log("val_acc", acc)
        self.log("val_prec3", prec3)
        self.log("val_loss", loss)
        
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)
