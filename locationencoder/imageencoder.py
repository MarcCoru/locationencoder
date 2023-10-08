from torch import optim, nn
import lightning.pytorch as pl
import torchvision.models as models
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import numpy as np

class ImageEncoder(pl.LightningModule):
    def __init__(self, num_classes, lr=0.0045, momentum=0.9, weight_decay=1e-4):
        super().__init__()

        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(2048, num_classes)
        self.model.aux_logits = False

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss()

        # this enables SpatialEncoder.load_from_checkpoint(path)
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        images, _, labels, _ = batch
        output = self.model(images)
        return self.criterion(output, labels), output

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        loss, output = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _, labels, _ = batch
        loss, output = self.common_step(batch, batch_idx)

        prec1 = top_k_accuracy_score(labels.cpu(), output.cpu(), labels=np.arange(output.shape[1]), k=1)
        acc = accuracy_score(y_true=labels.cpu(), y_pred=output.argmax(1).cpu()) # should be same as prec1
        prec3 = top_k_accuracy_score(labels.cpu(), output.cpu(), labels=np.arange(output.shape[1]), k=3)

        self.log("val_prec1", prec1)
        self.log("val_acc", acc)
        self.log("val_prec3", prec3)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), self.lr,
                                momentum=self.momentum,
                                weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
        return [optimizer], [scheduler]

