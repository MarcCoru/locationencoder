import lightning.pytorch as pl
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_predictions
import os
import locationencoder.pe as pe
import locationencoder.nn as nn

def plot_weights(pl_module,
                 current_epoch,
                 vmin=None,
                 vmax=None,
                 savepath=None,
                 show=False):
    L = pl_module.positional_encoder.L
    if isinstance(pl_module.neural_network, nn.SirenNet):
        weights_flat_all = pl_module.neural_network.layers[0].weight.detach().numpy()
    else: # linear
        weights_flat_all = pl_module.neural_network.weight.detach().numpy()

    for i, weights_flat in enumerate(weights_flat_all):

        lm = []
        for l in range(L):
            for m in range(-l, l + 1):
                lm.append((l, m))

        W = np.ones((L, 2 * L)) * np.nan
        for (l, m), w in zip(lm, weights_flat):
            W[l, m + L] = w

        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks(np.arange(-L, L, 5)[1:] + L)
        ax.set_xticklabels(np.arange(-L, L, 5)[1:])
        ax.set_xlabel("order m")
        ax.set_ylabel("degree l")

        ax.imshow(W, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"weight {i} epoch {current_epoch}")
        if show:
            plt.show()

        if savepath is not None:
            path, ext = os.path.splitext(savepath)
            fig.savefig(path + f"_{i}" + ext, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.clf()
        plt.cla()
        plt.close(fig)



class PlotIntermediateResultsCallback(pl.Callback):

    def __init__(self, savepath="/Users/marc/Desktop", plot_every_n_epochs=50):
        super(PlotIntermediateResultsCallback, self).__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.savepath = savepath

    def plot_all_figures(self, trainer, pl_module, prefix=""):
        # Called when the validation loop ends

        if trainer.current_epoch % self.plot_every_n_epochs == 0:
            savepath = os.path.join(self.savepath, prefix + str(trainer.current_epoch))
            os.makedirs(savepath, exist_ok=True)
            print(f"saving {savepath}")

            if isinstance(pl_module.positional_encoder, pe.SphericalHarmonics):
                plot_weights(pl_module,
                             trainer.current_epoch,
                             savepath=os.path.join(savepath, "weights.pdf")
                             )

            plot_predictions(pl_module,
                             savepath=os.path.join(savepath, "predictions.pdf"))


    def on_validation_end(self, trainer, pl_module):
        self.plot_all_figures(trainer, pl_module, prefix="val_end_")

    def on_fit_start(self, trainer, pl_module):
        self.plot_all_figures(trainer, pl_module, prefix="fit_start_")
