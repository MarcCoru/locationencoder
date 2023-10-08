import torch
import numpy as np
import os

def plot_longitudinal_accuracy(trainer, model, matplotlib=False, show=False,
    N_bins = 9, savepath=None):

    if not matplotlib and not show and savepath is None:
        print("warning plot_longitudinal_accuracy: no output will be saved, "
              "as show and savepath are both False and None")

    outputs = trainer.predict(model, dataloaders=trainer.datamodule.test_dataloader())
    logits, lonlats, labels = list(zip(*outputs))

    logits, lonlats, labels = torch.vstack(logits), torch.vstack(lonlats), torch.hstack(labels)

    #slogits, lonlats, labels = logits[:1000], lonlats[:1000], labels[:1000]
    lats = lonlats[:,1]

    (logits.argmax(1) == labels).float().mean()
    correct = (logits.argmax(1) == labels).float()

    bin_edges = np.linspace(-90,90, N_bins+1)
    bin_width = np.diff(bin_edges)[0]
    hist_correct, _ = np.histogram(lats, bins=bin_edges, weights=correct)
    hist_total, _ = np.histogram(lats, bins=bin_edges)

    hist_accuracy = hist_correct / hist_total

    if savepath is not None:
        os.makedirs(savepath, exist_ok=True)
        np.savez(os.path.join(savepath, "histogram.npz"),
                 bin_width=bin_width,
                 bin_edges=bin_edges,
                 hist_correct=hist_correct,
                 hist_total=hist_total,
                 hist_accuracy=hist_accuracy)



    if matplotlib:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.barh(bin_edges[:-1] + bin_width/2, hist_accuracy, height=np.diff(bin_edges) * 0.9, align='center')
        ax.set_xlabel("accuracy")
        ax.set_ylabel("latitude")
        ax.set_yticks(bin_edges[:-1] + bin_width/2)

        if show:
            plt.show()

        if savepath is not None:
            fig.savefig(os.path.join(savepath, "barplot.pdf"), bbox_inches="tight", pad_inches=0, transparent=True)