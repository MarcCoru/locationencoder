import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from argparse import Namespace
from train import fit

def main():
    resultsdir = "results/train/checkerboard/exp_longitudinal"

    fit_models()

    for neuralnet in ["siren", "fcnet", "linear"]:
        savepath = os.path.join(resultsdir, neuralnet + ".pdf")
        plot_summary(resultsdir, neuralnet, savepath=savepath)

    plt.show()

    extract_df_to_csv(resultsdir)


def fit_models():
    for pe in ["grid", "spherecplus", "sphericalharmonics"]:
        for nn in ["fcnet", "siren"]:
            for seed in np.arange(5):
                args = Namespace(dataset='checkerboard',
                          pe=pe,
                          nn=nn,
                          save_model=False,
                          log_wandb=False,
                          hparams='hparams.yaml',
                          results_dir='results/train',
                          expname=f'exp_longitudinal/{seed}',
                          legendre_polys=None, # take from hparams file
                          seed=seed,
                          harmonics_calculation="analytic",
                          resume_ckpt_from_results_dir=False,
                          matplotlib=True,
                          matplotlib_show=False,
                          checkerboard_scale=1,
                          use_expnamehps=False,
                          max_epochs=None,
                          accelerator="cpu",
                          gpus=-1,
                          min_radius=None)

                fit(args)
    return args

def extract_df_to_csv(resultsdir):
    seeds = [s for s in os.listdir(resultsdir) if os.path.isdir(os.path.join(resultsdir, s))]
    dfs = []
    for seed in seeds:
        rdir = os.path.join(resultsdir, seed, "longitudinalaccuracy")
        runs = os.listdir(rdir)

        for run in runs:
            pe, nn = run.split("-")

            file = np.load(os.path.join(rdir, run, "histogram.npz"))
            hist_accuracy = file["hist_accuracy"]
            bin_edges = file["bin_edges"] #np.linspace(-90,90, 10)
            bin_centers = bin_edges[:-1] + np.diff(bin_edges)[0] / 2

            df = pd.DataFrame([hist_accuracy, bin_centers], index=["accuracy", "bin_center"]).T
            df["seed"] = int(seed)
            df["run"] = run

            dfs.append(df)

    mean_df = pd.concat(dfs).groupby(["run","bin_center"]).mean()["accuracy"]
    std_df = pd.concat(dfs).groupby(["run", "bin_center"]).std()["accuracy"]

    for run in mean_df.index.unique(0):
        csvfile = os.path.join(resultsdir, run + "_mean.csv")
        mean_ = mean_df.loc[run]
        mean_.to_csv(csvfile)
        print(f"writing {csvfile}")

        csvfile = os.path.join(resultsdir, run + "_std.csv")
        std_ = std_df.loc[run]
        std_.to_csv(csvfile)
        print(f"writing {csvfile}")

        csvfile = os.path.join(resultsdir, run + "_lower.csv")
        (mean_-std_).to_csv(csvfile)
        print(f"writing {csvfile}")

        csvfile = os.path.join(resultsdir, run + "_upper.csv")
        (mean_+std_).to_csv(csvfile)
        print(f"writing {csvfile}")

def plot_summary(resultsdir, neuralnet, savepath=None):

    runs = os.listdir(resultsdir)
    runs = [r for r in runs if os.path.isdir(os.path.join(resultsdir,r))]

    colors = ['#e41a1c',  # Red
              '#377eb8',  # Blue
              '#4daf4a',  # Green
              '#984ea3']  # Purple

    runs = [r for r in runs if neuralnet in r]
    legend_names = [n.replace(f"-{neuralnet}", "") for n in runs]

    N = len(runs)

    fig, ax = plt.subplots()

    for o, run, color in zip(np.linspace(-N/2, N/2,N), runs, colors):
        file = np.load(os.path.join(resultsdir, run, "histogram.npz"))
        hist_accuracy = file["hist_accuracy"]
        bin_width = file["bin_width"]

        bin_edges = file["bin_edges"] #np.linspace(-90,90, 10)
        heights = np.diff(bin_edges) * 0.9 / N

        offset = o * heights[0]/1.5

        ax.barh(bin_edges[:-1] + bin_width / 2 + offset, hist_accuracy,
                height=heights, align='center', color=color)

    ax.legend(legend_names, ncols=1, loc="lower left")

    ax.set_xlabel("accuracy")
    ax.set_ylabel("latitude")
    ax.set_yticks(bin_edges[:-1] + bin_width / 2)

    if savepath is not None:
        print(f"writing {savepath}")
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0, transparent=True)

if __name__ == '__main__':
    main()
