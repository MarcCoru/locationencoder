from train import fit
from argparse import Namespace
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

SCALES = [0.25, 0.5, 1, 2, 3, 4, 5, 8, 16]
POLYS = [10, 20]
NN = ["linear", "siren"]

def main():
    for scale in SCALES:
        for poly in POLYS:
            for nn in NN:
                pe = 'sphericalharmonics'
                args = Namespace(dataset='checkerboard',
                          pe=pe,
                          nn=nn,
                          save_model=False,
                          log_wandb=False,
                          hparams='hparams.yaml',
                          results_dir='results/train',
                          expname=f'{pe}-{nn}-scale{scale}-{poly}poly',
                          seed=0,
                          resume_ckpt_from_results_dir=False,
                          matplotlib=True,
                          matplotlib_show=False,
                          checkerboard_scale=float(scale),
                          legendre_polys=poly,
                          use_expnamehps=False,
                          max_epochs=None,
                          accelerator="cpu",
                          gpus=-1,
                          harmonics_calculation="analytic",
                          min_radius=None)

            fit(args)

    resultsdir = os.path.join(args.results_dir, args.dataset)
    results = os.listdir(resultsdir)
    runs = [os.path.join(resultsdir, r) for r in results if os.path.isdir(os.path.join(resultsdir, r))]
    stats = []
    for run in runs:
        if len(os.path.basename(run).split("-")) != 4:
            continue

        pe, nn, scalestr, polystr = os.path.basename(run).split("-")
        scale = float(scalestr.replace("scale", ""))
        poly = int(polystr.replace("poly", ""))
        with open(os.path.join(run, f"{pe:1.8}-{nn:1.6}.json")) as f:
            stat = json.load(f)
        stats.append(
            dict(
                accuracy=stat["accuracy"],
                testloss=stat["testloss"],
                iou=stat["iou"],
                mean_dist=stat["mean_dist"],
                poly=poly,
                scale=scale,
                nn=nn,
                pe=pe
            )
        )
    df = pd.DataFrame(stats)

    fig, ax = plt.subplots()

    for nn in df.nn.unique():
        df_nn = df.loc[df.nn == nn]

        polys = df.poly.unique()
        for poly in polys:
            df_ = df_nn.loc[df_nn.poly == poly].set_index("mean_dist")["accuracy"].sort_index()
            ax.plot(df_.index, df_.values, "o-")
            ax.axvline(180 / poly)

    ax.legend(polys)
    ax.set_xlabel("mean distance between voronoi centers in degrees")
    ax.set_ylabel("accuracy")
    plt.show()

    for nn in df.nn.unique():
        for poly in df.poly.unique():
            df.loc[(df.poly == poly) & (df.nn == nn)].sort_values(by="mean_dist").to_csv(os.path.join(resultsdir, f"sphericalharmonics-{nn}-{poly}.csv"))

if __name__ == '__main__':
    main()

