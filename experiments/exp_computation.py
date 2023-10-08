from train import fit
from argparse import Namespace

import os
import json
import pandas as pd

SCALES = [1]
POLYS = [2, 5, 10, 20, 40]
NN = "linear"
PE = "sphericalharmonic"

def fit_comparison():

    # comparison
    for comparison in ["wrap", "spherecplus"]:
        for poly in POLYS:
            min_radius = int(180 / poly)
            args = Namespace(dataset='checkerboard',
                             pe=comparison,
                             nn=NN,
                             min_radius = min_radius,
                             legendre_polys=poly,
                             save_model=False,
                             log_wandb=False,
                             hparams='hparams.yaml',
                             results_dir='results/train/exp_computation',
                             expname=f"comparison/{comparison}-minr{min_radius}-poly{poly}",
                             harmonics_calculation="analytic",
                             seed=0,
                             resume_ckpt_from_results_dir=False,
                             matplotlib=True,
                             matplotlib_show=False,
                             use_expnamehps=False,
                             max_epochs=None,
                             accelerator="cpu",
                             gpus=-1,
                             checkerboard_scale=1)
            fit(args)

def fit_sh():
    for poly in POLYS:
        for calc in CALC:
            calc_name = calc.replace("-","_") # closed-form to closed_form

            pe = 'sphericalharmonics'
            args = Namespace(dataset='checkerboard',
                      pe=pe,
                      nn=NN,
                      save_model=False,
                      log_wandb=False,
                      hparams='hparams.yaml',
                      results_dir='results/train/exp_computation',
                      expname=f'{calc_name}-{PE}-{poly}poly',
                      seed=0,
                      harmonics_calculation=calc,
                      resume_ckpt_from_results_dir=False,
                      matplotlib=True,
                      matplotlib_show=False,
                      checkerboard_scale=1,
                      legendre_polys=poly,
                      min_radius=None)

            fit(args)
    return args


def extract_sh_df(results_dir, dataset):
    resultsdir = os.path.join(results_dir, dataset)
    results = os.listdir(resultsdir)
    runs = [os.path.join(resultsdir, r) for r in results if os.path.isdir(os.path.join(resultsdir, r))]
    stats = []
    for run in runs:
        if len(os.path.basename(run).split("-")) != 3:
            continue

        calc, pe, polystr = os.path.basename(run).split("-")
        poly = int(polystr.replace("poly", ""))
        with open(os.path.join(run, f"{pe:1.8}-{NN:1.6}.json")) as f:
            stat = json.load(f)
        stats.append(
            dict(
                accuracy=stat["accuracy"],
                testloss=stat["testloss"],
                iou=stat["iou"],
                harmonics_calculation=stat["harmonics_calculation"],
                test_duration=stat["test_duration"],
                train_duration=stat["test_duration"],
                mean_dist=stat["mean_dist"],
                poly=poly,
                nn=NN,
                pe=pe
            )
        )
    return pd.DataFrame(stats)

def extract_comp_df(results_dir, dataset):
    resultsdir = os.path.join(results_dir, dataset, "comparison")
    results = os.listdir(resultsdir)
    runs = [os.path.join(resultsdir, r) for r in results if os.path.isdir(os.path.join(resultsdir, r))]
    stats = []
    for run in runs:
        if len(os.path.basename(run).split("-")) != 3:
            continue

        pe, min_radius_str, polystr = os.path.basename(run).split("-")
        poly = int(polystr.replace("poly", ""))
        min_radius = int(min_radius_str.replace("minr", ""))

        with open(os.path.join(run, f"{pe:1.8}-{NN:1.6}.json")) as f:
            stat = json.load(f)
        stats.append(
            dict(
                accuracy=stat["accuracy"],
                testloss=stat["testloss"],
                iou=stat["iou"],
                harmonics_calculation=stat["harmonics_calculation"],
                test_duration=stat["test_duration"],
                train_duration=stat["train_duration"],
                mean_dist=stat["mean_dist"],
                embedding_dim=stat["embedding_dim"],
                min_radius=min_radius,
                poly=poly,
                nn=NN,
                pe=pe
            )
        )
    return pd.DataFrame(stats)


def plot_sh(df, ydim="test_duration"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    calcs = df.harmonics_calculation.unique()
    for calc in calcs:
        df_calc = df.loc[df.harmonics_calculation == calc]

        df_ = df_calc.set_index("poly")[ydim].sort_index()
        ax.plot(df_.index, df_.values, "o-")

    ax.legend(calcs)
    ax.set_xlabel("legendre polynomials")
    ax.set_ylabel(ydim)
    plt.show()

def plot_comp(df, ydim="test_duration"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    pes = df.pe.unique()
    for pe in pes:
        df_pe = df.loc[df.pe == pe]

        df_ = df_pe.set_index("poly")[ydim].sort_index()
        ax.plot(df_.index, df_.values, "o-")

    ax.legend(pes)
    ax.set_xlabel("legendre polynomials")
    ax.set_ylabel(ydim)
    plt.show()

def main():
    results_dir = 'results/train/exp_computation'
    dataset = 'checkerboard'
    # args = fit_sh()
    #fit_comparison()

    df = extract_comp_df(results_dir, dataset)
    plot_comp(df, ydim="test_duration")
    plot_comp(df, ydim="accuracy")

    for pe in df.pe.unique():
        #for poly in df.poly.unique(): (df.poly == poly) &
        df.loc[(df.pe == pe)].sort_values(by="poly").to_csv(os.path.join(results_dir, f"{pe}.csv"))


    df = extract_sh_df(results_dir, dataset)
    plot_sh(df, ydim="test_duration")
    plot_sh(df, ydim="accuracy")

    for comp in df.harmonics_calculation.unique():
        #for poly in df.poly.unique(): (df.poly == poly) &
        df.loc[(df.harmonics_calculation == comp)].sort_values(by="poly").to_csv(os.path.join(results_dir, f"sphericalharmonics-{comp}.csv"))

    #for nn in df.nn.unique():
    #    for poly in df.poly.unique():
    #        df.loc[(df.poly == poly) & (df.nn == nn)].sort_values(by="mean_dist").to_csv(os.path.join(resultsdir, f"sphericalharmonics-{nn}-{poly}.csv"))

if __name__ == '__main__':
    main()

