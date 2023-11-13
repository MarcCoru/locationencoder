import numpy as np
from argparse import Namespace
from train import fit
import argparse
import os
import json
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()

    # Add your arguments here
    parser.add_argument('--dataset', default="landoceandataset", type=str, choices=["landoceandataset", "inat2018", "checkerboard"])
    parser.add_argument('--pe', default=["sphericalharmonics"], type=str, nargs='+', help='positional encoder(s)', choices=["sphericalharmonics", "theory", "grid", "spherec", "spherecplus", "spherem", "spheremplus", "direct", "cartesian3d", "wrap"])
    parser.add_argument('--nn', default=["siren"], type=str, nargs='+', help='neural network(s)', choices=["linear", "siren", "fcnet", "mlp"])
    parser.add_argument('--num-seeds', default=5, type=int, help='number of random seeds')
    parser.add_argument('--gpus', default='-1', type=int, nargs='+', help='which gpus to use; if unset uses -1 which we map to auto')
    parser.add_argument('--accelerator', default='auto', type=str,
                        help='lightning accelerator')

    parser.add_argument('-r', '--resume-ckpt-from-results-dir', action="store_true",
                        help="searches through provided results dir and resumes from suitable checkpoint "
                             "that matches pe and nn")
    parser.add_argument('--no-fit', action="store_true", help='skip the fitting of the model and only compute results tables')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.no_fit:
        fit_models(args)

    expdir = os.path.join("results/train", args.dataset, "exp_quantiative")
    seeds = [s for s in os.listdir(expdir) if s.isdigit()]

    stats = []
    for seed in seeds:
        runs = [r for r in os.listdir(os.path.join(expdir, seed)) if r.endswith(".json")]

        for run in runs:
            runname, ext = os.path.splitext(run)
            if args.dataset == "inat2018":
                pe, nn = runname.split("-")[:2] # format "direct-mlp-val_loss=6.56_inat2018_result.jsom"
            else:
                pe, nn = runname.split("-") # format "direct-mlp.json"

            with open(os.path.join(expdir, seed, runname + ".json")) as f:
                stat = json.load(f)

            if isinstance(stat, list):
                stat = stat[0]

            if args.dataset == "inat2018":
                # rename some metrics to match checkerboard and landoceandataset results
                stat["accuracy"] = stat["val_acc"]

            stat.update(dict(
                pe = pe,
                nn = nn,
                seed = int(seed)
            ))
            stats.append(stat)

    df = pd.DataFrame(stats)
    csvfile = os.path.join(expdir, "runs.csv")
    df.to_csv(csvfile)

    df_mean = df[["pe", "nn", "accuracy", "seed"]].groupby(["pe", "nn"]).mean()["accuracy"]
    df_std = df[["pe", "nn", "accuracy", "seed"]].groupby(["pe", "nn"]).std()["accuracy"]

    # iterate over every entry
    cells = []
    for mean, std in zip(df_mean, df_std):
        cells.append(f"${mean*100:.1f} \pm {std*100:.1f}$")

    df_cells = pd.Series(cells)
    df_cells.name = "accuracy"
    df_cells.index = df_mean.index

    mean_table = pd.pivot_table(df_mean.reset_index(), "accuracy", "pe", "nn")
    std_table = pd.pivot_table(df_std.reset_index(), "accuracy", "pe", "nn")
    cells_table = pd.pivot(df_cells.reset_index(), values="accuracy", columns="nn", index="pe")

    csvfile = os.path.join(expdir, f"{args.dataset}_mean_table.csv")
    mean_table.to_csv(csvfile)
    print(f"writing {csvfile}")

    csvfile = os.path.join(expdir, f"{args.dataset}_std_table.csv")
    std_table.to_csv(csvfile)
    print(f"writing {csvfile}")

    texfile = os.path.join(expdir, f"{args.dataset}.tex")
    with open(texfile, "w") as f:
        print(cells_table.to_latex(), file=f)
    print(f"writing {texfile}")
    print()
    print(cells_table.to_latex())



def fit_models(args):
    dataset = args.dataset
    # ["spherec", "spherecplus", "wrap", "sphericalharmonics", "theory", "grid", "direct", "cartesian3d"]
    # ["linear", "siren", "fcnet", "mlp"]

    if not isinstance(args.pe, list):
        args.pe = list(args.pe)

    if not isinstance(args.nn, list):
        args.nn = list(args.nn)

    for pe in args.pe:
        for nn in args.nn: #
            for seed in np.arange(args.num_seeds):
                args_fit = Namespace(dataset=dataset,
                          pe=pe,
                          nn=nn,
                          save_model=True if dataset == "inat2018" else False, # inat needs checkpoints for evaluation
                          log_wandb=False,
                          hparams='hparams.yaml',
                          results_dir='results/train',
                          expname=f'exp_quantiative/{seed}',
                          legendre_polys=None, # take from hparams file
                          seed=seed,
                          harmonics_calculation="analytic",
                          resume_ckpt_from_results_dir=args.resume_ckpt_from_results_dir,
                          gpus=args.gpus,
                          use_expnamehps=False,
                          matplotlib=False,
                          matplotlib_show=False,
                          accelerator=args.accelerator,
                          checkerboard_scale=1,
                          max_epochs=None,
                          min_radius=None) # take from hparams file

                fit(args_fit)
    return args

if __name__ == '__main__':
    main()
