import os
import yaml
import argparse
import json

import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Timer

from data import (
    LandOceanDataModule,
    Inat2018DataModule,
    CheckerboardDataModule,
    ERA5DataModule)
from utils import (
    plot_predictions,
    plot_predictions_at_points,
    plot_longitudinal_accuracy,
    parse_resultsdir,
    count_parameters,
    find_best_checkpoint,
    set_default_if_unset,
)

from locationencoder import LocationImageEncoder, LocationEncoder

from lightning.pytorch.loggers import WandbLogger
import torch
import numpy as np
import random

#torch.set_float32_matmul_precision('medium')


def overwrite_hparams_with_args(hparams, args):
    # overwrites some hparams if specified in arguments
    if "legendre_polys" in hparams.keys() and args.legendre_polys is not None:
        hparams["legendre_polys"] = args.legendre_polys
        print(f"using legendre-polys={args.legendre_polys}, as specified in args")
    if "min_radius" in hparams.keys() and args.min_radius is not None:
        hparams["min_radius"] = args.min_radius
        print(f"using min-radius={args.min_radius}, as specified in args")
    if args.harmonics_calculation is not None:
        hparams["harmonics_calculation"] = args.harmonics_calculation
        print(f"using harmonics_calculation={args.harmonics_calculation}, as specified in args")
    if args.max_epochs is not None:
        hparams["max_epochs"] = args.max_epochs
        print(f"using max_epochs={args.max_epochs}, as specified in args")
    return hparams


def parse_args():
    parser = argparse.ArgumentParser()

    # Add your arguments here
    parser.add_argument('--dataset', default="landoceandataset", type=str, choices=["checkerboard",
                                                                                    "inat2018",
                                                                                    "landoceandataset",
                                                                                    "era5dataset"
                                                                                    ])
    parser.add_argument('--pe', default=["sphericalharmonics"], type=str, nargs='+', help='positional encoder(s)',
                        choices=["sphericalharmonics", "theory", "grid", "spherec", "spherecplus", "direct",
                                 "cartesian3d", "wrap", "spherem", "spheremplus"])
    parser.add_argument('--nn', default=["siren"], type=str, nargs='+', help='neural network(s)',
                        choices=["linear", "siren", "fcnet"])

    # optional configs
    parser.add_argument('--save-model', action="store_true", help='save model checkpoint to results-dir')
    parser.add_argument('--log-wandb', action="store_true", help='log run to wandb')
    parser.add_argument('--hparams', default="hparams.yaml", type=str, help='hypereparameter yaml')
    parser.add_argument('--results-dir', default="results/train", type=str, help='results directory')
    parser.add_argument('--expname', default=None, type=str,
                        help='experiment name. If specified, saves results in subfolder')
    parser.add_argument('--seed', default=0, type=int, help='global random seed')
    parser.add_argument('--max-epochs', default=None, type=int,
                        help='maximum number of epochs. If unset, uses value in hparams.yaml')
    parser.add_argument('--gpus', default='-1', type=int, nargs='+',
                        help='which gpus to use; if unset uses -1 which we map to auto')
    parser.add_argument('--accelerator', default='auto', type=str,
                        help='lightning accelerator')

    parser.add_argument('-r', '--resume-ckpt-from-results-dir', action="store_true",
                        help="searches through provided results dir and resumes from suitable checkpoint "
                             "that matches pe and nn")
    parser.add_argument('--matplotlib', action="store_true",
                        help="plot maps with matplotlib")
    parser.add_argument('--matplotlib-show', action="store_true",
                        help="shows matplotlib plots (can cause freezing when called remotely)")

    parser.add_argument('--use-expnamehps', default=False, type=bool,
                        help='whether expname is part of the hp file names')

    # checkerboard
    parser.add_argument('--checkerboard-scale', default=1, type=float,
                        help="scales the number of support points for the checkerboard dataset (specificed in hparams.yaml) "
                             "by this factor. This is useful to vary the scale to test different resolutions of encoders")

    # overwrite certain hparams
    parser.add_argument('--legendre-polys', default=None, type=int)
    parser.add_argument('--min-radius', default=None, type=float)
    parser.add_argument('--harmonics-calculation', default="analytic", type=str,
                        choices=["analytic", "closed-form", "discretized"],
                        help='calculation of spherical harmonics: ' +
                             'analytic uses pre-computed equations. This is exact, but works only up to degree 50, ' +
                             'closed-form uses one equation but is computationally slower (especially for high degrees)' +
                             'discretized pre-computes harmonics on a grid and interpolates these later')

    args = parser.parse_args()
    return args

def fit(args):
    positional_encoding_name = args.pe
    neural_network_name = args.nn
    dataset = args.dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.hparams) as f:
        hparams = yaml.safe_load(f)

    dataset_hparams = hparams[dataset]["dataset"]

    hparams = hparams[dataset]
    print(args)
    if args.use_expnamehps:
        if 'seed' in args.expname:
            appender_in_yaml = args.expname.split('_seed')[0]
        else:
            appender_in_yaml = args.expname
        hparams = hparams[f"{positional_encoding_name}-{neural_network_name}-{appender_in_yaml}"]
    else:
        hparams = hparams[f"{positional_encoding_name}-{neural_network_name}"]
    hparams.update(dataset_hparams)

    hparams = overwrite_hparams_with_args(hparams, args)
    hparams = set_default_if_unset(hparams, "max_radius", 360)

    if args.dataset == "landoceandataset":
        datamodule = LandOceanDataModule(batch_size=hparams["batch_size"])
    elif args.dataset == "inat2018":
        datamodule = Inat2018DataModule(hparams["inat_directory"], batch_size=hparams["batch_size"], mode="location")
    elif args.dataset == "checkerboard":
        datamodule = CheckerboardDataModule(num_samples=hparams["num_samples"],
                                            num_classes=hparams["num_classes"],
                                            num_support=int(hparams["num_support"] * args.checkerboard_scale),
                                            batch_size=hparams["batch_size"])
    elif args.dataset == 'era5dataset':
        datamodule = ERA5DataModule(batch_size=hparams["batch_size"],
                                    data_root=hparams["era5_directory"],
                                    label_key="t2m")
    elif args.dataset == 'era5dataset_multi':
        datamodule = ERA5DataModule(batch_size=hparams["batch_size"],
                                    data_root=hparams["era5_directory"],
                                    label_key=['u10', 'v10', 't2m', 'sp', 'd2m', 'ssr', 'str', 'tp'])

    if args.resume_ckpt_from_results_dir:
        resume_checkpoint = find_best_checkpoint(parse_resultsdir(args),
                                                 f"{positional_encoding_name}-{neural_network_name}",
                                                 verbose=True)
    else:
        resume_checkpoint = None

    locationencoder = LocationEncoder(
        positional_encoding_name,
        neural_network_name,
        hparams=hparams
    )

    timer = Timer()
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=hparams["patience"]),
        timer
    ]
    if args.save_model:
        callbacks += [ModelCheckpoint(
            dirpath=parse_resultsdir(args),
            monitor='val_loss',
            filename=f"{positional_encoding_name}-{neural_network_name}" + '-{val_loss:.2f}',
            save_last=False
        )]

    if args.log_wandb:
        logger = WandbLogger(project="sphericalharmonics",
                             name=f"{args.dataset}/{positional_encoding_name}-{neural_network_name}")
    else:
        logger = None

    # use GPU if it is available
    accelerator = args.accelerator
    devices = 1
    if args.gpus == -1 or args.gpus == [-1]:
        devices = 'auto'
    else:
        devices = args.gpus

    if torch.cuda.is_available():
        accelerator = 'gpu'


    print(f"using gpus: {devices}")

    trainer = pl.Trainer(
        max_epochs=hparams["max_epochs"],
        log_every_n_steps=5,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        precision=64)

    trainer.fit(model=locationencoder,
                datamodule=datamodule,
                ckpt_path=resume_checkpoint
                )

    if "landoceandataset" in dataset or dataset == "checkerboard":
        # Evaluation on test set
        testresults = trainer.test(model=locationencoder, datamodule=datamodule)
        testloss = testresults[0]["test_loss"]
        testaccuracy = testresults[0]["test_accuracy"]
        testiou = testresults[0]["test_IoU"]

        title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6}"
        resultsfile = f"{parse_resultsdir(args)}/{title}.json".replace(" ", "_").replace("%", "")
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)

        print(f"writing {resultsfile}")
        result = dict(
            iou=testiou,
            accuracy=testaccuracy,
            testloss=testloss,
            num_params=count_parameters(locationencoder),
            mean_dist=datamodule.mean_dist if hasattr(datamodule, "mean_dist") else None,
            test_duration=timer.time_elapsed("test"),
            train_duration=timer.time_elapsed("train"),
            test_samples=len(datamodule.test_dataloader().dataset),
            train_samples=len(datamodule.train_dataloader().dataset),
            embedding_dim=locationencoder.positional_encoder.embedding_dim
        )
        result.update(hparams)
        with open(resultsfile, "w") as json_file:
            json.dump(result, json_file)

        if dataset == "checkerboard":
            savepath = f"{parse_resultsdir(args)}/longitudinalaccuracy/{title}".replace(" ", "_").replace("%", "")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plot_longitudinal_accuracy(trainer, locationencoder,
                                       matplotlib=args.matplotlib, show=args.matplotlib_show,
                                       savepath=savepath)

        if args.matplotlib:

            # plotting of world map
            title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6} loss {testloss:.3f} acc {testaccuracy * 100:.2f} IoU {testiou * 100:.2f}"

            savepath = f"{parse_resultsdir(args)}/{title}.pdf".replace(" ", "_").replace("%", "")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            plot_predictions(locationencoder, title=title, show=True, savepath=savepath)
            plot_predictions(locationencoder, title=title, show=True, savepath=savepath.replace('.pdf', '.png'))

    elif dataset == "inat2018":
        ### Quantitative Comparison
        image_location_datamodule = Inat2018DataModule(hparams["inat_directory"], mode="all", batch_size=128,
                                                       num_workers=8)
        imageencoder_checkpoint = hparams["imageencoder_checkpoint"]
        locationencoder_checkpoint = find_best_checkpoint(parse_resultsdir(args),
                                                          f"{positional_encoding_name}-{neural_network_name}",
                                                          verbose=True)

        model = LocationImageEncoder(imageencoder_checkpoint=imageencoder_checkpoint,
                                     locationencoder_checkpoint=locationencoder_checkpoint,
                                     use_logits=False)

        result = pl.Trainer().test(model=model, datamodule=image_location_datamodule)

        with open(locationencoder_checkpoint.replace(".ckpt", "_inat2018_result.json"), "w") as json_file:
            json.dump(result, json_file)

        if args.matplotlib:
            ### Qualitative Maps
            from data.inat2018_loader import QUALITATIVE_SPECIES, QUALITATIVE_SPECIES_NAMES

            for species, name in zip(QUALITATIVE_SPECIES, QUALITATIVE_SPECIES_NAMES):
                class_idx = datamodule.name2id[species]

                # get samples for scatter plot
                samples, classes = datamodule.train_ds.dataset.tensors
                # samples, classes = datamodule.test_ds.tensors
                samples = samples[(classes == class_idx).bool().squeeze()]

                # plot and save figure
                savepath = os.path.join(parse_resultsdir(args), f"exp_qualitative/{args.pe}_{args.nn}/")
                os.makedirs(os.path.dirname(savepath), exist_ok=True)
                # with red sample points
                plot_predictions(model.locationencoder, plot_points=samples, class_idx=class_idx, title=name,
                                 savepath=os.path.join(savepath, name.replace(" ", "_") + ".png"))

                # without red sample points
                plot_predictions(model.locationencoder, class_idx=class_idx, title=name,
                                 savepath=os.path.join(savepath, name.replace(" ", "_") + "_nosamples" + ".png"))

    elif dataset == "era5dataset":
        # Evaluation on test set
        testresults = trainer.test(model=locationencoder, datamodule=datamodule)
        print(testresults)
        testloss = testresults[0]["test_loss"]
        testmae = testresults[0]["test_MAE"]
        testmse = testresults[0]["test_MSE"]
        testr2score = testresults[0]["test_r2"]

        title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6}"
        resultsfile = f"{parse_resultsdir(args)}/{title}.json".replace(" ", "_").replace("%", "")
        os.makedirs(os.path.dirname(resultsfile), exist_ok=True)

        print(f"writing {resultsfile}")
        result = dict(
            mae=testmae,
            mse=testmse,
            r2=testr2score,
            testloss=testloss,
            num_params=count_parameters(locationencoder),
            mean_dist=datamodule.mean_dist if hasattr(datamodule, "mean_dist") else None,
            test_duration=timer.time_elapsed("test"),
            train_duration=timer.time_elapsed("train"),
            test_samples=len(datamodule.test_dataloader().dataset),
            train_samples=len(datamodule.train_dataloader().dataset),
            embedding_dim=locationencoder.positional_encoder.embedding_dim
        )
        result.update(hparams)
        with open(resultsfile, "w") as json_file:
            json.dump(result, json_file)

        if args.matplotlib:
            # plot point predictions
            lonlats_test = datamodule.get_test_locs()

            title = f"{positional_encoding_name:1.8}-{neural_network_name:1.6} loss {testloss:.3f}"

            savepath = f"{parse_resultsdir(args)}/{title}.png".replace(" ", "_").replace("%", "")
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            era5_plot_kwargs = {'markersize': 4, 'legend': True, 'cmap': 'cool_r',
                                'legend_kwds': {'shrink': 0.6}, 'vmin': 0, 'vmax': 6.5}

            plot_predictions_at_points(locationencoder,
                                       lonlats_test,
                                       title=title,
                                       show=True,
                                       savepath=savepath,
                                       plot_kwargs=era5_plot_kwargs,
                                       lonlatscrs="4326",
                                       plot_crs='4326',
                                       )

    return locationencoder


if __name__ == '__main__':
    args = parse_args()

    positional_encoders = args.pe
    neural_networks = args.nn

    for pe in positional_encoders:
        for nn in neural_networks:
            # overwrite lists with single argument
            args.nn = nn
            args.pe = pe
            fit(args)

