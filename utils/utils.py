import os

def parse_resultsdir(args):
    if args.expname is None:
        rsdir = os.path.join(args.results_dir, args.dataset)
    else:
        rsdir = os.path.join(args.results_dir, args.dataset, args.expname)

    os.makedirs(rsdir, exist_ok=True)
    return rsdir


def find_best_checkpoint(directory, pattern, verbose=False):
    """searches a directory for checkpoints following a pattern (e.g., sphericalharmonics-siren) and returns
    the one with lowest val_loss.
    checkpoint format example: sphericalharmonics-siren-val_lossval_loss=6.69.ckpt
    """
    checkpoints = [c for c in os.listdir(directory) if c.endswith("ckpt")]
    checkpoints = [c for c in checkpoints if pattern in c]

    if len(checkpoints) == 0:
        if verbose:
            print("no suitable checkpoint found. returning None")
        return None
    else:
        if verbose:
            print(f"resuming from checkpoints in results-dir. Found candidates {' '.join(checkpoints)}")
        val_loss = [float(c.split("val_loss=")[-1].replace(".ckpt", "")) for c in checkpoints]

        # this line sorts checkpoints according to their validation loss and takes first (lowest val loss)
        resume_checkpoint = [c for _, c in sorted(zip(val_loss, checkpoints))][0]
        if verbose:
            print(f"taking: {resume_checkpoint}")

        return os.path.join(directory, resume_checkpoint)


def set_default_if_unset(hparams, key, value):
    if not key in hparams.keys():
        hparams[key] = value
    return hparams


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
