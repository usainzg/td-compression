import argparse
import time
import os

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.callbacks import RichModelSummary
import torch

from models import resnet, vgg
from utils import factorizations, data, utils
from modules import model_module


POSSIBLE_MODELS = ["resnet18", "resnet34", "resnet50", "vgg11", "vgg16", "vgg19"]
TN_DECOMPOSITIONS = ["tucker", "cp", "tt"]
TN_IMPLEMENTATIONS = ["reconstructed", "factorized", "mobilenet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model",
        type=str,
        choices=POSSIBLE_MODELS,
        default="resnet18",
        help="model name",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--log-dir", type=str, default="logs", help="log directory")
    parser.add_argument(
        "--out-dir", type=str, default="output", help="output directory"
    )
    parser.add_argument(
        "--tn-decomp",
        type=str,
        default=None,
        choices=TN_DECOMPOSITIONS,
        help="tensor decomposition",
    )
    parser.add_argument("--tn-rank", type=float, default=None, help="tensor rank")
    parser.add_argument(
        "--tn-implementation",
        type=str,
        default="reconstructed",
        choices=TN_IMPLEMENTATIONS,
        help="implementation",
    )
    parser.add_argument("--precision", type=int, default=32, help="precision")
    return parser.parse_args()


def create_model(
    model_name
):
    if model_name == "resnet18":
        model = resnet.ResNet18()
    elif model_name == "resnet34":
        model = resnet.ResNet34()
    elif model_name == "resnet50":
        model = resnet.ResNet50()
    elif model_name == "vgg11":
        model = vgg.VGG("vgg11")
    elif model_name == "vgg16":
        model = vgg.VGG("vgg16")
    elif model_name == "vgg19":
        model = vgg.VGG("vgg19")
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    
    return model


if __name__ == "__main__":
    NUM_WORKERS = 4
    # reproducibility
    SEED = 42
    pl.seed_everything(seed=SEED)
    # allow tf32 (TENSOR CORES)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.backends.cudnn.deterministic = True  # deterministic cudnn
    # parse args
    args = parse_args()
    print(args)
    # create model and count parameters
    model = create_model(args.model)
    # count parameters of model
    n_params = utils.count_parameters(model)
    print(f"Number of parameters (model): {n_params}")
    # factorize model
    if args.tn_decomp is not None:
        model = factorizations.factorize_network(
            args.model,
            model,
            args.tn_decomp,
            args.tn_rank,
            decompose_weights=False,
            implementation=args.tn_implementation,
        )
        # count parameters of fact_model
        n_params_fact = utils.count_parameters(model)
        print(f"Number of parameters (fact_model): {n_params_fact}")
        # get compression ratio
        compression_ratio = n_params / n_params_fact
        print(f"Compression ratio: {compression_ratio}")
        n_params = n_params_fact # to log n_params_fact
    # init lightining module
    pl_module = model_module.Model(model, init_lr=args.lr)
    # get and prepare data
    data_dict = data.prepare_data(
        batch_size=args.batch_size, num_workers=NUM_WORKERS, seed=SEED
    )
    # create log dir if not exists
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # create out dir if not exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # wandb run name
    if args.tn_decomp is not None:
        log_name = f"{args.model}-{args.tn_decomp}-{args.tn_rank}-{time.time()}"
    else:
        log_name = f"{args.model}-{time.time()}"
    # init loggers
    wandb_logger = pl_loggers.WandbLogger(
        name=log_name, project="td-compression", save_dir=args.log_dir
    )
    config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "tn_decomp": args.tn_decomp,
        "tn_rank": args.tn_rank,
        "implementation": args.tn_implementation if args.tn_decomp is not None else None,
        "n_params": n_params,
        "precision": args.precision,
        "compression_ratio": compression_ratio if args.tn_decomp is not None else None,
    }
    # update run config
    wandb_logger.experiment.config.update(config)

    # init trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=args.epochs,
        default_root_dir=args.log_dir,
        logger=wandb_logger,
        precision=args.precision,
        callbacks=RichModelSummary(max_depth=2),
    )
    # train
    trainer.fit(pl_module, data_dict["train"], data_dict["val"])
    # test
    trainer.test(pl_module, data_dict["test"])
    # save model
    trainer.save_checkpoint(
        os.path.join(args.out_dir, f"{log_name}.ckpt"), weights_only=True
    )
