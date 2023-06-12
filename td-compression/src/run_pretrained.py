import argparse
import time
import os

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from torchvision import models
import torch

from models import resnet, vgg
from utils import factorizations, data, utils
from modules import model_module


TN_DECOMPOSITIONS = ["tucker", "cp", "tt"]
VBMF = [0, 1]
TN_IMPLEMENTATIONS = ["reconstructed", "factorized", "mobilenet"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, default="resnet18", help="model name")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
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
    parser.add_argument(
        "--vbmf",
        type=int,
        default=0,
        choices=VBMF,
        help="rank selection",
    )
    parser.add_argument("--precision", type=int, default=32, help="precision")
    parser.add_argument("--finetune", type=int, default=0, help="finetune")
    parser.add_argument("--accelerator", type=str, default="gpu", help="accelerator")
    return parser.parse_args()


def factorize_pretrained(model, tn_decomp, tn_rank, rank_selection):
    if tn_decomp is not None:
        model = factorizations.factorize_network(
            model, tn_decomp, tn_rank, rank_selection
        )
    return model


def create_model(model_name):
    if model_name == "resnet18":
        model = resnet.ResNet18()
        model.load_state_dict(torch.load(args.out_dir + "/resnet18-pretrained.pth"))
    elif model_name == "resnet34":
        model = resnet.ResNet34()
        model.load_state_dict(torch.load(args.out_dir + "/resnet34-pretrained.pth"))
    elif model_name == "resnet50":
        model = resnet.ResNet50()
        model.load_state_dict(torch.load(args.out_dir + "/resnet50-pretrained.pth"))
    elif model_name == "vgg11":
        model = vgg.VGG("vgg11")
        model.load_state_dict(torch.load(args.out_dir + "/vgg11-pretrained.pth"))
    elif model_name == "vgg16":
        model = vgg.VGG("vgg16")
        model.load_state_dict(torch.load(args.out_dir + "/vgg16-pretrained.pth"))
    elif model_name == "vgg19":
        model = vgg.VGG("vgg19")
        model.load_state_dict(torch.load(args.out_dir + "/vgg19-pretrained.pth"))
    else:
        raise ValueError("Unknown model name: {}".format(model_name))

    if args.finetune == 0:
        model.eval()
    else:
        model.train()

    return model


if __name__ == "__main__":
    NUM_WORKERS = 4
    # reproducibility
    SEED = 42
    # reproducibility
    pl.seed_everything(42)
    # allow tf32 (TENSOR CORES)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.backends.cudnn.deterministic = True  # deterministic cudnn
    args = parse_args()
    print(args)
    # load pretrained model
    pretrained_model = create_model(args.model)
    # count parameters of pretrained model
    n_params = utils.count_parameters(pretrained_model)
    print(f"Number of parameters (pretrained_model): {n_params}")
    # factorize pretrained model
    if args.tn_decomp is not None:
        pretrained_model.eval()
        # factorize pretrained model
        pretrained_model = factorizations.factorize_network(
            args.model,
            pretrained_model,
            args.tn_decomp,
            args.tn_rank,
            vbmf=args.vbmf,
            decompose_weights=True,  # decompose weights from pretrained model
            implementation=args.tn_implementation,
        )
        # count parameters of fact_model
        n_params_fact = utils.count_parameters(pretrained_model)
        print(f"Number of parameters (fact_model): {n_params_fact}")
        # get compression ratio
        compression_ratio = n_params / n_params_fact
        print(f"Compression ratio: {compression_ratio}")
        n_params = n_params_fact  # to log n_params_fact
    # init lightining module
    pl_module = model_module.Model(pretrained_model)
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
    # logger run name
    if args.tn_decomp is not None:
        log_name = f"p-{args.model}-{args.tn_decomp}-{args.tn_rank}-{time.time()}"
    else:
        log_name = f"p-{args.model}-{time.time()}"
    # init logger
    wandb_logger = pl_loggers.WandbLogger(
        name=log_name, project="td-compression", save_dir=args.log_dir
    )
    tensorboard_logger = pl_loggers.TensorBoardLogger(
        name=log_name, save_dir=f"{args.log_dir}/tensorboard"
    )
    # log config
    config = {
        "model": args.model,
        "batch_size": args.batch_size,
        "tn_decomp": args.tn_decomp,
        "tn_rank": args.tn_rank,
        "implementation": args.tn_implementation
        if args.tn_decomp is not None
        else None,
        "n_params": n_params,
        "precision": args.precision,
        "compression_ratio": compression_ratio if args.tn_decomp is not None else 1.0,
        "model_size": n_params * 4 * 1e-6,
        "fine_tune": args.finetune,
        "pretrained": 1
    }
    # update run config
    wandb_logger.experiment.config.update(config)
    # init trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        default_root_dir=args.log_dir,
        logger=[wandb_logger, tensorboard_logger],
        precision=args.precision,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    # TODO: add fine-tuning
    # test
    trainer.test(pl_module, data_dict["test"])
