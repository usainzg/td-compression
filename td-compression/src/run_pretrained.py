import argparse
import time
import os

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers

from models import resnet, vgg
from utils import factorizations, data
from modules import model_module


TN_DECOMPOSITIONS = ['tucker', 'cp', 'tt']
RANK_SELECTION = ['manual', 'auto']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--pretrained-path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--log-dir', type=str, default='../logs', help='log directory')
    parser.add_argument('--out-dir', type=str, default='../output', help='output directory')
    parser.add_argument('--tn-decomp', type=str, default=None, choices=TN_DECOMPOSITIONS, help='tensor decomposition')
    parser.add_argument('--tn-rank', type=int, default=None, help='tensor rank')
    parser.add_argument('--rank-selection', type=str, default='manual', choices=RANK_SELECTION, help='rank selection')
    return parser.parse_args()

def load_module(pretrained_path):
    loaded_module= model_module.Model.load_from_checkpoint(checkpoint_path=pretrained_path)
    return loaded_module

def factorize_pretrained(model, tn_decomp, tn_rank, rank_selection):
    if tn_decomp is not None:
        model = factorizations.factorize_network(model, tn_decomp, tn_rank, rank_selection)
    return model

if __name__ == '__main__':
    # reproducibility
    pl.seed_everything(42)
    args = parse_args()
    print(args)
    # load pretrained model
    pretrained_module = load_module(args.pretrained_path)
    # factorize pretrained model
    model = factorize_pretrained(pretrained_module.model, args.tn_decomp, args.tn_rank, args.rank_selection)
    # init lightining module
    pl_module = model_module.Model(model)
    # get and prepare data
    data_dict = data.get_data(args.batch_size)
    # logger run name
    run_name = 'pretrained_{}_{}_{}_{}'.format(args.model, args.tn_decomp, args.tn_rank, args.rank_selection)
    # init logger
    wandb_logger = pl_loggers.WandbLogger(name=run_name, project='td-compression')
    # init trainer
    trainer = pl.Trainer(
        accelerator=1,
        default_root_dir=args.log_dir,
        logger=wandb_logger
    )
    # TODO: add fine-tuning
    # test
    trainer.test(pl_module, datamodule=data_dict['test'])

