import argparse
import time
import os

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers

from models import resnet, vgg
from utils import factorizations, data
from modules import model_module


POSSIBLE_MODELS = ['resnet18', 'resnet34', 'resnet50', 'vgg11', 'vgg16', 'vgg19']
TN_DECOMPOSITIONS = ['tucker', 'cp', 'tt']

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', type=str, choices=POSSIBLE_MODELS, default='resnet18', help='model name')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--log-dir', type=str, default='../logs', help='log directory')
    parser.add_argument('--out-dir', type=str, default='../output', help='output directory')
    parser.add_argument('--tn-decomp', type=str, default=None, choices=TN_DECOMPOSITIONS, help='tensor decomposition')
    parser.add_argument('--tn-rank', type=int, default=None, help='tensor rank')
    return parser.parse_args()

def create_model(model_name, tn_decomp=None, tn_rank=None):
    if model_name == 'resnet18':
        model = resnet.resnet18()
    elif model_name == 'resnet34':
        model = resnet.resnet34()
    elif model_name == 'resnet50':
        model = resnet.resnet50()
    elif model_name == 'vgg11':
        model = vgg.VGG('vgg11')
    elif model_name == 'vgg16':
        model = vgg.VGG('vgg16')
    elif model_name == 'vgg19':
        model = vgg.VGG('vgg19')
    else:
        raise ValueError('Unknown model name: {}'.format(model_name))
    
    if tn_decomp is not None:
        model = factorizations.factorize_network(model, tn_decomp, tn_rank)

    return model


if __name__ == '__main__':
    # reproducibility
    pl.seed_everything(42)
    # parse args
    args = parse_args()
    print(args)
    # create model
    model = create_model(args.model, args.tn_decomp, args.tn_rank)
    # init lightining module
    pl_module = model_module.Model(model, init_lr=args.lr)
    # get and prepare data
    data_dict = data.prepare_data()
    # wandb run name
    if args.tn_decomp is not None:
        log_name = f'{args.model}-{args.tn_decomp}-{args.tn_rank}-{time.time()}'
    else:
        log_name = f'{args.model}-{time.time()}'
    # init loggers
    wandb_logger = pl_loggers.WandbLogger(name=log_name, project="td-compression")
    # create log dir if not exists
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # init trainer
    trainer = pl.Trainer(
        accelerator=1,
        max_epochs=args.epochs,
        default_root_dir=args.log_dir,
        logger=wandb_logger
    )
    # train
    trainer.fit(pl_module, data_dict['train'], data_dict['val'])
    # test
    trainer.test(pl_module, data_dict['test'])

