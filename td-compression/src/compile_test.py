import argparse

import lightning.pytorch as pl
import torch
import numpy as np

from models import resnet, vgg
from utils import factorizations, data, utils


POSSIBLE_MODELS = ["resnet18", "resnet34", "resnet50", "vgg11", "vgg16", "vgg19"]
TN_DECOMPOSITIONS = ["tucker", "cp", "tt"]
TN_IMPLEMENTATIONS = ["reconstructed", "factorized", "mobilenet"]


def parse_args():
    parser = argparse.ArgumentParser(description="model")
    parser.add_argument(
        "--model",
        type=str,
        choices=POSSIBLE_MODELS,
        default="resnet18",
        help="model name",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--tn-decomp",
        type=str,
        default="tucker",
        choices=TN_DECOMPOSITIONS,
        help="tensor decomposition",
    )
    parser.add_argument("--tn-rank", type=float, default=0.8, help="tensor rank")
    parser.add_argument(
        "--tn-implementation",
        type=str,
        default="reconstructed",
        choices=TN_IMPLEMENTATIONS,
        help="implementation",
    )
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

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def generate_data(b):
    return (
        torch.randn(b, 3, 32, 32).to(torch.float32).cuda(),
        torch.randint(10, (b,)).cuda(),
    )

def evaluate(mod, inp):
    return mod(inp)

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
        fact_model = factorizations.factorize_network(
            args.model,
            model,
            args.tn_decomp,
            args.tn_rank,
            decompose_weights=False,
            implementation=args.tn_implementation,
        )
        # count parameters of fact_model
        n_params_fact = utils.count_parameters(fact_model)
        print(f"Number of parameters (fact_model): {n_params_fact}")
        # get compression ratio
        compression_ratio = n_params / n_params_fact
        print(f"Compression ratio: {compression_ratio}")
        n_params = n_params_fact # to log n_params_fact
        print("~" * 10)
    
    N_ITERS = 10
    model = model.cuda()
    fact_model = fact_model.cuda()

    model_times = []
    fact_times = []
    for i in range(N_ITERS):
        inp = generate_data(args.batch_size)[0]
        _, model_time = timed(lambda: evaluate(model, inp))
        model_times.append(model_time)
        print(f"model eval time {i}: {model_time}")

    print("~" * 10)

    fact_times = []
    for i in range(N_ITERS):
        inp = generate_data(args.batch_size)[0]
        _, fact_time = timed(lambda: evaluate(fact_model, inp))
        fact_times.append(fact_time)
        print(f"compile eval time {i}: {fact_time}")
    print("~" * 10)

    model_med = np.median(model_times)
    fact_med = np.median(fact_times)
    speedup = model_med / fact_med
    print(f"(eval) model median: {model_med}, factorized model median: {fact_med}, speedup: {speedup}x")
    print("~" * 10)
