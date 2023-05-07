import argparse


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

if __name__ == '__main__':
    args = parse_args()
    print(args)