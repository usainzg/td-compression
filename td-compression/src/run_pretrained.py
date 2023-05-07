import argparse

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

if __name__ == '__main__':
    args = parse_args()
    print(args)