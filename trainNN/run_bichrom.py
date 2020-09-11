import argparse
import yaml
from subprocess import call
from train import train_bichrom

if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser(description='Train and compare BichromSEQ\
                                     and Bichrom')
    parser.add_argument('training_schema_yaml',
                        help='YAML file with paths to train, test and val data')
    parser.add_argument('window_size', help='Size of genomic windows')
    parser.add_argument('bin_size', help='size of bins for chromatin data')
    parser.add_argument('outdir', help='Output directory')
    args = parser.parse_args()

    # load the yaml file with input data paths:
    with open(args.training_schema_yaml, 'r') as f:
        try:
            data_paths = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    # create the output directory:
    outdir = args.outdir
    call(['mkdir', outdir])

    train_bichrom(data_paths=data_paths, outdir=outdir)