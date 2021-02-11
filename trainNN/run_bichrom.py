import argparse
import yaml
from subprocess import call
from train import train_bichrom

if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser(description='Train and compare BichromSEQ\
                                     and Bichrom')
    parser.add_argument('-training_schema_yaml', required=True,
                        help='YAML file with paths to train, test and val data')
    parser.add_argument('-len', help='Size of genomic windows',
                        required=True, type=int)
    parser.add_argument('-outdir', required=True, help='Output directory')
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

    train_bichrom(data_paths=data_paths, outdir=outdir, seq_len=args.window_size,
                  bin_size=10)