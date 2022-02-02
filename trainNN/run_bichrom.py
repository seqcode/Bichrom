import argparse
import yaml
from subprocess import call
from train import train_bichrom

from tensorboard import program

tracking_address = "tensorboard_logs/"

if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser(description='Train and Evaluate Bichrom')
    parser.add_argument('-training_schema_yaml', required=True,
                        help='YAML file with paths to train, test and val data')
    parser.add_argument('-len', help='Size of genomic windows',
                        required=True, type=int)
    parser.add_argument('-outdir', required=True, help='Output directory')
    parser.add_argument('-nbins', type=int, required=True, help='Number of bins')
    args = parser.parse_args()

    # enable tensorboard    
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # load the yaml file with input data paths:
    with open(args.training_schema_yaml, 'r') as f:
        try:
            data_paths = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    # create the output directory:
    outdir = args.outdir
    call(['mkdir', outdir])

    train_bichrom(data_paths=data_paths, outdir=outdir, 
                  seq_len=args.len, bin_size=int(args.len/args.nbins))